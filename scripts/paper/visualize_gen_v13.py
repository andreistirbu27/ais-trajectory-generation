#!/usr/bin/env python3
"""
visualize_gen_v13.py — Grid visualization of v13 (TrajDiff) sampling.

For a sample of val trajectories, plots a small grid showing:
  - ground truth (green, solid)
  - K retrieved neighbor routes from train (thin blue)
  - v13 sample (red dashed)
  - v13 sample with CFG guidance (purple, only if --guidance > 0)
  - great-circle baseline (orange dotted)

Optional: overlay the coarse land mask shade if --land_sdf is given.

Uses the EMA weights from the checkpoint. Safe to run after training
finishes (or once a best.pt exists mid-training, but expects the trainer
to be idle to avoid GPU contention).

Usage:
    paper/.venv/bin/python3 scripts/paper/visualize_gen_v13.py \\
        --data_npz   data/processed/trajgen_128_clean.npz \\
        --checkpoint runs/paper/v13_trajdiff_base/best.pt \\
        --knn_cache  data/processed/trajgen_128_clean_knn_k5.npz \\
        --land_sdf   data/processed/land_sdf_050deg.npz \\
        --out_dir    outputs/viz_v13 \\
        --n 12 --guidance 2.0
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.data_gen import Scaler, TrajGenScalers, build_vtype_vocab, train_val_split_gen
from src.data_gen_retrieval import RetrievalConfig, get_retrieval_trajgen_loader
from src.diffusion import EDMSchedule, edm_sample, make_endpoint_anchor
from src.land_mask import LandMask
from src.metrics_gen import great_circle_trajectory, haversine_meters
from src.model_gen_v13 import TrajDiff, TrajDiffConfig, make_uncond


VTYPE_NAME = {
    **{c: "Passenger" for c in range(60, 70)},
    **{c: "Cargo"     for c in range(70, 80)},
    **{c: "Tanker"    for c in range(80, 90)},
}


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_v13(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device)
    cfg = ck["config"]
    mdl_cfg = TrajDiffConfig(
        seq_len=cfg["seq_len"], in_dim=cfg["in_dim"],
        n_vessel_types=cfg["n_vessel_types"], vessel_emb_dim=cfg["vessel_emb_dim"],
        d_model=cfg["d_model"], n_heads=cfg["n_heads"], ffn_mult=cfg["ffn_mult"],
        n_layer=cfg["n_layer"], dropout=cfg["dropout"], p_uncond=cfg["p_uncond"],
        max_retrieved=cfg["k_retrieval"], t_retr=cfg["t_retr"],
        time_emb_dim=cfg["time_emb_dim"],
    )
    model = TrajDiff(mdl_cfg).to(device).eval()
    if "ema" in ck and ck["ema"]:
        sd = model.state_dict()
        for n, v in ck["ema"].items():
            if n in sd:
                sd[n].copy_(v)
        model.load_state_dict(sd)
    else:
        model.load_state_dict(ck["model"])

    schedule = EDMSchedule(
        sigma_min=cfg["sigma_min"], sigma_max=cfg["sigma_max"],
        sigma_data=cfg["sigma_data"], rho=cfg["rho"],
    )
    sc = ck["scalers"]
    scalers = TrajGenScalers(
        pos=Scaler(
            mean=np.asarray(sc["pos_mean"],   dtype=np.float32),
            std =np.asarray(sc["pos_std"],    dtype=np.float32),
        ),
        delta=Scaler(
            mean=np.asarray(sc["delta_mean"], dtype=np.float32),
            std =np.asarray(sc["delta_std"],  dtype=np.float32),
        ),
    )
    return model, schedule, scalers, ck.get("vtype_vocab"), cfg


def build_val_batches(args, cfg, scalers, vtype_vocab, n):
    cache = np.load(args.knn_cache)
    val_knn = cache["val_knn"]
    data = np.load(args.data_npz, allow_pickle=True)
    trajectories = data["trajectories"].astype(np.float32)
    vessel_types = data["vessel_types"].astype(np.int32)
    track_ids    = data["track_ids"]
    (train_traj, train_vt, _,
     val_traj,   val_vt,   _) = train_val_split_gen(
        trajectories, vessel_types, track_ids,
        cfg["val_frac"], cfg["seed"])
    if vtype_vocab is None:
        vtype_vocab = build_vtype_vocab(train_vt)
    retr_cfg = RetrievalConfig(
        k_retrieval=cfg["k_retrieval"],
        vtype_weight=cfg["retrieval_vtype_weight"],
        exclude_self=cfg["exclude_self"],
    )
    loader = get_retrieval_trajgen_loader(
        trajectories=val_traj, vessel_types=val_vt,
        corpus_trajectories=train_traj, knn_indices=val_knn,
        vtype_vocab=vtype_vocab, scalers=scalers, config=retr_cfg,
        batch_size=n, shuffle=False, drop_last=False,
        num_workers=0, pin_memory=False, persistent_workers=False,
    )
    return next(iter(loader)), train_traj, val_traj, val_vt


@torch.no_grad()
def sample_v13(model, schedule, scalers, batch, *, n_steps, guidance, device):
    pos_mean_t = torch.tensor(scalers.pos.mean, device=device)
    pos_std_t  = torch.tensor(scalers.pos.std,  device=device)
    x_gt  = batch["traj_norm"].permute(1, 0, 2).to(device)         # (B, T, 2)
    cond = {
        "start": batch["start_norm"].to(device),
        "end":   batch["end_norm"].to(device),
        "vessel_type": batch["vessel_type"].to(device),
        "retrieved":   batch["retrieved_norm"].to(device),
    }
    B, T, _ = x_gt.shape
    anchor_mask, anchor_values = make_endpoint_anchor(
        cond["start"], cond["end"], T, device=device)
    uncond = make_uncond(cond, device) if guidance != 0.0 else None
    x = edm_sample(
        model, (B, T, 2), cond, schedule,
        n_steps=n_steps, device=device,
        anchor_mask=anchor_mask, anchor_values=anchor_values,
        guidance_scale=guidance, uncond_cond=uncond,
    )
    x_deg = (x * pos_std_t + pos_mean_t).cpu().numpy()
    gt_deg = (x_gt * pos_std_t + pos_mean_t).cpu().numpy()
    retr_deg = (batch["retrieved_norm"] * pos_std_t.cpu() + pos_mean_t.cpu()).numpy()
    return x_deg, gt_deg, retr_deg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_npz", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--knn_cache", required=True)
    ap.add_argument("--land_sdf", default=None)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n", type=int, default=12,
                    help="Number of trajectories to visualize (≤ batch size).")
    ap.add_argument("--n_cols", type=int, default=4)
    ap.add_argument("--n_steps", type=int, default=50)
    ap.add_argument("--guidance", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = get_device()

    model, schedule, scalers, vtype_vocab, cfg = load_v13(args.checkpoint, device)
    batch, train_traj, val_traj, val_vt = build_val_batches(
        args, cfg, scalers, vtype_vocab, n=args.n)

    pred, gt, retrieved = sample_v13(
        model, schedule, scalers, batch,
        n_steps=args.n_steps, guidance=args.guidance, device=device)

    # Optional CFG sample for side-by-side comparison.
    pred_cfg = None
    if args.guidance != 0.0:
        # `sample_v13` already used `args.guidance`, so we additionally do an
        # uncond pass for comparison. Skip if guidance was already zero.
        pred_cfg = pred  # rename: the just-computed pred IS the CFG sample
        pred, _, _ = sample_v13(
            model, schedule, scalers, batch,
            n_steps=args.n_steps, guidance=0.0, device=device)

    land_mask = LandMask.load(args.land_sdf) if args.land_sdf else None

    n_rows = (args.n + args.n_cols - 1) // args.n_cols
    fig, axes = plt.subplots(n_rows, args.n_cols, figsize=(4 * args.n_cols, 4 * n_rows))
    axes = np.array(axes).reshape(-1)

    for i in range(args.n):
        ax = axes[i]
        gt_i = gt[i]
        pred_i = pred[i]

        # Bbox for this trajectory
        all_xy = np.concatenate([gt_i, pred_i, retrieved[i].reshape(-1, 2)])
        if pred_cfg is not None:
            all_xy = np.concatenate([all_xy, pred_cfg[i]])
        margin = 0.5
        x_min, x_max = all_xy[:, 0].min() - margin, all_xy[:, 0].max() + margin
        y_min, y_max = all_xy[:, 1].min() - margin, all_xy[:, 1].max() + margin

        # Light land shading if SDF provided
        if land_mask is not None:
            ax.imshow(
                (land_mask.sdf_km > 0).astype(np.float32),
                extent=(land_mask.bbox[0], land_mask.bbox[2],
                        land_mask.bbox[1], land_mask.bbox[3]),
                origin="lower", cmap="Greys", alpha=0.15, zorder=0, aspect="auto",
            )

        # Retrieved neighbors
        for k in range(retrieved[i].shape[0]):
            ax.plot(retrieved[i, k, :, 0], retrieved[i, k, :, 1],
                    color="#90caf9", linewidth=0.7, alpha=0.6, zorder=1)

        # GC baseline
        gc = great_circle_trajectory(gt_i[0], gt_i[-1], gt_i.shape[0])
        ax.plot(gc[:, 0], gc[:, 1], color="#ff9800", linestyle=":", linewidth=1.0,
                zorder=2, label="GC")

        # GT
        ax.plot(gt_i[:, 0], gt_i[:, 1], color="#2e7d32", linewidth=1.8,
                zorder=3, label="GT")

        # v13 sample
        ax.plot(pred_i[:, 0], pred_i[:, 1], color="#c62828", linestyle="--",
                linewidth=1.4, zorder=4, label=f"v13 (g=0)")
        if pred_cfg is not None:
            ax.plot(pred_cfg[i, :, 0], pred_cfg[i, :, 1], color="#6a1b9a",
                    linestyle="-", linewidth=1.2, zorder=5,
                    label=f"v13 (g={args.guidance})")

        # Start/end markers
        ax.scatter(gt_i[0, 0], gt_i[0, 1], color="#2e7d32",
                   marker="o", s=60, zorder=10, edgecolor="black", linewidth=0.5)
        ax.scatter(gt_i[-1, 0], gt_i[-1, 1], color="#c62828",
                   marker="s", s=60, zorder=10, edgecolor="black", linewidth=0.5)

        # Per-trajectory ADE annotation
        per_pt_err = haversine_meters(
            gt_i[:, 1], gt_i[:, 0], pred_i[:, 1], pred_i[:, 0])
        ade_m = float(per_pt_err.mean())

        vtype = int(val_vt[i]) if i < len(val_vt) else -1
        ax.set_title(
            f"#{i} {VTYPE_NAME.get(vtype, f'type {vtype}')}  ADE={ade_m/1000:.2f} km",
            fontsize=9)
        ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)
        ax.set_aspect("auto")
        ax.grid(alpha=0.2); ax.set_xticks([]); ax.set_yticks([])
        if i == 0:
            ax.legend(fontsize=6, loc="lower right")

    for j in range(args.n, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    out_path = os.path.join(args.out_dir, f"v13_grid_n{args.n}_g{args.guidance:g}.png")
    plt.savefig(out_path, dpi=130)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
