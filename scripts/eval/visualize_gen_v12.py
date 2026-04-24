#!/usr/bin/env python3
"""
visualize_gen_v12.py — Plot v12 cell-graph pointer rollouts.

For each val sample plots a panel with:
  - Ground truth trajectory (green)
  - K retrieved neighbors (thin grey)
  - v12 model rollout (orange)
  - Great-circle baseline (dashed red)

Optional --tiles adds a basemap (contextily).
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.data_gen import Scaler, TrajGenScalers
from src.data_gen_v12 import load_v12_splits
from src.metrics_gen import great_circle_trajectory
from src.model_gen_v12 import TrajectoryGeneratorV12, V12Config


def get_device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def cells_to_lonlat(cix, ciy, off, bbox, dlon, dlat):
    minx, _, _, maxy = bbox
    lon = minx + (cix.astype(np.float64) + 0.5) * dlon + off[..., 0] * dlon
    lat = maxy - (ciy.astype(np.float64) + 0.5) * dlat + off[..., 1] * dlat
    return np.stack([lon, lat], axis=-1).astype(np.float32)


def load_ckpt(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    args = argparse.Namespace(**ckpt["args"])
    scalers = TrajGenScalers(
        pos=Scaler(mean=ckpt["pos_mean"], std=ckpt["pos_std"]),
        delta=Scaler(mean=ckpt["delta_mean"], std=ckpt["delta_std"]),
    )
    H, W = ckpt["grid_shape"]
    cfg = V12Config(
        N_x=W, N_y=H, K=ckpt["K"],
        max_retrieved=ckpt["k_retrieval"], t_retr=ckpt["t_retr"],
        d_model=args.d_model, nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        dim_feedforward=args.dim_feedforward, dropout=args.dropout,
        k_past=ckpt.get("k_past"), num_vessel_types=ckpt["num_vessel_types"],
    )
    model = TrajectoryGeneratorV12(cfg).to(device)
    sd = ckpt["model"]
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd); model.eval()
    return model, scalers, ckpt, cfg


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--checkpoint",    required=True)
    ap.add_argument("--quantized_npz", default="data/processed/trajgen_128_cells_005deg.npz")
    ap.add_argument("--knn_cache",     default="data/processed/trajgen_128_knn_k5.npz")
    ap.add_argument("--water_graph",   default="data/processed/water_graph_005deg.npz")
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--seed",     type=int,   default=42)
    ap.add_argument("--n_plots",  type=int,   default=16)
    ap.add_argument("--out_dir",  default="outputs/viz_v12")
    ap.add_argument("--tiles",    action="store_true")
    args = ap.parse_args()

    device = get_device()
    model, scalers, ckpt, cfg = load_ckpt(args.checkpoint, device)
    vtype_vocab = ckpt["vtype_vocab"]

    g = np.load(args.water_graph, allow_pickle=True)
    water = torch.from_numpy(g["water_mask"]).bool().to(device)
    model.attach_water_mask(water)

    splits = load_v12_splits(args.quantized_npz, args.knn_cache,
                             args.val_frac, args.seed)
    val = splits["val"]; corpus = splits["train_corpus_traj"]
    meta = splits["meta"]; bbox = meta["bbox"]; dlon = meta["dlon"]; dlat = meta["dlat"]

    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(val["traj"]), size=args.n_plots, replace=False)

    # Batch rollout for selected samples
    T = val["traj"].shape[1]
    gt   = val["traj"][idx]
    knn  = val["knn"][idx]
    ncols = 4; nrows = (args.n_plots + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))
    axes = axes.flatten()

    # Forward in one batch
    with torch.no_grad():
        start_norm = torch.tensor(scalers.pos.transform(gt[:, 0]),
                                   dtype=torch.float32, device=device)
        end_norm   = torch.tensor(scalers.pos.transform(gt[:, -1]),
                                   dtype=torch.float32, device=device)
        retr_raw = corpus[knn]
        B, K_r, T_r, _ = retr_raw.shape
        retr_norm = scalers.pos.transform(retr_raw.reshape(-1, 2)).reshape(B, K_r, T_r, 2)
        retr_norm = torch.tensor(retr_norm, dtype=torch.float32, device=device)
        retr_mask = torch.zeros(B, K_r, dtype=torch.bool, device=device)

        vt_idx = [vtype_vocab.get(int(v), 0) for v in val["vt"][idx]]
        vt_tensor = torch.tensor(vt_idx, dtype=torch.long, device=device)

        start_ix = torch.tensor(val["cix"][idx, 0].astype(np.int64), device=device)
        start_iy = torch.tensor(val["ciy"][idx, 0].astype(np.int64), device=device)
        start_off = torch.tensor(val["off"][idx, 0], dtype=torch.float32, device=device)
        end_ix = torch.tensor(val["cix"][idx, -1].astype(np.int64), device=device)
        end_iy = torch.tensor(val["ciy"][idx, -1].astype(np.int64), device=device)

        out = model.generate(
            start_norm=start_norm, end_norm=end_norm, vessel_type=vt_tensor,
            retrieved_norm=retr_norm, retrieval_mask=retr_mask,
            start_ix=start_ix, start_iy=start_iy, start_offset=start_off,
            end_ix=end_ix, end_iy=end_iy, n_steps=T)

    ix_bt  = out["cell_ix"].cpu().numpy().transpose(1, 0)          # (B, L)
    iy_bt  = out["cell_iy"].cpu().numpy().transpose(1, 0)
    off_bt = out["offset"].cpu().numpy().transpose(1, 0, 2)        # (B, L, 2)
    L = ix_bt.shape[1]
    if L < T:
        pad_ix  = np.repeat(ix_bt[:, -1:],  T - L, axis=1)
        pad_iy  = np.repeat(iy_bt[:, -1:],  T - L, axis=1)
        pad_off = np.repeat(off_bt[:, -1:], T - L, axis=1)
        ix_bt  = np.concatenate([ix_bt,  pad_ix],  axis=1)
        iy_bt  = np.concatenate([iy_bt,  pad_iy],  axis=1)
        off_bt = np.concatenate([off_bt, pad_off], axis=1)

    pred = cells_to_lonlat(ix_bt, iy_bt, off_bt, bbox, dlon, dlat)

    for p in range(args.n_plots):
        ax = axes[p]
        # GT
        ax.plot(gt[p, :, 0], gt[p, :, 1], color="#2e7d32", linewidth=2, label="GT")
        ax.scatter(gt[p, 0, 0], gt[p, 0, 1], color="#2e7d32", marker="o", s=40, zorder=5)
        ax.scatter(gt[p, -1, 0], gt[p, -1, 1], color="#2e7d32", marker="s", s=40, zorder=5)
        # Retrieved neighbors
        for k in range(min(5, corpus[knn[p]].shape[0])):
            r = corpus[knn[p, k]]
            ax.plot(r[:, 0], r[:, 1], color="#999999", linewidth=0.6, alpha=0.5)
        # v12 prediction
        ax.plot(pred[p, :, 0], pred[p, :, 1], color="#e65100", linewidth=1.6,
                label="v12", alpha=0.9)
        # Great-circle
        gc = great_circle_trajectory(gt[p, 0], gt[p, -1], T)
        ax.plot(gc[:, 0], gc[:, 1], color="#c62828", linewidth=1.0,
                linestyle="--", label="GC", alpha=0.7)

        ax.set_title(f"sample {idx[p]}  (n_steps={int(out['length'][p])})", fontsize=9)
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=0.2)
        if p == 0: ax.legend(fontsize=7, loc="best")

        if args.tiles:
            try:
                import contextily as ctx
                ctx.add_basemap(ax, crs="EPSG:4326",
                                 source=ctx.providers.OpenStreetMap.Mapnik)
            except Exception as e:
                print(f"  tiles failed for panel {p}: {e}")

    for p in range(args.n_plots, nrows * ncols):
        axes[p].axis("off")

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "trajectories.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
