#!/usr/bin/env python3
"""
eval_gen_v13.py — v13 (TrajDiff) evaluation: EDM sampling + CFG sweep
+ optional land guidance.

Loads a checkpoint saved by `train_gen_v13.py`, samples trajectories using
the EDM DDIM-style sampler from `src/diffusion.py`, and reports the
standard metrics suite used for v10 (ADE/FDE/endpoint, length-bucketed
ADE, land%@10km, cross%@10km, GC baseline).

Usage (single config):
    paper/.venv/bin/python3 scripts/paper/eval_gen_v13.py \\
        --data_npz   data/processed/trajgen_128_clean.npz \\
        --checkpoint runs/paper/v13_trajdiff_base/best.pt \\
        --knn_cache  data/processed/trajgen_128_clean_knn_k5.npz \\
        --land_sdf   data/processed/land_sdf_050deg.npz \\
        --n_eval 500 --n_steps 50 --guidance 0.0 --land_strength 0.0

CFG sweep (compare different guidance scales on the same val subset):
    paper/.venv/bin/python3 scripts/paper/eval_gen_v13.py \\
        --data_npz   data/processed/trajgen_128_clean.npz \\
        --checkpoint runs/paper/v13_trajdiff_base/best.pt \\
        --knn_cache  data/processed/trajgen_128_clean_knn_k5.npz \\
        --land_sdf   data/processed/land_sdf_050deg.npz \\
        --n_eval 500 --n_steps 50 \\
        --sweep_guidance 0.0,1.0,2.0,3.0 \\
        --out_csv results/v13_cfg_sweep.csv

Land guidance (additive correction to x_0 during sampling, pushing predicted
points off the SDF land surface). Strength is on top of the unconditional
diffusion score:
    paper/.venv/bin/python3 scripts/paper/eval_gen_v13.py \\
        --data_npz   data/processed/trajgen_128_clean.npz \\
        --checkpoint runs/paper/v13_trajdiff_base/best.pt \\
        --knn_cache  data/processed/trajgen_128_clean_knn_k5.npz \\
        --land_sdf   data/processed/land_sdf_050deg.npz \\
        --n_eval 500 --land_strength 0.5
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.data_gen import Scaler, TrajGenScalers, build_vtype_vocab, train_val_split_gen
from src.data_gen_retrieval import (
    RetrievalConfig, get_retrieval_trajgen_loader,
)
from src.diffusion import EDMSchedule, edm_sample, make_endpoint_anchor
from src.land_mask import LandMask
from src.metrics_gen import evaluate_generation, evaluate_great_circle_baseline
from src.model_gen_v13 import TrajDiff, TrajDiffConfig, make_uncond


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_checkpoint(path: str, device: torch.device):
    """Load v13 ckpt, prefer EMA weights for sampling."""
    ckpt = torch.load(path, map_location=device)
    cfg = ckpt["config"]

    mdl_cfg = TrajDiffConfig(
        seq_len=cfg["seq_len"], in_dim=cfg["in_dim"],
        n_vessel_types=cfg["n_vessel_types"], vessel_emb_dim=cfg["vessel_emb_dim"],
        d_model=cfg["d_model"], n_heads=cfg["n_heads"], ffn_mult=cfg["ffn_mult"],
        n_layer=cfg["n_layer"], dropout=cfg["dropout"], p_uncond=cfg["p_uncond"],
        max_retrieved=cfg["k_retrieval"], t_retr=cfg["t_retr"],
        time_emb_dim=cfg["time_emb_dim"],
    )
    model = TrajDiff(mdl_cfg).to(device)
    # Prefer EMA weights for sampling — they're what the paper reports.
    if "ema" in ckpt and ckpt["ema"]:
        sd = model.state_dict()
        for n, v in ckpt["ema"].items():
            if n in sd:
                sd[n].copy_(v)
        model.load_state_dict(sd)
    else:
        model.load_state_dict(ckpt["model"])
    model.eval()

    schedule = EDMSchedule(
        sigma_min=cfg["sigma_min"], sigma_max=cfg["sigma_max"],
        sigma_data=cfg["sigma_data"], rho=cfg["rho"],
    )

    sc = ckpt["scalers"]
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

    return model, schedule, scalers, ckpt.get("vtype_vocab"), cfg


def build_val_loader(args, cfg, scalers, vtype_vocab):
    if not os.path.exists(args.knn_cache):
        raise FileNotFoundError(
            f"KNN cache not found: {args.knn_cache}. "
            "Build it with build_and_cache_knn."
        )
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

    return get_retrieval_trajgen_loader(
        trajectories=val_traj, vessel_types=val_vt,
        corpus_trajectories=train_traj, knn_indices=val_knn,
        vtype_vocab=vtype_vocab, scalers=scalers, config=retr_cfg,
        batch_size=args.batch_size, shuffle=False, drop_last=False,
        num_workers=0, pin_memory=True, persistent_workers=False,
    )


def make_land_guidance(land_mask: LandMask, scalers: TrajGenScalers,
                       strength: float, threshold_km: float, device: torch.device):
    """Return a callable `(x_0_hat, sigma) -> x_0_hat_corrected` that nudges
    points off land at sampling time. x_0_hat is in normalized-position space.
    """
    pos_mean = torch.tensor(scalers.pos.mean, device=device, dtype=torch.float32)
    pos_std  = torch.tensor(scalers.pos.std,  device=device, dtype=torch.float32)

    def guidance(x_norm: torch.Tensor, sigma):
        if strength == 0.0:
            return x_norm
        # Denormalize → raw degrees. Under @torch.no_grad on edm_sample, we
        # locally re-enable autograd to differentiate the SDF penalty.
        with torch.enable_grad():
            x_deg = x_norm.detach() * pos_std + pos_mean        # (B, T, 2)
            x_deg_g = x_deg.detach().clone().requires_grad_(True)
            sdf_g = land_mask.sample_km_torch(x_deg_g[..., 0], x_deg_g[..., 1])
            penalty = torch.relu(sdf_g - threshold_km).pow(2).sum()
            (grad,) = torch.autograd.grad(penalty, x_deg_g, retain_graph=False)
        # `grad` is in km²/degree — large and dimensionally wrong as a
        # correction. Normalize to a unit direction (in degree space), then
        # interpret `strength` as kilometres-of-water-movement per sampling
        # step. Conversion: 1 km lat ≈ 1/111 deg; 1 km lon ≈ 1/(111·cos(lat))
        # deg, but the gradient already encodes the right anisotropy, so we
        # just normalize the gradient itself.
        grad_mag = grad.pow(2).sum(dim=-1, keepdim=True).clamp_min(1e-8).sqrt()
        direction_deg = grad / grad_mag                          # unit vec in deg-space
        active = (grad_mag > 0).to(direction_deg.dtype)         # zero out null-grad points
        km_per_deg = 111.0                                       # rough; lon scaled by pos_std anyway
        corr_deg = direction_deg * (strength / km_per_deg) * active
        corr_norm = corr_deg / pos_std
        return x_norm - corr_norm

    return guidance


@torch.no_grad()
def sample_v13(model, schedule, scalers, batches, *, n_steps, guidance,
               land_guidance, device):
    """Sample full trajectories for a list of pre-loaded batches.

    Returns (pred_deg (N, T, 2), gt_deg (N, T, 2))."""
    pos_mean_t = torch.tensor(scalers.pos.mean, device=device)
    pos_std_t  = torch.tensor(scalers.pos.std,  device=device)

    preds, gts = [], []
    for batch in batches:
        start = batch["start_norm"].to(device)
        end   = batch["end_norm"].to(device)
        vtype = batch["vessel_type"].to(device)
        retr  = batch["retrieved_norm"].to(device)
        x_gt  = batch["traj_norm"].permute(1, 0, 2).to(device)   # (B, T, 2)

        B, T, _ = x_gt.shape
        cond = {"start": start, "end": end, "vessel_type": vtype, "retrieved": retr}
        anchor_mask, anchor_values = make_endpoint_anchor(start, end, T, device=device)
        uncond = make_uncond(cond, device) if guidance != 0.0 else None

        # Re-enable grad for the land-guidance callable when it's active.
        if land_guidance is not None:
            with torch.enable_grad():
                x_pred_norm = edm_sample(
                    model, (B, T, 2), cond, schedule,
                    n_steps=n_steps, device=device,
                    anchor_mask=anchor_mask, anchor_values=anchor_values,
                    guidance_scale=guidance, uncond_cond=uncond,
                    score_guidance=land_guidance,
                )
        else:
            x_pred_norm = edm_sample(
                model, (B, T, 2), cond, schedule,
                n_steps=n_steps, device=device,
                anchor_mask=anchor_mask, anchor_values=anchor_values,
                guidance_scale=guidance, uncond_cond=uncond,
            )

        x_pred_deg = x_pred_norm * pos_std_t + pos_mean_t
        x_gt_deg   = x_gt        * pos_std_t + pos_mean_t

        preds.append(x_pred_deg.cpu().numpy())
        gts.append(x_gt_deg.cpu().numpy())

    return np.concatenate(preds, 0), np.concatenate(gts, 0)


def length_bucket_ade(pred_deg, gt_deg):
    """Match v10's length-bucketed ADE: short <50km, medium 50–500km, long >500km."""
    from src.metrics_gen import haversine_meters, path_length_m
    N = pred_deg.shape[0]
    pe = haversine_meters(gt_deg[:, :, 1], gt_deg[:, :, 0],
                          pred_deg[:, :, 1], pred_deg[:, :, 0])
    per_traj = pe.mean(axis=1)
    lengths_m = np.array([path_length_m(gt_deg[i]) for i in range(N)])
    lengths_km = lengths_m / 1000.0
    out = {}
    for name, lo, hi in (("short_<50km", 0, 50),
                         ("medium_50_500km", 50, 500),
                         ("long_>500km", 500, np.inf)):
        m = (lengths_km >= lo) & (lengths_km < hi)
        out[name] = (int(m.sum()), float(per_traj[m].mean()) if m.any() else float("nan"))
    return out


def land_stats(traj_deg, land_mask: LandMask, threshold_km: float = 10.0):
    """traj_crossing_rate (cross%@threshold) + on-land fraction (land%@threshold)."""
    return land_mask.trajectory_stats(traj_deg, threshold_km=threshold_km)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_npz", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--knn_cache", required=True)
    ap.add_argument("--land_sdf", required=True)
    ap.add_argument("--n_eval", type=int, default=500)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--n_steps", type=int, default=50)
    ap.add_argument("--guidance", type=float, default=0.0)
    ap.add_argument("--land_strength", type=float, default=0.0)
    ap.add_argument("--land_threshold_km", type=float, default=10.0)
    ap.add_argument("--no_frechet", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sweep_guidance", type=str, default=None,
                    help="Comma-separated guidance scales, e.g. '0,1,2,3'.")
    ap.add_argument("--sweep_land", type=str, default=None,
                    help="Comma-separated land-guidance strengths, e.g. '0,0.25,0.5'.")
    ap.add_argument("--out_csv", type=str, default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = get_device()
    print(f"Device: {device}")

    model, schedule, scalers, vtype_vocab, train_cfg = load_checkpoint(args.checkpoint, device)
    print(f"Loaded v13 from {args.checkpoint}")
    print(f"  trained sigma=({train_cfg['sigma_min']}, {train_cfg['sigma_max']}), "
          f"d_model={train_cfg['d_model']}, layers={train_cfg['n_layer']}")

    val_loader = build_val_loader(args, train_cfg, scalers, vtype_vocab)

    # Pre-load batches up to n_eval so we sweep over the same subset across configs.
    batches = []
    n_so_far = 0
    for batch in val_loader:
        batches.append(batch)
        n_so_far += batch["start_norm"].shape[0]
        if n_so_far >= args.n_eval:
            break
    print(f"Evaluating on {n_so_far} val trajectories")

    land_mask = LandMask.load(args.land_sdf).as_torch(device=device)

    # Build configurations to sweep.
    g_list = [float(g) for g in args.sweep_guidance.split(",")] if args.sweep_guidance else [args.guidance]
    l_list = [float(l) for l in args.sweep_land.split(",")] if args.sweep_land else [args.land_strength]

    rows = []
    for guidance in g_list:
        for l_strength in l_list:
            land_guidance = make_land_guidance(
                land_mask, scalers, l_strength,
                args.land_threshold_km, device,
            ) if l_strength != 0.0 else None

            t0 = time.time()
            pred_deg, gt_deg = sample_v13(
                model, schedule, scalers, batches,
                n_steps=args.n_steps, guidance=guidance,
                land_guidance=land_guidance, device=device,
            )
            dt = time.time() - t0

            metrics = evaluate_generation(pred_deg, gt_deg,
                                          compute_frechet=not args.no_frechet)
            buckets = length_bucket_ade(pred_deg, gt_deg)
            lstats_pred = land_stats(pred_deg, land_mask, args.land_threshold_km)
            lstats_gt   = land_stats(gt_deg,   land_mask, args.land_threshold_km)

            print()
            print(f"=== guidance={guidance:.2f}  land_strength={l_strength:.3f}  "
                  f"(sample={dt:.1f}s) ===")
            print(f"  ADE        {metrics['ade_m']:.0f} m")
            print(f"  FDE        {metrics['fde_m']:.0f} m")
            print(f"  endpoint   {metrics['endpoint_error_m']:.0f} m")
            print(f"  path-ratio {metrics['path_length_ratio']:.3f}")
            print(f"  norm_ADE   {metrics['normalized_ade']:.5f}")
            for k, (n, ade) in buckets.items():
                print(f"  bucket {k:18s}  n={n:4d}  ADE={ade:.0f} m")
            print(f"  land%@{args.land_threshold_km:g}km: "
                  f"pred {lstats_pred['land_frac']*100:.2f} | gt {lstats_gt['land_frac']*100:.2f}")
            print(f"  cross%@{args.land_threshold_km:g}km: "
                  f"pred {lstats_pred['traj_crossing_rate']*100:.2f} | "
                  f"gt {lstats_gt['traj_crossing_rate']*100:.2f}")

            row = {
                "checkpoint": args.checkpoint,
                "n_eval": gt_deg.shape[0],
                "n_steps": args.n_steps,
                "guidance": guidance,
                "land_strength": l_strength,
                "ade_m": metrics["ade_m"],
                "fde_m": metrics["fde_m"],
                "endpoint_m": metrics["endpoint_error_m"],
                "path_ratio": metrics["path_length_ratio"],
                "norm_ade": metrics["normalized_ade"],
                "ade_short": buckets["short_<50km"][1],
                "n_short":   buckets["short_<50km"][0],
                "ade_medium": buckets["medium_50_500km"][1],
                "n_medium":   buckets["medium_50_500km"][0],
                "ade_long": buckets["long_>500km"][1],
                "n_long":   buckets["long_>500km"][0],
                "land_pct_pred": lstats_pred["land_frac"] * 100,
                "land_pct_gt":   lstats_gt["land_frac"] * 100,
                "cross_pct_pred": lstats_pred["traj_crossing_rate"] * 100,
                "cross_pct_gt":   lstats_gt["traj_crossing_rate"] * 100,
                "sample_time_s": dt,
            }
            rows.append(row)

    # GC baseline once.
    gc = evaluate_great_circle_baseline(gt_deg)
    print(f"\nGreat-circle baseline ADE: {gc['ade_m']:.0f} m  FDE: {gc['fde_m']:.0f} m")

    if args.out_csv:
        os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
        with open(args.out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nWrote sweep results → {args.out_csv}")


if __name__ == "__main__":
    main()
