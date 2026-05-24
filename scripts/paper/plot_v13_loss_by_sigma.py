#!/usr/bin/env python3
"""
plot_v13_loss_by_sigma.py — Per-sigma-bin EDM loss diagnostic for v13.

Why this exists
---------------
The EDM training loss reported in metrics.csv is an average over a log-normal
distribution of noise levels (sigma_sample_mean=-1.2, sigma_sample_std=1.2 by
default). When training collapses to fitting *only* the easy low-sigma samples
(near-clean reconstruction) and ignores the hard high-sigma samples, the
*average* loss still looks good but the model can't denoise from pure noise at
sampling time — ADE blows up.

This script samples a fixed val mini-batch, evaluates the denoiser at multiple
sigmas (in log-spaced bins from sigma_min to sigma_max), and plots the
per-bin EDM-weighted MSE. A healthy training curve is roughly flat or only
mildly increasing across bins; a collapsed model shows a tight low-sigma well
with the curve exploding upward past sigma ~ 0.5.

Run mid-training (uses the checkpoint at runs/paper/v13_trajdiff_base/best.pt
without touching the run directory in any other way) or after training
finishes.

Usage:
    paper/.venv/bin/python3 scripts/paper/plot_v13_loss_by_sigma.py \\
        --data_npz   data/processed/trajgen_128_clean.npz \\
        --checkpoint runs/paper/v13_trajdiff_base/best.pt \\
        --knn_cache  data/processed/trajgen_128_clean_knn_k5.npz \\
        --n_samples 64 --n_bins 16 \\
        --out runs/paper/v13_trajdiff_base/loss_by_sigma.png
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
from src.data_gen_retrieval import (
    RetrievalConfig, get_retrieval_trajgen_loader,
)
from src.diffusion import EDMSchedule, edm_preconditioning, make_endpoint_anchor
from src.model_gen_v13 import TrajDiff, TrajDiffConfig


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_v13(ckpt_path: str, device: torch.device):
    """Load v13 weights (prefer EMA) and return (model, schedule, scalers,
    vtype_vocab, train_cfg)."""
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


def build_val_batch(args, cfg, scalers, vtype_vocab):
    """Pull a single mini-batch from the val set."""
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
        batch_size=args.n_samples, shuffle=False, drop_last=False,
        num_workers=0, pin_memory=False, persistent_workers=False,
    )
    return next(iter(loader))


@torch.no_grad()
def per_bin_loss(model, schedule, batch, device, *, n_bins: int, n_repeats: int):
    """Evaluate EDM loss at log-spaced sigma bins.

    Returns (sigmas, weighted_losses, raw_mse). The weighted loss is what
    the training loop minimizes (and what naturally falls toward high σ
    because of EDM's loss-weight function); raw_mse is the unweighted
    denoising error and is the right quantity to check for the
    "easy-noise-collapse" failure mode (raw MSE should be roughly flat or
    only gently increasing with σ for a healthy model).
    """
    x = batch["traj_norm"].permute(1, 0, 2).to(device)         # (B, T, 2)
    cond = {
        "start": batch["start_norm"].to(device),
        "end":   batch["end_norm"].to(device),
        "vessel_type": batch["vessel_type"].to(device),
        "retrieved":   batch["retrieved_norm"].to(device),
    }
    B, T, _ = x.shape
    anchor_mask, _ = make_endpoint_anchor(
        cond["start"], cond["end"], T, device=device)

    # Log-spaced sigmas from schedule.sigma_min to sigma_max.
    sigmas = torch.tensor(
        np.exp(np.linspace(np.log(schedule.sigma_min),
                           np.log(schedule.sigma_max), n_bins)),
        dtype=torch.float32, device=device,
    )
    losses     = np.zeros(n_bins, dtype=np.float64)   # weighted (training-loss form)
    raw_mse    = np.zeros(n_bins, dtype=np.float64)   # unweighted denoising error
    for i, sigma_val in enumerate(sigmas.tolist()):
        accum_w = 0.0
        accum_r = 0.0
        for _ in range(n_repeats):
            sigma = torch.full((B,), sigma_val, device=device, dtype=x.dtype)
            sigma_b = sigma.view(B, 1, 1)
            eps = torch.randn_like(x)
            x_noisy = x + sigma_b * eps
            x_noisy = torch.where(anchor_mask.unsqueeze(-1), x, x_noisy)
            c_skip, c_out, c_in, c_noise = edm_preconditioning(sigma, schedule.sigma_data)
            c_skip_b = c_skip.view(B, 1, 1)
            c_out_b  = c_out.view(B, 1, 1)
            c_in_b   = c_in.view(B, 1, 1)
            F = model(c_in_b * x_noisy, c_noise, cond)
            D = c_skip_b * x_noisy + c_out_b * F
            w = schedule.loss_weight(sigma).view(B, 1, 1)
            err = (D - x) ** 2
            err = err * (~anchor_mask).unsqueeze(-1).to(err.dtype)
            accum_w += float((w * err).mean().item())
            accum_r += float(err.mean().item())
        losses[i]  = accum_w / n_repeats
        raw_mse[i] = accum_r / n_repeats
    return sigmas.cpu().numpy(), losses, raw_mse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_npz", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--knn_cache", required=True)
    ap.add_argument("--n_samples", type=int, default=64,
                    help="Val batch size for the diagnostic.")
    ap.add_argument("--n_bins", type=int, default=16,
                    help="Number of log-spaced sigma bins.")
    ap.add_argument("--n_repeats", type=int, default=4,
                    help="Noise resamples per bin (averages out the per-eps variance).")
    ap.add_argument("--out", default=None,
                    help="Output PNG; defaults to loss_by_sigma.png next to the checkpoint.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = get_device()
    print(f"Device: {device}")
    model, schedule, scalers, vtype_vocab, train_cfg = load_v13(args.checkpoint, device)
    print(f"Loaded v13 from {args.checkpoint}")
    print(f"  sigma_min={schedule.sigma_min}  sigma_max={schedule.sigma_max}  "
          f"sigma_data={schedule.sigma_data}  rho={schedule.rho}")
    print(f"  Training sample distribution: lognormal(mean={train_cfg['sigma_sample_mean']}, "
          f"std={train_cfg['sigma_sample_std']})")

    batch = build_val_batch(args, train_cfg, scalers, vtype_vocab)
    print(f"Val batch: B={args.n_samples} T={batch['traj_norm'].shape[0]}")

    sigmas, losses, raw_mse = per_bin_loss(
        model, schedule, batch, device,
        n_bins=args.n_bins, n_repeats=args.n_repeats,
    )

    out_path = args.out or os.path.join(
        os.path.dirname(args.checkpoint) or ".", "loss_by_sigma.png")

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Left: weighted training-form loss (what training optimizes — falls
    # toward high σ by construction; useful as a sanity check).
    ax.plot(sigmas, losses, marker="o", color="#1565c0", linewidth=1.8)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("sigma (noise level)")
    ax.set_ylabel("EDM-weighted MSE on val mini-batch")
    ax.set_title("Training-form loss")
    ax.grid(alpha=0.3, which="both")

    # Right: raw (unweighted) denoising MSE — the actually-informative
    # diagnostic. Healthy = roughly flat or gently increasing with σ.
    # Easy-noise-collapse = flat at low σ, spike at high σ.
    ax2.plot(sigmas, raw_mse, marker="o", color="#c62828", linewidth=1.8)
    ax2.set_xscale("log"); ax2.set_yscale("log")
    ax2.set_xlabel("sigma (noise level)")
    ax2.set_ylabel("raw MSE  (D(x_noisy) - x)²")
    ax2.set_title("Unweighted denoising MSE  ← the collapse diagnostic")
    ax2.grid(alpha=0.3, which="both")

    # Mark the training sigma distribution's bulk on both panels.
    s_mean = float(np.exp(train_cfg["sigma_sample_mean"]))
    s_lo   = float(np.exp(train_cfg["sigma_sample_mean"] - train_cfg["sigma_sample_std"]))
    s_hi   = float(np.exp(train_cfg["sigma_sample_mean"] + train_cfg["sigma_sample_std"]))
    for a in (ax, ax2):
        a.axvspan(s_lo, s_hi, color="#cccccc", alpha=0.3,
                  label=f"train sigmas (1σ): [{s_lo:.3f}, {s_hi:.3f}]")
        a.axvline(s_mean, color="#666", linestyle="--", linewidth=0.8)
        a.legend(loc="best", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    print(f"\nWrote {out_path}")

    # Print the numeric breakdown so the user can see the curve shape.
    print()
    print(f"{'sigma':>10}  {'weighted':>10}  {'raw_MSE':>10}")
    for s, L, R in zip(sigmas, losses, raw_mse):
        print(f"{s:>10.4f}  {L:>10.5f}  {R:>10.6f}")

    # Heuristic flag on the RAW MSE (the right quantity): is the highest-σ
    # quarter much worse than the lowest-σ quarter? Healthy: ratio ≲ 5×.
    lo = raw_mse[:max(1, len(raw_mse)//4)].mean()
    hi = raw_mse[-max(1, len(raw_mse)//4):].mean()
    ratio = hi / lo if lo > 0 else float("inf")
    print(f"\nhigh-sigma / low-sigma loss ratio: {ratio:.1f}")
    if ratio > 10:
        print("  WARNING: this is the easy-noise-collapse pattern. The model "
              "is fitting low-sigma well but failing at high-sigma denoising — "
              "expect bad sampling-time ADE. Try raising sigma_sample_mean "
              "(e.g. -0.4) or sigma_sample_std (e.g. 1.6).")
    else:
        print("  Curve looks healthy.")


if __name__ == "__main__":
    main()
