#!/usr/bin/env python3
"""
diagnose_high_mse.py — Isolate high-MSE batches for data quality analysis.

Loads a trained checkpoint, re-runs inference with shuffle=False, and flags
batches where MSE exceeds a threshold. For each flagged batch the worst
samples are saved to a CSV and plotted so you can inspect the raw positions
and displacements.

Usage:
    python3 scripts/diagnose_high_mse.py \\
        --csv data/processed/AIS_2024_sample.csv \\
        --checkpoint runs/ais_transformer/best.pt \\
        --threshold 1.0 \\
        --out_dir outputs/high_mse
"""

import argparse
import csv
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.data import (CausalDataset, Scaler, Scalers, causal_collate,
                      load_tracks)
from src.model import AISTransformer


# ── Checkpoint loading (mirrors visualize.py) ─────────────────────────────────

def load_checkpoint(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    a = ckpt["args"]
    scalers = Scalers(
        pos   = Scaler(mean=ckpt["pos_mean"],   std=ckpt["pos_std"]),
        logdt = Scaler(mean=ckpt["logdt_mean"], std=ckpt["logdt_std"]),
        disp  = Scaler(mean=ckpt["disp_mean"],  std=ckpt["disp_std"]),
    )
    input_dim = 5 if a.get("use_velocity", True) else 3
    model = AISTransformer(
        input_dim=input_dim,
        d_model=a.get("d_model", 128),
        nhead=a.get("nhead", 8),
        num_layers=a.get("num_layers", 2),
        dropout=0.0,
        pred_mode=a.get("pred_mode", "causal"),
        num_vessel_types=ckpt.get("num_vessel_types", 0),
        vessel_type_embed_dim=8,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, scalers, ckpt.get("vtype_vocab", {0: 0}), a, ckpt.get("epoch", "?")


# ── Helpers ───────────────────────────────────────────────────────────────────

def approx_km(dlon, dlat, lat_deg):
    """Rough displacement in km (flat-earth)."""
    dx = dlon * 111.0 * math.cos(math.radians(float(lat_deg)))
    dy = dlat * 111.0
    return math.sqrt(dx**2 + dy**2)


def plot_batch(batch_idx, sample_rows, out_path):
    """Plot up to 4 worst samples in a flagged batch."""
    n = min(len(sample_rows), 4)
    ncols = min(n, 4)
    fig, axes = plt.subplots(1, ncols, figsize=(ncols * 5, 4.5))
    if ncols == 1:
        axes = [axes]

    for i, row in enumerate(sample_rows[:ncols]):
        ax = axes[i]
        lons = row["lons"]
        lats = row["lats"]

        # Draw track
        for j in range(len(lons) - 1):
            t = j / max(len(lons) - 2, 1)
            ax.plot(lons[j:j+2], lats[j:j+2],
                    color=plt.cm.Blues(0.3 + 0.7 * t), linewidth=0.9, alpha=0.7)

        last_lon, last_lat = lons[-1], lats[-1]

        true_next = (last_lon + row["true_dlon"], last_lat + row["true_dlat"])
        pred_next = (last_lon + row["pred_dlon"], last_lat + row["pred_dlat"])

        ax.scatter(last_lon, last_lat, s=70, color="navy", zorder=6, marker="o")
        ax.scatter(*true_next, s=150, color="#2e7d32", zorder=8, marker="*")
        ax.scatter(*pred_next, s=90,  color="#c62828", zorder=7, marker="^")

        def arrow(src, dst, color):
            ax.annotate("", xy=dst, xytext=src,
                        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5, alpha=0.85))

        arrow((last_lon, last_lat), true_next, "#2e7d32")
        arrow((last_lon, last_lat), pred_next, "#c62828")

        err_km = approx_km(
            row["pred_dlon"] - row["true_dlon"],
            row["pred_dlat"] - row["true_dlat"],
            last_lat,
        )
        true_km = approx_km(row["true_dlon"], row["true_dlat"], last_lat)

        ax.set_title(f"{row['vessel_id']}  step {row['start_step']}", fontsize=8)
        ax.text(0.02, 0.02,
                f"sample_mse={row['sample_mse']:.3f}\n"
                f"pred_err≈{err_km:.1f}km  true_disp≈{true_km:.1f}km",
                transform=ax.transAxes, fontsize=7,
                color="white", bbox=dict(fc="#000000bb", boxstyle="round,pad=0.25"))
        ax.tick_params(labelsize=6)
        ax.set_xlabel("Lon", fontsize=7)
        ax.set_ylabel("Lat", fontsize=7)

    fig.suptitle(
        f"Batch {batch_idx} — high-MSE samples\n"
        "● last pos  ★ true next  ▲ predicted next",
        fontsize=10,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--csv",         required=True,              help="Processed AIS CSV")
    p.add_argument("--checkpoint",  default="runs/ais_transformer/best.pt")
    p.add_argument("--threshold",   type=float, default=1.0,    help="MSE threshold for flagging")
    p.add_argument("--out_dir",     default="outputs/high_mse")
    p.add_argument("--batch_size",  type=int,   default=64)
    p.add_argument("--max_batches", type=int,   default=None,   help="Cap batches (for quick tests)")
    p.add_argument("--top_k_plots", type=int,   default=4,      help="Worst samples to plot per batch")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cpu")

    # ── Load checkpoint ────────────────────────────────────────────────────────
    print(f"Loading checkpoint: {args.checkpoint}")
    model, scalers, vtype_vocab, model_args, epoch = load_checkpoint(args.checkpoint, device)
    print(f"  Epoch {epoch}  |  threshold={args.threshold}")

    seq_len     = model_args.get("seq_len", 120)
    stride      = model_args.get("stride", 1)
    use_vel     = model_args.get("use_velocity", True)
    max_gap_sec = model_args.get("max_gap_sec", 600.0)

    # ── Load data ──────────────────────────────────────────────────────────────
    print(f"\nLoading tracks: {args.csv}")
    tracks, vessel_types = load_tracks(args.csv)

    dataset = CausalDataset(
        tracks, vessel_types, vtype_vocab, scalers,
        seq_len=seq_len, stride=stride,
        use_velocity=use_vel, max_gap_sec=max_gap_sec,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=causal_collate, drop_last=False,
    )
    print(f"  {len(dataset):,} windows  |  {len(loader):,} batches")

    # ── Scan batches ───────────────────────────────────────────────────────────
    print(f"\nScanning for batches with MSE > {args.threshold} ...")
    summary_rows = []
    n_flagged = 0
    sample_offset = 0

    with torch.no_grad():
        for batch_idx, (x, y, gap_mask, vtype) in enumerate(loader):
            if args.max_batches and batch_idx >= args.max_batches:
                break

            B = x.shape[1]
            pred = model(x.to(device), gap_mask=gap_mask.to(device),
                         vessel_type=vtype.to(device))

            batch_mse = F.mse_loss(pred[1:], y[1:].to(device)).item()

            if batch_mse > args.threshold:
                n_flagged += 1

                # Per-sample MSE: mean over (seq_len-1) and 2 output dims
                per_sample_mse = (
                    (pred[1:] - y[1:].to(device)) ** 2
                ).mean(dim=(0, 2)).cpu().numpy()  # (B,)

                # Denormalize positions and displacements
                # x: (seq_len, B, D) — first 2 dims are normalized lon/lat
                pos_norm = x[:, :, :2].numpy()                     # (seq_len, B, 2)
                pos = scalers.pos.inverse(pos_norm.reshape(-1, 2)).reshape(pos_norm.shape)

                # y / pred: (seq_len, B, 2) normalized displacement
                disp_true_norm = y[1:].numpy()                     # (seq_len-1, B, 2)
                disp_pred_norm = pred[1:].cpu().numpy()
                disp_true = scalers.disp.inverse(
                    disp_true_norm.reshape(-1, 2)).reshape(disp_true_norm.shape)
                disp_pred = scalers.disp.inverse(
                    disp_pred_norm.reshape(-1, 2)).reshape(disp_pred_norm.shape)

                # Sort samples by descending MSE — take top_k for CSV + plot
                order = np.argsort(per_sample_mse)[::-1]
                sample_rows = []

                for rank, si in enumerate(order[:args.top_k_plots]):
                    global_idx = sample_offset + si
                    v_id, start_step = dataset.index[global_idx]

                    lons = pos[:, si, 0].tolist()
                    lats = pos[:, si, 1].tolist()

                    # Use last-step displacement as representative
                    true_dlon = float(disp_true[-1, si, 0])
                    true_dlat = float(disp_true[-1, si, 1])
                    pred_dlon = float(disp_pred[-1, si, 0])
                    pred_dlat = float(disp_pred[-1, si, 1])

                    last_lat = lats[-1]
                    true_km = approx_km(true_dlon, true_dlat, last_lat)
                    err_km  = approx_km(pred_dlon - true_dlon, pred_dlat - true_dlat, last_lat)
                    max_jump_km = max(
                        approx_km(float(disp_true[t, si, 0]),
                                  float(disp_true[t, si, 1]),
                                  lats[t])
                        for t in range(disp_true.shape[0])
                    )

                    row = dict(
                        batch_idx=batch_idx,
                        batch_mse=round(batch_mse, 4),
                        rank_in_batch=rank + 1,
                        sample_mse=round(float(per_sample_mse[si]), 4),
                        vessel_id=v_id,
                        start_step=start_step,
                        last_lon=round(lons[-1], 5),
                        last_lat=round(lats[-1], 5),
                        true_dlon=round(true_dlon, 6),
                        true_dlat=round(true_dlat, 6),
                        pred_dlon=round(pred_dlon, 6),
                        pred_dlat=round(pred_dlat, 6),
                        true_disp_km=round(true_km, 3),
                        pred_error_km=round(err_km, 3),
                        max_jump_km=round(max_jump_km, 3),
                    )
                    summary_rows.append(row)
                    sample_rows.append({**row, "lons": lons, "lats": lats})

                # Save plot
                plot_path = os.path.join(args.out_dir, f"batch_{batch_idx:05d}.png")
                plot_batch(batch_idx, sample_rows, plot_path)
                print(f"  batch {batch_idx:5d}  mse={batch_mse:.4f}  "
                      f"worst_sample={per_sample_mse[order[0]]:.4f}  "
                      f"vessel={dataset.index[sample_offset + order[0]][0]}  "
                      f"→ {os.path.basename(plot_path)}")

            sample_offset += B

    # ── Write summary CSV ──────────────────────────────────────────────────────
    csv_path = os.path.join(args.out_dir, "high_mse_batches.csv")
    if summary_rows:
        fieldnames = list(summary_rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"\nSaved CSV: {csv_path}  ({len(summary_rows)} rows)")
    else:
        print("\nNo batches exceeded threshold — nothing to save.")

    total = args.max_batches or len(loader)
    print(f"\nSummary: {n_flagged}/{total} batches flagged "
          f"({100 * n_flagged / total:.1f}%) with MSE > {args.threshold}")


if __name__ == "__main__":
    main()
