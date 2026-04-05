#!/usr/bin/env python3
"""
plot_curves.py — Plot training curves from metrics.csv.

Reads the per-step/per-epoch metrics saved by train_disp.py and produces a 2-panel figure:
  Left  : train MSE (mid-step) + val MSE (mid-step approx + epoch-end full) vs global step
  Right : val ADE + val FDE in metres vs global step (epoch-end only)

Epoch boundaries are marked with faint vertical grey lines.

Usage:
    python3 scripts/plot_curves.py --csv runs/ais_transformer/metrics.csv
    python3 scripts/plot_curves.py --csv runs/ais_transformer/metrics.csv \\
        --out outputs/training_curves.png
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--csv", required=True, help="Path to metrics.csv from training")
    p.add_argument("--out", default=None,
                   help="Output PNG path (default: training_curves.png next to CSV)")
    p.add_argument("--baseline_ade", type=float, default=None,
                   help="CV baseline ADE in metres (overrides baseline.json if provided)")
    p.add_argument("--baseline_fde", type=float, default=None,
                   help="CV baseline FDE in metres (overrides baseline.json if provided)")
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    if df.empty:
        print("No data in metrics.csv yet.")
        return

    out_path = args.out or os.path.join(os.path.dirname(args.csv), "training_curves.png")

    # Load baseline from baseline.json if present (CLI args override)
    baseline_ade = args.baseline_ade
    baseline_fde = args.baseline_fde
    baseline_json = os.path.join(os.path.dirname(args.csv), "baseline.json")
    if os.path.exists(baseline_json) and (baseline_ade is None or baseline_fde is None):
        with open(baseline_json) as _f:
            bl = json.load(_f)
        if baseline_ade is None:
            baseline_ade = bl.get("ade_m")
        if baseline_fde is None:
            baseline_fde = bl.get("fde_m")

    # Mid-step rows: have train_mse, may have val_mse; no ade/fde
    step_df = df[df["train_mse"].notna()].copy()
    # Epoch-end rows: have full val metrics; no train_mse
    epoch_df = df[df["train_mse"].isna() & df["val_mse"].notna()].copy()

    # Epoch boundary x-positions (from epoch-end rows)
    epoch_boundaries = epoch_df["global_step"].tolist() if not epoch_df.empty else []

    # Best epoch by full val MSE
    best_step = None
    if not epoch_df.empty:
        best_step = int(epoch_df.loc[epoch_df["val_mse"].idxmin(), "global_step"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # ── Left: MSE ─────────────────────────────────────────────────────────────
    # Faint epoch boundary lines
    for xb in epoch_boundaries:
        ax1.axvline(xb, color="#cccccc", linewidth=0.7, zorder=0)

    if not step_df.empty and step_df["train_mse"].notna().any():
        ax1.plot(step_df["global_step"], step_df["train_mse"],
                 label="Train MSE (raw)", color="#1565c0", linewidth=0.6, alpha=0.3)
        smoothed = step_df["train_mse"].rolling(window=20, min_periods=1).mean()
        ax1.plot(step_df["global_step"], smoothed,
                 label="Train MSE (smoothed)", color="#1565c0", linewidth=1.6, alpha=0.9)

    if not step_df.empty and "val_mse" in step_df.columns and step_df["val_mse"].notna().any():
        ax1.plot(step_df["global_step"], step_df["val_mse"],
                 label="Val MSE (~approx)", color="#c62828", linewidth=1.2, alpha=0.7)

    if not epoch_df.empty and epoch_df["val_mse"].notna().any():
        ax1.scatter(epoch_df["global_step"], epoch_df["val_mse"],
                    label="Val MSE (full epoch-end)", color="#c62828", s=25, zorder=5)

    if best_step is not None:
        ax1.axvline(best_step, color="#555", linewidth=1.0, linestyle="--",
                    label=f"Best step {best_step}")

    ax1.set_xlabel("Global step")
    ax1.set_ylabel("MSE (normalised space)")
    ax1.set_title("Train vs Val MSE")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    # ── Right: ADE / FDE ──────────────────────────────────────────────────────
    for xb in epoch_boundaries:
        ax2.axvline(xb, color="#cccccc", linewidth=0.7, zorder=0)

    if not epoch_df.empty:
        if "val_ade_m" in epoch_df.columns and epoch_df["val_ade_m"].notna().any():
            ax2.plot(epoch_df["global_step"], epoch_df["val_ade_m"],
                     label="Val ADE (m)", color="#2e7d32", linewidth=1.6, marker="o", ms=4)
        if "val_fde_m" in epoch_df.columns and epoch_df["val_fde_m"].notna().any():
            ax2.plot(epoch_df["global_step"], epoch_df["val_fde_m"],
                     label="Val FDE (m)", color="#e65100", linewidth=1.6,
                     linestyle="--", marker="o", ms=4)

    if baseline_ade is not None:
        ax2.axhline(baseline_ade, color="#2e7d32", linewidth=1.2, linestyle=":",
                    label=f"CV baseline ADE {baseline_ade:.1f}m")
    if baseline_fde is not None:
        ax2.axhline(baseline_fde, color="#e65100", linewidth=1.2, linestyle=":",
                    label=f"CV baseline FDE {baseline_fde:.1f}m")

    if best_step is not None:
        ax2.axvline(best_step, color="#555", linewidth=1.0, linestyle="--",
                    label=f"Best step {best_step}")

    ax2.set_xlabel("Global step")
    ax2.set_ylabel("Distance (metres)")
    ax2.set_title("Val ADE & FDE")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    total_steps = int(df["global_step"].max())
    n_epochs = len(epoch_boundaries)
    fig.suptitle(f"Training curves  ({total_steps:,} steps | {n_epochs} epochs)", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
