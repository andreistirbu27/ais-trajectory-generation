#!/usr/bin/env python3
"""
plot_curves_gen.py — Plot training curves for trajectory generation (v2).

Reads metrics.csv from train_gen.py and produces a 2-panel figure:
  Left  : train_loss + val_loss vs global step
  Right : val_delta + val_endpoint vs epoch (epoch-end rows only)

Usage:
    python3 scripts/eval/plot_curves_gen.py --csv runs/trajgen_mvp/metrics.csv
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--csv", required=True, help="Path to metrics.csv from train_gen.py")
    p.add_argument("--out", default=None,
                   help="Output PNG path (default: training_curves.png next to CSV)")
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    if df.empty:
        print("No data in metrics.csv yet.")
        return

    out_path = args.out or os.path.join(os.path.dirname(args.csv), "training_curves.png")

    # Mid-step rows: have train_loss and val_loss, no val_delta/val_endpoint
    step_df = df[df["train_loss"].notna()].copy()
    # Epoch-end rows: have val_delta and val_endpoint
    epoch_df = df[df["val_delta"].notna()].copy()

    epoch_boundaries = epoch_df["global_step"].tolist() if not epoch_df.empty else []

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # ── Left: Loss curves (log scale) ───────────────────────────────────────
    for xb in epoch_boundaries:
        ax1.axvline(xb, color="#cccccc", linewidth=0.5, zorder=0)

    if not step_df.empty:
        ax1.plot(step_df["global_step"], step_df["train_loss"],
                 color="#1565c0", linewidth=0.4, alpha=0.2)
        smoothed = step_df["train_loss"].rolling(window=20, min_periods=1).mean()
        ax1.plot(step_df["global_step"], smoothed,
                 label="Train loss (smoothed)", color="#1565c0", linewidth=1.6)

        if "val_loss" in step_df.columns and step_df["val_loss"].notna().any():
            ax1.plot(step_df["global_step"], step_df["val_loss"],
                     label="Val loss (~approx)", color="#c62828", linewidth=1.0, alpha=0.7)

    if not epoch_df.empty and "val_loss" in epoch_df.columns:
        ax1.scatter(epoch_df["global_step"], epoch_df["val_loss"],
                    label="Val loss (full)", color="#c62828", s=25, zorder=5)

    ax1.set_yscale("log")
    ax1.set_xlabel("Global step")
    ax1.set_ylabel("Loss (log scale)")
    ax1.set_title("Train vs Val Loss")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3, which="both")

    # ── Right: Component losses (epoch-end) ──────────────────────────────────
    if not epoch_df.empty:
        for xb in epoch_boundaries:
            ax2.axvline(xb, color="#cccccc", linewidth=0.5, zorder=0)

        if "val_delta" in epoch_df.columns and epoch_df["val_delta"].notna().any():
            ax2.plot(epoch_df["global_step"], epoch_df["val_delta"],
                     label="Val delta loss", color="#2e7d32", linewidth=1.6,
                     marker="o", ms=4)
        if "val_endpoint" in epoch_df.columns and epoch_df["val_endpoint"].notna().any():
            ax2.plot(epoch_df["global_step"], epoch_df["val_endpoint"],
                     label="Val endpoint loss", color="#e65100", linewidth=1.6,
                     linestyle="--", marker="o", ms=4)

        ax2.set_xlabel("Global step")
        ax2.set_ylabel("Loss")
        ax2.set_title("Validation Component Losses")
        ax2.legend(fontsize=8)
        ax2.grid(alpha=0.3)
    else:
        ax2.text(0.5, 0.5, "No epoch-end data yet", transform=ax2.transAxes,
                 ha="center", va="center", fontsize=12, color="gray")
        ax2.set_title("Validation Component Losses")

    total_steps = int(df["global_step"].max())
    n_epochs = len(epoch_boundaries)
    fig.suptitle(f"Trajectory Generation Training  ({total_steps:,} steps | {n_epochs} epochs)",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
