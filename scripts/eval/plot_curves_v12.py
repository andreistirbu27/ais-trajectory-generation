#!/usr/bin/env python3
"""
plot_curves_v12.py — Plot training curves for v12 (pointer over water graph).

v12 metrics.csv schema matches v11:
    global_step, epoch, train_loss, val_loss,
    val_ce, val_offset, val_smooth, val_top1

So this script is functionally identical to plot_curves_v11.py but
re-titled for v12. Kept separate so future-v12 changes don't break v11
plots.

Usage:
    python3 scripts/eval/plot_curves_v12.py --csv runs/trajgen_v12_lite/metrics.csv
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--csv", required=True, help="Path to metrics.csv from train_gen_v12.py")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    if df.empty:
        print("No data in metrics.csv yet.")
        return

    out_path = args.out or os.path.join(os.path.dirname(args.csv), "training_curves.png")

    step_df = df[df["train_loss"].notna()].copy()
    epoch_df = df[df["val_top1"].notna()].copy()
    epoch_boundaries = epoch_df["global_step"].tolist() if not epoch_df.empty else []

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for xb in epoch_boundaries:
        ax1.axvline(xb, color="#cccccc", linewidth=0.5, zorder=0)

    if not step_df.empty:
        ax1.plot(step_df["global_step"], step_df["train_loss"],
                 color="#1565c0", linewidth=0.4, alpha=0.2)
        smoothed = step_df["train_loss"].rolling(window=20, min_periods=1).mean()
        ax1.plot(step_df["global_step"], smoothed,
                 label="Train loss (smoothed)", color="#1565c0", linewidth=1.6)
        if step_df["val_loss"].notna().any():
            ax1.plot(step_df["global_step"], step_df["val_loss"],
                     label="Val loss (~approx)", color="#c62828",
                     linewidth=1.0, alpha=0.7)

    if not epoch_df.empty and epoch_df["val_loss"].notna().any():
        ax1.scatter(epoch_df["global_step"], epoch_df["val_loss"],
                    label="Val loss (full)", color="#c62828", s=25, zorder=5)

    ax1.set_yscale("log"); ax1.set_xlabel("Global step")
    ax1.set_ylabel("Loss (log scale)"); ax1.set_title("Train vs Val Loss")
    ax1.legend(fontsize=8); ax1.grid(alpha=0.3, which="both")

    if not epoch_df.empty:
        for xb in epoch_boundaries:
            ax2.axvline(xb, color="#cccccc", linewidth=0.5, zorder=0)
        for col, label, style in [
            ("val_ce",     "Val CE",     dict(color="#2e7d32", linestyle="-",  marker="o", ms=4)),
            ("val_offset", "Val offset", dict(color="#e65100", linestyle="--", marker="o", ms=4)),
            ("val_smooth", "Val smooth", dict(color="#1565c0", linestyle=":",  marker="o", ms=4)),
        ]:
            if col in epoch_df.columns and epoch_df[col].notna().any():
                ax2.plot(epoch_df["global_step"], epoch_df[col],
                         label=label, linewidth=1.6, **style)
        ax2.set_xlabel("Global step"); ax2.set_ylabel("Component loss")
        ax2.set_title("Val components + pointer top-1"); ax2.grid(alpha=0.3)

        ax2b = ax2.twinx()
        if epoch_df["val_top1"].notna().any():
            ax2b.plot(epoch_df["global_step"], epoch_df["val_top1"],
                      label="Val top-1 acc", color="#6a1b9a",
                      linewidth=1.6, marker="s", ms=4)
        ax2b.set_ylabel("Top-1 accuracy"); ax2b.set_ylim(0, 1)
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2b.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="best")
    else:
        ax2.text(0.5, 0.5, "No epoch-end data yet", transform=ax2.transAxes,
                 ha="center", va="center", fontsize=12, color="gray")
        ax2.set_title("Val components + pointer top-1")

    total_steps = int(df["global_step"].max())
    n_epochs = len(epoch_boundaries)
    fig.suptitle(f"v12 Pointer-over-water-graph  ({total_steps:,} steps | {n_epochs} epochs)",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
