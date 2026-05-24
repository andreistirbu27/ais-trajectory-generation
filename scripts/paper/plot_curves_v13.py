#!/usr/bin/env python3
"""
plot_curves_v13.py — Plot training curves for v13 (TrajDiff).

v13 metrics.csv schema:
    epoch, global_step, train_loss, val_loss

Two panels:
    Left  — train_loss vs global_step (log y), with rolling smoothing, and
            val_loss markers at each validation epoch.
    Right — val_loss over wall epochs only (so the user can see the
            EMA-validation trajectory without the train-step noise).

Safe to run mid-training; just re-reads the CSV.

Usage:
    paper/.venv/bin/python3 scripts/paper/plot_curves_v13.py \\
        --csv runs/paper/v13_trajdiff_base/metrics.csv
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--csv", required=True,
                   help="Path to metrics.csv from train_gen_v13.py.")
    p.add_argument("--out", default=None,
                   help="Output PNG path; defaults to training_curves.png next to --csv.")
    p.add_argument("--smooth", type=int, default=50,
                   help="Rolling window for train_loss smoothing (in step rows).")
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    if df.empty:
        print("metrics.csv has no rows yet.")
        return

    step_df  = df[df["train_loss"].notna() & df["val_loss"].isna()].copy()
    val_df   = df[df["val_loss"].notna()].copy()

    out_path = args.out or os.path.join(os.path.dirname(args.csv) or ".",
                                         "training_curves.png")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # ── Left: train loss vs step, with val markers ────────────────────
    if not val_df.empty:
        for xb in val_df["global_step"]:
            ax1.axvline(xb, color="#cccccc", linewidth=0.5, zorder=0)

    if not step_df.empty:
        ax1.plot(step_df["global_step"], step_df["train_loss"],
                 color="#1565c0", linewidth=0.4, alpha=0.2,
                 label="Train loss (raw)")
        smoothed = step_df["train_loss"].rolling(window=args.smooth, min_periods=1).mean()
        ax1.plot(step_df["global_step"], smoothed,
                 color="#1565c0", linewidth=1.6,
                 label=f"Train loss (smooth w={args.smooth})")

    if not val_df.empty:
        ax1.scatter(val_df["global_step"], val_df["val_loss"],
                    color="#c62828", s=30, zorder=5,
                    label="Val loss (EMA, full set)")

    ax1.set_yscale("log")
    ax1.set_xlabel("Global step")
    ax1.set_ylabel("EDM training loss (log)")
    ax1.set_title("Train / Val loss")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3, which="both")

    # ── Right: val loss per epoch only ────────────────────────────────
    if not val_df.empty:
        ax2.plot(val_df["epoch"], val_df["val_loss"],
                 color="#c62828", linewidth=1.6, marker="o", ms=6,
                 label="Val loss (EMA)")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Val EDM loss")
        ax2.set_title("Validation loss per epoch")
        ax2.grid(alpha=0.3)
        ax2.legend(fontsize=9)
        # Note: train loss can fall fast due to easy-noise dominance in the
        # EDM training objective; flat val loss while train loss collapses
        # is a known failure mode — see scripts/paper/plot_v13_loss_by_sigma.py
        # for the diagnostic.
    else:
        ax2.text(0.5, 0.5, "No validation rows yet\n(val every 5 epochs by default)",
                 ha="center", va="center", transform=ax2.transAxes, fontsize=11)
        ax2.set_axis_off()

    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    print(f"Wrote {out_path}")
    print()
    print(f"Last train_loss : {step_df['train_loss'].iloc[-1]:.4f}  "
          f"(step {int(step_df['global_step'].iloc[-1])})" if not step_df.empty else "no train rows")
    if not val_df.empty:
        print(f"Last val_loss   : {val_df['val_loss'].iloc[-1]:.4f}  "
              f"(epoch {int(val_df['epoch'].iloc[-1])})")
        print(f"Best val_loss   : {val_df['val_loss'].min():.4f}  "
              f"(epoch {int(val_df.loc[val_df['val_loss'].idxmin(), 'epoch'])})")


if __name__ == "__main__":
    main()
