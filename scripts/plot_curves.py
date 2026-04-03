#!/usr/bin/env python3
"""
plot_curves.py — Plot training curves from metrics.csv.

Reads the per-epoch metrics saved by train_disp.py and produces a 2-panel figure:
  Left  : train MSE + val MSE vs epoch
  Right : val ADE + val FDE in metres vs epoch

Can be run at any point during or after training.

Usage:
    python3 scripts/plot_curves.py --csv runs/ais_transformer/metrics.csv
    python3 scripts/plot_curves.py --csv runs/ais_transformer/metrics.csv \
        --out outputs/training_curves.png
"""

import argparse
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
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    if df.empty:
        print("No data in metrics.csv yet.")
        return

    out_path = args.out or os.path.join(os.path.dirname(args.csv), "training_curves.png")

    # Best epoch by val MSE
    best_epoch = None
    if "val_mse" in df.columns and df["val_mse"].notna().any():
        best_epoch = int(df.loc[df["val_mse"].idxmin(), "epoch"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # ── Left: MSE ─────────────────────────────────────────────────────────────
    ax1.plot(df["epoch"], df["train_mse"], label="Train MSE",
             color="#1565c0", linewidth=1.6)
    if "val_mse" in df.columns and df["val_mse"].notna().any():
        ax1.plot(df["epoch"], df["val_mse"], label="Val MSE",
                 color="#c62828", linewidth=1.6)
    if best_epoch is not None:
        ax1.axvline(best_epoch, color="#555", linewidth=1.0, linestyle="--",
                    label=f"Best epoch {best_epoch}")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("MSE (normalised space)")
    ax1.set_title("Train vs Val MSE")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    # ── Right: ADE / FDE ──────────────────────────────────────────────────────
    if "val_ade_m" in df.columns and df["val_ade_m"].notna().any():
        ax2.plot(df["epoch"], df["val_ade_m"], label="Val ADE (m)",
                 color="#2e7d32", linewidth=1.6)
    if "val_fde_m" in df.columns and df["val_fde_m"].notna().any():
        ax2.plot(df["epoch"], df["val_fde_m"], label="Val FDE (m)",
                 color="#e65100", linewidth=1.6, linestyle="--")
    if best_epoch is not None:
        ax2.axvline(best_epoch, color="#555", linewidth=1.0, linestyle="--",
                    label=f"Best epoch {best_epoch}")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Distance (metres)")
    ax2.set_title("Val ADE & FDE")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    epochs_done = int(df["epoch"].max())
    fig.suptitle(f"Training curves  ({epochs_done} epochs logged)", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
