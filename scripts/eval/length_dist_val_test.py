"""Compare arc-length distribution across train/val/test partitions of the cleaned
corpus.  Tests the leading hypothesis for the val-to-test ADE gap (§6.5).

Outputs:
    figures/length_dist_val_test.png
    report/figures/length_dist_val_test.png
    results/length_dist_val_test.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data_gen import train_val_split_gen


def arc_length_km(traj: np.ndarray) -> float:
    """Sum of haversine distances between consecutive waypoints (T, 2) -> km."""
    R = 6371.0088
    lon, lat = np.deg2rad(traj[:, 0]), np.deg2rad(traj[:, 1])
    dlon = np.diff(lon)
    dlat = np.diff(lat)
    a = np.sin(dlat / 2) ** 2 + np.cos(lat[:-1]) * np.cos(lat[1:]) * np.sin(dlon / 2) ** 2
    return float(2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1))).sum())


def summarise(name: str, arr: np.ndarray) -> dict:
    return {
        "partition": name,
        "n_routes": int(arr.size),
        "mean_km": float(arr.mean()),
        "median_km": float(np.median(arr)),
        "p25_km": float(np.percentile(arr, 25)),
        "p75_km": float(np.percentile(arr, 75)),
        "p95_km": float(np.percentile(arr, 95)),
        "frac_long_500km": float((arr > 500.0).mean()),
        "frac_short_50km": float((arr < 50.0).mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_npz", default="data/processed/trajgen_128_clean.npz")
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--test_frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_csv", default="results/length_dist_val_test.csv")
    ap.add_argument("--out_png", default="figures/length_dist_val_test.png")
    ap.add_argument("--out_png_report", default="report/figures/length_dist_val_test.png")
    args = ap.parse_args()

    data = np.load(args.data_npz, allow_pickle=True)
    trajs = data["trajectories"].astype(np.float32)
    vts = data["vessel_types"]
    tids = data["track_ids"]

    (tr_t, _, _, va_t, _, _, te_t, _, _) = train_val_split_gen(
        trajs, vts, tids, args.val_frac, args.seed, test_frac=args.test_frac
    )

    train_km = np.array([arc_length_km(t) for t in tr_t], dtype=np.float64)
    val_km = np.array([arc_length_km(t) for t in va_t], dtype=np.float64)
    test_km = np.array([arc_length_km(t) for t in te_t], dtype=np.float64)

    rows = [summarise("train", train_km), summarise("val", val_km), summarise("test", test_km)]
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    for r in rows:
        print(
            f"{r['partition']:>5}  n={r['n_routes']:>5}  "
            f"mean={r['mean_km']:7.1f} km  median={r['median_km']:6.1f} km  "
            f"P95={r['p95_km']:7.1f} km  long%={100 * r['frac_long_500km']:5.2f}  "
            f"short%={100 * r['frac_short_50km']:5.2f}"
        )

    log_bins = np.logspace(np.log10(5), np.log10(3000), 50)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for name, arr, colour in [
        ("train", train_km, "#1f77b4"),
        ("val",   val_km,   "#2ca02c"),
        ("test",  test_km,  "#d62728"),
    ]:
        ax.hist(arr, bins=log_bins, density=True, histtype="step", linewidth=2,
                label=f"{name} (n={arr.size}, median {np.median(arr):.0f} km)", color=colour)
    ax.axvline(50, ls=":", color="gray", lw=0.8)
    ax.axvline(500, ls=":", color="gray", lw=0.8)
    ax.text(50, ax.get_ylim()[1] * 0.92, " short/medium", fontsize=8, color="gray")
    ax.text(500, ax.get_ylim()[1] * 0.92, " medium/long", fontsize=8, color="gray")
    ax.set_xscale("log")
    ax.set_xlabel("Trajectory arc length (km, log scale)")
    ax.set_ylabel("Density")
    ax.set_title("Arc-length distribution across train / val / test partitions"
                 " (cleaned 128-pt corpus, 80/15/5 MMSI-grouped split, seed=42)")
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    ax.grid(alpha=0.25, which="both")
    fig.tight_layout()

    os.makedirs(os.path.dirname(args.out_png) or ".", exist_ok=True)
    fig.savefig(args.out_png, dpi=160)
    if args.out_png_report:
        os.makedirs(os.path.dirname(args.out_png_report) or ".", exist_ok=True)
        fig.savefig(args.out_png_report, dpi=160)
    plt.close(fig)
    print(f"\nWrote {args.out_csv}, {args.out_png}")


if __name__ == "__main__":
    main()
