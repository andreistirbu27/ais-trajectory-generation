#!/usr/bin/env python3
"""
regen_report_data_figs.py -- regenerate the Section 3.1 data overview
subfigures with paper-quality formatting:

  data_start_end.png       2-up: dirty vs clean datasets, same colourmap
  data_distributions.png   2-panel: start-end distance + arc length, with
                           x-axis clipped at 99th-percentile and a labelled
                           ">x" tail bin so bars don't terminate mid-axis.

Run from the repo root:
    python3 scripts/eval/regen_report_data_figs.py \
        --dirty data/processed/trajgen_128.npz \
        --clean data/processed/trajgen_128_clean.npz \
        --out_dir report/figures
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1r = np.radians(lat1); lat2r = np.radians(lat2)
    dlat = np.radians(lat2 - lat1); dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.minimum(1, np.sqrt(a)))


def arc_lengths_km(trajs):
    step_km = haversine_km(
        trajs[:, :-1, 1].reshape(-1), trajs[:, :-1, 0].reshape(-1),
        trajs[:,  1:, 1].reshape(-1), trajs[:,  1:, 0].reshape(-1),
    ).reshape(len(trajs), -1)
    return step_km.sum(axis=1)


def start_end_km(trajs):
    s = trajs[:, 0, :]; e = trajs[:, -1, :]
    return haversine_km(s[:, 1], s[:, 0], e[:, 1], e[:, 0])


def hist_with_tail(ax, data, label, color, n_bins=40, cap_pct=99):
    cap = np.percentile(data, cap_pct)
    cap = float(np.round(cap / 50) * 50) if cap > 100 else float(np.round(cap))
    capped = np.minimum(data, cap)
    counts, edges, _ = ax.hist(capped, bins=np.linspace(0, cap, n_bins + 1),
                                color=color, edgecolor="white", alpha=0.85,
                                label=label)
    # ">cap" bar
    n_tail = int((data > cap).sum())
    if n_tail > 0:
        width = (edges[1] - edges[0])
        ax.bar(cap + width * 0.55, n_tail, width=width * 0.9, color=color,
               edgecolor="black", linewidth=0.8, hatch="//", alpha=0.85)
        ax.text(cap + width * 0.55, n_tail, f"$>${int(cap)}\nn={n_tail:,}",
                ha="center", va="bottom", fontsize=8)
    ax.set_xlim(0, cap + (edges[1] - edges[0]) * 1.4)
    ax.axvline(np.median(data), color="black", linestyle="--", lw=1, alpha=0.7,
               label=f"median {np.median(data):.0f} km")


def plot_distributions(dirty_trajs, clean_trajs, out_path):
    dirty_se = start_end_km(dirty_trajs)
    dirty_arc = arc_lengths_km(dirty_trajs)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    hist_with_tail(axes[0], dirty_se, "trajgen_128 (dirty)", "#2ecc71")
    axes[0].set_xlabel("Straight-line distance (km)", fontsize=10)
    axes[0].set_ylabel("Count (tracks)", fontsize=10)
    axes[0].set_title(f"Start--end distance (n={len(dirty_trajs):,})", fontsize=11)
    axes[0].legend(loc="upper right", fontsize=8)

    hist_with_tail(axes[1], dirty_arc, "trajgen_128 (dirty)", "#e74c3c")
    axes[1].set_xlabel("Arc length (km)", fontsize=10)
    axes[1].set_title(f"Arc length (n={len(dirty_trajs):,})", fontsize=11)
    axes[1].legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_path}")


def plot_start_end_2up(dirty_trajs, dirty_vt, clean_trajs, clean_vt, out_path):
    # Vessel class buckets (broad)
    def vtype_to_class(v):
        v = int(v)
        if 60 <= v <= 69: return "passenger"
        if 70 <= v <= 79: return "cargo"
        if 80 <= v <= 89: return "tanker"
        return "other"

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0))
    for ax, trajs, vts, title in [
        (axes[0], dirty_trajs, dirty_vt, f"trajgen_128 (dirty, n={len(dirty_trajs):,})"),
        (axes[1], clean_trajs, clean_vt, f"trajgen_128_clean (E1, n={len(clean_trajs):,})"),
    ]:
        cls = np.array([vtype_to_class(v) for v in vts])
        starts = trajs[:, 0, :]
        ends = trajs[:, -1, :]
        for c, col in [("passenger", "#3498db"),
                        ("cargo",     "#e67e22"),
                        ("tanker",    "#9b59b6"),
                        ("other",     "#7f8c8d")]:
            mask = cls == c
            if mask.sum() == 0:
                continue
            ax.scatter(starts[mask, 0], starts[mask, 1], s=2, c=col, alpha=0.25,
                       label=f"start ({c}, n={mask.sum():,})", marker="o",
                       linewidths=0)
            ax.scatter(ends[mask, 0],   ends[mask, 1],   s=2, c=col, alpha=0.25,
                       marker="^", linewidths=0)
        ax.set_xlabel("lon", fontsize=9)
        ax.set_ylabel("lat", fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlim(-126, -59)
        ax.set_ylim(9, 56)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower left", fontsize=7, framealpha=0.85,
                  markerscale=4)
    fig.suptitle("Start (circle) and end (triangle) points per vessel class --- "
                  "before and after E1 land filtering",
                  fontsize=11, y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dirty", default="data/processed/trajgen_128.npz")
    p.add_argument("--clean", default="data/processed/trajgen_128_clean.npz")
    p.add_argument("--out_dir", default="report/figures")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Loading {args.dirty}")
    d = np.load(args.dirty, allow_pickle=True)
    dirty_trajs = d["trajectories"].astype(np.float32)
    dirty_vt = d["vessel_types"]
    print(f"  {len(dirty_trajs):,} tracks")
    print(f"Loading {args.clean}")
    c = np.load(args.clean, allow_pickle=True)
    clean_trajs = c["trajectories"].astype(np.float32)
    clean_vt = c["vessel_types"]
    print(f"  {len(clean_trajs):,} tracks")

    plot_distributions(dirty_trajs, clean_trajs,
                       os.path.join(args.out_dir, "data_distributions.png"))
    plot_start_end_2up(dirty_trajs, dirty_vt, clean_trajs, clean_vt,
                       os.path.join(args.out_dir, "data_start_end.png"))


if __name__ == "__main__":
    main()
