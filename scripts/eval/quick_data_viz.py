#!/usr/bin/env python3
"""
quick_data_viz.py — Quick visual sanity check of trajectory generation data.

Reads the NPZ dataset and produces diagnostic plots:
  1. Sample trajectories on a map (colored by vessel type)
  2. Start/end points overview
  3. Track length and distance distributions
  4. Individual track samples with point spacing
  5. Vessel type breakdown

Usage:
    python3 scripts/eval/quick_data_viz.py \
        --data_npz data/processed/trajgen_128.npz \
        --out_dir outputs/data_viz
"""

import argparse
import os
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

EARTH_RADIUS_KM = 6371.0

VTYPE_NAME = {
    **{c: "Passenger" for c in range(60, 70)},
    **{c: "Cargo"     for c in range(70, 80)},
    **{c: "Tanker"    for c in range(80, 90)},
}

VTYPE_COLOR = {
    "Passenger": "#e74c3c",
    "Cargo":     "#2ecc71",
    "Tanker":    "#3498db",
}


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    return EARTH_RADIUS_KM * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def plot_all_tracks(trajs, vtypes, out_path, n_plot=2000):
    """Plot sampled tracks on a map, colored by vessel type."""
    fig, ax = plt.subplots(figsize=(16, 10))

    n = min(n_plot, len(trajs))
    alpha = max(0.05, min(0.4, 30.0 / n))

    for i in range(n):
        vt = int(vtypes[i])
        vname = VTYPE_NAME.get(vt, "Other")
        color = VTYPE_COLOR.get(vname, "#999999")
        ax.plot(trajs[i, :, 0], trajs[i, :, 1], color=color,
                linewidth=0.4, alpha=alpha)

    for vname, color in VTYPE_COLOR.items():
        ax.plot([], [], color=color, linewidth=2, label=vname)
    ax.legend(fontsize=11, loc="upper left")

    ax.set_xlabel("Longitude", fontsize=12)
    ax.set_ylabel("Latitude", fontsize=12)
    ax.set_title(f"Sample of {n:,} Resampled Tracks (128 pts each)", fontsize=14)
    ax.set_aspect("equal")

    try:
        import contextily as ctx
        ctx.add_basemap(ax, crs="EPSG:4326",
                        source=ctx.providers.CartoDB.Positron,
                        zoom="auto", alpha=0.5)
    except Exception:
        pass

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_path}")


def plot_start_end(trajs, out_path):
    """Plot start and end points."""
    fig, ax = plt.subplots(figsize=(16, 10))

    ax.scatter(trajs[:, 0, 0], trajs[:, 0, 1], c="green", s=4, alpha=0.15,
               label="Start", zorder=3)
    ax.scatter(trajs[:, -1, 0], trajs[:, -1, 1], c="red", s=4, alpha=0.15,
               label="End", zorder=3)

    ax.legend(fontsize=11)
    ax.set_xlabel("Longitude", fontsize=12)
    ax.set_ylabel("Latitude", fontsize=12)
    ax.set_title(f"Start / End Points ({len(trajs):,} tracks)", fontsize=14)
    ax.set_aspect("equal")

    try:
        import contextily as ctx
        ctx.add_basemap(ax, crs="EPSG:4326",
                        source=ctx.providers.CartoDB.Positron,
                        zoom="auto", alpha=0.5)
    except Exception:
        pass

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_path}")


def plot_distributions(trajs, vtypes, out_path):
    """Histograms of track statistics."""
    N = len(trajs)

    # Start-end distances (vectorized)
    starts = trajs[:, 0, :]   # (N, 2) [lon, lat]
    ends = trajs[:, -1, :]
    dists_km = haversine_km(starts[:, 1], starts[:, 0], ends[:, 1], ends[:, 0])

    # Arc lengths (vectorized)
    deltas = np.diff(trajs, axis=1)  # (N, 127, 2)
    step_km = haversine_km(
        trajs[:, :-1, 1].reshape(-1), trajs[:, :-1, 0].reshape(-1),
        trajs[:, 1:, 1].reshape(-1), trajs[:, 1:, 0].reshape(-1),
    ).reshape(N, -1)
    arc_km = step_km.sum(axis=1)

    # Sinuosity
    sinuosity = arc_km / np.maximum(dists_km, 0.1)

    # Step size distribution (sample)
    sample_steps = step_km[:5000].reshape(-1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Start-end distance
    axes[0, 0].hist(dists_km, bins=80, color="#2ecc71", edgecolor="white", alpha=0.8)
    axes[0, 0].set_xlabel("Start-End Distance (km)")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].set_title(f"Start-End Distance (median={np.median(dists_km):.0f} km)")
    axes[0, 0].axvline(np.median(dists_km), color="red", linestyle="--", alpha=0.7)

    # Arc length
    axes[0, 1].hist(arc_km, bins=80, color="#e74c3c", edgecolor="white", alpha=0.8)
    axes[0, 1].set_xlabel("Arc Length (km)")
    axes[0, 1].set_ylabel("Count")
    axes[0, 1].set_title(f"Arc Length (median={np.median(arc_km):.0f} km)")
    axes[0, 1].axvline(np.median(arc_km), color="red", linestyle="--", alpha=0.7)

    # Sinuosity
    sin_clipped = np.clip(sinuosity, 1.0, 5.0)
    axes[1, 0].hist(sin_clipped, bins=80, color="#9b59b6", edgecolor="white", alpha=0.8)
    axes[1, 0].set_xlabel("Sinuosity (arc / straight-line, clipped at 5)")
    axes[1, 0].set_ylabel("Count")
    axes[1, 0].set_title(f"Sinuosity (median={np.median(sinuosity):.2f})")
    axes[1, 0].axvline(np.median(sinuosity), color="red", linestyle="--", alpha=0.7)

    # Per-step distances
    axes[1, 1].hist(sample_steps, bins=100, color="#3498db", edgecolor="white", alpha=0.8)
    axes[1, 1].set_xlabel("Per-Step Distance (km)")
    axes[1, 1].set_ylabel("Count")
    axes[1, 1].set_title(f"Step Size (median={np.median(sample_steps):.2f} km)")
    axes[1, 1].axvline(np.median(sample_steps), color="red", linestyle="--", alpha=0.7)

    fig.suptitle(f"Dataset Statistics ({N:,} tracks, 128 pts each)", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_path}")


def plot_individual_tracks(trajs, vtypes, out_path, n=16, seed=42):
    """Plot individual tracks in a grid."""
    rng = np.random.RandomState(seed)
    N = len(trajs)

    # Pick a diverse set: sort by arc length, sample from quartiles
    deltas = np.diff(trajs, axis=1)
    arc_deg = np.sqrt((deltas**2).sum(axis=2)).sum(axis=1)
    order = np.argsort(arc_deg)

    picks = []
    quarter = max(1, N // 4)
    for q in range(4):
        start = q * quarter
        end = min(start + quarter, N)
        idxs = rng.choice(order[start:end], min(4, end - start), replace=False)
        picks.extend(idxs)
    picks = picks[:n]

    ncols = 4
    nrows = max(1, (n + ncols - 1) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes[None, :]
    elif ncols == 1:
        axes = axes[:, None]

    for idx_i in tqdm(range(nrows * ncols), desc="Plotting tracks"):
        ax = axes[idx_i // ncols, idx_i % ncols]
        if idx_i >= len(picks):
            ax.set_visible(False)
            continue

        i = picks[idx_i]
        traj = trajs[i]  # (128, 2) [lon, lat]
        vt = int(vtypes[i])
        vname = VTYPE_NAME.get(vt, f"type_{vt}")
        color = VTYPE_COLOR.get(vname, "#999999")

        dist_km = haversine_km(traj[0, 1], traj[0, 0], traj[-1, 1], traj[-1, 0])

        # Track line
        ax.plot(traj[:, 0], traj[:, 1], "-", color=color, linewidth=1.2, alpha=0.8)
        # Point markers (every 4th point)
        ax.plot(traj[::4, 0], traj[::4, 1], ".", color=color, markersize=3, alpha=0.5)

        # Start/end
        ax.plot(traj[0, 0], traj[0, 1], "gs", markersize=8, zorder=5)
        ax.plot(traj[-1, 0], traj[-1, 1], "ro", markersize=8, zorder=5)

        ax.set_title(f"{vname} | {dist_km:.0f} km", fontsize=9)
        ax.tick_params(labelsize=7)
        ax.set_aspect("equal")

        # Add map tiles with coastlines
        try:
            import contextily as ctx
            # Add padding around track extent
            pad = 0.5  # degrees
            ax.set_xlim(traj[:, 0].min() - pad, traj[:, 0].max() + pad)
            ax.set_ylim(traj[:, 1].min() - pad, traj[:, 1].max() + pad)
            ctx.add_basemap(ax, crs="EPSG:4326",
                            source=ctx.providers.CartoDB.Positron,
                            zoom="auto", alpha=0.6)
        except Exception:
            pass

    fig.suptitle("Individual Track Samples (green=start, red=end)", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_path}")


def plot_vessel_type_breakdown(vtypes, out_path):
    """Bar chart of vessel type distribution."""
    counts = defaultdict(int)
    for vt in vtypes:
        vname = VTYPE_NAME.get(int(vt), f"type_{vt}")
        counts[vname] += 1

    labels = sorted(counts.keys(), key=lambda x: -counts[x])
    values = [counts[l] for l in labels]
    colors = [VTYPE_COLOR.get(l, "#999999") for l in labels]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, values, color=colors, edgecolor="white")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:,}", ha="center", va="bottom", fontsize=11)

    ax.set_ylabel("Count")
    ax.set_title(f"Vessel Type Distribution ({len(vtypes):,} tracks)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_path}")


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_npz", required=True,
        help="NPZ file from prepare_trajgen.py")
    parser.add_argument("--out_dir", default="outputs/data_viz")
    parser.add_argument("--n_plot", type=int, default=2000,
        help="Max tracks on the overview map")
    parser.add_argument("--n_individual", type=int, default=16,
        help="Number of individual track plots")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading {args.data_npz}...")
    data = np.load(args.data_npz, allow_pickle=True)
    trajs = data["trajectories"]      # (N, 128, 2)
    vtypes = data["vessel_types"]     # (N,)
    print(f"  {len(trajs):,} trajectories, {trajs.shape[1]} points each\n")

    # Shuffle for plotting
    rng = np.random.RandomState(args.seed)
    perm = rng.permutation(len(trajs))
    trajs_shuffled = trajs[perm]
    vtypes_shuffled = vtypes[perm]

    plot_all_tracks(trajs_shuffled, vtypes_shuffled,
                    os.path.join(args.out_dir, "all_tracks.png"), n_plot=args.n_plot)
    plot_start_end(trajs_shuffled[:args.n_plot],
                   os.path.join(args.out_dir, "start_end.png"))
    plot_distributions(trajs, vtypes,
                       os.path.join(args.out_dir, "distributions.png"))
    plot_individual_tracks(trajs, vtypes,
                           os.path.join(args.out_dir, "individual_tracks.png"),
                           n=args.n_individual, seed=args.seed)
    plot_vessel_type_breakdown(vtypes,
                               os.path.join(args.out_dir, "vessel_types.png"))

    print(f"\nAll plots saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
