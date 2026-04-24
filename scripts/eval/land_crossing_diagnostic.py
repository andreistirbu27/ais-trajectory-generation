#!/usr/bin/env python3
"""
land_crossing_diagnostic.py — Report land-crossing statistics for candidate
trajectory sources.

Runs (model-free) diagnostics on:
  - Ground truth val trajectories (should be near-zero on land)
  - Retrieval-top-1 KNN predictions
  - Great-circle predictions

Requires a prebuilt land SDF NPZ (see src/land_mask.py).

Usage:
    python3 scripts/eval/land_crossing_diagnostic.py \
        --data_npz data/processed/trajgen_128.npz \
        --knn_cache data/processed/trajgen_128_knn_k5.npz \
        --land_sdf data/processed/land_sdf_050deg.npz \
        --n_eval 500
"""
import argparse
import os
import sys

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.data_gen import train_val_split_gen
from src.land_mask import LandMask
from src.metrics_gen import great_circle_trajectory


def print_stats(name: str, stats: dict) -> None:
    print(f"  {name:<30s}  "
          f"land%={stats['land_frac']*100:5.2f}  "
          f"cross%={stats['traj_crossing_rate']*100:5.1f}  "
          f"mean_pts={stats['mean_on_land_pts']:5.2f}  "
          f"max_pen={stats['mean_max_penetration_km']:5.1f} km")


def sdf_histogram(traj: np.ndarray, mask: LandMask) -> None:
    """Print a text histogram of SDF values across all waypoints."""
    sdf = mask.sample_km_np(traj[:, :, 0], traj[:, :, 1]).ravel()
    bins = [-np.inf, -100, -50, -20, -10, -5, 0, 5, 10, 25, 50, 100, np.inf]
    labels = ["<-100", "-100..-50", "-50..-20", "-20..-10", "-10..-5",
              "-5..0", "0..5", "5..10", "10..25", "25..50", "50..100", ">100"]
    counts, _ = np.histogram(sdf, bins=bins)
    total = counts.sum()
    for lab, c in zip(labels, counts):
        pct = 100 * c / total
        bar = "#" * int(pct)
        print(f"    sdf_km {lab:>11} : {pct:5.2f}%  {bar}")


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data_npz",  required=True)
    p.add_argument("--knn_cache", required=True)
    p.add_argument("--land_sdf",  default="data/processed/land_sdf_050deg.npz")
    p.add_argument("--val_frac",  type=float, default=0.15)
    p.add_argument("--seed",      type=int,   default=42)
    p.add_argument("--n_eval",    type=int,   default=500)
    p.add_argument("--traj_npz",  default=None,
                   help="Optional path to a (N,T,2) trajectory NPZ under key "
                        "'pred' to include model outputs (v5, v9, v10, …)")
    p.add_argument("--traj_label", default="model",
                   help="Label for --traj_npz trajectories in the report")
    p.add_argument("--thresholds", type=float, nargs="+",
                   default=[0.0, 5.0, 10.0, 25.0],
                   help="km thresholds for 'on land' classification")
    args = p.parse_args()

    print(f"Loading land SDF: {args.land_sdf}")
    mask = LandMask.load(args.land_sdf)
    print(f"  Grid: {mask.shape}  bbox: {mask.bbox}\n")

    data = np.load(args.data_npz, allow_pickle=True)
    trajectories = data["trajectories"].astype(np.float32)
    vessel_types = data["vessel_types"].astype(np.int32)
    track_ids    = data["track_ids"]
    train_traj, _, _, val_traj, _, _ = train_val_split_gen(
        trajectories, vessel_types, track_ids, args.val_frac, args.seed)

    knn = np.load(args.knn_cache)
    val_knn = knn["val_knn"]

    # Subsample val consistently with eval_gen_retrieval.py (seed 42)
    if args.n_eval > 0 and args.n_eval < len(val_traj):
        rng = np.random.RandomState(42)
        idx = rng.choice(len(val_traj), args.n_eval, replace=False)
        val_traj = val_traj[idx]
        val_knn = val_knn[idx]

    N, T, _ = val_traj.shape
    print(f"Evaluating on {N} val trajectories (T={T})\n")

    # Pre-compute GC (slow loop only once)
    gc_traj = np.zeros_like(val_traj)
    for i in tqdm(range(N), desc="GC", leave=False):
        gc_traj[i] = great_circle_trajectory(val_traj[i, 0], val_traj[i, -1], T)
    retr = train_traj[val_knn[:, 0]]

    sources = [
        ("Ground truth (val)", val_traj),
        ("Retrieval-top-1 (KNN)", retr),
        ("Great-circle", gc_traj),
    ]
    if args.traj_npz is not None:
        ext = np.load(args.traj_npz)
        pred = ext["pred"].astype(np.float32)
        assert pred.shape == val_traj.shape, \
            f"Traj shape {pred.shape} must match val_traj {val_traj.shape}"
        sources.append((args.traj_label, pred))

    for threshold in args.thresholds:
        print("=" * 95)
        print(f"  LAND-CROSSING DIAGNOSTIC  (threshold = {threshold:.1f} km)")
        print("=" * 95)
        for label, traj in sources:
            print_stats(label, mask.trajectory_stats(traj, threshold_km=threshold))
        print()

    # Histogram of SDF values on GT trajectories to pick a threshold visually
    print("=" * 95)
    print("  SDF HISTOGRAM — ground truth waypoints (negative = water, positive = land)")
    print("=" * 95)
    sdf_histogram(val_traj, mask)

    print("\nNotes:")
    print("  threshold  = SDF_km value above which a waypoint is 'on land'")
    print("  Use a threshold ≥ a few km to discount rasterization noise in")
    print("  harbors/bays; use 0 km for strict classification.")


if __name__ == "__main__":
    main()
