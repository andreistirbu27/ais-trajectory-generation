#!/usr/bin/env python3
"""
filter_trajgen_npz.py — drop land-crossing tracks from an existing trajgen NPZ.

Loads `data/processed/trajgen_128.npz`, samples the coarse SDF at every
waypoint, and keeps only tracks whose on-land content is below the given
thresholds. Used for the E1/E2 land-filter experiments when re-running the
full prepare_trajgen pipeline from CSV is not an option.

Default rule: a track is dropped if MORE than `--land5_max_frac` of its
points sit > 5 km on land OR if its deepest land penetration exceeds
`--max_pen_km`. Both are evaluated on the LandMask coarse SDF.

Usage:
    python3 scripts/data/filter_trajgen_npz.py \\
        --in_npz   data/processed/trajgen_128.npz \\
        --out_npz  data/processed/trajgen_128_clean.npz \\
        --land_sdf data/processed/land_sdf_050deg.npz
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.land_mask import LandMask


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--in_npz",   default="data/processed/trajgen_128.npz")
    ap.add_argument("--out_npz",  required=True)
    ap.add_argument("--land_sdf", default="data/processed/land_sdf_050deg.npz")
    ap.add_argument("--land5_max_frac", type=float, default=0.02,
                    help="Max fraction of points allowed > 5 km on land.")
    ap.add_argument("--max_pen_km",     type=float, default=10.0,
                    help="Max single-point land penetration (km).")
    args = ap.parse_args()

    print(f"Loading {args.in_npz}")
    z = np.load(args.in_npz, allow_pickle=True)
    trajs = z["trajectories"].astype(np.float32)
    vts   = z["vessel_types"]
    tids  = z["track_ids"]
    N = trajs.shape[0]
    print(f"  N = {N:,}, T = {trajs.shape[1]}")

    print(f"Loading {args.land_sdf}")
    land = LandMask.load(args.land_sdf)
    sdf = land.sample_km_np(trajs[:, :, 0], trajs[:, :, 1])  # (N, T)

    land5_frac = (sdf > 5.0).mean(axis=1)
    max_pen = sdf.max(axis=1)
    keep = (land5_frac <= args.land5_max_frac) & (max_pen <= args.max_pen_km)
    n_kept = int(keep.sum())
    print(f"\nFilter: land5_frac <= {args.land5_max_frac} AND max_pen_km <= {args.max_pen_km}")
    print(f"  kept {n_kept:,} / {N:,}  ({100 * n_kept / N:.1f}%)")
    print(f"  dropped: {N - n_kept:,}")

    # GT diagnostic on kept subset
    sub = trajs[keep]
    s10 = land.trajectory_stats(sub, threshold_km=10.0)
    s0  = land.trajectory_stats(sub, threshold_km=0.0)
    print(f"\nGT land-crossing on kept subset:")
    print(f"  threshold 10 km: land%={100*s10['land_frac']:5.2f}  cross%={100*s10['traj_crossing_rate']:5.2f}")
    print(f"  threshold  0 km: land%={100*s0['land_frac']:5.2f}  cross%={100*s0['traj_crossing_rate']:5.2f}")

    out_dir = os.path.dirname(args.out_npz) or "."
    os.makedirs(out_dir, exist_ok=True)
    print(f"\nSaving {args.out_npz}")
    np.savez(args.out_npz,
             trajectories=trajs[keep].astype(np.float32),
             vessel_types=vts[keep],
             track_ids=tids[keep])
    sz = os.path.getsize(args.out_npz) / 1e6
    print(f"  size: {sz:.1f} MB")


if __name__ == "__main__":
    main()
