#!/usr/bin/env python3
"""
retrieval_baseline.py — Gate for v9 (retrieval-augmented generation).

For each val trajectory, find the single nearest train trajectory by
(start, end, vessel_type) and report its ADE against val GT. No model,
no training — just tells us whether the corpus has usable templates
for the typical query.

Decision rule:
  - ADE < 6.7 km  → retrieval beats v5/MVP; build v9.
  - ADE ≈ 10 km+  → corpus too sparse; skip v9.

Usage:
    python3 scripts/eval/retrieval_baseline.py \
        --data_npz data/processed/trajgen_128.npz \
        --val_frac 0.15 --seed 42
"""

import argparse
import os
import sys

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.data_gen import train_val_split_gen
from src.metrics_gen import (
    evaluate_generation,
    great_circle_trajectory,
    haversine_meters,
)


def build_query(trajectories: np.ndarray, vessel_types: np.ndarray) -> np.ndarray:
    """Query vector per trajectory: [start_lon, start_lat, end_lon, end_lat, vtype].

    Positions are in raw degrees; vtype is the raw AIS code.
    """
    start = trajectories[:, 0, :]    # (N, 2)
    end   = trajectories[:, -1, :]   # (N, 2)
    return np.concatenate(
        [start, end, vessel_types.astype(np.float32)[:, None]], axis=1
    )  # (N, 5)


def retrieve_top1(val_q: np.ndarray, train_q: np.ndarray,
                  vtype_weight: float, use_vtype: bool) -> np.ndarray:
    """For each val query, return the index of the nearest train query.

    Distance is weighted Euclidean in the 5-D space. Position dims are in
    degrees (~111 km per degree of lat); vtype is categorical, so we scale
    it by vtype_weight degrees-equivalent.

    Args:
        val_q:        (Nv, 5)
        train_q:      (Nt, 5)
        vtype_weight: degrees-equivalent penalty for mismatched vtype
        use_vtype:    if False, ignore vtype entirely (positions only)

    Returns:
        nn_idx: (Nv,) index into train_q of the nearest train sample.
    """
    # Split position vs vtype columns
    val_pos,   val_vt   = val_q[:, :4],   val_q[:, 4]
    train_pos, train_vt = train_q[:, :4], train_q[:, 4]

    # Position distance (batched): Euclidean in degrees
    # (Nv, 4) vs (Nt, 4) -> (Nv, Nt). Memory: Nv*Nt floats.
    # For a typical 30k val x 180k train = 5.4B floats = 21 GB — too big.
    # Chunk over val instead.
    Nv = len(val_q)
    nn_idx = np.empty(Nv, dtype=np.int64)
    chunk = 256
    for i in tqdm(range(0, Nv, chunk), desc="KNN", unit="chunk"):
        j = min(i + chunk, Nv)
        # Position squared distance (chunk, Nt)
        diff = val_pos[i:j, None, :] - train_pos[None, :, :]    # (chunk, Nt, 4)
        d2 = (diff ** 2).sum(axis=-1)                            # (chunk, Nt)
        if use_vtype:
            mismatch = (val_vt[i:j, None] != train_vt[None, :]).astype(np.float32)
            d2 = d2 + (vtype_weight ** 2) * mismatch
        nn_idx[i:j] = d2.argmin(axis=1)
    return nn_idx


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data_npz", required=True, help="Path to trajgen NPZ")
    p.add_argument("--val_frac", type=float, default=0.15,
                   help="Must match training val_frac to reproduce split")
    p.add_argument("--seed",     type=int,   default=42,
                   help="Must match training seed to reproduce split")
    p.add_argument("--n_eval",   type=int,   default=500,
                   help="Evaluate this many val samples (0 = all)")
    p.add_argument("--vtype_weight", type=float, default=5.0,
                   help="Vtype-mismatch penalty in degrees (positions are also in degrees)")
    args = p.parse_args()

    # Load data
    print(f"Loading {args.data_npz}...")
    data = np.load(args.data_npz, allow_pickle=True)
    trajectories = data["trajectories"].astype(np.float32)   # (N, 128, 2)
    vessel_types = data["vessel_types"].astype(np.int32)
    track_ids    = data["track_ids"]

    (train_traj, train_vt, _train_ids,
     val_traj,   val_vt,   _val_ids) = train_val_split_gen(
        trajectories, vessel_types, track_ids, args.val_frac, args.seed)

    print(f"  Train: {len(train_traj):,}   Val: {len(val_traj):,}")

    # Optionally subsample val for speed — matches eval_gen.py default (500)
    if args.n_eval > 0 and args.n_eval < len(val_traj):
        # Deterministic subsample; matches eval_gen.py which takes first N
        val_traj = val_traj[:args.n_eval]
        val_vt   = val_vt[:args.n_eval]

    # Build query vectors
    train_q = build_query(train_traj, train_vt)
    val_q   = build_query(val_traj,   val_vt)

    # Two variants: with vtype and without
    for use_vtype, label in [(True, "with vtype"), (False, "positions only")]:
        print(f"\n─── Retrieval ({label}) ──────────────────────────────")
        nn_idx = retrieve_top1(val_q, train_q, args.vtype_weight, use_vtype)

        # Diagnostic: how close are the retrieved queries?
        diffs = val_q[:, :4] - train_q[nn_idx, :4]
        mean_start_km = haversine_meters(
            val_q[:, 1], val_q[:, 0], train_q[nn_idx, 1], train_q[nn_idx, 0]
        ).mean() / 1000.0
        mean_end_km = haversine_meters(
            val_q[:, 3], val_q[:, 2], train_q[nn_idx, 3], train_q[nn_idx, 2]
        ).mean() / 1000.0
        vtype_match = (val_q[:, 4] == train_q[nn_idx, 4]).mean() * 100.0
        print(f"  Mean start distance to retrieved: {mean_start_km:.2f} km")
        print(f"  Mean end   distance to retrieved: {mean_end_km:.2f} km")
        print(f"  Vtype-match rate                : {vtype_match:.1f}%")

        # Evaluate retrieved trajectory vs val GT
        pred = train_traj[nn_idx]     # (Nv, 128, 2) — retrieved route as prediction
        metrics = evaluate_generation(pred, val_traj, compute_frechet=False)
        print(f"  ADE.............................. {metrics['ade_m']:>9.1f} m")
        print(f"  FDE.............................. {metrics['fde_m']:>9.1f} m")
        print(f"  Path length ratio................ {metrics['path_length_ratio']:.3f}")

    # Great-circle baseline for context (skip Frechet — slow on 500 samples)
    print(f"\n─── Great-circle baseline ────────────────────────────")
    T = val_traj.shape[1]
    gc_traj = np.zeros_like(val_traj)
    for i in range(len(val_traj)):
        gc_traj[i] = great_circle_trajectory(val_traj[i, 0], val_traj[i, -1], T)
    gc = evaluate_generation(gc_traj, val_traj, compute_frechet=False)
    print(f"  ADE.............................. {gc['ade_m']:>9.1f} m")
    print(f"  FDE.............................. {gc['fde_m']:>9.1f} m")

    # Decision reminder
    print(f"\n─── Decision ─────────────────────────────────────────")
    print(f"  v5 (MVP) model ADE: 6753 m")
    print(f"  If retrieval ADE <  ~6700 m → v9 is worth building")
    print(f"  If retrieval ADE > ~10000 m → corpus too sparse, skip v9")


if __name__ == "__main__":
    main()
