#!/usr/bin/env python3
"""Quantify v10's failure modes: bucket the worst-ADE trajectories by cause.

Turns the appendix's narrative failure-mode discussion into counts. Runs v10
(raw) on the cleaned US corpus, ranks trajectories by ADE, and categorises the
worst-N tail by measurable geometric proxies of the GT route:

  - narrow / coastal : closest approach to land < clearance_km (tight passages,
                       where small lateral error crosses the shoreline)
  - maneuvering      : tortuosity = path_len / great-circle dist > tortuosity_max
                       (winding routes, port approaches, course changes)
  - long-route       : great-circle start->end distance > long_km
  - other            : none of the above

A trajectory can trip several flags; we report per-flag counts plus a single
priority assignment (narrow > maneuvering > long > other).

Usage:
    python3 scripts/paper/failure_mode_analysis.py \
        --data_npz data/processed/trajgen_128_clean.npz \
        --checkpoint runs/trajgen_v10/best.pt \
        --knn_cache data/processed/trajgen_128_clean_knn_k5.npz \
        --land_sdf data/processed/land_sdf_050deg.npz \
        --n_eval 1000 --worst_n 25 --seed 42
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from scripts.eval.eval_all_clean import _gen_v10, get_device, load_v10  # noqa: E402
from src.data_gen import train_val_split_gen  # noqa: E402
from src.land_mask import LandMask  # noqa: E402
from src.metrics_gen import haversine_meters  # noqa: E402


def per_traj_ade(pred, gt):
    return haversine_meters(pred[..., 1], pred[..., 0],
                            gt[..., 1], gt[..., 0]).mean(axis=1)  # (N,)


def great_circle_km(traj):
    return haversine_meters(traj[:, 0, 1], traj[:, 0, 0],
                            traj[:, -1, 1], traj[:, -1, 0]) / 1000.0  # (N,)


def path_len_km(traj):
    seg = haversine_meters(traj[:, :-1, 1], traj[:, :-1, 0],
                           traj[:, 1:, 1], traj[:, 1:, 0])
    return seg.sum(axis=1) / 1000.0  # (N,)


def min_clearance_km(traj, mask):
    """Closest approach to land (km) along each route. Positive = water side."""
    N, T, _ = traj.shape
    sdf = mask.sample_km_np(traj[..., 0].ravel(), traj[..., 1].ravel())
    sdf = np.asarray(sdf).reshape(N, T)        # positive = land
    return (-sdf).min(axis=1)                   # min distance to land


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_npz", default="data/processed/trajgen_128_clean.npz")
    ap.add_argument("--checkpoint", default="runs/trajgen_v10/best.pt")
    ap.add_argument("--knn_cache",
                    default="data/processed/trajgen_128_clean_knn_k5.npz")
    ap.add_argument("--land_sdf", default="data/processed/land_sdf_050deg.npz")
    ap.add_argument("--n_eval", type=int, default=1000)
    ap.add_argument("--worst_n", type=int, default=25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--clearance_km", type=float, default=5.0)
    ap.add_argument("--tortuosity_max", type=float, default=1.3)
    ap.add_argument("--long_km", type=float, default=500.0)
    ap.add_argument("--out_csv", default="results/paper/failure_modes.csv")
    args = ap.parse_args()

    device = get_device()
    model, scalers, vocab, T, ckpt = load_v10(args.checkpoint, device)
    cargs = ckpt["args"]
    val_frac = cargs.get("val_frac", 0.15) if isinstance(cargs, dict) \
        else getattr(cargs, "val_frac", 0.15)
    split_seed = cargs.get("seed", 42) if isinstance(cargs, dict) \
        else getattr(cargs, "seed", 42)
    mask = LandMask.load(args.land_sdf)

    data = np.load(args.data_npz, allow_pickle=True)
    trajs = data["trajectories"].astype(np.float32)
    vts = data["vessel_types"].astype(np.int32)
    tids = data["track_ids"]
    train_traj, _, _, val_traj, val_vt, _ = train_val_split_gen(
        trajs, vts, tids, val_frac, split_seed)

    knn = np.load(args.knn_cache)
    val_knn = knn["val_knn"]

    rng = np.random.RandomState(args.seed)
    n = min(args.n_eval, len(val_traj))
    idx = rng.choice(len(val_traj), n, replace=False)
    gt, vt, kn = val_traj[idx], val_vt[idx], val_knn[idx]

    print(f"Generating v10 raw on n={n} (split_seed={split_seed})...")
    pred = _gen_v10(model, scalers, vocab, gt, vt, train_traj, kn, T, device,
                    land_mask=None)

    ade = per_traj_ade(pred, gt)
    gc = great_circle_km(gt)
    tort = np.where(gc > 1e-6, path_len_km(gt) / np.maximum(gc, 1e-6), 1.0)
    clr = min_clearance_km(gt, mask)

    order = np.argsort(ade)[::-1]
    worst = order[:args.worst_n]
    print(f"\nOverall median ADE: {np.median(ade):.0f} m   "
          f"worst-{args.worst_n} median ADE: {np.median(ade[worst]):.0f} m\n")

    # Per-flag counts among the worst-N (flags can overlap).
    is_narrow = clr[worst] < args.clearance_km
    is_maneuver = tort[worst] > args.tortuosity_max
    is_long = gc[worst] > args.long_km
    flags = {
        "narrow_or_coastal": int(is_narrow.sum()),
        "maneuvering": int(is_maneuver.sum()),
        "long_route": int(is_long.sum()),
    }
    # Single priority assignment: narrow > maneuvering > long > other.
    primary = []
    for j in range(len(worst)):
        if is_narrow[j]:
            primary.append("narrow_or_coastal")
        elif is_maneuver[j]:
            primary.append("maneuvering")
        elif is_long[j]:
            primary.append("long_route")
        else:
            primary.append("other")
    prim_counts = {c: primary.count(c) for c in
                   ["narrow_or_coastal", "maneuvering", "long_route", "other"]}

    print(f"Worst-{args.worst_n} flag counts (overlapping):")
    for k, v in flags.items():
        print(f"  {k:20s} {v:3d} / {args.worst_n}")
    print(f"\nPrimary cause (priority: narrow > maneuver > long > other):")
    for k, v in prim_counts.items():
        print(f"  {k:20s} {v:3d} / {args.worst_n}")

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["category", "overlapping_count", "primary_count", "worst_n",
                    "overall_median_ade_m", "worst_median_ade_m"])
        om, wm = float(np.median(ade)), float(np.median(ade[worst]))
        for c in ["narrow_or_coastal", "maneuvering", "long_route"]:
            w.writerow([c, flags[c], prim_counts[c], args.worst_n,
                        round(om, 1), round(wm, 1)])
        w.writerow(["other", "", prim_counts["other"], args.worst_n,
                    round(om, 1), round(wm, 1)])
    print(f"\nWrote {args.out_csv}")


if __name__ == "__main__":
    main()
