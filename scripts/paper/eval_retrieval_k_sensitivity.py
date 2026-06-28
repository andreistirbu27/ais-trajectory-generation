#!/usr/bin/env python3
"""Retrieval-K sensitivity: how many retrieved analogues does v10 actually use?

Eval-only ablation on the production v10 checkpoint — no retraining. The model
was trained with max_retrieved=5, so we sweep K in {1,2,3,5} by subsetting the
cached 5-NN columns (K>5 would overflow the route-index embedding). Reports
v10-raw ADE per K on the cleaned US corpus. WaterRouter post-processing is
K-independent and omitted here (it adds ~0 ADE on clean data).

Usage:
    python3 scripts/paper/eval_retrieval_k_sensitivity.py \
        --data_npz data/processed/trajgen_128_clean.npz \
        --checkpoint runs/trajgen_v10/best.pt \
        --knn_cache data/processed/trajgen_128_clean_knn_k5.npz \
        --n_eval 500 --ks 1 2 3 5 --seeds 42
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
from src.metrics_gen import haversine_meters  # noqa: E402


def ade_m(pred, gt):
    """Mean per-point haversine error (m) over (N, T, 2) [lon, lat] arrays."""
    d = haversine_meters(pred[..., 1], pred[..., 0], gt[..., 1], gt[..., 0])
    return float(d.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_npz", default="data/processed/trajgen_128_clean.npz")
    ap.add_argument("--checkpoint", default="runs/trajgen_v10/best.pt")
    ap.add_argument("--knn_cache",
                    default="data/processed/trajgen_128_clean_knn_k5.npz")
    ap.add_argument("--n_eval", type=int, default=500)
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 2, 3, 5])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 42, 123])
    ap.add_argument("--out_csv", default="results/paper/retrieval_k_sensitivity.csv")
    args = ap.parse_args()

    device = get_device()
    model, scalers, vocab, T, ckpt = load_v10(args.checkpoint, device)
    cargs = ckpt["args"]
    val_frac = cargs.get("val_frac", 0.15) if isinstance(cargs, dict) \
        else getattr(cargs, "val_frac", 0.15)
    seed = cargs.get("seed", 42) if isinstance(cargs, dict) \
        else getattr(cargs, "seed", 42)
    print(f"Loaded {args.checkpoint}  (val_frac={val_frac}, split_seed={seed})")

    data = np.load(args.data_npz, allow_pickle=True)
    trajs = data["trajectories"].astype(np.float32)
    vts = data["vessel_types"].astype(np.int32)
    tids = data["track_ids"]
    train_traj, _, _, val_traj, val_vt, _ = train_val_split_gen(
        trajs, vts, tids, val_frac, seed)

    knn = np.load(args.knn_cache)
    val_knn_full = knn["val_knn"]
    max_k = val_knn_full.shape[1]
    ks = [k for k in args.ks if k <= max_k]
    print(f"Val: {len(val_traj):,}   sweeping K={ks}   seeds={args.seeds}\n")

    rows = []
    for K in ks:
        per_seed = []
        for sd in args.seeds:
            rng = np.random.RandomState(sd)
            n = min(args.n_eval, len(val_traj))
            idx = rng.choice(len(val_traj), n, replace=False)
            vt_g = val_traj[idx]
            vt_v = val_vt[idx]
            vt_knn = val_knn_full[idx, :K]               # subset to K neighbours
            pred = _gen_v10(model, scalers, vocab, vt_g, vt_v, train_traj,
                            vt_knn, T, device, land_mask=None)
            per_seed.append(ade_m(pred, vt_g))
        mean, std = float(np.mean(per_seed)), float(np.std(per_seed))
        rows.append(dict(K=K, ade_m_mean=round(mean, 1), ade_m_std=round(std, 1),
                         n_seeds=len(args.seeds), n_eval=args.n_eval))
        print(f"  K={K}:  ADE = {mean:7.1f} +/- {std:5.1f} m")

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["K", "ade_m_mean", "ade_m_std",
                                          "n_seeds", "n_eval"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {args.out_csv}")


if __name__ == "__main__":
    main()
