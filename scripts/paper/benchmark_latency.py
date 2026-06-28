#!/usr/bin/env python3
"""Inference-latency micro-benchmark: v10 (single-shot) vs TrAISformer (best-of-16).

Strengthens the deployment story and defuses the "best-of-16 is unfair" critique:
v10 produces a full route in one deterministic forward rollout, while TrAISformer's
reported numbers need 16 stochastic samples per track. This script times both on
the same GPU and reports milliseconds per trajectory.

Generation cost depends on tensor *shapes* (batch, sequence length, retrieval K /
sample count), not on the actual coordinate values, so we drive the real model
generation paths with correctly-shaped synthetic inputs — no DMA data files needed.
We reuse the exact generation functions used by the K-anchor harness:
  - v10:         scripts.eval.eval_all_clean._gen_v10   (single-shot, deterministic)
  - TrAISformer: scripts.paper.eval_traisformer.predict_trajectories (n_samples=16)

Usage:
    paper/.venv/bin/python3 scripts/paper/benchmark_latency.py \
        --v10_checkpoint runs/paper/v10_dma/best.pt \
        --traisformer_checkpoint paper/external/TrAISformer/results/<run>/model.pt \
        --out_csv results/paper/latency.csv
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from scripts.eval.eval_all_clean import _gen_v10, get_device, load_v10  # noqa: E402
from scripts.paper.eval_traisformer import (  # noqa: E402
    load_traisformer, predict_trajectories,
)

# DMA (Kattegat) bbox — synthetic positions live here so v10's scaler sees
# in-distribution magnitudes. Values are irrelevant to timing; shapes are not.
DMA_LON = (10.3, 13.0)
DMA_LAT = (55.5, 58.0)


def _param_millions(model) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def _time_call(fn, n_warmup, n_iter, device):
    """Return (mean_seconds_per_call) over n_iter timed calls after n_warmup."""
    for _ in range(n_warmup):
        fn()
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(n_iter):
        fn()
    _sync(device)
    return (time.perf_counter() - t0) / n_iter


def bench_v10(ckpt, device, batch_sizes, n_warmup, n_iter):
    model, scalers, vocab, T, _ = load_v10(ckpt, device)
    params_m = _param_millions(model)
    K = 5  # retrieval bank size (ckpt k_retrieval); _gen_v10 infers K from val_knn
    rng = np.random.default_rng(0)
    pool = rng.uniform(
        [DMA_LON[0], DMA_LAT[0]], [DMA_LON[1], DMA_LAT[1]],
        size=(max(64, K), T, 2)).astype(np.float32)  # retrieval corpus
    rows = []
    for B in batch_sizes:
        val_traj = rng.uniform(
            [DMA_LON[0], DMA_LAT[0]], [DMA_LON[1], DMA_LAT[1]],
            size=(B, T, 2)).astype(np.float32)
        val_vt = np.zeros(B, dtype=np.int64)
        val_knn = rng.integers(0, len(pool), size=(B, K))

        def call():
            _gen_v10(model, scalers, vocab, val_traj, val_vt, pool, val_knn,
                     T, device, land_mask=None, batch_size=B)

        sec = _time_call(call, n_warmup, n_iter, device)
        rows.append(dict(method="v10", batch_size=B, n_samples=1,
                         params_m=round(params_m, 2),
                         ms_per_traj=round(1000 * sec / B, 3)))
        print(f"  v10            B={B:<3d} {1000*sec/B:8.2f} ms/traj")
    return rows


def bench_traisformer(ckpt, device, batch_sizes, n_warmup, n_iter, n_samples):
    model, cf = load_traisformer(ckpt, device)
    params_m = _param_millions(model)
    init_seqlen = cf.init_seqlen
    max_seqlen = cf.max_seqlen
    rng = np.random.default_rng(0)
    rows = []
    for B in batch_sizes:
        # TrAISformer's 4 native features in their normalised [0,1) space.
        hist = rng.uniform(0.0, 0.9999, size=(B, init_seqlen, 4)).astype(np.float32)

        def call():
            predict_trajectories(model, cf, hist, max_seqlen, device,
                                 n_samples=n_samples)

        # Autoregressive ×n_samples is slow; fewer timed iters is plenty.
        sec = _time_call(call, max(1, n_warmup // 2), max(3, n_iter // 4), device)
        rows.append(dict(method=f"traisformer_b{n_samples}", batch_size=B,
                         n_samples=n_samples, params_m=round(params_m, 2),
                         ms_per_traj=round(1000 * sec / B, 3)))
        print(f"  traisformer×{n_samples} B={B:<3d} {1000*sec/B:8.2f} ms/traj")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v10_checkpoint", default="runs/paper/v10_dma/best.pt")
    ap.add_argument("--traisformer_checkpoint", default=None,
                    help="path to TrAISformer model.pt (skip its row if omitted)")
    ap.add_argument("--batch_sizes", type=int, nargs="+", default=[1, 32])
    ap.add_argument("--n_warmup", type=int, default=3)
    ap.add_argument("--n_iter", type=int, default=20)
    ap.add_argument("--traisformer_n_samples", type=int, default=16)
    ap.add_argument("--out_csv", default="results/paper/latency.csv")
    args = ap.parse_args()

    device = get_device()
    print(f"Device: {device}")

    # load_traisformer chdir's into the submodule before torch.load, so a
    # relative ckpt path would break — resolve to absolute up front.
    v10_ckpt = os.path.abspath(args.v10_checkpoint)
    tf_ckpt = (os.path.abspath(args.traisformer_checkpoint)
               if args.traisformer_checkpoint else None)

    rows = []
    print("v10 (single-shot, deterministic):")
    rows += bench_v10(v10_ckpt, device, args.batch_sizes,
                      args.n_warmup, args.n_iter)

    if tf_ckpt:
        print(f"TrAISformer (best-of-{args.traisformer_n_samples}):")
        rows += bench_traisformer(
            tf_ckpt, device, args.batch_sizes,
            args.n_warmup, args.n_iter, args.traisformer_n_samples)
    else:
        print("[skip] no --traisformer_checkpoint given")

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    cols = ["method", "batch_size", "n_samples", "params_m", "ms_per_traj"]
    import csv
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {args.out_csv}")


if __name__ == "__main__":
    main()
