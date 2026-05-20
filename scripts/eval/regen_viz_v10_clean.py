#!/usr/bin/env python3
"""regen_viz_v10_clean.py — regenerate the qualitative figure
``figures/viz_v10_clean.png`` (referenced by ``report/report_v2.tex`` Fig. 8)
from the v10 checkpoint and the cleaned corpus.

Previously this PNG existed in ``figures/`` and ``report/figures/`` with no
generation script. This file plugs that reproducibility gap by reusing the
helpers in ``scripts/eval/visualize_gen_v10.py`` so a single command
reproduces the figure end-to-end from raw artefacts.

Usage:
    python3 scripts/eval/regen_viz_v10_clean.py
"""

import argparse
import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.data_gen import train_val_split_gen
from src.land_mask import LandMask
from src.metrics_gen import great_circle_trajectory
# Re-export helpers from the existing visualiser.
from scripts.eval.visualize_gen_v10 import (
    get_device, load_checkpoint, generate_one, plot_grid)


def main() -> None:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data_npz",   default="data/processed/trajgen_128_clean.npz")
    p.add_argument("--checkpoint", default="runs/trajgen_v10/best.pt")
    p.add_argument("--knn_cache",  default="data/processed/trajgen_128_clean_knn_k5.npz")
    p.add_argument("--land_sdf",   default="data/processed/land_sdf_050deg.npz")
    p.add_argument("--hard_threshold_km", type=float, default=10.0)
    p.add_argument("--n_plots", type=int, default=16,
                   help="Number of routes in the qualitative grid")
    p.add_argument("--sample_seed", type=int, default=42,
                   help="Seed for sampling which val routes to plot")
    p.add_argument("--out", default="figures/viz_v10_clean.png")
    p.add_argument("--mirror_to", default="report/figures/viz_v10_clean.png",
                   help="Also copy the rendered PNG here (set empty to skip)")
    args = p.parse_args()

    device = get_device()
    print(f"Device: {device}")

    print(f"Loading checkpoint: {args.checkpoint}")
    model, scalers, vtype_vocab, n_resample, train_args = load_checkpoint(
        args.checkpoint, device)
    print(f"  k_retrieval={train_args.k_retrieval}  t_retr={train_args.t_retr}  "
          f"k_past={getattr(train_args, 'k_past', None)}")

    print(f"Loading SDF: {args.land_sdf}")
    land_mask = LandMask.load(args.land_sdf)

    print(f"Loading data:     {args.data_npz}")
    data = np.load(args.data_npz, allow_pickle=True)
    trajectories = data["trajectories"].astype(np.float32)
    vessel_types = data["vessel_types"].astype(np.int32)
    track_ids    = data["track_ids"]
    train_traj, _, _, val_traj, val_vt, _ = train_val_split_gen(
        trajectories, vessel_types, track_ids,
        train_args.val_frac, train_args.seed)
    print(f"  Val: {len(val_traj):,} routes")

    print(f"Loading KNN cache: {args.knn_cache}")
    knn = np.load(args.knn_cache)
    val_knn = knn["val_knn"]

    rng = np.random.RandomState(args.sample_seed)
    n_plots = min(args.n_plots, len(val_traj))
    sample_idx = rng.choice(len(val_traj), n_plots, replace=False)
    print(f"Sampling {n_plots} routes for the qualitative grid (seed={args.sample_seed})")

    gt_list, raw_list, proj_list, gc_list, retr_list, vt_list = [], [], [], [], [], []
    for i in sample_idx:
        gt = val_traj[i]
        vt = val_vt[i]
        retrieved = train_traj[val_knn[i]]
        raw = generate_one(model, gt[0], gt[-1], vt, retrieved,
                           vtype_vocab, scalers, n_resample, device,
                           land_mask=None)
        proj = generate_one(model, gt[0], gt[-1], vt, retrieved,
                            vtype_vocab, scalers, n_resample, device,
                            land_mask=land_mask,
                            hard_threshold_km=args.hard_threshold_km)
        gc = great_circle_trajectory(gt[0], gt[-1], n_resample)
        gt_list.append(gt)
        raw_list.append(raw)
        proj_list.append(proj)
        gc_list.append(gc)
        retr_list.append(retrieved)
        vt_list.append(vt)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plot_grid(
        np.array(gt_list), np.array(raw_list),
        np.array(proj_list), np.array(gc_list),
        np.array(retr_list), np.array(vt_list),
        args.out,
        ncols=4)

    if args.mirror_to:
        os.makedirs(os.path.dirname(args.mirror_to) or ".", exist_ok=True)
        shutil.copy(args.out, args.mirror_to)
        print(f"Mirrored → {args.mirror_to}")


if __name__ == "__main__":
    main()
