#!/usr/bin/env python3
"""
eval_gen.py — Evaluate a trajectory generation checkpoint.

Loads a trained TrajectoryGenerator checkpoint, generates trajectories for
the validation set using autoregressive inference, and computes metrics
against ground truth and the great-circle baseline.

Usage:
    python3 scripts/eval/eval_gen.py \
        --data_npz data/processed/trajgen_128.npz \
        --checkpoint runs/trajgen_mvp/best.pt

    # Limit evaluation to N trajectories (faster)
    python3 scripts/eval/eval_gen.py \
        --data_npz data/processed/trajgen_128.npz \
        --checkpoint runs/trajgen_mvp/best.pt \
        --n_eval 500
"""

import argparse
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.data_gen import (TrajGenScalers, Scaler, train_val_split_gen,
                          build_vtype_vocab, get_trajgen_loader)
from src.model_gen import TrajectoryGenerator
from src.metrics_gen import evaluate_generation, evaluate_great_circle_baseline


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_checkpoint(checkpoint_path, device):
    """Load checkpoint and reconstruct model + scalers."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = argparse.Namespace(**ckpt["args"])

    scalers = TrajGenScalers(
        pos=Scaler(mean=ckpt["pos_mean"], std=ckpt["pos_std"]),
        delta=Scaler(mean=ckpt["delta_mean"], std=ckpt["delta_std"]),
    )

    model = TrajectoryGenerator(
        d_model=args.d_model,
        nhead=args.nhead,
        num_decoder_layers=args.num_decoder_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        num_vessel_types=ckpt["num_vessel_types"],
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    return model, scalers, ckpt["vtype_vocab"], ckpt.get("n_resample", 128), args


def generate_trajectories(model, val_traj, val_vt, vtype_vocab, scalers,
                          n_resample, device, n_eval=None):
    """Generate trajectories for val set using autoregressive inference.

    Returns:
        pred_traj: (N, T, 2) [lon, lat] degrees
        gt_traj:   (N, T, 2) [lon, lat] degrees
    """
    if n_eval is not None and n_eval < len(val_traj):
        rng = np.random.RandomState(42)
        idx = rng.choice(len(val_traj), n_eval, replace=False)
        val_traj = val_traj[idx]
        val_vt = val_vt[idx]

    N = len(val_traj)
    gt_traj = val_traj.copy()  # (N, T, 2) raw [lon, lat]
    pred_traj = np.zeros_like(gt_traj)

    batch_size = 32
    for start_idx in tqdm(range(0, N, batch_size), desc="Generating"):
        end_idx = min(start_idx + batch_size, N)
        batch_gt = gt_traj[start_idx:end_idx]   # (B, T, 2)
        batch_vt = val_vt[start_idx:end_idx]     # (B,)

        B = len(batch_gt)

        # Normalize start/end
        starts_raw = batch_gt[:, 0]               # (B, 2) [lon, lat]
        ends_raw = batch_gt[:, -1]                 # (B, 2) [lon, lat]
        starts_norm = scalers.pos.transform(starts_raw)
        ends_norm = scalers.pos.transform(ends_raw)

        # Vessel type indices
        vtypes = np.array([vtype_vocab.get(int(v), 0) for v in batch_vt])

        start_t = torch.tensor(starts_norm, dtype=torch.float32, device=device)
        end_t = torch.tensor(ends_norm, dtype=torch.float32, device=device)
        vtype_t = torch.tensor(vtypes, dtype=torch.long, device=device)

        # Generate
        with torch.no_grad():
            gen_norm = model.generate(
                start_t, end_t, vtype_t, n_steps=n_resample,
                pos_scaler=scalers.pos, delta_scaler=scalers.delta)
            # gen_norm: (T, B, 2) normalized

        gen_norm_np = gen_norm.permute(1, 0, 2).cpu().numpy()  # (B, T, 2)

        # Inverse normalize to raw [lon, lat]
        for i in range(B):
            pred_traj[start_idx + i] = scalers.pos.inverse(gen_norm_np[i])

    return pred_traj, gt_traj


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_npz", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--n_eval", type=int, default=None,
        help="Max trajectories to evaluate (None = all val)")
    parser.add_argument("--no_frechet", action="store_true",
        help="Skip Frechet distance computation (slow)")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}\n")

    # Load checkpoint
    model, scalers, vtype_vocab, n_resample, train_args = load_checkpoint(
        args.checkpoint, device)
    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"  Epoch {train_args.epochs}, d_model={train_args.d_model}, "
          f"layers={train_args.num_decoder_layers}\n")

    # Load data and reproduce val split
    data = np.load(args.data_npz, allow_pickle=True)
    trajectories = data["trajectories"]
    vessel_types = data["vessel_types"]
    track_ids = data["track_ids"]

    _, _, _, val_traj, val_vt, _ = train_val_split_gen(
        trajectories, vessel_types, track_ids,
        train_args.val_frac, train_args.seed)
    print(f"Val set: {len(val_traj):,} trajectories, {n_resample} points each\n")

    # Generate trajectories
    pred_traj, gt_traj = generate_trajectories(
        model, val_traj, val_vt, vtype_vocab, scalers,
        n_resample, device, n_eval=args.n_eval)

    N = len(pred_traj)
    compute_frechet = not args.no_frechet

    # ── Model metrics ────────────────────────────────────────────────────────
    print("=" * 65)
    print("  MODEL EVALUATION")
    print("=" * 65)
    model_metrics = evaluate_generation(pred_traj, gt_traj,
                                         compute_frechet=compute_frechet)
    for k, v in model_metrics.items():
        if isinstance(v, float):
            print(f"  {k:.<30} {v:>10.2f}")
        else:
            print(f"  {k:.<30} {v}")

    # ── Great-circle baseline ────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("  GREAT-CIRCLE BASELINE")
    print("=" * 65)
    gc_metrics = evaluate_great_circle_baseline(gt_traj)
    for k, v in gc_metrics.items():
        if isinstance(v, float):
            print(f"  {k:.<30} {v:>10.2f}")
        else:
            print(f"  {k:.<30} {v}")

    # ── Comparison ───────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("  COMPARISON (model vs great-circle)")
    print("=" * 65)
    for metric in ["ade_m", "fde_m", "endpoint_error_m"]:
        m_val = model_metrics.get(metric, float("nan"))
        gc_val = gc_metrics.get(metric, float("nan"))
        diff_pct = 100 * (m_val - gc_val) / gc_val if gc_val != 0 else 0
        better = "better" if m_val < gc_val else "worse"
        print(f"  {metric:.<25} model={m_val:>10.1f}  "
              f"GC={gc_val:>10.1f}  ({diff_pct:+.1f}% {better})")

    if compute_frechet:
        m_f = model_metrics.get("frechet_m", float("nan"))
        gc_f = gc_metrics.get("frechet_m", float("nan"))
        diff_pct = 100 * (m_f - gc_f) / gc_f if gc_f != 0 else 0
        better = "better" if m_f < gc_f else "worse"
        print(f"  {'frechet_m':.<25} model={m_f:>10.1f}  "
              f"GC={gc_f:>10.1f}  ({diff_pct:+.1f}% {better})")

    print("=" * 65)


if __name__ == "__main__":
    main()
