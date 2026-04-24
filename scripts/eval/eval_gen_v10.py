#!/usr/bin/env python3
"""
eval_gen_v10.py — Evaluate v10 trajectory generator.

Reports ADE/FDE/Frechet + normalized + length-bucketed metrics, plus
land-crossing stats, for:
  - v10 raw model output (no post-processing)
  - v10 with hard land projection (sdf > threshold → nearest water)
  - Retrieval-top-1 baseline
  - Great-circle baseline

Usage:
    python3 scripts/eval/eval_gen_v10.py \
        --data_npz data/processed/trajgen_128.npz \
        --checkpoint runs/trajgen_v10/best.pt \
        --knn_cache data/processed/trajgen_128_knn_k5.npz \
        --land_sdf data/processed/land_sdf_050deg.npz \
        --n_eval 500 --no_frechet
"""

import argparse
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.data_gen import TrajGenScalers, Scaler, train_val_split_gen
from src.land_mask import LandMask
from src.metrics_gen import evaluate_generation, evaluate_great_circle_baseline
from src.model_gen_v10 import TrajectoryGeneratorV10


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_checkpoint(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = argparse.Namespace(**ckpt["args"])
    scalers = TrajGenScalers(
        pos=Scaler(mean=ckpt["pos_mean"], std=ckpt["pos_std"]),
        delta=Scaler(mean=ckpt["delta_mean"], std=ckpt["delta_std"]),
    )
    model = TrajectoryGeneratorV10(
        d_model=args.d_model, nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        dim_feedforward=args.dim_feedforward, dropout=args.dropout,
        num_vessel_types=ckpt["num_vessel_types"],
        max_retrieved=ckpt["k_retrieval"],
        t_retr=ckpt["t_retr"],
        n_points=ckpt.get("n_resample", 128),
        k_past=ckpt.get("k_past", None),
    ).to(device)
    state_dict = ckpt["model"]
    if any(k.startswith("_orig_mod.") for k in state_dict):
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    return model, scalers, ckpt["vtype_vocab"], ckpt.get("n_resample", 128), args, ckpt


def generate(model, val_traj, val_vt, train_traj, val_knn, vtype_vocab,
             scalers, n_resample, device, land_mask=None,
             hard_threshold_km=10.0):
    N = len(val_traj)
    pred_traj = np.zeros_like(val_traj)
    batch_size = 32

    for start_idx in tqdm(range(0, N, batch_size), desc="Gen"):
        end_idx = min(start_idx + batch_size, N)
        batch_gt = val_traj[start_idx:end_idx]
        batch_vt = val_vt[start_idx:end_idx]
        batch_knn = val_knn[start_idx:end_idx]
        B = len(batch_gt)

        starts_raw = batch_gt[:, 0]
        ends_raw = batch_gt[:, -1]
        starts_norm = scalers.pos.transform(starts_raw)
        ends_norm = scalers.pos.transform(ends_raw)
        vtypes = np.array([vtype_vocab.get(int(v), 0) for v in batch_vt])

        retrieved_raw = train_traj[batch_knn]
        K = retrieved_raw.shape[1]
        retrieved_norm = scalers.pos.transform(
            retrieved_raw.reshape(-1, 2)).reshape(B, K, n_resample, 2)
        retrieval_mask = np.zeros((B, K), dtype=bool)

        start_t = torch.tensor(starts_norm, dtype=torch.float32, device=device)
        end_t   = torch.tensor(ends_norm,   dtype=torch.float32, device=device)
        vtype_t = torch.tensor(vtypes,      dtype=torch.long,    device=device)
        retr_t  = torch.tensor(retrieved_norm, dtype=torch.float32, device=device)
        rmask_t = torch.tensor(retrieval_mask, dtype=torch.bool,    device=device)

        with torch.no_grad():
            gen_norm = model.generate(
                start_t, end_t, vtype_t, retr_t, rmask_t,
                n_steps=n_resample,
                pos_scaler=scalers.pos, delta_scaler=scalers.delta,
                land_mask=land_mask,
                hard_threshold_km=hard_threshold_km)

        gen_norm_np = gen_norm.permute(1, 0, 2).cpu().numpy()
        for i in range(B):
            pred_traj[start_idx + i] = scalers.pos.inverse(gen_norm_np[i])

    return pred_traj


def _print_metrics(m):
    order = ["ade_m", "fde_m", "endpoint_error_m", "normalized_ade",
             "path_length_ratio", "smoothness", "gt_smoothness",
             "frechet_m", "frechet_median_m", "n_trajectories",
             "n_short", "ade_short_m", "normalized_ade_short",
             "n_medium", "ade_medium_m", "normalized_ade_medium",
             "n_long", "ade_long_m", "normalized_ade_long"]
    for k in order:
        if k not in m:
            continue
        v = m[k]
        if isinstance(v, float):
            fmt = f"{v:>10.4f}" if k.startswith("normalized_") else f"{v:>10.2f}"
            print(f"  {k:.<30} {fmt}")
        else:
            print(f"  {k:.<30} {v:>10}")


def _print_land(label, stats):
    print(f"  {label:<26s}  land%={stats['land_frac']*100:5.2f}"
          f"  cross%={stats['traj_crossing_rate']*100:5.1f}"
          f"  mean_pts={stats['mean_on_land_pts']:5.2f}"
          f"  max_pen={stats['mean_max_penetration_km']:5.1f} km")


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_npz",   required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--knn_cache",  required=True)
    parser.add_argument("--land_sdf",   default="data/processed/land_sdf_050deg.npz")
    parser.add_argument("--n_eval",     type=int, default=500)
    parser.add_argument("--no_frechet", action="store_true")
    parser.add_argument("--hard_threshold_km", type=float, default=10.0,
        help="SDF > this is projected to water in post-processing")
    parser.add_argument("--land_stats_threshold", type=float, default=10.0,
        help="'on land' threshold for diagnostic stats (separate from hard proj)")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}\n")

    model, scalers, vtype_vocab, n_resample, train_args, ckpt = load_checkpoint(
        args.checkpoint, device)
    print(f"Loaded: {args.checkpoint}")
    print(f"  epoch={ckpt['epoch']}  d_model={train_args.d_model}  "
          f"k_retrieval={train_args.k_retrieval}  t_retr={ckpt['t_retr']}  "
          f"k_past={ckpt.get('k_past')}\n")

    print(f"Loading land SDF: {args.land_sdf}")
    land_mask = LandMask.load(args.land_sdf)

    data = np.load(args.data_npz, allow_pickle=True)
    trajectories = data["trajectories"].astype(np.float32)
    vessel_types = data["vessel_types"].astype(np.int32)
    track_ids    = data["track_ids"]

    train_traj, _, _, val_traj, val_vt, _ = train_val_split_gen(
        trajectories, vessel_types, track_ids,
        train_args.val_frac, train_args.seed)
    print(f"Train: {len(train_traj):,}  Val: {len(val_traj):,}\n")

    knn = np.load(args.knn_cache)
    val_knn = knn["val_knn"]

    if args.n_eval < len(val_traj):
        rng = np.random.RandomState(42)
        idx = rng.choice(len(val_traj), args.n_eval, replace=False)
        val_traj = val_traj[idx]
        val_vt = val_vt[idx]
        val_knn = val_knn[idx]
        print(f"Subsampled to {args.n_eval} (seed=42)\n")

    compute_frechet = not args.no_frechet

    # Generate v10 raw + v10 with hard projection
    print("─ v10 raw output ─────────────────────────────────")
    pred_raw = generate(model, val_traj, val_vt, train_traj, val_knn,
                         vtype_vocab, scalers, n_resample, device,
                         land_mask=None)

    print("\n─ v10 + hard land projection ────────────────────")
    pred_proj = generate(model, val_traj, val_vt, train_traj, val_knn,
                         vtype_vocab, scalers, n_resample, device,
                         land_mask=land_mask,
                         hard_threshold_km=args.hard_threshold_km)

    retr_pred = train_traj[val_knn[:, 0]]

    # Core metrics
    print("\n" + "=" * 70)
    print("  v10 RAW")
    print("=" * 70)
    m_v10 = evaluate_generation(pred_raw, val_traj, compute_frechet=compute_frechet)
    _print_metrics(m_v10)

    print("\n" + "=" * 70)
    print("  v10 + HARD LAND PROJECTION")
    print("=" * 70)
    m_v10p = evaluate_generation(pred_proj, val_traj, compute_frechet=compute_frechet)
    _print_metrics(m_v10p)

    print("\n" + "=" * 70)
    print("  RETRIEVAL-TOP-1 (zero training)")
    print("=" * 70)
    m_retr = evaluate_generation(retr_pred, val_traj, compute_frechet=compute_frechet)
    _print_metrics(m_retr)

    print("\n" + "=" * 70)
    print("  GREAT-CIRCLE")
    print("=" * 70)
    m_gc = evaluate_great_circle_baseline(val_traj)
    _print_metrics(m_gc)

    # Comparison table
    print("\n" + "=" * 82)
    print("  COMPARISON  (lower is better)")
    print("=" * 82)
    cmp_keys = ["ade_m", "normalized_ade", "fde_m", "endpoint_error_m",
                "ade_short_m", "ade_medium_m", "ade_long_m"]
    if compute_frechet:
        cmp_keys.append("frechet_m")
    print(f"  {'metric':<22} {'v10':>10} {'v10+proj':>10} {'retr-top1':>12} {'GC':>10}")
    print("  " + "-" * 78)
    for k in cmp_keys:
        vals = [m_v10.get(k, float("nan")), m_v10p.get(k, float("nan")),
                m_retr.get(k, float("nan")), m_gc.get(k, float("nan"))]
        if k.startswith("normalized_"):
            print(f"  {k:.<22} " + " ".join(f"{v:>10.4f}" if i!=2 else f"{v:>12.4f}"
                                             for i, v in enumerate(vals)))
        else:
            print(f"  {k:.<22} " + " ".join(f"{v:>10.1f}" if i!=2 else f"{v:>12.1f}"
                                             for i, v in enumerate(vals)))

    # Land-crossing stats
    print("\n" + "=" * 82)
    print(f"  LAND-CROSSING  (threshold = {args.land_stats_threshold} km)")
    print("=" * 82)
    _print_land("Ground truth",       land_mask.trajectory_stats(val_traj, args.land_stats_threshold))
    _print_land("v10 raw",            land_mask.trajectory_stats(pred_raw, args.land_stats_threshold))
    _print_land("v10 + hard proj",    land_mask.trajectory_stats(pred_proj, args.land_stats_threshold))
    _print_land("Retrieval-top-1",    land_mask.trajectory_stats(retr_pred, args.land_stats_threshold))
    gc_traj = np.zeros_like(val_traj)
    from src.metrics_gen import great_circle_trajectory
    for i in range(len(val_traj)):
        gc_traj[i] = great_circle_trajectory(val_traj[i, 0], val_traj[i, -1], val_traj.shape[1])
    _print_land("Great-circle",       land_mask.trajectory_stats(gc_traj, args.land_stats_threshold))
    print("=" * 82)


if __name__ == "__main__":
    main()
