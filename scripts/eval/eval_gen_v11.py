#!/usr/bin/env python3
"""
eval_gen_v11.py — Evaluate v11 pointer trajectory generator.

Reports ADE/FDE/Frechet + normalized + length-bucketed metrics + land-crossing
stats + infeasibility and fallback rates for:
  - v11 model output (water by construction)
  - Retrieval-top-1 baseline
  - Great-circle baseline

Usage:
    python3 scripts/eval/eval_gen_v11.py \
        --data_npz data/processed/trajgen_128.npz \
        --checkpoint runs/trajgen_v11_lite/best.pt \
        --knn_cache data/processed/trajgen_128_knn_k5.npz \
        --land_sdf data/processed/land_sdf_005deg.npz \
        --n_eval 500 --no_frechet
"""
import argparse
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.data_gen import Scaler, TrajGenScalers, train_val_split_gen
from src.land_mask import LandMask
from src.metrics_gen import (evaluate_generation, evaluate_great_circle_baseline,
                              great_circle_trajectory)
from src.model_gen_v11 import TrajectoryGeneratorV11


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
    model = TrajectoryGeneratorV11(
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
    return model, scalers, ckpt, args


def generate(model, val_traj, val_vt, train_traj, val_knn, vtype_vocab,
             scalers, n_resample, device, land_mask, ckpt_args,
             batch_size=16):
    """Batched v11 rollout. Returns (pred, info_dict_aggregated)."""
    N = len(val_traj)
    pred = np.zeros_like(val_traj)
    fallback_counts = np.zeros(N, dtype=np.int32)
    infeasible_flags = np.zeros(N, dtype=bool)
    endpoint_snap = np.zeros(N, dtype=np.float32)

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
        K_r = retrieved_raw.shape[1]
        retrieved_norm = scalers.pos.transform(
            retrieved_raw.reshape(-1, 2)).reshape(B, K_r, n_resample, 2)
        retrieval_mask = np.zeros((B, K_r), dtype=bool)

        start_t = torch.tensor(starts_norm, dtype=torch.float32, device=device)
        end_t   = torch.tensor(ends_norm,   dtype=torch.float32, device=device)
        vtype_t = torch.tensor(vtypes,      dtype=torch.long,    device=device)
        retr_t  = torch.tensor(retrieved_norm, dtype=torch.float32, device=device)
        rmask_t = torch.tensor(retrieval_mask, dtype=torch.bool,    device=device)

        traj, info = model.generate(
            start=start_t, end=end_t, vessel_type=vtype_t,
            retrieved_norm=retr_t, retrieval_mask=rmask_t,
            land_mask=land_mask, pos_scaler=scalers.pos,
            n_steps=n_resample,
            K=getattr(ckpt_args, 'n_candidates', 32),
            cone_deg=getattr(ckpt_args, 'cand_cone_deg', 60.0),
            d_min_km=getattr(ckpt_args, 'cand_d_min_km', 3.0),
            d_max_km=getattr(ckpt_args, 'cand_d_max_km', 25.0),
            tau_km=getattr(ckpt_args, 'water_threshold_km', 2.0),
            n_segment_samples=getattr(ckpt_args, 'n_segment_samples', 5),
            use_segment_check=getattr(ckpt_args, 'use_segment_check', True),
        )
        traj_np = traj.permute(1, 0, 2).cpu().numpy()    # (B, T, 2) normalized
        for i in range(B):
            pred[start_idx + i] = scalers.pos.inverse(traj_np[i])
        fallback_counts[start_idx:end_idx] = info["fallback_flags"].sum(axis=0)
        infeasible_flags[start_idx:end_idx] = info["infeasible_mask"]
        endpoint_snap[start_idx:end_idx] = info["endpoint_snap_km"]

    return pred, {
        "fallback_counts": fallback_counts,
        "infeasible_flags": infeasible_flags,
        "endpoint_snap": endpoint_snap,
    }


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
    parser.add_argument("--land_sdf",   default="data/processed/land_sdf_005deg.npz")
    parser.add_argument("--n_eval",     type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--no_frechet", action="store_true")
    parser.add_argument("--land_stats_threshold", type=float, default=10.0)
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}\n")

    model, scalers, ckpt, train_args = load_checkpoint(args.checkpoint, device)
    print(f"Loaded: {args.checkpoint}")
    print(f"  epoch={ckpt['epoch']}  d_model={train_args.d_model}  "
          f"k_retrieval={train_args.k_retrieval}  t_retr={ckpt['t_retr']}  "
          f"k_past={ckpt.get('k_past')}")
    print(f"  K={ckpt.get('n_candidates')}  cone={ckpt.get('cand_cone_deg')}°  "
          f"τ={ckpt.get('water_threshold_km')}km  "
          f"seg_check={ckpt.get('use_segment_check')}\n")

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

    vtype_vocab = ckpt["vtype_vocab"]
    n_resample = ckpt.get("n_resample", 128)

    print("─ v11 generation ────────────────────────────────")
    pred, info = generate(
        model, val_traj, val_vt, train_traj, val_knn,
        vtype_vocab, scalers, n_resample, device, land_mask,
        train_args, batch_size=args.batch_size)

    retr_pred = train_traj[val_knn[:, 0]]

    # Core metrics
    print("\n" + "=" * 70)
    print("  v11  (pointer, water-by-construction)")
    print("=" * 70)
    m_v11 = evaluate_generation(pred, val_traj, compute_frechet=compute_frechet)
    _print_metrics(m_v11)

    print("\n" + "=" * 70)
    print("  RETRIEVAL-TOP-1")
    print("=" * 70)
    m_retr = evaluate_generation(retr_pred, val_traj, compute_frechet=compute_frechet)
    _print_metrics(m_retr)

    print("\n" + "=" * 70)
    print("  GREAT-CIRCLE")
    print("=" * 70)
    m_gc = evaluate_great_circle_baseline(val_traj)
    _print_metrics(m_gc)

    # Comparison
    print("\n" + "=" * 76)
    print("  COMPARISON  (lower is better)")
    print("=" * 76)
    cmp_keys = ["ade_m", "normalized_ade", "fde_m", "endpoint_error_m",
                "ade_short_m", "ade_medium_m", "ade_long_m"]
    if compute_frechet:
        cmp_keys.append("frechet_m")
    print(f"  {'metric':<22} {'v11':>12} {'retr-top1':>12} {'GC':>12}")
    print("  " + "-" * 72)
    for k in cmp_keys:
        vals = [m_v11.get(k, float("nan")),
                m_retr.get(k, float("nan")),
                m_gc.get(k, float("nan"))]
        if k.startswith("normalized_"):
            print(f"  {k:.<22} " + " ".join(f"{v:>12.4f}" for v in vals))
        else:
            print(f"  {k:.<22} " + " ".join(f"{v:>12.1f}" for v in vals))

    # Constraint checks
    print("\n" + "=" * 76)
    print(f"  LAND-CROSSING  (threshold = {args.land_stats_threshold} km)")
    print("=" * 76)
    _print_land("Ground truth",     land_mask.trajectory_stats(val_traj, args.land_stats_threshold))
    _print_land("v11",              land_mask.trajectory_stats(pred, args.land_stats_threshold))
    _print_land("Retrieval-top-1",  land_mask.trajectory_stats(retr_pred, args.land_stats_threshold))
    gc_traj = np.zeros_like(val_traj)
    for i in range(len(val_traj)):
        gc_traj[i] = great_circle_trajectory(
            val_traj[i, 0], val_traj[i, -1], val_traj.shape[1])
    _print_land("Great-circle",     land_mask.trajectory_stats(gc_traj, args.land_stats_threshold))

    # Strict (tau_km = 0) for v11 — the binding constraint check
    print("\n" + "=" * 76)
    print(f"  STRICT LAND-CROSSING  (threshold = 0 km, water-by-construction check)")
    print("=" * 76)
    _print_land("v11 (strict 0 km)", land_mask.trajectory_stats(pred, 0.0))
    strict_stats = land_mask.trajectory_stats(pred, 0.0)
    if strict_stats["traj_crossing_rate"] > 0:
        print("  *** WARNING: v11 has non-zero strict-crossing rate. "
              "Candidate filter is leaking — investigate segment check.")

    # Infeasibility / fallback rates
    print("\n" + "=" * 76)
    print("  FEASIBILITY")
    print("=" * 76)
    n = len(val_traj)
    n_fb_samples = int((info["fallback_counts"] > 0).sum())
    total_steps = (n_resample - 1) * n
    total_fb = int(info["fallback_counts"].sum())
    n_inf = int(info["infeasible_flags"].sum())
    print(f"  Samples using fallback  : {n_fb_samples:5d} / {n} "
          f"({100 * n_fb_samples / n:.1f}%)")
    print(f"  Fallback step rate      : {total_fb:5d} / {total_steps} "
          f"({100 * total_fb / total_steps:.3f}%)")
    print(f"  Infeasible samples      : {n_inf:5d} / {n} "
          f"({100 * n_inf / n:.1f}%)")
    print(f"  Endpoint-snap mean_km   : {info['endpoint_snap'].mean():.2f}")
    print(f"  Endpoint-snap max_km    : {info['endpoint_snap'].max():.2f}")
    print("=" * 76)


if __name__ == "__main__":
    main()
