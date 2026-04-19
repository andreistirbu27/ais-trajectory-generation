#!/usr/bin/env python3
"""
eval_obstacle_gen.py — Evaluate obstacle-conditioned trajectory generation.

Tests whether the model actually avoids obstacles by:
1. Generating baseline trajectories (no obstacles)
2. Placing blocking obstacles on the baseline route midpoints
3. Generating avoidance trajectories (with obstacles)
4. Measuring avoidance rate, clearance distance, and path length overhead

Also computes standard ADE/FDE on clean data (no obstacles) to check for regression.

Usage:
    python3 scripts/eval/eval_obstacle_gen.py \
        --data_npz data/processed/trajgen_128.npz \
        --checkpoint runs/trajgen_v6_obstacle/best.pt

    # Quick eval
    python3 scripts/eval/eval_obstacle_gen.py \
        --data_npz data/processed/trajgen_128.npz \
        --checkpoint runs/trajgen_v6_obstacle/best.pt \
        --n_eval 200
"""

import argparse
import math
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.data_gen import (TrajGenScalers, Scaler, train_val_split_gen,
                          build_vtype_vocab)
from src.model_gen_obs import ObstacleTrajectoryGenerator
from src.metrics_gen import (haversine_meters, evaluate_generation,
                              evaluate_great_circle_baseline, path_length_m)


DEG_PER_KM_LAT = 1.0 / 111.0


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

    model = ObstacleTrajectoryGenerator(
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=getattr(args, "num_encoder_layers", 2),
        num_decoder_layers=args.num_decoder_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        num_vessel_types=ckpt["num_vessel_types"],
        max_obstacles=ckpt.get("max_obstacles", 3),
    ).to(device)

    state_dict = ckpt["model"]
    if any(k.startswith("_orig_mod.") for k in state_dict):
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()

    return model, scalers, ckpt["vtype_vocab"], ckpt.get("n_resample", 128), args


def generate_batch(model, starts_raw, ends_raw, vtypes, vtype_vocab, scalers,
                   n_resample, device, obstacles_raw=None):
    """Generate trajectories for a batch. Returns (B, T, 2) raw [lon, lat]."""
    B = len(starts_raw)
    starts_norm = scalers.pos.transform(starts_raw)
    ends_norm = scalers.pos.transform(ends_raw)
    vtype_idx = np.array([vtype_vocab.get(int(v), 0) for v in vtypes])

    start_t = torch.tensor(starts_norm, dtype=torch.float32, device=device)
    end_t = torch.tensor(ends_norm, dtype=torch.float32, device=device)
    vtype_t = torch.tensor(vtype_idx, dtype=torch.long, device=device)

    # Prepare obstacles
    obs_t = None
    omask_t = None
    if obstacles_raw is not None:
        # obstacles_raw: (B, K, 3) [center_lon, center_lat, radius_deg]
        K = obstacles_raw.shape[1]
        obs_norm = obstacles_raw.copy()
        omask = np.ones((B, K), dtype=bool)
        for b in range(B):
            for k in range(K):
                if obstacles_raw[b, k, 2] > 0:  # has valid radius
                    obs_norm[b, k, :2] = scalers.pos.transform(
                        obstacles_raw[b, k, :2].reshape(1, 2))[0]
                    omask[b, k] = False
        obs_t = torch.tensor(obs_norm, dtype=torch.float32, device=device)
        omask_t = torch.tensor(omask, dtype=torch.bool, device=device)

    with torch.no_grad():
        gen_norm = model.generate(
            start_t, end_t, vtype_t, n_steps=n_resample,
            pos_scaler=scalers.pos, delta_scaler=scalers.delta,
            obstacles=obs_t, obstacle_mask=omask_t)

    gen_np = gen_norm.permute(1, 0, 2).cpu().numpy()  # (B, T, 2)
    result = np.zeros_like(gen_np)
    for i in range(B):
        result[i] = scalers.pos.inverse(gen_np[i])
    return result


def place_blocking_obstacle(traj_raw, radius_km=50.0):
    """Place an obstacle blocking the trajectory midpoint.

    Returns:
        obstacle: (3,) [center_lon, center_lat, radius_deg]
    """
    T = len(traj_raw)
    mid_idx = T // 2
    center = traj_raw[mid_idx].copy()
    radius_deg = radius_km * DEG_PER_KM_LAT
    return np.array([center[0], center[1], radius_deg], dtype=np.float32)


def measure_avoidance(traj_raw, obstacle):
    """Measure how well a trajectory avoids an obstacle.

    Args:
        traj_raw: (T, 2) [lon, lat]
        obstacle: (3,) [center_lon, center_lat, radius_deg]

    Returns:
        dict with avoidance metrics
    """
    center = obstacle[:2]
    radius_deg = obstacle[2]

    # Approximate distances in degrees
    diffs = traj_raw - center
    dists_deg = np.sqrt(diffs[:, 0]**2 + diffs[:, 1]**2)

    min_dist_deg = dists_deg.min()
    penetration = max(0, radius_deg - min_dist_deg)
    avoids = min_dist_deg > radius_deg

    # Convert to km for reporting
    min_dist_km = min_dist_deg / DEG_PER_KM_LAT
    radius_km = radius_deg / DEG_PER_KM_LAT
    clearance_km = min_dist_km - radius_km

    return {
        "avoids": avoids,
        "min_dist_km": min_dist_km,
        "clearance_km": clearance_km,
        "penetration_deg": penetration,
    }


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_npz", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--n_eval", type=int, default=500)
    parser.add_argument("--obstacle_radius_km", type=float, default=50.0,
        help="Radius of blocking obstacle (km)")
    parser.add_argument("--no_frechet", action="store_true")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}\n")

    model, scalers, vtype_vocab, n_resample, train_args = load_checkpoint(
        args.checkpoint, device)
    print(f"Loaded: {args.checkpoint}")
    print(f"  max_obstacles={getattr(train_args, 'max_obstacles', 3)}\n")

    # Load data
    data = np.load(args.data_npz, allow_pickle=True)
    trajectories = data["trajectories"]
    vessel_types = data["vessel_types"]
    track_ids = data["track_ids"]

    _, _, _, val_traj, val_vt, _ = train_val_split_gen(
        trajectories, vessel_types, track_ids,
        train_args.val_frac, train_args.seed)
    print(f"Val set: {len(val_traj):,} trajectories\n")

    # Sample
    rng = np.random.RandomState(42)
    n_eval = min(args.n_eval, len(val_traj))
    idx = rng.choice(len(val_traj), n_eval, replace=False)
    eval_traj = val_traj[idx]
    eval_vt = val_vt[idx]

    max_obs = getattr(train_args, "max_obstacles", 3)

    # ── 1. Clean ADE (no obstacles) ─────────────────────────────────────────
    print("Generating clean trajectories (no obstacles)...")
    batch_size = 32
    clean_pred = np.zeros_like(eval_traj)
    for s in tqdm(range(0, n_eval, batch_size)):
        e = min(s + batch_size, n_eval)
        clean_pred[s:e] = generate_batch(
            model, eval_traj[s:e, 0], eval_traj[s:e, -1],
            eval_vt[s:e], vtype_vocab, scalers, n_resample, device)

    print("\n" + "=" * 65)
    print("  CLEAN EVALUATION (no obstacles)")
    print("=" * 65)
    clean_metrics = evaluate_generation(clean_pred, eval_traj,
                                         compute_frechet=not args.no_frechet)
    for k, v in clean_metrics.items():
        if isinstance(v, float):
            print(f"  {k:.<30} {v:>10.2f}")

    # ── 2. Obstacle avoidance test ───���──────────────────────────────────────
    print(f"\nPlacing blocking obstacles (r={args.obstacle_radius_km}km) "
          f"and generating avoidance trajectories...")

    avoid_pred = np.zeros_like(eval_traj)
    all_obstacles = np.zeros((n_eval, max_obs, 3), dtype=np.float32)

    # Place one blocking obstacle per trajectory (on the clean baseline midpoint)
    for i in range(n_eval):
        obs = place_blocking_obstacle(clean_pred[i], args.obstacle_radius_km)
        all_obstacles[i, 0] = obs  # slot 0

    for s in tqdm(range(0, n_eval, batch_size)):
        e = min(s + batch_size, n_eval)
        avoid_pred[s:e] = generate_batch(
            model, eval_traj[s:e, 0], eval_traj[s:e, -1],
            eval_vt[s:e], vtype_vocab, scalers, n_resample, device,
            obstacles_raw=all_obstacles[s:e])

    # ── 3. Avoidance metrics ────────────────────────────────────────────────
    avoidance_results = []
    for i in range(n_eval):
        result = measure_avoidance(avoid_pred[i], all_obstacles[i, 0])
        result["path_length_clean_km"] = path_length_m(clean_pred[i]) / 1000
        result["path_length_avoid_km"] = path_length_m(avoid_pred[i]) / 1000
        avoidance_results.append(result)

    avoidance_rate = np.mean([r["avoids"] for r in avoidance_results])
    mean_clearance = np.mean([r["clearance_km"] for r in avoidance_results])
    mean_min_dist = np.mean([r["min_dist_km"] for r in avoidance_results])
    clean_lengths = [r["path_length_clean_km"] for r in avoidance_results]
    avoid_lengths = [r["path_length_avoid_km"] for r in avoidance_results]
    length_overhead = np.mean(np.array(avoid_lengths) / np.array(clean_lengths))

    # ADE of avoidance trajectories vs GT
    avoid_metrics = evaluate_generation(avoid_pred, eval_traj, compute_frechet=False)

    print("\n" + "=" * 65)
    print(f"  OBSTACLE AVOIDANCE TEST (r={args.obstacle_radius_km}km)")
    print("=" * 65)
    print(f"  Avoidance rate ............ {avoidance_rate * 100:.1f}%")
    print(f"  Mean min distance ......... {mean_min_dist:.1f} km")
    print(f"  Mean clearance ............ {mean_clearance:.1f} km")
    print(f"  Mean path length (clean) .. {np.mean(clean_lengths):.1f} km")
    print(f"  Mean path length (avoid) .. {np.mean(avoid_lengths):.1f} km")
    print(f"  Path length overhead ...... {length_overhead:.3f}x")
    print(f"  ADE (avoid vs GT) ......... {avoid_metrics['ade_m']:.1f} m")
    print(f"  ADE (clean vs GT) ......... {clean_metrics['ade_m']:.1f} m")

    # ── 4. Great-circle baseline ───────��────────────────────────────────────
    print(f"\n{'='*65}")
    print("  GREAT-CIRCLE BASELINE")
    print("=" * 65)
    gc_metrics = evaluate_great_circle_baseline(eval_traj)
    for k, v in gc_metrics.items():
        if isinstance(v, float):
            print(f"  {k:.<30} {v:>10.2f}")

    print("=" * 65)


if __name__ == "__main__":
    main()
