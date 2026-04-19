#!/usr/bin/env python3
"""
visualize_obstacle_gen.py — Visualize obstacle avoidance trajectories.

For each sample:
  1. Generate baseline trajectory (no obstacles)
  2. Place a blocking obstacle on the baseline midpoint
  3. Generate avoidance trajectory (with obstacle)
  4. Plot: obstacle circle, baseline, avoidance, ground truth

Usage:
    python3 scripts/eval/visualize_obstacle_gen.py \
        --data_npz data/processed/trajgen_128.npz \
        --checkpoint runs/trajgen_v6_obstacle/best.pt \
        --out_dir outputs/viz_obstacle
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.data_gen import (TrajGenScalers, Scaler, train_val_split_gen,
                          build_vtype_vocab)
from src.model_gen_obs import ObstacleTrajectoryGenerator
from src.metrics_gen import haversine_meters, path_length_m

DEG_PER_KM_LAT = 1.0 / 111.0

VTYPE_NAME = {
    **{c: "Passenger" for c in range(60, 70)},
    **{c: "Cargo"     for c in range(70, 80)},
    **{c: "Tanker"    for c in range(80, 90)},
}


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
        d_model=args.d_model, nhead=args.nhead,
        num_encoder_layers=getattr(args, "num_encoder_layers", 2),
        num_decoder_layers=args.num_decoder_layers,
        dim_feedforward=args.dim_feedforward, dropout=args.dropout,
        num_vessel_types=ckpt["num_vessel_types"],
        max_obstacles=ckpt.get("max_obstacles", 3),
    ).to(device)
    state_dict = ckpt["model"]
    if any(k.startswith("_orig_mod.") for k in state_dict):
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    return model, scalers, ckpt["vtype_vocab"], ckpt.get("n_resample", 128), args


def generate_one(model, start_raw, end_raw, vtype_code, vtype_vocab,
                 scalers, n_resample, device, obstacles_raw=None, max_obs=3):
    """Generate a single trajectory, optionally with obstacles."""
    start_norm = scalers.pos.transform(start_raw.reshape(1, 2))
    end_norm = scalers.pos.transform(end_raw.reshape(1, 2))
    vtype_idx = vtype_vocab.get(int(vtype_code), 0)

    start_t = torch.tensor(start_norm, dtype=torch.float32, device=device)
    end_t = torch.tensor(end_norm, dtype=torch.float32, device=device)
    vtype_t = torch.tensor([vtype_idx], dtype=torch.long, device=device)

    obs_t = None
    omask_t = None
    if obstacles_raw is not None:
        # obstacles_raw: (K, 3) [center_lon, center_lat, radius_deg]
        K = obstacles_raw.shape[0]
        obs_norm = obstacles_raw.copy()
        omask = np.ones(K, dtype=bool)
        for k in range(K):
            if obstacles_raw[k, 2] > 0:
                obs_norm[k, :2] = scalers.pos.transform(
                    obstacles_raw[k, :2].reshape(1, 2))[0]
                omask[k] = False
        obs_t = torch.tensor(obs_norm, dtype=torch.float32, device=device).unsqueeze(0)
        omask_t = torch.tensor(omask, dtype=torch.bool, device=device).unsqueeze(0)

    with torch.no_grad():
        gen_norm = model.generate(
            start_t, end_t, vtype_t, n_steps=n_resample,
            pos_scaler=scalers.pos, delta_scaler=scalers.delta,
            obstacles=obs_t, obstacle_mask=omask_t)
    return scalers.pos.inverse(gen_norm[:, 0, :].cpu().numpy())


def plot_obstacle_grid(gt_trajs, clean_trajs, avoid_trajs, obstacles_list,
                       vtypes, out_path, ncols=4):
    """Plot grid comparing clean vs avoidance trajectories."""
    n = len(gt_trajs)
    nrows = max(1, (n + ncols - 1) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes[None, :]
    elif ncols == 1:
        axes = axes[:, None]

    for idx in range(nrows * ncols):
        ax = axes[idx // ncols, idx % ncols]
        if idx >= n:
            ax.set_visible(False)
            continue

        gt = gt_trajs[idx]
        clean = clean_trajs[idx]
        avoid = avoid_trajs[idx]
        obs = obstacles_list[idx]  # (K, 3)
        vt = int(vtypes[idx])

        # Plot trajectories
        ax.plot(gt[:, 0], gt[:, 1], 'g-', linewidth=1.5, alpha=0.7,
                label='Ground truth')
        ax.plot(clean[:, 0], clean[:, 1], 'b--', linewidth=1.2, alpha=0.7,
                label='Baseline (no obs)')
        ax.plot(avoid[:, 0], avoid[:, 1], 'r-', linewidth=1.5, alpha=0.8,
                label='Avoidance')

        # Start/end markers
        ax.plot(gt[0, 0], gt[0, 1], 'gs', markersize=8, zorder=5)
        ax.plot(gt[-1, 0], gt[-1, 1], 'ro', markersize=8, zorder=5)

        # Draw obstacles as circles
        for k in range(obs.shape[0]):
            if obs[k, 2] > 0:
                circle = patches.Circle(
                    (obs[k, 0], obs[k, 1]), obs[k, 2],
                    facecolor='red', alpha=0.15,
                    edgecolor='red', linewidth=1.5, linestyle='--')
                ax.add_patch(circle)

        # Metrics
        clean_ade = haversine_meters(
            gt[:, 1], gt[:, 0], clean[:, 1], clean[:, 0]).mean()
        avoid_ade = haversine_meters(
            gt[:, 1], gt[:, 0], avoid[:, 1], avoid[:, 0]).mean()

        # Check avoidance
        obs_center = obs[0, :2]
        obs_r = obs[0, 2]
        diffs = avoid - obs_center
        min_dist = np.sqrt(diffs[:, 0]**2 + diffs[:, 1]**2).min()
        avoids = "Y" if min_dist > obs_r else "N"

        vname = VTYPE_NAME.get(vt, f"type_{vt}")
        ax.set_title(
            f"{vname} | clean {clean_ade/1000:.1f}km  avoid {avoid_ade/1000:.1f}km\n"
            f"avoids obstacle: {avoids}  "
            f"(clearance {(min_dist - obs_r) / DEG_PER_KM_LAT:.0f}km)",
            fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_aspect('equal')

        if idx == 0:
            ax.legend(fontsize=7, loc='best')

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved -> {out_path}")


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_npz", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out_dir", default="outputs/viz_obstacle")
    parser.add_argument("--n_plots", type=int, default=16)
    parser.add_argument("--obstacle_radius_km", type=float, default=50.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = get_device()
    os.makedirs(args.out_dir, exist_ok=True)

    model, scalers, vtype_vocab, n_resample, train_args = load_checkpoint(
        args.checkpoint, device)
    max_obs = getattr(train_args, "max_obstacles", 3)

    data = np.load(args.data_npz, allow_pickle=True)
    trajectories = data["trajectories"]
    vessel_types = data["vessel_types"]
    track_ids = data["track_ids"]

    _, _, _, val_traj, val_vt, _ = train_val_split_gen(
        trajectories, vessel_types, track_ids,
        train_args.val_frac, train_args.seed)
    print(f"Val set: {len(val_traj):,} trajectories\n")

    rng = np.random.RandomState(args.seed)
    n_total = min(args.n_plots, len(val_traj))
    sample_idx = rng.choice(len(val_traj), n_total, replace=False)

    gt_list, clean_list, avoid_list, obs_list, vt_list = [], [], [], [], []
    for i in tqdm(sample_idx, desc="Generating"):
        gt = val_traj[i]
        vt = val_vt[i]

        # Baseline (no obstacles)
        clean = generate_one(model, gt[0], gt[-1], vt, vtype_vocab,
                             scalers, n_resample, device)

        # Place blocking obstacle on baseline midpoint
        mid = clean[len(clean) // 2]
        radius_deg = args.obstacle_radius_km * DEG_PER_KM_LAT
        obstacles = np.zeros((max_obs, 3), dtype=np.float32)
        obstacles[0] = [mid[0], mid[1], radius_deg]

        # Avoidance trajectory
        avoid = generate_one(model, gt[0], gt[-1], vt, vtype_vocab,
                             scalers, n_resample, device,
                             obstacles_raw=obstacles, max_obs=max_obs)

        gt_list.append(gt)
        clean_list.append(clean)
        avoid_list.append(avoid)
        obs_list.append(obstacles)
        vt_list.append(vt)

    # Plot grid
    plot_obstacle_grid(
        gt_list, clean_list, avoid_list, obs_list,
        vt_list, os.path.join(args.out_dir, "obstacle_avoidance.png"))

    print(f"\nAll plots saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
