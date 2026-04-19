"""
Obstacle-conditioned trajectory generation data loading.

Extends data_gen.py with synthetic obstacle augmentation.
For each training trajectory, with configurable probability, places 1-3
circular exclusion zones near (but not crossing) the trajectory, so the
GT trajectory is an "avoidance" example by construction.

Obstacle representation per sample:
    obstacles     : (K, 3) float32 [center_lon_norm, center_lat_norm, radius_deg]
    obstacle_mask : (K,)   bool    — True for padded (unused) slots
"""

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.data_gen import (
    Scaler, TrajGenScalers, TrajectoryGenDataset, build_vtype_vocab,
    train_val_split_gen,
)


# ─── Obstacle augmenter ────────────────────────────────────────────────────

# Approximate degrees per km at mid-latitudes (US Atlantic ~35°N)
DEG_PER_KM_LON = 1.0 / 91.0   # ~91 km per degree longitude at 35°N
DEG_PER_KM_LAT = 1.0 / 111.0  # ~111 km per degree latitude


@dataclass
class ObstacleConfig:
    max_obstacles: int = 3
    p_obstacle: float = 0.5
    radius_min_km: float = 10.0
    radius_max_km: float = 100.0
    min_clearance_factor: float = 0.1   # obstacle center at least r * factor from GT traj
    max_distance_factor: float = 3.0    # obstacle center within r * factor of GT traj


class ObstacleAugmenter:
    """Generate synthetic circular exclusion zones near a trajectory."""

    def __init__(self, config: ObstacleConfig):
        self.cfg = config

    def generate(
        self,
        traj_raw: np.ndarray,   # (T, 2) [lon, lat] degrees
        rng: np.random.RandomState,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate obstacles for one trajectory.

        Returns:
            obstacles: (K, 3) [center_lon, center_lat, radius_deg]
            mask: (K,) bool — True for padding (no obstacle)
        """
        K = self.cfg.max_obstacles
        obstacles = np.zeros((K, 3), dtype=np.float32)
        mask = np.ones(K, dtype=bool)  # all padded initially

        # Skip with probability (1 - p_obstacle)
        if rng.rand() > self.cfg.p_obstacle:
            return obstacles, mask

        n_obstacles = rng.randint(1, K + 1)  # 1 to K
        start, end = traj_raw[0], traj_raw[-1]

        for k in range(n_obstacles):
            obs = self._place_one_obstacle(traj_raw, start, end, rng)
            if obs is not None:
                obstacles[k] = obs
                mask[k] = False

        return obstacles, mask

    def _place_one_obstacle(
        self,
        traj_raw: np.ndarray,
        start: np.ndarray,
        end: np.ndarray,
        rng: np.random.RandomState,
        max_attempts: int = 50,
    ) -> Optional[np.ndarray]:
        """Try to place one obstacle that the trajectory avoids.

        Strategy: pick a random point laterally offset from the trajectory's
        midpoint region, with a random radius. Verify the trajectory doesn't
        cross the obstacle circle.
        """
        T = len(traj_raw)
        radius_km = rng.uniform(self.cfg.radius_min_km, self.cfg.radius_max_km)
        radius_deg = radius_km * DEG_PER_KM_LAT  # approximate

        for _ in range(max_attempts):
            # Pick a random trajectory point as anchor
            anchor_idx = rng.randint(1, T - 1)  # avoid start/end
            anchor = traj_raw[anchor_idx]

            # Offset laterally (perpendicular to local trajectory direction)
            if anchor_idx < T - 1:
                direction = traj_raw[anchor_idx + 1] - traj_raw[anchor_idx]
            else:
                direction = traj_raw[anchor_idx] - traj_raw[anchor_idx - 1]
            norm = np.sqrt(direction[0]**2 + direction[1]**2)
            if norm < 1e-10:
                continue

            # Perpendicular direction (rotate 90 degrees)
            perp = np.array([-direction[1], direction[0]]) / norm

            # Random lateral offset: between 1× and 3× radius
            offset_dist_deg = rng.uniform(1.0, self.cfg.max_distance_factor) * radius_deg
            side = rng.choice([-1, 1])
            center = anchor + side * perp * offset_dist_deg

            # Check: obstacle doesn't overlap start or end
            d_start = self._approx_dist_deg(center, start)
            d_end = self._approx_dist_deg(center, end)
            if d_start < radius_deg * 1.5 or d_end < radius_deg * 1.5:
                continue

            # Check: trajectory does NOT enter the obstacle
            dists = self._approx_dists_deg(center, traj_raw)  # (T,)
            min_dist = dists.min()
            if min_dist < radius_deg * (1.0 + self.cfg.min_clearance_factor):
                continue  # trajectory crosses or is too close to obstacle

            # Check: obstacle is close enough to be relevant
            if min_dist > radius_deg * self.cfg.max_distance_factor * 2:
                continue  # too far away, not interesting

            return np.array([center[0], center[1], radius_deg], dtype=np.float32)

        return None  # couldn't place after max_attempts

    @staticmethod
    def _approx_dist_deg(a: np.ndarray, b: np.ndarray) -> float:
        """Approximate distance in degrees (Euclidean — fine for nearby points)."""
        return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

    @staticmethod
    def _approx_dists_deg(center: np.ndarray, points: np.ndarray) -> np.ndarray:
        """Approximate distances in degrees from center to all points."""
        diff = points - center
        return np.sqrt(diff[:, 0]**2 + diff[:, 1]**2)


# ─── Dataset ────────────────────────────────────────────────────────────────

class ObstacleTrajectoryGenDataset(Dataset):
    """Dataset for obstacle-conditioned trajectory generation.

    Returns everything TrajectoryGenDataset returns, plus:
        obstacles     : (K, 3) [center_lon_norm, center_lat_norm, radius_deg]
        obstacle_mask : (K,)   bool — True for padding
    """

    def __init__(
        self,
        trajectories: np.ndarray,    # (N, T, 2) [lon, lat]
        vessel_types: np.ndarray,    # (N,)
        vtype_vocab: Dict[int, int],
        scalers: TrajGenScalers,
        obstacle_config: ObstacleConfig,
        seed: int = 42,
    ):
        self.trajectories = trajectories
        self.vessel_types = vessel_types
        self.vtype_vocab = vtype_vocab
        self.scalers = scalers
        self.augmenter = ObstacleAugmenter(obstacle_config)
        self.max_obstacles = obstacle_config.max_obstacles
        self.seed = seed

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        traj = self.trajectories[idx]               # (T, 2) [lon, lat]
        start = traj[0]
        end = traj[-1]
        vtype = self.vtype_vocab.get(int(self.vessel_types[idx]), 0)

        # Normalize trajectory
        traj_norm = self.scalers.pos.transform(traj)
        start_norm = self.scalers.pos.transform(start[None])[0]
        end_norm = self.scalers.pos.transform(end[None])[0]

        # Deltas (target)
        deltas = np.diff(traj, axis=0)
        deltas_norm = self.scalers.delta.transform(deltas)

        # Generate obstacles in raw space
        rng = np.random.RandomState(self.seed + idx)
        obstacles_raw, obstacle_mask = self.augmenter.generate(traj, rng)
        # obstacles_raw: (K, 3) [center_lon, center_lat, radius_deg]

        # Normalize obstacle centers (same scaler as positions)
        obstacles_norm = obstacles_raw.copy()
        for k in range(self.max_obstacles):
            if not obstacle_mask[k]:  # not padding
                center = obstacles_raw[k, :2]
                obstacles_norm[k, :2] = self.scalers.pos.transform(center[None])[0]
                # radius stays in degrees (not normalized — model needs absolute scale)

        return {
            "start_norm":    torch.tensor(start_norm,    dtype=torch.float32),
            "end_norm":      torch.tensor(end_norm,      dtype=torch.float32),
            "vessel_type":   torch.tensor(vtype,         dtype=torch.long),
            "traj_norm":     torch.tensor(traj_norm,     dtype=torch.float32),
            "deltas_norm":   torch.tensor(deltas_norm,   dtype=torch.float32),
            "obstacles":     torch.tensor(obstacles_norm, dtype=torch.float32),
            "obstacle_mask": torch.tensor(obstacle_mask, dtype=torch.bool),
        }


def obstacle_trajgen_collate(batch):
    """Collate into seq-first tensors, including obstacles.

    Returns:
        start_norm    : (B, 2)
        end_norm      : (B, 2)
        vessel_type   : (B,)
        traj_norm     : (T, B, 2)
        deltas_norm   : (T-1, B, 2)
        obstacles     : (B, K, 3)
        obstacle_mask : (B, K)
    """
    start = torch.stack([b["start_norm"]    for b in batch])
    end   = torch.stack([b["end_norm"]      for b in batch])
    vtype = torch.stack([b["vessel_type"]   for b in batch])
    traj  = torch.stack([b["traj_norm"]     for b in batch])
    delta = torch.stack([b["deltas_norm"]   for b in batch])
    obs   = torch.stack([b["obstacles"]     for b in batch])
    omask = torch.stack([b["obstacle_mask"] for b in batch])

    return {
        "start_norm":    start,
        "end_norm":      end,
        "vessel_type":   vtype,
        "traj_norm":     traj.permute(1, 0, 2),    # (T, B, 2)
        "deltas_norm":   delta.permute(1, 0, 2),    # (T-1, B, 2)
        "obstacles":     obs,                        # (B, K, 3)
        "obstacle_mask": omask,                      # (B, K)
    }


def get_obstacle_trajgen_loader(
    trajectories: np.ndarray,
    vessel_types: np.ndarray,
    vtype_vocab: Dict[int, int],
    scalers: TrajGenScalers,
    obstacle_config: ObstacleConfig,
    batch_size: int,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: int = 0,
    seed: int = 42,
) -> DataLoader:
    """Create a DataLoader for obstacle-conditioned trajectory generation."""
    ds = ObstacleTrajectoryGenDataset(
        trajectories, vessel_types, vtype_vocab, scalers,
        obstacle_config, seed=seed,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        collate_fn=obstacle_trajgen_collate,
        num_workers=num_workers,
    )
