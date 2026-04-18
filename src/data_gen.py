"""
Trajectory generation data loading, normalization, and dataset classes.

Trajectories are stored as float32 arrays of shape (N, n_points, 2): [lon, lat].
Resampled to equal arc-length spacing by prepare_trajgen.py.

Two scalers are maintained (fit on training data only):
  pos   — [lon, lat] absolute position
  delta — [dlon, dlat] per-step displacement between resampled points
"""

import random
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


# ─── Scalers ─────────────────────────────────────────────────────────────────

@dataclass
class Scaler:
    mean: np.ndarray
    std:  np.ndarray

    @staticmethod
    def fit(x: np.ndarray) -> "Scaler":
        mean = x.mean(axis=0).astype(np.float32)
        std  = x.std(axis=0).astype(np.float32)
        std  = np.where(std < 1e-6, 1.0, std)
        return Scaler(mean=mean, std=std)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.std).astype(np.float32)

    def inverse(self, x: np.ndarray) -> np.ndarray:
        return (x * self.std + self.mean).astype(np.float32)


@dataclass
class TrajGenScalers:
    pos:   Scaler   # [lon, lat] absolute position
    delta: Scaler   # [dlon, dlat] per-step displacement

    @staticmethod
    def fit(trajectories: np.ndarray) -> "TrajGenScalers":
        """Fit scalers on training trajectories.

        Args:
            trajectories: (N, T, 2) float32 [lon, lat]
        """
        all_pos = trajectories.reshape(-1, 2)
        all_delta = np.diff(trajectories, axis=1).reshape(-1, 2)
        return TrajGenScalers(
            pos   = Scaler.fit(all_pos),
            delta = Scaler.fit(all_delta),
        )


# ─── Train/val split ────────────────────────────────────────────────────────

def _root_id(vid: str) -> str:
    """Strip batch prefix (b000_) and all trailing segment suffixes (_0, _1_2, etc.)
    to recover the original MMSI."""
    s = re.sub(r'^b\d+_', '', str(vid))
    s = re.sub(r'(_\d+)+$', '', s)
    return s


def train_val_split_gen(
    trajectories: np.ndarray,
    vessel_types: np.ndarray,
    track_ids: np.ndarray,
    val_frac: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray, np.ndarray]:
    """Split trajectory arrays by root MMSI (no leakage between vessel segments).

    Returns:
        (train_traj, train_vtypes, train_ids,
         val_traj,   val_vtypes,   val_ids)
    """
    N = len(trajectories)
    if N <= 1 or val_frac <= 0:
        empty = np.array([], dtype=trajectories.dtype).reshape(0, *trajectories.shape[1:])
        return (trajectories, vessel_types, track_ids,
                empty, np.array([], dtype=vessel_types.dtype),
                np.array([], dtype=track_ids.dtype))

    # Group indices by root MMSI
    root_to_indices: Dict[str, List[int]] = {}
    for i, tid in enumerate(track_ids):
        root = _root_id(str(tid))
        root_to_indices.setdefault(root, []).append(i)

    root_ids = sorted(root_to_indices.keys())
    random.Random(seed).shuffle(root_ids)
    n_val = max(1, int(len(root_ids) * val_frac))
    val_roots = set(root_ids[:n_val])

    val_idx = []
    train_idx = []
    for root, indices in root_to_indices.items():
        if root in val_roots:
            val_idx.extend(indices)
        else:
            train_idx.extend(indices)

    train_idx = np.array(sorted(train_idx), dtype=np.intp)
    val_idx = np.array(sorted(val_idx), dtype=np.intp)

    return (
        trajectories[train_idx], vessel_types[train_idx], track_ids[train_idx],
        trajectories[val_idx],   vessel_types[val_idx],   track_ids[val_idx],
    )


# ─── Vessel type vocab ──────────────────────────────────────────────────────

def build_vtype_vocab(vessel_types: np.ndarray) -> Dict[int, int]:
    """Map raw AIS vessel type codes to contiguous indices 0..K-1."""
    unique = sorted(set(int(v) for v in vessel_types))
    return {code: idx for idx, code in enumerate(unique)}


# ─── Dataset ─────────────────────────────────────────────────────────────────

class TrajectoryGenDataset(Dataset):
    """Dataset for conditional trajectory generation.

    Each sample provides:
        start_norm   : (2,)   normalized start [lon, lat]
        end_norm     : (2,)   normalized end [lon, lat]
        vessel_type  : ()     int64 vocab index
        traj_norm    : (T, 2) normalized positions
        deltas_norm  : (T-1, 2) normalized step deltas (target)
    """

    def __init__(
        self,
        trajectories: np.ndarray,    # (N, T, 2) [lon, lat]
        vessel_types: np.ndarray,    # (N,) int
        vtype_vocab: Dict[int, int],
        scalers: TrajGenScalers,
    ):
        self.trajectories = trajectories
        self.vessel_types = vessel_types
        self.vtype_vocab = vtype_vocab
        self.scalers = scalers

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        traj = self.trajectories[idx]               # (T, 2) [lon, lat]
        start = traj[0]                              # (2,)
        end = traj[-1]                               # (2,)
        vtype = self.vtype_vocab.get(int(self.vessel_types[idx]), 0)

        # Normalize
        traj_norm = self.scalers.pos.transform(traj)          # (T, 2)
        start_norm = self.scalers.pos.transform(start[None])[0]  # (2,)
        end_norm = self.scalers.pos.transform(end[None])[0]      # (2,)

        # Deltas (target)
        deltas = np.diff(traj, axis=0)                         # (T-1, 2)
        deltas_norm = self.scalers.delta.transform(deltas)     # (T-1, 2)

        return {
            "start_norm":  torch.tensor(start_norm, dtype=torch.float32),
            "end_norm":    torch.tensor(end_norm,   dtype=torch.float32),
            "vessel_type": torch.tensor(vtype,      dtype=torch.long),
            "traj_norm":   torch.tensor(traj_norm,  dtype=torch.float32),
            "deltas_norm": torch.tensor(deltas_norm, dtype=torch.float32),
        }


def trajgen_collate(batch):
    """Collate into seq-first tensors for transformer.

    Returns:
        start_norm  : (B, 2)
        end_norm    : (B, 2)
        vessel_type : (B,)
        traj_norm   : (T, B, 2)     — seq-first
        deltas_norm : (T-1, B, 2)   — seq-first
    """
    start = torch.stack([b["start_norm"]  for b in batch])    # (B, 2)
    end   = torch.stack([b["end_norm"]    for b in batch])    # (B, 2)
    vtype = torch.stack([b["vessel_type"] for b in batch])    # (B,)
    traj  = torch.stack([b["traj_norm"]   for b in batch])    # (B, T, 2)
    delta = torch.stack([b["deltas_norm"] for b in batch])    # (B, T-1, 2)

    return {
        "start_norm":  start,
        "end_norm":    end,
        "vessel_type": vtype,
        "traj_norm":   traj.permute(1, 0, 2),    # (T, B, 2)
        "deltas_norm": delta.permute(1, 0, 2),    # (T-1, B, 2)
    }


def get_trajgen_loader(
    trajectories: np.ndarray,
    vessel_types: np.ndarray,
    vtype_vocab: Dict[int, int],
    scalers: TrajGenScalers,
    batch_size: int,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Create a DataLoader for trajectory generation."""
    ds = TrajectoryGenDataset(trajectories, vessel_types, vtype_vocab, scalers)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        collate_fn=trajgen_collate,
        num_workers=num_workers,
    )
