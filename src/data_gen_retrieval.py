"""
Retrieval-augmented trajectory generation data pipeline (v9).

Two-stage design:
  1. build_knn_index() — one-shot pre-computation. For every sample in the
     NPZ, find its K nearest neighbors (by start/end positions + optional
     vtype weight), excluding self. Cache to disk. Separate indices for
     train-self-exclusion and val-from-train.
  2. RetrievalTrajectoryGenDataset — at __getitem__, look up K cached
     retrieval indices and return the retrieved trajectories alongside
     the base sample.

Why pre-compute: the index is deterministic per (data_npz, val_frac, seed,
k, vtype_weight) and doesn't need to change during training. An on-the-fly
KNN would dominate dataloader cost.

Extends data_gen.py with retrieval fields:
    retrieved_traj : (K, T, 2) float32  — raw [lon, lat] of K retrieved routes
    retrieval_mask : (K,)      bool     — True for padding (not enough neighbors)
"""

import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.data_gen import (
    TrajGenScalers, TrajectoryGenDataset, train_val_split_gen,
)


@dataclass
class RetrievalConfig:
    k_retrieval: int = 5
    vtype_weight: float = 0.5     # degrees-equivalent
    exclude_self: bool = True


# ─── KNN index build ────────────────────────────────────────────────────────

def _build_query_vectors(trajectories: np.ndarray,
                          vessel_types: np.ndarray) -> np.ndarray:
    """(N, 5): [start_lon, start_lat, end_lon, end_lat, vtype_as_float]."""
    start = trajectories[:, 0, :]
    end   = trajectories[:, -1, :]
    return np.concatenate(
        [start, end, vessel_types.astype(np.float32)[:, None]], axis=1
    )


def build_knn_index(
    query_trajectories: np.ndarray,    # (Nq, T, 2) — who is asking
    query_vessel_types: np.ndarray,
    corpus_trajectories: np.ndarray,   # (Nc, T, 2) — who to retrieve from
    corpus_vessel_types: np.ndarray,
    k: int,
    vtype_weight: float,
    exclude_self: bool,
    chunk: int = 256,
) -> np.ndarray:
    """Return (Nq, k) int64 array of corpus indices for each query.

    If exclude_self=True, Nq must equal Nc and row i in the corpus will be
    excluded for query i.
    """
    Nq = len(query_trajectories)
    Nc = len(corpus_trajectories)
    if exclude_self and Nq != Nc:
        raise ValueError(f"exclude_self requires Nq == Nc, got {Nq} vs {Nc}")

    q = _build_query_vectors(query_trajectories, query_vessel_types)
    c = _build_query_vectors(corpus_trajectories, corpus_vessel_types)

    nn_idx = np.empty((Nq, k), dtype=np.int64)
    for i in tqdm(range(0, Nq, chunk), desc="KNN", unit="chunk"):
        j = min(i + chunk, Nq)
        # Position distance in degrees (chunk, Nc)
        pos_diff = q[i:j, None, :4] - c[None, :, :4]
        d2 = (pos_diff ** 2).sum(axis=-1)
        if vtype_weight > 0:
            mismatch = (q[i:j, None, 4] != c[None, :, 4]).astype(np.float32)
            d2 = d2 + (vtype_weight ** 2) * mismatch
        if exclude_self:
            # Mask the diagonal for rows [i:j]: absolute indices are i..j-1
            for row, abs_idx in enumerate(range(i, j)):
                d2[row, abs_idx] = np.inf
        # Top-k smallest
        topk = np.argpartition(d2, kth=k, axis=1)[:, :k]
        # Sort top-k by actual distance (argpartition doesn't sort)
        for row in range(j - i):
            order = np.argsort(d2[row, topk[row]])
            topk[row] = topk[row, order]
        nn_idx[i:j] = topk
    return nn_idx


def build_and_cache_knn(
    data_npz_path: str,
    cache_path: str,
    val_frac: float,
    seed: int,
    k: int,
    vtype_weight: float,
) -> Dict[str, np.ndarray]:
    """Build KNN indices for train-self-exclusion and val-from-train. Cache to disk.

    Writes an NPZ with:
        train_knn : (Ntrain, k) int64 — retrieved indices within train set
        val_knn   : (Nval,   k) int64 — retrieved indices within train set

    Cached by (data_npz_path, val_frac, seed, k, vtype_weight) — recompute
    if any of these change.
    """
    if os.path.exists(cache_path):
        d = np.load(cache_path, allow_pickle=True)
        print(f"  [retrieval] loaded KNN cache from {cache_path}")
        return {"train_knn": d["train_knn"], "val_knn": d["val_knn"]}

    print(f"  [retrieval] building KNN index → {cache_path}")
    data = np.load(data_npz_path, allow_pickle=True)
    trajectories = data["trajectories"].astype(np.float32)
    vessel_types = data["vessel_types"].astype(np.int32)
    track_ids    = data["track_ids"]

    (train_traj, train_vt, _,
     val_traj,   val_vt,   _) = train_val_split_gen(
        trajectories, vessel_types, track_ids, val_frac, seed)

    # Train queries retrieve from train (self-exclusion)
    print(f"  [retrieval] train KNN: {len(train_traj):,} queries × corpus")
    train_knn = build_knn_index(
        train_traj, train_vt, train_traj, train_vt,
        k=k, vtype_weight=vtype_weight, exclude_self=True)

    # Val queries retrieve from train (no self, no leakage)
    print(f"  [retrieval] val KNN: {len(val_traj):,} queries × train corpus")
    val_knn = build_knn_index(
        val_traj, val_vt, train_traj, train_vt,
        k=k, vtype_weight=vtype_weight, exclude_self=False)

    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    np.savez(cache_path, train_knn=train_knn, val_knn=val_knn)
    return {"train_knn": train_knn, "val_knn": val_knn}


# ─── Dataset ────────────────────────────────────────────────────────────────

class RetrievalTrajectoryGenDataset(Dataset):
    """Dataset returning base sample + K retrieved routes."""

    def __init__(
        self,
        trajectories: np.ndarray,       # (Nq, T, 2) — queries (train or val)
        vessel_types: np.ndarray,       # (Nq,)
        corpus_trajectories: np.ndarray, # (Nc, T, 2) — retrieval corpus (always train)
        knn_indices: np.ndarray,        # (Nq, K) int64 — precomputed
        vtype_vocab: Dict[int, int],
        scalers: TrajGenScalers,
        config: RetrievalConfig,
    ):
        self.trajectories = trajectories
        self.vessel_types = vessel_types
        self.corpus = corpus_trajectories
        self.knn = knn_indices
        self.vtype_vocab = vtype_vocab
        self.scalers = scalers
        self.k = config.k_retrieval

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        traj = self.trajectories[idx]    # (T, 2)
        start = traj[0]
        end = traj[-1]
        vtype = self.vtype_vocab.get(int(self.vessel_types[idx]), 0)

        traj_norm = self.scalers.pos.transform(traj)
        start_norm = self.scalers.pos.transform(start[None])[0]
        end_norm = self.scalers.pos.transform(end[None])[0]
        deltas = np.diff(traj, axis=0)
        deltas_norm = self.scalers.delta.transform(deltas)

        # Retrieved routes — normalize with the same pos scaler
        nn_idx = self.knn[idx, :self.k]   # (K,)
        retrieved_raw = self.corpus[nn_idx]   # (K, T, 2)
        K, T, _ = retrieved_raw.shape
        retrieved_flat = retrieved_raw.reshape(-1, 2)
        retrieved_norm = self.scalers.pos.transform(retrieved_flat).reshape(K, T, 2)

        # Padding mask: all retrieved slots valid (KNN was pre-filtered)
        retrieval_mask = np.zeros(K, dtype=bool)

        return {
            "start_norm":     torch.tensor(start_norm,      dtype=torch.float32),
            "end_norm":       torch.tensor(end_norm,        dtype=torch.float32),
            "vessel_type":    torch.tensor(vtype,           dtype=torch.long),
            "traj_norm":      torch.tensor(traj_norm,       dtype=torch.float32),
            "deltas_norm":    torch.tensor(deltas_norm,     dtype=torch.float32),
            "retrieved_norm": torch.tensor(retrieved_norm,  dtype=torch.float32),
            "retrieval_mask": torch.tensor(retrieval_mask,  dtype=torch.bool),
        }


def retrieval_trajgen_collate(batch):
    """Collate into seq-first tensors, including retrieved routes."""
    start = torch.stack([b["start_norm"]     for b in batch])
    end   = torch.stack([b["end_norm"]       for b in batch])
    vtype = torch.stack([b["vessel_type"]    for b in batch])
    traj  = torch.stack([b["traj_norm"]      for b in batch])
    delta = torch.stack([b["deltas_norm"]    for b in batch])
    retr  = torch.stack([b["retrieved_norm"] for b in batch])  # (B, K, T, 2)
    rmask = torch.stack([b["retrieval_mask"] for b in batch])  # (B, K)

    return {
        "start_norm":     start,
        "end_norm":       end,
        "vessel_type":    vtype,
        "traj_norm":      traj.permute(1, 0, 2),    # (T, B, 2)
        "deltas_norm":    delta.permute(1, 0, 2),    # (T-1, B, 2)
        "retrieved_norm": retr,                       # (B, K, T, 2)
        "retrieval_mask": rmask,                      # (B, K)
    }


def get_retrieval_trajgen_loader(
    trajectories: np.ndarray,
    vessel_types: np.ndarray,
    corpus_trajectories: np.ndarray,
    knn_indices: np.ndarray,
    vtype_vocab: Dict[int, int],
    scalers: TrajGenScalers,
    config: RetrievalConfig,
    batch_size: int,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    ds = RetrievalTrajectoryGenDataset(
        trajectories, vessel_types, corpus_trajectories,
        knn_indices, vtype_vocab, scalers, config,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        collate_fn=retrieval_trajgen_collate,
        num_workers=num_workers,
    )
