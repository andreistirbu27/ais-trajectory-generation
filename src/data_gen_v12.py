"""
v12 dataset — retrieval + quantized cell sequences.

Extends v9's RetrievalTrajectoryGenDataset by adding per-sample:
    cell_ix  (T,)       int32  — grid col index per waypoint
    cell_iy  (T,)       int32  — grid row index per waypoint
    offset   (T, 2)     float  — sub-cell offset in cell-edge units ∈ [-0.5, 0.5]
    end_ix, end_iy      int32  — end-waypoint cell coords
    start_ix, start_iy, start_offset — convenience for inference rollout

Cell sequences are pre-quantized by `scripts/data/quantize_trajgen.py` and
stored in `data/processed/trajgen_128_cells_005deg.npz`. The dataset just
indexes into that file plus the NPZ's KNN cache.

Because quantization is deterministic in (trajgen NPZ, water graph), there
is no on-the-fly compute in __getitem__ beyond the same normalization that
v9 does for retrieved routes.
"""
from __future__ import annotations

import os
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.data_gen import TrajGenScalers
from src.data_gen_retrieval import RetrievalConfig


class RetrievalV12Dataset(Dataset):
    """v9 retrieval dataset + quantized cell sequences for v12 pointer training."""

    def __init__(
        self,
        trajectories: np.ndarray,            # (Nq, T, 2) raw lon/lat — for start/end/normalization
        vessel_types: np.ndarray,            # (Nq,)
        cell_ix: np.ndarray,                 # (Nq, T) int32
        cell_iy: np.ndarray,                 # (Nq, T) int32
        offset: np.ndarray,                  # (Nq, T, 2) float
        corpus_trajectories: np.ndarray,     # (Nc, T, 2) raw — retrieved routes corpus
        knn_indices: np.ndarray,             # (Nq, K)
        vtype_vocab: Dict[int, int],
        scalers: TrajGenScalers,
        config: RetrievalConfig,
    ):
        self.trajectories = trajectories
        self.vessel_types = vessel_types
        self.cell_ix = cell_ix
        self.cell_iy = cell_iy
        self.offset = offset
        self.corpus = corpus_trajectories
        self.knn = knn_indices
        self.vtype_vocab = vtype_vocab
        self.scalers = scalers
        self.k = config.k_retrieval

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        traj = self.trajectories[idx]                    # (T, 2) raw
        start = traj[0]
        end = traj[-1]
        vtype = self.vtype_vocab.get(int(self.vessel_types[idx]), 0)

        start_norm = self.scalers.pos.transform(start[None])[0]
        end_norm   = self.scalers.pos.transform(end[None])[0]

        nn_idx = self.knn[idx, :self.k]                   # (K,)
        retrieved_raw = self.corpus[nn_idx]               # (K, T, 2)
        K_r, T, _ = retrieved_raw.shape
        retrieved_flat = retrieved_raw.reshape(-1, 2)
        retrieved_norm = self.scalers.pos.transform(retrieved_flat).reshape(K_r, T, 2)

        retrieval_mask = np.zeros(K_r, dtype=bool)

        cix = self.cell_ix[idx]                           # (T,)
        ciy = self.cell_iy[idx]
        off = self.offset[idx]                            # (T, 2)

        return {
            "start_norm":     torch.tensor(start_norm,     dtype=torch.float32),
            "end_norm":       torch.tensor(end_norm,       dtype=torch.float32),
            "vessel_type":    torch.tensor(vtype,          dtype=torch.long),
            "retrieved_norm": torch.tensor(retrieved_norm, dtype=torch.float32),
            "retrieval_mask": torch.tensor(retrieval_mask, dtype=torch.bool),
            "cell_ix":        torch.tensor(cix,            dtype=torch.long),
            "cell_iy":        torch.tensor(ciy,            dtype=torch.long),
            "offset":         torch.tensor(off,            dtype=torch.float32),
            "start_ix":       torch.tensor(int(cix[0]),    dtype=torch.long),
            "start_iy":       torch.tensor(int(ciy[0]),    dtype=torch.long),
            "start_offset":   torch.tensor(off[0],         dtype=torch.float32),
            "end_ix":         torch.tensor(int(cix[-1]),   dtype=torch.long),
            "end_iy":         torch.tensor(int(ciy[-1]),   dtype=torch.long),
        }


def v12_collate(batch):
    """Seq-first tensors for the transformer; scalar end/start batched normally."""
    def stack(key):  return torch.stack([b[key] for b in batch])
    out = {
        "start_norm":     stack("start_norm"),
        "end_norm":       stack("end_norm"),
        "vessel_type":    stack("vessel_type"),
        "retrieved_norm": stack("retrieved_norm"),
        "retrieval_mask": stack("retrieval_mask"),
        "cell_ix":        stack("cell_ix").permute(1, 0),          # (T, B)
        "cell_iy":        stack("cell_iy").permute(1, 0),          # (T, B)
        "offset":         stack("offset").permute(1, 0, 2),         # (T, B, 2)
        "start_ix":       stack("start_ix"),
        "start_iy":       stack("start_iy"),
        "start_offset":   stack("start_offset"),
        "end_ix":         stack("end_ix"),
        "end_iy":         stack("end_iy"),
    }
    return out


def get_v12_loader(
    trajectories: np.ndarray,
    vessel_types: np.ndarray,
    cell_ix: np.ndarray,
    cell_iy: np.ndarray,
    offset: np.ndarray,
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
    ds = RetrievalV12Dataset(
        trajectories, vessel_types, cell_ix, cell_iy, offset,
        corpus_trajectories, knn_indices, vtype_vocab, scalers, config,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        collate_fn=v12_collate,
        num_workers=num_workers,
    )


# ─── Loader helper: load the quantized NPZ + KNN cache + apply val split ────

def load_v12_splits(
    quantized_npz: str,
    knn_cache_npz: str,
    val_frac: float,
    split_seed: int,
):
    """Return everything needed to build train/val loaders for v12.

    Shapes:
        train_traj  (Ntr, T, 2)     raw lon/lat
        train_vt    (Ntr,)
        train_cix   (Ntr, T) int32
        train_ciy   (Ntr, T) int32
        train_off   (Ntr, T, 2)
        train_knn   (Ntr, K)        retrieves within train corpus
        val_*       — analogous
        train_corpus_traj — the train raw lon/lat corpus (for retrieval lookup)
        meta        — dict with bbox, dlon, dlat, grid_shape
    """
    from src.data_gen import train_val_split_gen

    if not os.path.exists(quantized_npz):
        raise FileNotFoundError(
            f"{quantized_npz} not found. Run "
            f"`python3 scripts/data/quantize_trajgen.py` first.")
    if not os.path.exists(knn_cache_npz):
        raise FileNotFoundError(
            f"{knn_cache_npz} not found. Build the v9/v10 KNN cache first.")

    z = np.load(quantized_npz, allow_pickle=True)
    traj = z["trajectories"]
    vt = z["vessel_types"]
    tids = z["track_ids"]
    cix = z["cell_ix"]
    ciy = z["cell_iy"]
    off = z["offset"]
    meta = {
        "bbox": z["bbox"],
        "dlon": float(z["dlon"]),
        "dlat": float(z["dlat"]),
        "grid_shape": z["grid_shape"],
    }

    (train_traj, train_vt, train_ids,
     val_traj,   val_vt,   val_ids) = train_val_split_gen(
        traj, vt, tids, val_frac, split_seed)

    # Recompute the train/val index arrays to slice cell arrays in lockstep
    # (train_val_split_gen returns views, not indices).
    def _pick(ids_subset, all_ids):
        mask = np.isin(all_ids, ids_subset)
        return np.nonzero(mask)[0]

    train_idx = _pick(train_ids, tids)
    val_idx   = _pick(val_ids,   tids)

    train_cix = cix[train_idx]; train_ciy = ciy[train_idx]; train_off = off[train_idx]
    val_cix   = cix[val_idx];   val_ciy   = ciy[val_idx];   val_off   = off[val_idx]

    knn = np.load(knn_cache_npz)
    train_knn = knn["train_knn"]
    val_knn   = knn["val_knn"]

    return {
        "train": {
            "traj": train_traj, "vt": train_vt,
            "cix": train_cix,   "ciy": train_ciy, "off": train_off,
            "knn": train_knn,
        },
        "val": {
            "traj": val_traj, "vt": val_vt,
            "cix": val_cix,   "ciy": val_ciy, "off": val_off,
            "knn": val_knn,
        },
        "train_corpus_traj": train_traj,
        "meta": meta,
    }
