"""
v11 dataset — retrieval + per-step candidate pool + GT pointer indices.

Extends v9's RetrievalTrajectoryGenDataset by adding, per sample:
  candidates_norm : (T-1, K, 2) float32  — K candidates per step, normalized
  candidate_feats : (T-1, K, 2) float32  — [sdf_km_raw, storm_dist_km_raw]
  candidate_mask  : (T-1, K)    bool     — True = valid (water + segment)
  gt_indices      : (T-1,)      int64    — pointer target; -100 = ignore

Candidates are generated deterministically from (p_t_raw, end_raw) via the
Halton grid in src/candidates.py, so caching to disk is safe. For v11-lite
we skip disk caching (in-RAM lazy __getitem__) — candidate generation for
128 steps at K=32 with 5 segment samples per candidate is ~30 ms/sample on
numpy. With num_workers>0 that amortizes fine.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.candidates import (filter_candidates, generate_candidates,
                             insert_gt_into_candidates)
from src.data_gen import TrajGenScalers
from src.data_gen_retrieval import RetrievalConfig


@dataclass
class CandidateConfig:
    K: int = 32
    cone_deg: float = 60.0
    d_min_km: float = 3.0
    d_max_km: float = 25.0
    tau_km: float = 2.0
    n_segment_samples: int = 5
    use_segment_check: bool = True
    use_storm_check: bool = False


class RetrievalV11Dataset(Dataset):
    """v9 retrieval dataset + per-step water-valid candidate pool + GT pointer labels."""

    def __init__(
        self,
        trajectories: np.ndarray,           # (Nq, T, 2)
        vessel_types: np.ndarray,           # (Nq,)
        corpus_trajectories: np.ndarray,    # (Nc, T, 2)
        knn_indices: np.ndarray,            # (Nq, K_retr)
        vtype_vocab: Dict[int, int],
        scalers: TrajGenScalers,
        retr_config: RetrievalConfig,
        cand_config: CandidateConfig,
        land_mask,
        storms: Optional[List[Tuple[float, float, float]]] = None,
    ):
        self.trajectories = trajectories
        self.vessel_types = vessel_types
        self.corpus = corpus_trajectories
        self.knn = knn_indices
        self.vtype_vocab = vtype_vocab
        self.scalers = scalers
        self.k_retr = retr_config.k_retrieval
        self.cand = cand_config
        self.land_mask = land_mask
        self.storms = storms

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        traj = self.trajectories[idx]                   # (T, 2) raw
        T = traj.shape[0]
        start = traj[0]
        end = traj[-1]
        vtype = self.vtype_vocab.get(int(self.vessel_types[idx]), 0)

        traj_norm = self.scalers.pos.transform(traj)
        start_norm = self.scalers.pos.transform(start[None])[0]
        end_norm = self.scalers.pos.transform(end[None])[0]
        deltas = np.diff(traj, axis=0)
        deltas_norm = self.scalers.delta.transform(deltas)

        # Retrieved routes
        nn_idx = self.knn[idx, :self.k_retr]
        retrieved_raw = self.corpus[nn_idx]               # (K, T, 2)
        K_r, T_r, _ = retrieved_raw.shape
        retrieved_norm = self.scalers.pos.transform(
            retrieved_raw.reshape(-1, 2)).reshape(K_r, T_r, 2)
        retrieval_mask = np.zeros(K_r, dtype=bool)

        # ── Per-step candidates + GT indices ─────────────────────────────
        cc = self.cand
        # We need candidates at each step 0..T-2 centered at traj[t]
        # targeting end. gt_next = traj[t+1].
        p_t_all = traj[:-1]                               # (T-1, 2) raw
        gt_all = traj[1:]                                  # (T-1, 2) raw
        end_rep = np.broadcast_to(end[None], p_t_all.shape).copy()

        cands_raw = generate_candidates(
            p_t_all, end_rep, K=cc.K, cone_deg=cc.cone_deg,
            d_min_km=cc.d_min_km, d_max_km=cc.d_max_km)    # (T-1, K, 2)
        valid = filter_candidates(
            p_t_all, cands_raw, self.land_mask,
            tau_km=cc.tau_km,
            n_segment_samples=cc.n_segment_samples,
            storms=self.storms,
            use_segment_check=cc.use_segment_check,
            use_storm_check=cc.use_storm_check,
        )                                                  # (T-1, K)

        cands_raw, valid, gt_indices = insert_gt_into_candidates(
            cands_raw, valid, p_t_all, gt_all, self.land_mask,
            tau_km=cc.tau_km,
            n_segment_samples=cc.n_segment_samples,
            segment_check=cc.use_segment_check,
        )

        # Normalize candidates; compute per-candidate features (sdf_km, storm_km)
        flat = cands_raw.reshape(-1, 2)
        cands_norm_flat = self.scalers.pos.transform(flat)
        cands_norm = cands_norm_flat.reshape(T - 1, cc.K, 2)

        sdf_flat = self.land_mask.sample_km_np(flat[:, 0], flat[:, 1]
                                                ).astype(np.float32)
        sdf = sdf_flat.reshape(T - 1, cc.K)
        storm = np.full_like(sdf, 1e6)      # stub: no storms → large distance
        if self.storms:
            for c_lon, c_lat, r_km in self.storms:
                km_per_deg_lat = 111.0
                cos_lat = np.cos(np.deg2rad(c_lat))
                dlat_km = (flat[:, 1] - c_lat) * km_per_deg_lat
                dlon_km = (flat[:, 0] - c_lon) * km_per_deg_lat * cos_lat
                d_km = np.sqrt(dlat_km ** 2 + dlon_km ** 2) - r_km
                storm = np.minimum(storm, d_km.reshape(T - 1, cc.K))
        cand_feats = np.stack([sdf, storm], axis=-1).astype(np.float32
                                                              )   # (T-1, K, 2)

        return {
            "start_norm":      torch.tensor(start_norm,     dtype=torch.float32),
            "end_norm":        torch.tensor(end_norm,       dtype=torch.float32),
            "vessel_type":     torch.tensor(vtype,          dtype=torch.long),
            "traj_norm":       torch.tensor(traj_norm,      dtype=torch.float32),
            "deltas_norm":     torch.tensor(deltas_norm,    dtype=torch.float32),
            "retrieved_norm":  torch.tensor(retrieved_norm, dtype=torch.float32),
            "retrieval_mask":  torch.tensor(retrieval_mask, dtype=torch.bool),
            "candidates_norm": torch.tensor(cands_norm,     dtype=torch.float32),
            "candidate_feats": torch.tensor(cand_feats,     dtype=torch.float32),
            "candidate_mask":  torch.tensor(valid,          dtype=torch.bool),
            "gt_indices":      torch.tensor(gt_indices,     dtype=torch.long),
        }


def retrieval_v11_collate(batch):
    """Collate into seq-first tensors including per-step candidates."""
    start = torch.stack([b["start_norm"]      for b in batch])
    end   = torch.stack([b["end_norm"]        for b in batch])
    vtype = torch.stack([b["vessel_type"]     for b in batch])
    traj  = torch.stack([b["traj_norm"]       for b in batch])
    delta = torch.stack([b["deltas_norm"]     for b in batch])
    retr  = torch.stack([b["retrieved_norm"]  for b in batch])
    rmask = torch.stack([b["retrieval_mask"]  for b in batch])
    cands = torch.stack([b["candidates_norm"] for b in batch])   # (B, T-1, K, 2)
    cfeat = torch.stack([b["candidate_feats"] for b in batch])   # (B, T-1, K, 2)
    cmask = torch.stack([b["candidate_mask"]  for b in batch])   # (B, T-1, K)
    gti   = torch.stack([b["gt_indices"]      for b in batch])   # (B, T-1)

    return {
        "start_norm":      start,
        "end_norm":        end,
        "vessel_type":     vtype,
        "traj_norm":       traj.permute(1, 0, 2),       # (T, B, 2)
        "deltas_norm":     delta.permute(1, 0, 2),      # (T-1, B, 2)
        "retrieved_norm":  retr,                         # (B, K_retr, T, 2)
        "retrieval_mask":  rmask,
        "candidates_norm": cands.permute(1, 0, 2, 3),   # (T-1, B, K, 2)
        "candidate_feats": cfeat.permute(1, 0, 2, 3),   # (T-1, B, K, 2)
        "candidate_mask":  cmask.permute(1, 0, 2),      # (T-1, B, K)
        "gt_indices":      gti.permute(1, 0),           # (T-1, B)
    }


def get_retrieval_v11_loader(
    trajectories: np.ndarray,
    vessel_types: np.ndarray,
    corpus_trajectories: np.ndarray,
    knn_indices: np.ndarray,
    vtype_vocab: Dict[int, int],
    scalers: TrajGenScalers,
    retr_config: RetrievalConfig,
    cand_config: CandidateConfig,
    land_mask,
    batch_size: int,
    storms: Optional[List[Tuple[float, float, float]]] = None,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    ds = RetrievalV11Dataset(
        trajectories, vessel_types, corpus_trajectories,
        knn_indices, vtype_vocab, scalers,
        retr_config, cand_config, land_mask, storms=storms,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        collate_fn=retrieval_v11_collate,
        num_workers=num_workers,
    )
