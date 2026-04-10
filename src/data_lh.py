"""
Lighthouse-aware extensions to src/data.py.

Adds distance-to-nearest-lighthouses as input features. All original classes
and functions are importable from here for convenience; lighthouse-specific
versions have an _lh suffix or LH prefix.

Track arrays are extended from (T, 3) to (T, 6):
    [lon, lat, dt_seconds, lh_dist_1_km, lh_dist_2_km, lh_dist_3_km]

The 3 lighthouse distances are encoded as log1p(dist_km) and normalised by
a dedicated scaler, matching the existing log1p(dt) pattern.
"""

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

# Re-export everything from the base module for convenience
from .data import (
    Scaler,
    Scalers,
    causal_collate,
    single_collate,
    make_input,
    train_val_split,
    CausalDataset,
    SingleStepDataset,
)


# ─── Lighthouse Scalers ──────────────────────────────────────────────────────

@dataclass
class LHScalers:
    """Scalers extended with a lighthouse distance scaler."""
    pos:   Scaler   # [lon, lat]
    logdt: Scaler   # log1p(dt_seconds)
    disp:  Scaler   # [dlon, dlat] per-step displacement
    lh:    Scaler   # log1p(lighthouse_dist_km), shape (3,)

    @staticmethod
    def fit(tracks: Dict[str, np.ndarray]) -> "LHScalers":
        """Fit all scalers including lighthouse distances.

        Expects tracks with shape (T, 6): [lon, lat, dt, lh1, lh2, lh3].
        """
        all_pos, all_logdt, all_disp, all_lh = [], [], [], []
        for pts in tracks.values():
            all_pos.append(pts[:, :2])
            all_logdt.append(np.log1p(pts[:, 2]).reshape(-1, 1))
            all_disp.append(np.diff(pts[:, :2], axis=0))
            all_lh.append(np.log1p(pts[:, 3:6]))  # log1p(dist_km)
        return LHScalers(
            pos   = Scaler.fit(np.concatenate(all_pos, axis=0)),
            logdt = Scaler.fit(np.concatenate(all_logdt, axis=0)),
            disp  = Scaler.fit(np.concatenate(all_disp, axis=0)),
            lh    = Scaler.fit(np.concatenate(all_lh, axis=0)),
        )


# ─── Data loading ────────────────────────────────────────────────────────────

def load_tracks_lh(
    csv_path: str,
    lighthouse_cols: List[str],
    id_col:          str = "MMSI",
    time_col:        str = "BaseDateTime",
    lat_col:         str = "LAT",
    lon_col:         str = "LON",
    vessel_type_col: str = "VesselType",
) -> Tuple[Dict[str, np.ndarray], Dict[str, int]]:
    """Load tracks with lighthouse distance columns.

    Returns (tracks, vessel_types) where:
      tracks       : vessel_id -> float32 (T, 6) = [lon, lat, dt, lh1, lh2, lh3]
      vessel_types : vessel_id -> int AIS type code (0 = unknown/missing)
    """
    df = pd.read_csv(csv_path)
    required = {id_col, time_col, lat_col, lon_col} | set(lighthouse_cols)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[id_col, time_col, lat_col, lon_col])
    df = df.sort_values([id_col, time_col])

    tracks: Dict[str, np.ndarray] = {}
    vessel_types: Dict[str, int] = {}
    for v_id, g in df.groupby(id_col):
        lon = g[lon_col].to_numpy(dtype=np.float32)
        lat = g[lat_col].to_numpy(dtype=np.float32)
        ts  = g[time_col].values.astype("datetime64[s]").astype(np.float64)
        dt  = np.concatenate([[0.0], np.diff(ts)]).astype(np.float32)

        # Lighthouse distances
        lh = g[lighthouse_cols].to_numpy(dtype=np.float32)  # (T, 3)

        pts = np.column_stack([lon, lat, dt, lh])  # (T, 6)
        if pts.shape[0] >= 5:
            key = str(v_id)
            tracks[key] = pts
            if vessel_type_col in g.columns:
                vt = g[vessel_type_col].dropna()
                vessel_types[key] = int(vt.iloc[0]) if len(vt) > 0 else 0
            else:
                vessel_types[key] = 0

    lengths = [v.shape[0] for v in tracks.values()]
    print(f"Loaded {len(tracks):,} vessels (with lighthouse features)")
    print(f"  Track length -- min:{min(lengths)}  "
          f"median:{np.median(lengths):.0f}  max:{max(lengths)}")
    return tracks, vessel_types


# ─── Feature building ────────────────────────────────────────────────────────

def make_input_lh(pts: np.ndarray, scalers: LHScalers,
                  use_velocity: bool) -> np.ndarray:
    """Build normalised input features from a (T, 6) trajectory window.

    Returns (T, 6) without velocity or (T, 8) with velocity:
        [lon_norm, lat_norm, log_dt_norm, (vel_lon, vel_lat,) lh1_norm, lh2_norm, lh3_norm]
    """
    # Base features (reuse the same logic as make_input)
    pos_norm   = scalers.pos.transform(pts[:, :2])
    logdt_norm = scalers.logdt.transform(np.log1p(pts[:, 2]).reshape(-1, 1))
    x = np.concatenate([pos_norm, logdt_norm], axis=1)

    if use_velocity:
        dpos     = np.diff(pts[:, :2], axis=0, prepend=pts[0:1, :2])
        vel_norm = scalers.disp.transform(dpos)
        x = np.concatenate([x, vel_norm], axis=1)

    # Lighthouse features: log1p(dist_km), normalised
    lh_norm = scalers.lh.transform(np.log1p(pts[:, 3:6]))  # (T, 3)
    x = np.concatenate([x, lh_norm], axis=1)

    return x


# ─── Datasets ────────────────────────────────────────────────────────────────

class CausalDatasetLH(Dataset):
    """CausalDataset with lighthouse features.

    Identical to CausalDataset but uses make_input_lh and LHScalers.
    """

    def __init__(self, tracks: Dict[str, np.ndarray],
                 vessel_types: Dict[str, int],
                 vtype_vocab: Dict[int, int],
                 scalers: LHScalers,
                 seq_len: int, stride: int = 1,
                 max_windows_per_track: Optional[int] = None,
                 use_velocity: bool = True, max_gap_sec: float = 600):
        self.tracks       = tracks
        self.vessel_types = vessel_types
        self.vtype_vocab  = vtype_vocab
        self.scalers      = scalers
        self.seq_len      = seq_len
        self.use_velocity = use_velocity
        self.max_gap_sec  = max_gap_sec
        self.index: List[Tuple[str, int]] = []

        for v_id, pts in tracks.items():
            T = pts.shape[0]
            if T < seq_len + 1:
                continue
            starts = list(range(0, T - seq_len, stride))
            if max_windows_per_track and len(starts) > max_windows_per_track:
                starts = sorted(random.sample(starts, max_windows_per_track))
            for s in starts:
                self.index.append((v_id, s))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        v_id, s = self.index[idx]
        pts = self.tracks[v_id][s : s + self.seq_len + 1]   # (seq_len+1, 6)

        x        = make_input_lh(pts[:-1], self.scalers, self.use_velocity)
        disp_raw = pts[1:, :2] - pts[:-1, :2]               # (seq_len, 2)
        y        = self.scalers.disp.transform(disp_raw)

        dt       = pts[:-1, 2]
        gap_mask = torch.tensor(dt > self.max_gap_sec, dtype=torch.bool)

        raw_code  = self.vessel_types.get(v_id, 0)
        vtype_idx = torch.tensor(self.vtype_vocab.get(raw_code, 0), dtype=torch.long)

        return torch.tensor(x), torch.tensor(y), gap_mask, vtype_idx


# ─── DataLoader ──────────────────────────────────────────────────────────────

def get_loader_lh(
    tracks: Dict[str, np.ndarray],
    vessel_types: Dict[str, int],
    vtype_vocab: Dict[int, int],
    scalers: LHScalers,
    seq_len: int,
    batch_size: int,
    stride: int = 1,
    max_windows_per_track: Optional[int] = None,
    shuffle: bool = True,
    drop_last: bool = True,
    use_velocity: bool = True,
    num_workers: int = 0,
    max_gap_sec: float = 600,
) -> DataLoader:
    dataset = CausalDatasetLH(
        tracks, vessel_types, vtype_vocab, scalers,
        seq_len, stride, max_windows_per_track,
        use_velocity, max_gap_sec,
    )
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        collate_fn=causal_collate, drop_last=drop_last,
        num_workers=num_workers, pin_memory=False,
    )
