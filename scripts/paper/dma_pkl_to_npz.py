#!/usr/bin/env python3
"""
dma_pkl_to_npz.py
-----------------
Convert TrAISformer's shipped Kattegat DMA pickles into v10's NPZ format,
preserving TrAISformer's train/valid/test split assignment so the K-anchor
benchmark can score both methods on the exact same test set.

Input  : paper/external/TrAISformer/data/ct_dma/ct_dma_{train,valid,test}.pkl
         Each: list[{"mmsi": int, "traj": (T, 6) float64}]
         Cols of traj: [lat_norm, lon_norm, sog_norm, cog_norm, unix_ts, mmsi].

Output : data/processed/dma_128.npz  (single file containing all 13,679 tracks)
         Fields:
           trajectories     (N, 128, 2) float32  [lon_deg, lat_deg]
           vessel_types     (N,)        int32    placeholder 70 (cargo) for all
           track_ids        (N,)        <U32     "<mmsi>_<segidx>" (root = mmsi)
           original_split   (N,)        <U8      one of "train"/"valid"/"test"
           original_length  (N,)        int32    original # of 10-min steps in
                                                 the pickle (needed to match
                                                 TrAISformer's time-uniform
                                                 rollout against arc-length GT)

DMA pickles have no vessel-type column; we use a single placeholder for all so
v10's vtype-conditioned encoder still functions (the embedding contribution is
just a constant bias across all DMA tracks).

Usage:
    paper/.venv/bin/python3 scripts/paper/dma_pkl_to_npz.py \\
        --out data/processed/dma_128.npz
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.data.prepare_trajgen import resample_track  # arc-length interpolator

DMA_LAT_MIN, DMA_LAT_MAX = 55.5, 58.0
DMA_LON_MIN, DMA_LON_MAX = 10.3, 13.0
VTYPE_PLACEHOLDER = 70   # generic cargo, valid index in v10's 28-class embedding


def denorm_track(traj_norm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lat = traj_norm[:, 0] * (DMA_LAT_MAX - DMA_LAT_MIN) + DMA_LAT_MIN
    lon = traj_norm[:, 1] * (DMA_LON_MAX - DMA_LON_MIN) + DMA_LON_MIN
    return lon, lat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/processed/dma_128.npz")
    ap.add_argument("--n_resample", type=int, default=128)
    ap.add_argument("--min_input_len", type=int, default=24,
                    help="Skip input tracks shorter than this (default: DMA min).")
    ap.add_argument("--min_total_km", type=float, default=2.0,
                    help="Drop tracks whose total arc length is below this. "
                         "DMA bbox is ~280 km wide so 2 km is permissive.")
    ap.add_argument("--src_dir",
                    default="paper/external/TrAISformer/data/ct_dma")
    args = ap.parse_args()

    src = Path(args.src_dir)
    all_trajs, all_vtypes, all_ids, all_splits, all_lengths = [], [], [], [], []
    counts = {"train": 0, "valid": 0, "test": 0}
    dropped = {"short": 0, "stationary": 0}

    for split in ("train", "valid", "test"):
        pkl_path = src / f"ct_dma_{split}.pkl"
        with open(pkl_path, "rb") as f:
            tracks = pickle.load(f)
        print(f"Loading {pkl_path.name}: {len(tracks)} tracks")

        for seg_idx, t in enumerate(tracks):
            traj = t["traj"]
            mmsi = int(t["mmsi"])
            if len(traj) < args.min_input_len:
                dropped["short"] += 1
                continue
            lon, lat = denorm_track(traj)
            # Quick stationary check before the resample helper bails to a
            # constant point.
            step_d = np.sqrt(np.diff(lon)**2 + np.diff(lat)**2)
            # Convert the (lon,lat) degree distance to a rough km — at lat 57 a
            # degree-lon is ~60 km, degree-lat ~111 km. Use a worst-case scale
            # of 60 km/deg so the threshold filters out only truly degenerate
            # tracks.
            if step_d.sum() * 60.0 < args.min_total_km:
                dropped["stationary"] += 1
                continue
            lon_r, lat_r = resample_track(lon, lat, n_points=args.n_resample)
            all_trajs.append(np.stack([lon_r, lat_r], axis=-1).astype(np.float32))
            all_vtypes.append(VTYPE_PLACEHOLDER)
            all_ids.append(f"{mmsi}_{seg_idx}")
            all_splits.append(split)
            all_lengths.append(len(traj))
            counts[split] += 1

    if not all_trajs:
        sys.exit("No tracks survived filtering.")

    trajectories = np.stack(all_trajs, axis=0).astype(np.float32)
    vessel_types = np.array(all_vtypes, dtype=np.int32)
    track_ids = np.array(all_ids, dtype="<U32")
    original_split = np.array(all_splits, dtype="<U8")
    original_length = np.array(all_lengths, dtype=np.int32)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        trajectories=trajectories,
        vessel_types=vessel_types,
        track_ids=track_ids,
        original_split=original_split,
        original_length=original_length,
    )

    print(f"\nWrote {out_path}")
    print(f"  total tracks: {len(trajectories)}")
    print(f"  by split: train={counts['train']}, valid={counts['valid']}, "
          f"test={counts['test']}")
    print(f"  dropped: short={dropped['short']}, stationary={dropped['stationary']}")
    print(f"  trajectories: shape={trajectories.shape}, dtype={trajectories.dtype}")
    print(f"  lon range: [{trajectories[..., 0].min():.4f}, "
          f"{trajectories[..., 0].max():.4f}]")
    print(f"  lat range: [{trajectories[..., 1].min():.4f}, "
          f"{trajectories[..., 1].max():.4f}]")


if __name__ == "__main__":
    main()
