#!/usr/bin/env python3
"""
csv_to_traisformer_pkl.py
-------------------------
Convert raw NOAA AIS CSV(s) into TrAISformer's pickle format
([lat_norm, lon_norm, sog_norm, cog_norm, unix_timestamp] per row,
grouped per trajectory, pickled as a list of dicts).

The expected output matches what TrAISformer's `AISDataset` consumes
(see paper/notes/traisformer_data_format.md). Note that lat/lon/SOG/COG are
pre-normalized to [0, 1) within the chosen ROI; the model further filters out
short trajectories at training time.

This script is the paper-track equivalent of `scripts/data/prepare_data.py`,
but produces the time-uniform 10-min-spaced pkl format TrAISformer needs.
It does NOT modify the existing arc-length-resampled NPZ pipeline used by v10.

Usage
-----
    # Single CSV (e.g. already merged):
    python3 scripts/paper/csv_to_traisformer_pkl.py \\
        --csv data/processed/AIS_2024_full.csv \\
        --out_dir data/processed/traisformer_us_gulf \\
        --bbox -90.0 28.0 -87.3 30.5 \\
        --resample_minutes 10

    # Multiple day-CSVs straight from NOAA:
    python3 scripts/paper/csv_to_traisformer_pkl.py \\
        --csv data/raw/AIS_2024_01_*.csv \\
        --out_dir data/processed/traisformer_us_gulf \\
        --bbox -90.0 28.0 -87.3 30.5

Outputs three pickles per run:
    <out_dir>/ct_us_train.pkl
    <out_dir>/ct_us_valid.pkl
    <out_dir>/ct_us_test.pkl
plus <out_dir>/meta.json (bbox, normalization constants, counts, seed).

Train/val/test split is MMSI-grouped (root MMSI) to match the discipline used
elsewhere in the project (`src/data.py::_root_id`).
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


# -----------------------------------------------------------------------------
# NOAA AIS schema
# -----------------------------------------------------------------------------
# The full set of columns in NOAA Marine Cadastre 2024 AIS:
#   MMSI, BaseDateTime, LAT, LON, SOG, COG, Heading, VesselName, IMO,
#   CallSign, VesselType, Status, Length, Width, Draft, Cargo, TransceiverClass
NOAA_COLS = ["MMSI", "BaseDateTime", "LAT", "LON", "SOG", "COG", "VesselType"]

# TrAISformer's normalization conventions (from their config + dataset code).
SOG_RANGE = 30.0    # knots — anything above is clipped before normalization
COG_RANGE = 360.0


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _root_mmsi(mmsi: str | int) -> str:
    """Strip batch-prefix and segment-suffix (matches src/data.py::_root_id)."""
    s = str(mmsi)
    # Drop b{NNN}_ prefix if present
    if s.startswith("b") and "_" in s:
        parts = s.split("_", 1)
        if parts[0][1:].isdigit():
            s = parts[1]
    # Drop _<segment_id> suffix if present
    if "_" in s:
        root, tail = s.rsplit("_", 1)
        if tail.isdigit():
            s = root
    return s


def _interpolate_cog(cog_deg: np.ndarray, dt_seconds: np.ndarray) -> np.ndarray:
    """Linear interp of course-over-ground, respecting circular wrap.

    Input: irregular samples + their relative times.
    Output: interpolated values at the relative times provided in `dt_seconds`.
    (Caller resamples positions/times to a uniform grid and asks for cog at the
    new times.)
    """
    # Convert to sin/cos, interp those, convert back.
    rad = np.deg2rad(cog_deg)
    s = np.sin(rad)
    c = np.cos(rad)
    s_interp = np.interp(dt_seconds, dt_seconds, s)  # placeholder; caller does interp
    c_interp = np.interp(dt_seconds, dt_seconds, c)
    return (np.rad2deg(np.arctan2(s_interp, c_interp)) + 360.0) % 360.0


def resample_track_uniform(
    df: pd.DataFrame,
    resample_seconds: int,
    min_points: int,
) -> np.ndarray | None:
    """Resample a single track's (LAT, LON, SOG, COG, timestamp) rows to a
    fixed time grid via linear interpolation.

    COG is interpolated via sin/cos to respect the 0/360 wrap.

    Returns
    -------
    np.ndarray of shape (T, 5) with columns
        [LAT, LON, SOG, COG, unix_timestamp]
    or None if the resampled track has fewer than `min_points` rows.
    Caller will optionally append an MMSI-repeat column to match TrAISformer's
    6-column shipped format.
    """
    df = df.sort_values("BaseDateTime")
    t = pd.to_datetime(df["BaseDateTime"]).astype("int64").to_numpy() // 10**9  # unix seconds
    lat = df["LAT"].to_numpy(dtype=np.float64)
    lon = df["LON"].to_numpy(dtype=np.float64)
    sog = df["SOG"].to_numpy(dtype=np.float64)
    cog = df["COG"].to_numpy(dtype=np.float64)

    if len(t) < 2:
        return None
    t0, t1 = t[0], t[-1]
    if (t1 - t0) < (min_points - 1) * resample_seconds:
        return None

    new_t = np.arange(t0, t1 + 1, resample_seconds, dtype=np.int64)
    new_lat = np.interp(new_t, t, lat)
    new_lon = np.interp(new_t, t, lon)
    new_sog = np.clip(np.interp(new_t, t, sog), 0.0, SOG_RANGE - 1e-6)
    # Circular interp for COG: sin/cos then atan2.
    cog_rad = np.deg2rad(cog)
    sin_new = np.interp(new_t, t, np.sin(cog_rad))
    cos_new = np.interp(new_t, t, np.cos(cog_rad))
    new_cog = (np.rad2deg(np.arctan2(sin_new, cos_new)) + 360.0) % 360.0

    if len(new_t) < min_points:
        return None
    return np.stack([new_lat, new_lon, new_sog, new_cog, new_t.astype(np.float64)], axis=1)


def split_at_gaps(
    df: pd.DataFrame, max_gap_minutes: float
) -> list[pd.DataFrame]:
    """Split a single-MMSI dataframe into sub-tracks wherever consecutive
    timestamps are more than `max_gap_minutes` apart.
    """
    if len(df) < 2:
        return [df]
    df = df.sort_values("BaseDateTime").reset_index(drop=True)
    t = pd.to_datetime(df["BaseDateTime"]).astype("int64").to_numpy() // 10**9
    gaps = np.diff(t)
    cut = np.where(gaps > max_gap_minutes * 60)[0]
    if len(cut) == 0:
        return [df]
    segments = []
    start = 0
    for c in cut:
        segments.append(df.iloc[start : c + 1])
        start = c + 1
    segments.append(df.iloc[start:])
    return segments


def normalize_traj(
    traj: np.ndarray,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
) -> np.ndarray:
    """Normalize lat/lon/SOG/COG to [0, 1) in-place columns 0..3.
    Returns a new array; timestamp column 4 is untouched.
    """
    out = traj.copy()
    out[:, 0] = (out[:, 0] - lat_min) / (lat_max - lat_min)
    out[:, 1] = (out[:, 1] - lon_min) / (lon_max - lon_min)
    out[:, 2] = out[:, 2] / SOG_RANGE
    out[:, 3] = out[:, 3] / COG_RANGE
    # Clip into [0, 1) the way TrAISformer expects.
    out[:, :4] = np.clip(out[:, :4], 0.0, 1.0 - 1e-4)
    return out


# -----------------------------------------------------------------------------
# Pipeline
# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", nargs="+", required=True,
                    help="One or more NOAA AIS CSV files (globs OK).")
    ap.add_argument("--out_dir", required=True,
                    help="Directory to write {ct_us_train,valid,test}.pkl + meta.json.")
    ap.add_argument("--bbox", nargs=4, type=float, required=True,
                    metavar=("LON_MIN", "LAT_MIN", "LON_MAX", "LAT_MAX"),
                    help="ROI bbox in degrees. For Gulf example: -90.0 28.0 -87.3 30.5")
    ap.add_argument("--resample_minutes", type=float, default=10.0,
                    help="Uniform sampling cadence (default 10 min, matches TrAISformer).")
    ap.add_argument("--max_gap_minutes", type=float, default=60.0,
                    help="Split a track wherever consecutive timestamps exceed this gap.")
    ap.add_argument("--min_points", type=int, default=37,
                    help="Minimum samples per output track (>=37 satisfies "
                         "TrAISformer's min_seqlen=36; default 37).")
    ap.add_argument("--keep_vessel_types", type=int, nargs="*",
                    default=list(range(60, 90)),
                    help="VesselType codes to keep (default: 60..89 = commercial).")
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--test_frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_rows", type=int, default=None,
                    help="If given, cap rows after concat (dev/debug).")
    args = ap.parse_args()

    lon_min, lat_min, lon_max, lat_max = args.bbox
    assert lon_min < lon_max and lat_min < lat_max, "Bad bbox"

    # ------ Load CSVs ------
    paths = []
    for pat in args.csv:
        matched = sorted(glob(pat))
        if not matched:
            print(f"  [warn] no files matched {pat}", file=sys.stderr)
        paths.extend(matched)
    if not paths:
        sys.exit("No input CSV files found.")
    print(f"Loading {len(paths)} CSV file(s)...")
    dfs = []
    for p in paths:
        df = pd.read_csv(p, usecols=lambda c: c in NOAA_COLS,
                         dtype={"MMSI": str}, low_memory=False)
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    if args.max_rows:
        df = df.iloc[: args.max_rows]
    print(f"  total rows: {len(df):,}")

    # ------ Filter rows ------
    df = df.dropna(subset=["LAT", "LON", "SOG", "COG", "BaseDateTime", "MMSI"])
    df = df[(df["LAT"].between(lat_min, lat_max)) &
            (df["LON"].between(lon_min, lon_max))]
    if args.keep_vessel_types:
        df = df[df["VesselType"].isin(args.keep_vessel_types)]
    df = df[(df["SOG"] >= 0) & (df["SOG"] <= SOG_RANGE) &
            (df["COG"] >= 0) & (df["COG"] <= COG_RANGE)]
    print(f"  rows after bbox+vtype+valid filter: {len(df):,}")
    if len(df) == 0:
        sys.exit("No rows survived filtering. Check bbox/vtype/CSV format.")

    # ------ Group, split, resample ------
    resample_seconds = int(args.resample_minutes * 60)
    print(f"Resampling each track to {args.resample_minutes}-min grid...")
    all_tracks: list[dict] = []
    n_dropped_short = 0
    n_dropped_resample = 0
    for mmsi, sub in df.groupby("MMSI"):
        for seg_idx, seg_df in enumerate(split_at_gaps(sub, args.max_gap_minutes)):
            traj_raw = resample_track_uniform(
                seg_df, resample_seconds=resample_seconds,
                min_points=args.min_points)
            if traj_raw is None:
                n_dropped_resample += 1
                continue
            if len(traj_raw) < args.min_points:
                n_dropped_short += 1
                continue
            traj_norm = normalize_traj(traj_raw, lat_min, lat_max, lon_min, lon_max)
            mmsi_int = int(mmsi) if str(mmsi).isdigit() else hash(mmsi) & 0xFFFFFFFF
            # Append an MMSI-repeat column to match the 6-column shipped DMA format.
            mmsi_col = np.full((traj_norm.shape[0], 1), mmsi_int, dtype=np.float32)
            traj_6col = np.concatenate([traj_norm.astype(np.float32), mmsi_col], axis=1)
            all_tracks.append({
                "mmsi": mmsi_int,
                "mmsi_str": str(mmsi),
                "root_mmsi": _root_mmsi(mmsi),
                "seg_idx": seg_idx,
                "traj": traj_6col,
            })
    print(f"  produced {len(all_tracks):,} tracks "
          f"({n_dropped_resample} dropped by resample, "
          f"{n_dropped_short} dropped by length).")
    if len(all_tracks) == 0:
        sys.exit("No tracks produced. Loosen min_points or bbox.")

    # ------ MMSI-grouped split ------
    rng = np.random.default_rng(args.seed)
    roots = sorted({t["root_mmsi"] for t in all_tracks})
    rng.shuffle(roots)
    n_total = len(roots)
    n_test = int(round(n_total * args.test_frac))
    n_val = int(round(n_total * args.val_frac))
    test_roots = set(roots[:n_test])
    val_roots = set(roots[n_test : n_test + n_val])
    train_roots = set(roots[n_test + n_val:])

    splits = {"train": [], "valid": [], "test": []}
    for t in all_tracks:
        if t["root_mmsi"] in test_roots:
            splits["test"].append(t)
        elif t["root_mmsi"] in val_roots:
            splits["valid"].append(t)
        else:
            splits["train"].append(t)

    # ------ Write pickles in TrAISformer's expected schema ------
    # Their loader expects each entry as {"mmsi": int, "traj": (T, 5) ndarray}.
    # Drop our extra bookkeeping keys before writing.
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split in ("train", "valid", "test"):
        out_path = out_dir / f"ct_us_{split}.pkl"
        compact = [{"mmsi": t["mmsi"], "traj": t["traj"]} for t in splits[split]]
        with open(out_path, "wb") as f:
            pickle.dump(compact, f, protocol=pickle.HIGHEST_PROTOCOL)
        counts[split] = len(compact)
        print(f"  wrote {out_path}  ({len(compact):,} tracks)")

    # ------ Metadata ------
    meta = {
        "source_csv": [str(Path(p).name) for p in paths],
        "bbox": {"lon_min": lon_min, "lat_min": lat_min,
                 "lon_max": lon_max, "lat_max": lat_max,
                 "lon_range_deg": lon_max - lon_min,
                 "lat_range_deg": lat_max - lat_min},
        "resample_minutes": args.resample_minutes,
        "max_gap_minutes": args.max_gap_minutes,
        "min_points": args.min_points,
        "keep_vessel_types": args.keep_vessel_types,
        "split": {"train": counts["train"], "valid": counts["valid"],
                  "test": counts["test"]},
        "split_policy": "root-MMSI grouped (matches src/data.py::_root_id)",
        "seed": args.seed,
        "sog_range": SOG_RANGE,
        "cog_range": COG_RANGE,
        "format_note": "Each pkl is a list of dicts {mmsi:int, traj: (T,6) float32}. "
                       "Columns: lat_norm, lon_norm, sog_norm, cog_norm, "
                       "unix_timestamp, mmsi_repeat. Column 5 (mmsi_repeat) "
                       "matches TrAISformer's shipped DMA format and is ignored "
                       "by their loader.",
    }
    meta_path = out_dir / "meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  wrote {meta_path}")
    print("Done.")


if __name__ == "__main__":
    main()
