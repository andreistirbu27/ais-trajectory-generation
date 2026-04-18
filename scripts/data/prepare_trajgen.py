#!/usr/bin/env python3
"""
prepare_trajgen.py — Build trajectory-generation dataset from processed AIS CSV.

Takes an already-filtered CSV (from prepare_data.py or the full pipeline) and
produces an NPZ file with fixed-length resampled trajectories suitable for
conditional trajectory generation (start, end, ship_type → full route).

Key differences from the next-step prediction pipeline:
  - No turn-based segmentation (we want full voyages with turns)
  - Tracks are resampled to a fixed number of equally-spaced-along-arc points
  - Filtered by start→end straight-line distance (keep meaningful voyages)
  - Output is NPZ (fast loading) not CSV

Usage:
    # From an already-filtered CSV (e.g. without turn segmentation):
    python3 scripts/data/prepare_trajgen.py \
        --csv data/processed/AIS_2024_full.csv \
        --output data/processed/trajgen_128.npz

    # From raw AIS files (runs prepare_data.py filtering inline, no turn seg):
    python3 scripts/data/prepare_trajgen.py \
        --raw data/raw/AIS_2024_01_01.csv data/raw/AIS_2024_01_02.csv \
        --output data/processed/trajgen_128.npz

    # Adjust resampling and distance filters:
    python3 scripts/data/prepare_trajgen.py \
        --csv data/processed/AIS_2024_full.csv \
        --n_resample 128 --min_dist_km 20 --max_dist_km 2000 \
        --output data/processed/trajgen_128.npz
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# Allow imports from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

EARTH_RADIUS_KM = 6371.0


# ── Geometry helpers ─────────────────────────────────────────────────────────

def haversine_km(lat1, lon1, lat2, lon2):
    """Haversine distance in km between two points in degrees."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    return EARTH_RADIUS_KM * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def resample_track(lon, lat, n_points=128):
    """Resample a track to n_points equally spaced along cumulative arc length.

    Uses linear interpolation on the cumulative distance curve.
    Returns (lon_resampled, lat_resampled) each of shape (n_points,).
    """
    dlon = np.diff(lon)
    dlat = np.diff(lat)
    step_dist = np.sqrt(dlon**2 + dlat**2)
    cum_dist = np.concatenate([[0.0], np.cumsum(step_dist)])
    total = cum_dist[-1]

    if total < 1e-10:
        # Degenerate track — all points at same location
        return np.full(n_points, lon[0]), np.full(n_points, lat[0])

    target = np.linspace(0.0, total, n_points)
    lon_r = np.interp(target, cum_dist, lon)
    lat_r = np.interp(target, cum_dist, lat)
    return lon_r, lat_r


# ── Inline filtering (for --raw mode) ───────────────────────────────────────

def filter_raw_for_trajgen(df, args):
    """Apply prepare_data.py-style filtering without turn segmentation.

    This is a simplified version of prepare_data.filter_data() with
    max_turn_deg=0 (no turn splitting) and adjusted defaults for
    trajectory generation.
    """
    id_col = args.id_col
    time_col = args.time_col
    lat_col = args.lat_col
    lon_col = args.lon_col
    vtype_col = args.vessel_type_col
    n_before = len(df)

    # Missing values
    df = df.dropna(subset=[id_col, time_col, lat_col, lon_col])

    # Vessel type filter (60-89 = commercial)
    if args.keep_vessel_types and vtype_col in df.columns:
        allowed = set()
        for part in args.keep_vessel_types.split(","):
            part = part.strip()
            if "-" in part:
                lo, hi = part.split("-", 1)
                allowed.update(range(int(lo), int(hi) + 1))
            else:
                allowed.add(int(part))
        df = df[df[vtype_col].isin(allowed)].reset_index(drop=True)

    # Parse timestamps
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col])

    # Valid coordinates
    df = df[df[lat_col].between(-90, 90) & df[lon_col].between(-180, 180)]

    # Bounding box
    df = df[
        (df[lat_col] >= args.lat_min) & (df[lat_col] <= args.lat_max) &
        (df[lon_col] >= args.lon_min) & (df[lon_col] <= args.lon_max)
    ].reset_index(drop=True)

    # Duplicate (vessel, timestamp)
    df = df.drop_duplicates(subset=[id_col, time_col])
    df = df.sort_values([id_col, time_col]).reset_index(drop=True)

    # Temporal gap segmentation (split at gaps > max_gap_min)
    dt_s = df.groupby(id_col)[time_col].diff().dt.total_seconds().fillna(0)
    gap_start = (dt_s > args.max_gap_min * 60)
    seg_idx = gap_start.groupby(df[id_col]).cumsum().astype(int)
    df[id_col] = df[id_col].astype(str) + "_" + seg_idx.astype(str)

    # NO turn-based segmentation — this is the key difference

    # Unrealistic speed filter
    dlon = df.groupby(id_col)[lon_col].diff()
    dlat = df.groupby(id_col)[lat_col].diff()
    jump_km = np.sqrt(dlon**2 + dlat**2) * 111
    dt_h = df.groupby(id_col)[time_col].diff().dt.total_seconds() / 3600
    speed = jump_km / dt_h.replace(0, np.nan)
    df = df[(speed <= args.max_speed_kmh) | speed.isna()].reset_index(drop=True)

    # Short tracks
    counts = df.groupby(id_col)[id_col].transform("count")
    df = df[counts >= args.min_points].reset_index(drop=True)

    n_after = len(df)
    n_vessels = df[id_col].nunique()
    print(f"Filtering: {n_before:,} → {n_after:,} rows ({n_vessels:,} tracks)")
    return df


# ── Main pipeline ────────────────────────────────────────────────────────────

def build_trajgen_dataset(df, args):
    """Extract, filter, and resample trajectories from a DataFrame.

    Returns:
        trajectories: (N, n_resample, 2) float32 [lon, lat]
        vessel_types: (N,) int32
        track_ids:    list of str
    """
    id_col = args.id_col
    time_col = args.time_col
    lat_col = args.lat_col
    lon_col = args.lon_col
    vtype_col = args.vessel_type_col
    n_resample = args.n_resample

    # Sort and group
    df = df.sort_values([id_col, time_col]).reset_index(drop=True)
    groups = df.groupby(id_col)

    trajectories = []
    vtypes = []
    tids = []

    stats = {"total": 0, "too_short": 0, "too_close": 0,
             "too_far": 0, "zero_length": 0, "kept": 0}

    for vid, grp in tqdm(groups, desc="Processing tracks"):
        stats["total"] += 1
        lon = grp[lon_col].to_numpy(dtype=np.float64)
        lat = grp[lat_col].to_numpy(dtype=np.float64)

        # Need enough raw points for meaningful resampling
        if len(lon) < args.min_points:
            stats["too_short"] += 1
            continue

        # Start-end straight-line distance filter
        dist_km = haversine_km(lat[0], lon[0], lat[-1], lon[-1])
        if dist_km < args.min_dist_km:
            stats["too_close"] += 1
            continue
        if dist_km > args.max_dist_km:
            stats["too_far"] += 1
            continue

        # Resample to fixed number of points
        lon_r, lat_r = resample_track(lon, lat, n_resample)

        # Check for degenerate resampling
        arc = np.sqrt(np.diff(lon_r)**2 + np.diff(lat_r)**2).sum()
        if arc < 1e-8:
            stats["zero_length"] += 1
            continue

        traj = np.stack([lon_r, lat_r], axis=1).astype(np.float32)  # (N, 2)
        trajectories.append(traj)

        # Vessel type
        if vtype_col in grp.columns:
            vt = grp[vtype_col].dropna()
            vtypes.append(int(vt.iloc[0]) if len(vt) > 0 else 0)
        else:
            vtypes.append(0)

        tids.append(str(vid))
        stats["kept"] += 1

    print(f"\nTrack filtering stats:")
    print(f"  Total tracks:       {stats['total']:,}")
    print(f"  Too short (<{args.min_points} pts): {stats['too_short']:,}")
    print(f"  Start≈end (<{args.min_dist_km} km): {stats['too_close']:,}")
    print(f"  Too far (>{args.max_dist_km} km):   {stats['too_far']:,}")
    print(f"  Zero arc length:    {stats['zero_length']:,}")
    print(f"  Kept:               {stats['kept']:,}")

    trajectories = np.array(trajectories, dtype=np.float32)  # (N, n_resample, 2)
    vtypes = np.array(vtypes, dtype=np.int32)                # (N,)

    return trajectories, vtypes, tids


def print_dataset_stats(trajectories, vtypes, tids):
    """Print summary statistics of the built dataset."""
    N = len(trajectories)
    if N == 0:
        print("\nWARNING: No trajectories in dataset!")
        return

    # Start-end distances
    starts = trajectories[:, 0, :]   # (N, 2) [lon, lat]
    ends = trajectories[:, -1, :]    # (N, 2) [lon, lat]
    dists_km = np.array([
        haversine_km(starts[i, 1], starts[i, 0], ends[i, 1], ends[i, 0])
        for i in range(N)
    ])

    # Arc lengths (approximate, degree units)
    deltas = np.diff(trajectories, axis=1)  # (N, n-1, 2)
    step_deg = np.sqrt((deltas**2).sum(axis=2))  # (N, n-1)
    arc_deg = step_deg.sum(axis=1)
    arc_km = arc_deg * 111  # rough km

    # Vessel type distribution
    unique_vt, counts_vt = np.unique(vtypes, return_counts=True)

    print(f"\n{'='*60}")
    print(f"  TRAJECTORY GENERATION DATASET")
    print(f"{'='*60}")
    print(f"  Trajectories:  {N:,}")
    print(f"  Points/track:  {trajectories.shape[1]}")
    print(f"  Start→end (km): min={dists_km.min():.1f}  "
          f"median={np.median(dists_km):.1f}  max={dists_km.max():.1f}")
    print(f"  Arc length (km): min={arc_km.min():.1f}  "
          f"median={np.median(arc_km):.1f}  max={arc_km.max():.1f}")
    print(f"  Lon range:  [{trajectories[:,:,0].min():.2f}, "
          f"{trajectories[:,:,0].max():.2f}]")
    print(f"  Lat range:  [{trajectories[:,:,1].min():.2f}, "
          f"{trajectories[:,:,1].max():.2f}]")
    print(f"\n  Vessel types ({len(unique_vt)}):")
    VTYPE_NAME = {
        **{c: "Passenger" for c in range(60, 70)},
        **{c: "Cargo"     for c in range(70, 80)},
        **{c: "Tanker"    for c in range(80, 90)},
    }
    for vt, cnt in sorted(zip(unique_vt, counts_vt), key=lambda x: -x[1]):
        name = VTYPE_NAME.get(int(vt), f"type_{vt}")
        print(f"    {name:>12} (code {vt:>2}): {cnt:>6,}  ({100*cnt/N:.1f}%)")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Build trajectory-generation dataset (NPZ) from AIS data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input (one of --csv or --raw required)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--csv",
        help="Pre-filtered CSV (from prepare_data.py without turn segmentation)")
    input_group.add_argument("--raw", nargs="+",
        help="Raw AIS CSV files (filtering applied inline)")

    parser.add_argument("--output", "-o", required=True,
        help="Output NPZ path (e.g. data/processed/trajgen_128.npz)")

    # Resampling
    parser.add_argument("--n_resample", type=int, default=128,
        help="Number of points to resample each trajectory to")

    # Distance filters
    parser.add_argument("--min_dist_km", type=float, default=20.0,
        help="Min start→end straight-line distance in km")
    parser.add_argument("--max_dist_km", type=float, default=2000.0,
        help="Max start→end straight-line distance in km")

    # Track length filter
    parser.add_argument("--min_points", type=int, default=50,
        help="Min raw points per track (before resampling)")

    # Column names
    parser.add_argument("--id_col", default="MMSI")
    parser.add_argument("--time_col", default="BaseDateTime")
    parser.add_argument("--lat_col", default="LAT")
    parser.add_argument("--lon_col", default="LON")
    parser.add_argument("--vessel_type_col", default="VesselType")

    # Raw-mode filtering params (only used with --raw)
    parser.add_argument("--keep_vessel_types", default="60-89")
    parser.add_argument("--max_gap_min", type=float, default=60.0)
    parser.add_argument("--max_speed_kmh", type=float, default=40.0)
    parser.add_argument("--lon_min", type=float, default=-125.0)
    parser.add_argument("--lon_max", type=float, default=-60.0)
    parser.add_argument("--lat_min", type=float, default=10.0)
    parser.add_argument("--lat_max", type=float, default=55.0)

    args = parser.parse_args()

    # ── Load data ────────────────────────────────────────────────────────────
    if args.csv:
        print(f"Loading pre-filtered CSV: {args.csv}")
        df = pd.read_csv(args.csv, low_memory=False)
        df[args.time_col] = pd.to_datetime(df[args.time_col], errors="coerce")
        df = df.dropna(subset=[args.time_col])
        print(f"  {len(df):,} rows, {df[args.id_col].nunique():,} tracks")
    else:
        print(f"Loading {len(args.raw)} raw CSV files...")
        dfs = []
        for path in args.raw:
            print(f"  {path}")
            dfs.append(pd.read_csv(path, low_memory=False))
        df = pd.concat(dfs, ignore_index=True)
        df = df.drop_duplicates(subset=[args.id_col, args.time_col])
        print(f"  Combined: {len(df):,} rows")
        df = filter_raw_for_trajgen(df, args)

    # ── Build dataset ────────────────────────────────────────────────────────
    trajectories, vtypes, tids = build_trajgen_dataset(df, args)

    if len(trajectories) == 0:
        print("ERROR: No trajectories survived filtering. Check your data and thresholds.")
        sys.exit(1)

    print_dataset_stats(trajectories, vtypes, tids)

    # ── Save NPZ ─────────────────────────────────────────────────────────────
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        trajectories=trajectories,
        vessel_types=vtypes,
        track_ids=np.array(tids, dtype=str),
    )
    size_mb = out_path.stat().st_size / 1024**2
    print(f"\nSaved → {out_path}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
