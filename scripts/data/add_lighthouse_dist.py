#!/usr/bin/env python3
"""
add_lighthouse_dist.py — Enrich an AIS CSV with distances to the 3 nearest lighthouses.

Reads a processed AIS CSV and a lighthouse reference CSV, computes haversine
distances using a BallTree for efficiency, and writes a new CSV with three
additional columns: lh_dist_1_km, lh_dist_2_km, lh_dist_3_km.

Usage:
    python3 scripts/data/add_lighthouse_dist.py \
        --csv data/processed/AIS_2024_gt80.csv \
        --lighthouses data/lighthouses.csv \
        --output data/processed/AIS_2024_gt80_lh.csv
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree
from tqdm import tqdm

EARTH_RADIUS_KM = 6371.0

# US coastal bounding box (matches prepare_data.py defaults)
BBOX_LON = (-125.0, -60.0)
BBOX_LAT = (10.0, 55.0)

# Common lat/lon column name variants
LAT_NAMES = ["lat", "LAT", "latitude", "Latitude", "LATITUDE"]
LON_NAMES = ["lon", "LON", "longitude", "Longitude", "LONGITUDE"]


def detect_col(df_columns, candidates, label):
    """Find the first matching column name from a list of candidates."""
    for name in candidates:
        if name in df_columns:
            return name
    raise ValueError(
        f"Cannot detect {label} column. Tried {candidates}. "
        f"Available: {list(df_columns)}"
    )


def load_lighthouses(path, k_needed=3):
    """Load and filter lighthouse positions to the US coastal bounding box.

    Returns numpy array of shape (N, 2) in radians: [lat_rad, lon_rad].
    """
    df = pd.read_csv(path)
    lat_col = detect_col(df.columns, LAT_NAMES, "latitude")
    lon_col = detect_col(df.columns, LON_NAMES, "longitude")
    print(f"Lighthouse CSV: {len(df):,} rows, using columns {lat_col}/{lon_col}")

    # Filter to bounding box
    mask = (
        (df[lon_col] >= BBOX_LON[0]) & (df[lon_col] <= BBOX_LON[1]) &
        (df[lat_col] >= BBOX_LAT[0]) & (df[lat_col] <= BBOX_LAT[1])
    )
    df = df[mask].dropna(subset=[lat_col, lon_col])
    print(f"After bbox filter: {len(df):,} lighthouses in "
          f"lon {BBOX_LON}, lat {BBOX_LAT}")

    if len(df) < k_needed:
        raise ValueError(
            f"Only {len(df)} lighthouses after filtering — need at least {k_needed}. "
            f"Check your lighthouse CSV and bounding box."
        )

    # BallTree expects (lat_rad, lon_rad) — latitude first!
    coords_rad = np.deg2rad(df[[lat_col, lon_col]].to_numpy(dtype=np.float64))
    return coords_rad


def main():
    parser = argparse.ArgumentParser(
        description="Add lighthouse distance columns to an AIS CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--csv", required=True,
                        help="Input processed AIS CSV")
    parser.add_argument("--lighthouses", required=True,
                        help="Lighthouse reference CSV (must have lat/lon columns)")
    parser.add_argument("--output", required=True,
                        help="Output CSV path (input + 3 distance columns)")
    parser.add_argument("--k", type=int, default=3,
                        help="Number of nearest lighthouses")
    parser.add_argument("--chunk_size", type=int, default=500_000,
                        help="Rows per chunk for memory-safe processing")
    parser.add_argument("--lat_col", default="LAT",
                        help="Latitude column in the AIS CSV")
    parser.add_argument("--lon_col", default="LON",
                        help="Longitude column in the AIS CSV")
    args = parser.parse_args()

    # ── Load lighthouses and build spatial index ─────────────────────────────
    lh_coords = load_lighthouses(args.lighthouses, k_needed=args.k)
    tree = BallTree(lh_coords, metric="haversine")
    print(f"BallTree built with {len(lh_coords):,} lighthouses\n")

    # ── Count total rows for progress bar ────────────────────────────────────
    total_rows = sum(1 for _ in open(args.csv)) - 1  # subtract header
    print(f"AIS CSV: {total_rows:,} rows")

    # ── Process in chunks ────────────────────────────────────────────────────
    reader = pd.read_csv(args.csv, chunksize=args.chunk_size)
    first_chunk = True

    dist_col_names = [f"lh_dist_{i+1}_km" for i in range(args.k)]

    with tqdm(total=total_rows, unit="rows", desc="Computing distances") as pbar:
        for chunk in reader:
            lat = chunk[args.lat_col].to_numpy(dtype=np.float64)
            lon = chunk[args.lon_col].to_numpy(dtype=np.float64)

            # BallTree query: (lat_rad, lon_rad)
            query_rad = np.deg2rad(np.column_stack([lat, lon]))
            distances_rad, _ = tree.query(query_rad, k=args.k)

            # Convert radians to km
            distances_km = distances_rad * EARTH_RADIUS_KM

            for i, col_name in enumerate(dist_col_names):
                chunk[col_name] = distances_km[:, i].astype(np.float32)

            chunk.to_csv(
                args.output,
                mode="w" if first_chunk else "a",
                header=first_chunk,
                index=False,
            )
            first_chunk = False
            pbar.update(len(chunk))

    print(f"\nDone. Output: {args.output}")
    print(f"  Added columns: {dist_col_names}")

    # Quick sanity check on first few rows
    sample = pd.read_csv(args.output, nrows=5)
    for col in dist_col_names:
        vals = sample[col].values
        print(f"  {col} sample: {vals}")


if __name__ == "__main__":
    main()
