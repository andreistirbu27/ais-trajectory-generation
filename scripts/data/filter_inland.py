#!/usr/bin/env python3
"""
filter_inland.py — Remove tracks that traverse rivers, channels, ports, and
other non-open-sea areas.

Uses GSHHS L1 land polygons to classify each AIS point as inland, coastal
(< min_shore_dist_km from coastline), or open-sea. Drops entire tracks where
too many points are coastal/inland or where median shore distance is too low.

Download GSHHS shapefiles (intermediate resolution recommended):
    https://www.soest.hawaii.edu/pwessel/gshhg/
    Extract and point --shoreline to e.g. GSHHS_shp/i/GSHHS_i_L1.shp

Usage:
    python3 scripts/data/filter_inland.py \
        --csv data/processed/AIS_2024_gt80.csv \
        --shoreline data/gshhg/GSHHS_i_L1.shp \
        --output data/processed/AIS_2024_gt80_opensea.csv
"""

import argparse
from collections import defaultdict

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from sklearn.neighbors import BallTree
from tqdm import tqdm

EARTH_RADIUS_KM = 6371.0

# US coastal bounding box (matches prepare_data.py defaults)
BBOX = (-125.0, 10.0, -60.0, 55.0)  # (minx, miny, maxx, maxy)


def build_coastline_index(shoreline_path):
    """Load GSHHS L1 shapefile, build land polygon and coastline BallTree.

    Returns:
        land_union: prepared shapely MultiPolygon of all land within bbox
        tree: BallTree on densified coastline vertices (haversine, radians)
    """
    print(f"Loading shoreline: {shoreline_path}")
    gdf = gpd.read_file(shoreline_path, bbox=BBOX)
    print(f"  Polygons in bbox: {len(gdf):,}")

    gdf.geometry = gdf.geometry.make_valid()
    land_union = gdf.union_all()
    shapely.prepare(land_union)
    print(f"  Land union: {land_union.geom_type} "
          f"({shapely.get_num_geometries(land_union):,} parts)")

    # Densify coastline so BallTree nearest-vertex error is bounded
    # max_segment_length=0.005 degrees ≈ 500 m → error ≤ ~250 m
    boundary = land_union.boundary
    boundary_dense = shapely.segmentize(boundary, max_segment_length=0.005)

    # Extract all vertices as (lon, lat) array
    coords = shapely.get_coordinates(boundary_dense)  # (N, 2) = [x/lon, y/lat]
    print(f"  Coastline vertices (densified): {len(coords):,}")

    # BallTree expects (lat_rad, lon_rad) — latitude first
    coords_rad = np.deg2rad(coords[:, ::-1])  # flip to (lat, lon), then radians
    tree = BallTree(coords_rad, metric="haversine")
    print(f"  BallTree built\n")

    return land_union, tree


def main():
    parser = argparse.ArgumentParser(
        description="Filter inland/coastal AIS tracks, keep open-sea only.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--csv", required=True,
                        help="Input processed AIS CSV")
    parser.add_argument("--shoreline", required=True,
                        help="Path to GSHHS L1 shapefile (e.g. GSHHS_i_L1.shp)")
    parser.add_argument("--output", required=True,
                        help="Output CSV path (filtered)")
    parser.add_argument("--min_shore_dist_km", type=float, default=2.0,
                        help="Min distance to coast for a point to be 'open sea'")
    parser.add_argument("--max_coastal_frac", type=float, default=0.2,
                        help="Max fraction of coastal/inland points before track is dropped")
    parser.add_argument("--min_median_shore_dist_km", type=float, default=5.0,
                        help="Drop tracks with median shore distance below this")
    parser.add_argument("--chunk_size", type=int, default=500_000,
                        help="Rows per CSV chunk")
    parser.add_argument("--id_col", default="MMSI")
    parser.add_argument("--lat_col", default="LAT")
    parser.add_argument("--lon_col", default="LON")
    args = parser.parse_args()

    # ── Build spatial index ─────────────────────────────────────────────────
    land_union, tree = build_coastline_index(args.shoreline)

    # ── Count total rows ────────────────────────────────────────────────────
    total_rows = sum(1 for _ in open(args.csv)) - 1
    print(f"AIS CSV: {total_rows:,} rows\n")

    # ── Pass 1: compute per-track statistics ────────────────────────────────
    print("Pass 1: classifying points …")
    # Per-MMSI counters: total, bad (inland + coastal), inland, far_enough
    stats = defaultdict(lambda: [0, 0, 0, 0])  # [total, bad, inland, far_enough]

    reader = pd.read_csv(args.csv, chunksize=args.chunk_size)
    with tqdm(total=total_rows, unit=" rows", desc="Classifying") as pbar:
        for chunk in reader:
            lon = chunk[args.lon_col].to_numpy(dtype=np.float64)
            lat = chunk[args.lat_col].to_numpy(dtype=np.float64)
            ids = chunk[args.id_col].values

            # BallTree: nearest coastline vertex distance
            query_rad = np.deg2rad(np.column_stack([lat, lon]))  # (lat, lon)
            dist_rad, _ = tree.query(query_rad, k=1)
            shore_dist_km = dist_rad.ravel() * EARTH_RADIUS_KM

            # Inland check: points inside land polygons
            points = shapely.points(lon, lat)
            inland = shapely.contains(land_union, points)

            # Inland points get shore_dist = 0
            shore_dist_km[inland] = 0.0

            # Classify
            is_bad = inland | (shore_dist_km < args.min_shore_dist_km)
            is_far = shore_dist_km >= args.min_median_shore_dist_km

            # Accumulate per-MMSI
            for mmsi in np.unique(ids):
                mask = ids == mmsi
                s = stats[mmsi]
                s[0] += int(mask.sum())
                s[1] += int(is_bad[mask].sum())
                s[2] += int(inland[mask].sum())
                s[3] += int(is_far[mask].sum())

            pbar.update(len(chunk))

    # ── Filter decision ─────────────────────────────────────────────────────
    n_total = len(stats)
    dropped_coastal = set()
    dropped_median = set()

    for mmsi, (total, bad, inl, far) in stats.items():
        coastal_frac = bad / total if total > 0 else 1.0
        median_ok = (far / total) >= 0.5 if total > 0 else False

        if coastal_frac > args.max_coastal_frac:
            dropped_coastal.add(mmsi)
        if not median_ok:
            dropped_median.add(mmsi)

    dropped_all = dropped_coastal | dropped_median
    kept = set(stats.keys()) - dropped_all

    # ── Pass 2: write filtered CSV ──────────────────────────────────────────
    print(f"\nPass 2: writing filtered CSV …")
    reader = pd.read_csv(args.csv, chunksize=args.chunk_size)
    first_chunk = True
    rows_in = 0
    rows_out = 0

    with tqdm(total=total_rows, unit=" rows", desc="Writing") as pbar:
        for chunk in reader:
            rows_in += len(chunk)
            filtered = chunk[chunk[args.id_col].isin(kept)]
            rows_out += len(filtered)

            if len(filtered) > 0:
                filtered.to_csv(
                    args.output,
                    mode="w" if first_chunk else "a",
                    header=first_chunk,
                    index=False,
                )
                first_chunk = False

            pbar.update(len(chunk))

    # ── Audit summary ───────────────────────────────────────────────────────
    n_dropped_coastal_only = len(dropped_coastal - dropped_median)
    n_dropped_median_only = len(dropped_median - dropped_coastal)
    n_dropped_both = len(dropped_coastal & dropped_median)

    print(f"\n{'INLAND / COASTAL FILTER AUDIT':^50}")
    print("─" * 50)
    print(f"Total tracks in:          {n_total:>8,}")
    print(f"Dropped (coastal frac):   {len(dropped_coastal):>8,}  "
          f"({len(dropped_coastal)/n_total*100:5.1f}%)  "
          f"— >{args.max_coastal_frac*100:.0f}% of points within "
          f"{args.min_shore_dist_km} km of shore")
    print(f"Dropped (median dist):    {len(dropped_median):>8,}  "
          f"({len(dropped_median)/n_total*100:5.1f}%)  "
          f"— median shore dist < {args.min_median_shore_dist_km} km")
    print(f"  (both reasons):         {n_dropped_both:>8,}")
    print(f"Total dropped:            {len(dropped_all):>8,}  "
          f"({len(dropped_all)/n_total*100:5.1f}%)")
    print(f"Tracks kept:              {len(kept):>8,}  "
          f"({len(kept)/n_total*100:5.1f}%)")
    print("─" * 50)
    print(f"Rows in:            {rows_in:>12,}")
    print(f"Rows out:           {rows_out:>12,}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
