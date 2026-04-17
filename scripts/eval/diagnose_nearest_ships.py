#!/usr/bin/env python3
"""
diagnose_nearest_ships.py — Check whether nearby-vessel features would be
useful for open-sea trajectory prediction.

Samples N rows from the AIS CSV, groups by 1-minute time bins, and for each
row finds the distance to the nearest other vessel (different root MMSI) in
the same bin. Plots the distribution and prints summary statistics.

If median nearest-vessel distance is > 20 km, the feature is likely too sparse
to help for open-sea data.

Usage:
    python3 scripts/eval/diagnose_nearest_ships.py \
        --csv data/processed/AIS_2024_gt80_opensea.csv \
        --n_sample 50000

    # Save plot to specific directory
    python3 scripts/eval/diagnose_nearest_ships.py \
        --csv data/processed/AIS_2024_gt80_opensea.csv \
        --out_dir outputs/diagnostics
"""

import argparse
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree
from tqdm import tqdm

EARTH_RADIUS_KM = 6371.0


def _root_id(vid: str) -> str:
    """Strip batch prefix and segment suffixes to get original MMSI."""
    s = re.sub(r'^b\d+_', '', str(vid))
    s = re.sub(r'(_\d+)+$', '', s)
    return s


def main():
    p = argparse.ArgumentParser(
        description="Diagnose nearest-vessel distances in AIS data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--csv", required=True, help="Processed AIS CSV")
    p.add_argument("--n_sample", type=int, default=50_000,
                   help="Number of rows to sample for analysis")
    p.add_argument("--time_bin_sec", type=int, default=60,
                   help="Time bin size in seconds for grouping pings")
    p.add_argument("--out_dir", default="outputs/diagnostics")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--id_col", default="MMSI")
    p.add_argument("--time_col", default="BaseDateTime")
    p.add_argument("--lat_col", default="LAT")
    p.add_argument("--lon_col", default="LON")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ── Load and sample ────────────────────────────────────────────────────
    print(f"Loading {args.csv} ...")
    df = pd.read_csv(args.csv, usecols=[args.id_col, args.time_col,
                                         args.lat_col, args.lon_col])
    print(f"  Total rows: {len(df):,}")

    df[args.time_col] = pd.to_datetime(df[args.time_col], errors="coerce")
    df = df.dropna(subset=[args.time_col])

    # Add root MMSI column
    df["root_mmsi"] = df[args.id_col].astype(str).apply(_root_id)

    # Time bins: floor to nearest time_bin_sec
    df["time_bin"] = (
        df[args.time_col].astype("int64") // 10**9 // args.time_bin_sec
    )

    # Sample rows
    if len(df) > args.n_sample:
        df_sample = df.sample(n=args.n_sample, random_state=args.seed)
    else:
        df_sample = df
    print(f"  Sampled rows: {len(df_sample):,}")

    # ── Build per-bin spatial indices ──────────────────────────────────────
    # For each sampled row, we need all vessels in its time bin.
    # Get the set of time bins we need.
    needed_bins = set(df_sample["time_bin"].unique())
    print(f"  Unique time bins needed: {len(needed_bins):,}")

    # Build lookup: time_bin -> DataFrame of all vessels in that bin
    print("Building per-bin vessel index ...")
    bin_df = df[df["time_bin"].isin(needed_bins)]

    # Deduplicate: keep one row per (root_mmsi, time_bin) — latest position
    bin_df = bin_df.sort_values(args.time_col).drop_duplicates(
        subset=["root_mmsi", "time_bin"], keep="last"
    )
    print(f"  Unique (vessel, bin) entries: {len(bin_df):,}")

    bin_groups = dict(list(bin_df.groupby("time_bin")))

    # ── Query nearest vessel per sample row ───────────────────────────────
    print("Finding nearest vessel for each sampled row ...")
    nn_dists_km = []

    for _, row in tqdm(df_sample.iterrows(), total=len(df_sample),
                       desc="Querying", unit=" rows"):
        tbin = row["time_bin"]
        root = row["root_mmsi"]

        if tbin not in bin_groups:
            continue

        grp = bin_groups[tbin]
        # Exclude same vessel
        others = grp[grp["root_mmsi"] != root]
        if len(others) == 0:
            nn_dists_km.append(np.nan)
            continue

        # BallTree on other vessels in this bin
        coords_rad = np.deg2rad(
            others[[args.lat_col, args.lon_col]].to_numpy(dtype=np.float64)
        )
        tree = BallTree(coords_rad, metric="haversine")

        query = np.deg2rad(
            np.array([[row[args.lat_col], row[args.lon_col]]], dtype=np.float64)
        )
        dist_rad, _ = tree.query(query, k=1)
        nn_dists_km.append(float(dist_rad[0, 0] * EARTH_RADIUS_KM))

    nn_dists_km = np.array(nn_dists_km)
    valid = nn_dists_km[~np.isnan(nn_dists_km)]

    # ── Print statistics ──────────────────────────────────────────────────
    print(f"\n{'NEAREST-VESSEL DISTANCE DIAGNOSTIC':^50}")
    print("=" * 50)
    print(f"Sampled rows:     {len(df_sample):>10,}")
    print(f"With neighbours:  {len(valid):>10,}  "
          f"({len(valid)/len(df_sample)*100:.1f}%)")
    print(f"No neighbours:    {int(np.isnan(nn_dists_km).sum()):>10,}  "
          f"(alone in time bin)")
    print("-" * 50)
    if len(valid) > 0:
        print(f"Min:              {np.min(valid):>10.1f} km")
        print(f"P25:              {np.percentile(valid, 25):>10.1f} km")
        print(f"Median:           {np.median(valid):>10.1f} km")
        print(f"P75:              {np.percentile(valid, 75):>10.1f} km")
        print(f"P90:              {np.percentile(valid, 90):>10.1f} km")
        print(f"P95:              {np.percentile(valid, 95):>10.1f} km")
        print(f"Max:              {np.max(valid):>10.1f} km")
        print(f"Mean:             {np.mean(valid):>10.1f} km")
    print("=" * 50)

    if len(valid) > 0 and np.median(valid) > 20:
        print("\n>> Median > 20 km: nearest-vessel features likely too sparse")
        print("   for open-sea data. Consider skipping this feature.")
    elif len(valid) > 0:
        print(f"\n>> Median = {np.median(valid):.1f} km: signal may be useful!")
        print("   Proceed with building the full nearest-ships pipeline.")

    # ── Plot ──────────────────────────────────────────────────────────────
    if len(valid) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Histogram
        ax = axes[0]
        ax.hist(valid[valid < 100], bins=80, color="#1565c0", edgecolor="white",
                linewidth=0.3, alpha=0.85)
        ax.axvline(np.median(valid), color="#c62828", linestyle="--", linewidth=2,
                   label=f"Median: {np.median(valid):.1f} km")
        ax.axvline(np.percentile(valid, 25), color="#e65100", linestyle=":",
                   linewidth=1.5, label=f"P25: {np.percentile(valid, 25):.1f} km")
        ax.axvline(np.percentile(valid, 75), color="#e65100", linestyle=":",
                   linewidth=1.5, label=f"P75: {np.percentile(valid, 75):.1f} km")
        ax.set_xlabel("Distance to nearest vessel (km)")
        ax.set_ylabel("Count")
        ax.set_title("Nearest-vessel distance distribution\n(capped at 100 km)")
        ax.legend(fontsize=9)

        # CDF
        ax = axes[1]
        sorted_v = np.sort(valid)
        cdf = np.arange(1, len(sorted_v) + 1) / len(sorted_v)
        ax.plot(sorted_v, cdf, color="#1565c0", linewidth=1.5)
        ax.axhline(0.5, color="#c62828", linestyle="--", linewidth=1, alpha=0.5)
        ax.axvline(np.median(valid), color="#c62828", linestyle="--", linewidth=1,
                   alpha=0.5)
        for thresh in [5, 10, 20]:
            frac = (valid < thresh).mean()
            ax.axvline(thresh, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
            ax.text(thresh + 0.5, 0.02, f"<{thresh}km: {frac*100:.1f}%",
                    fontsize=7, rotation=90, va="bottom")
        ax.set_xlabel("Distance to nearest vessel (km)")
        ax.set_ylabel("Cumulative fraction")
        ax.set_title("CDF of nearest-vessel distances")
        ax.set_xlim(0, max(100, np.percentile(valid, 99)))

        plt.tight_layout()
        out_path = os.path.join(args.out_dir, "nearest_vessel_distances.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\nPlot saved: {out_path}")


if __name__ == "__main__":
    main()
