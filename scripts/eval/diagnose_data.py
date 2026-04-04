#!/usr/bin/env python3
"""
diagnose_data.py — AIS dataset diagnostic plots.

Plots distributions of 6 metrics that drive the prepare_data.py filters, with
dashed red threshold lines showing where each filter cuts.  Works on an already-
processed CSV — no model checkpoint required.

Usage:
    python3 scripts/diagnose_data.py \
        --csv data/processed/AIS_combined_processed.csv \
        --out_dir runs/
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ── Geometry helpers ───────────────────────────────────────────────────────────

def _haversine_km(lat1, lon1, lat2, lon2):
    """Vectorised haversine distance in km. Accepts numpy arrays."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = (np.radians(v) for v in (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(np.maximum(1 - a, 0)))


def _bearing(lat1, lon1, lat2, lon2):
    """Vectorised bearing from point 1 → 2 in degrees [0, 360). Accepts numpy arrays."""
    dlon = np.radians(lon2 - lon1)
    r1   = np.radians(lat1)
    r2   = np.radians(lat2)
    x    = np.sin(dlon) * np.cos(r2)
    y    = np.cos(r1) * np.sin(r2) - np.sin(r1) * np.cos(r2) * np.cos(dlon)
    return (np.degrees(np.arctan2(x, y)) + 360) % 360


# ── Metric computation ─────────────────────────────────────────────────────────

def compute_metrics(df, id_col, time_col, lat_col, lon_col):
    """
    Returns:
        ping_df  — per-ping DataFrame with columns: dt_s, dist_km, speed_kmh, turn_deg
        track_df — per-track DataFrame with columns: track_len, avg_speed_kmh
    """
    df = df.sort_values([id_col, time_col]).reset_index(drop=True)

    grp = df.groupby(id_col, sort=False)

    # Per-ping time delta and previous position (NaN for first ping per vessel)
    dt_s     = grp[time_col].diff().dt.total_seconds()
    lat_prev = grp[lat_col].shift(1)
    lon_prev = grp[lon_col].shift(1)
    lat_next = grp[lat_col].shift(-1)
    lon_next = grp[lon_col].shift(-1)

    lat = df[lat_col].values.astype(float)
    lon = df[lon_col].values.astype(float)

    dist_km = pd.Series(
        _haversine_km(lat_prev.values.astype(float), lon_prev.values.astype(float),
                      lat, lon),
        index=df.index)

    speed_kmh = dist_km / dt_s * 3600.0

    # Turn angle: angle between incoming and outgoing bearing at each ping
    brg_in  = _bearing(lat_prev.values.astype(float), lon_prev.values.astype(float),
                        lat, lon)
    brg_out = _bearing(lat, lon,
                        lat_next.values.astype(float), lon_next.values.astype(float))
    delta    = (brg_out - brg_in + 180) % 360 - 180
    turn_deg = pd.Series(np.abs(delta), index=df.index)

    ping_df = pd.DataFrame({
        "dt_s":      dt_s,
        "dist_km":   dist_km,
        "speed_kmh": speed_kmh,
        "turn_deg":  turn_deg,
    }).dropna()

    # Per-track aggregates
    track_len  = grp[lat_col].count().rename("track_len")
    df["_spd"] = speed_kmh
    avg_speed  = grp["_spd"].mean().rename("avg_speed_kmh")
    track_df   = pd.concat([track_len, avg_speed], axis=1).dropna()

    return ping_df, track_df


# ── Plotting ───────────────────────────────────────────────────────────────────

def _hist(ax, data, title, xlabel, threshold, color="steelblue", bins=80):
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    data = data[data > 0]
    ax.hist(data.values, bins=bins, color=color, alpha=0.85, edgecolor="none")
    ax.set_yscale("log")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel("count (log scale)", fontsize=8)
    ax.tick_params(labelsize=8)
    if threshold is not None:
        ax.axvline(threshold, color="crimson", linestyle="--", linewidth=1.5)
        ax.text(threshold, 0.97, f"  cutoff\n  {threshold}",
                color="crimson", fontsize=8, va="top",
                transform=ax.get_xaxis_transform())


def plot_diagnostics(ping_df, track_df, args, out_path):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()

    _hist(axes[0], ping_df["dt_s"],
          "Inter-ping time  (step 8: gap segmentation)",
          "seconds", threshold=args.max_gap_sec)

    _hist(axes[1], ping_df["dist_km"],
          "Inter-ping distance  (step 7: spatial jump filter)",
          "km", threshold=args.max_jump_km)

    _hist(axes[2], ping_df["speed_kmh"],
          "Per-step speed  (step 9: unrealistic speed filter)",
          "km/h", threshold=args.max_speed_kmh)

    _hist(axes[3], ping_df["turn_deg"],
          "Turn angle at each ping  (proposed angle-split threshold)",
          "degrees", threshold=args.max_turn_deg, bins=180)

    _hist(axes[4], track_df["track_len"].astype(float),
          "Track length  (step 10: minimum points filter)",
          "# pings", threshold=float(args.min_track_len))

    _hist(axes[5], track_df["avg_speed_kmh"],
          "Average speed per track  (step 11: avg-speed filter)",
          "km/h", threshold=args.min_avg_speed_kmh)

    fig.suptitle("AIS Dataset Diagnostics — distribution of filter metrics\n"
                 "(dashed red = filter cutoff; data shown is already post-filter)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Plot AIS data quality diagnostics with filter threshold lines",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--csv",      required=True)
    parser.add_argument("--out_dir",  default="runs/")
    parser.add_argument("--id_col",   default="MMSI")
    parser.add_argument("--time_col", default="BaseDateTime")
    parser.add_argument("--lat_col",  default="LAT")
    parser.add_argument("--lon_col",  default="LON")

    # Threshold overrides (match prepare_data.py defaults)
    parser.add_argument("--max_gap_sec",      type=float, default=3600.0,
                        help="Step 8 gap threshold in seconds")
    parser.add_argument("--max_jump_km",      type=float, default=2.0,
                        help="Step 7 spatial jump threshold in km")
    parser.add_argument("--max_speed_kmh",    type=float, default=40.0,
                        help="Step 9 speed filter threshold in km/h")
    parser.add_argument("--max_turn_deg",     type=float, default=150.0,
                        help="Proposed angle-split threshold in degrees")
    parser.add_argument("--min_track_len",    type=int,   default=50,
                        help="Step 10 minimum pings per track")
    parser.add_argument("--min_avg_speed_kmh", type=float, default=1.0,
                        help="Step 11 minimum average speed in km/h")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading {args.csv} ...")
    df = pd.read_csv(
        args.csv,
        usecols=[args.id_col, args.time_col, args.lat_col, args.lon_col],
        low_memory=False)
    df[args.time_col] = pd.to_datetime(df[args.time_col])
    n_vessels = df[args.id_col].nunique()
    print(f"  {len(df):,} rows, {n_vessels:,} vessels")

    print("Computing per-ping and per-track metrics ...")
    ping_df, track_df = compute_metrics(
        df, args.id_col, args.time_col, args.lat_col, args.lon_col)
    print(f"  {len(ping_df):,} ping pairs, {len(track_df):,} tracks")

    out_path = os.path.join(args.out_dir, "viz_data_diagnostics.png")
    plot_diagnostics(ping_df, track_df, args, out_path)


if __name__ == "__main__":
    main()
