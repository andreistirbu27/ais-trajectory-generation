#!/usr/bin/env python3
"""
preview_angle_split.py — Preview how angle-based segmentation would split tracks.

Shows N sample tracks (preferring those with heading reversals) in an N×2 grid:
  Left column  : original track in blue, reversal points marked as red stars (★)
  Right column : same track re-drawn, each resulting segment in a distinct colour

The CSV is never modified — this is a read-only preview.

Usage:
    python3 scripts/preview_angle_split.py \
        --csv data/processed/AIS_combined_processed.csv \
        --max_turn_deg 150 --n_tracks 5 --out_dir runs/
"""

import argparse
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ── Geometry helpers ───────────────────────────────────────────────────────────

def _bearing(lat1, lon1, lat2, lon2):
    """Vectorised bearing from point 1 → 2 in degrees [0, 360)."""
    dlon = np.radians(lon2 - lon1)
    r1   = np.radians(lat1)
    r2   = np.radians(lat2)
    x    = np.sin(dlon) * np.cos(r2)
    y    = np.cos(r1) * np.sin(r2) - np.sin(r1) * np.cos(r2) * np.cos(dlon)
    return (np.degrees(np.arctan2(x, y)) + 360) % 360


def compute_turns(lat, lon):
    """
    Turn angle (degrees) at each ping.  First and last pings are NaN
    (no incoming / outgoing bearing respectively).
    Shape: same length as lat/lon.
    """
    n = len(lat)
    if n < 3:
        return np.full(n, np.nan)
    brg_in  = _bearing(lat[:-1],  lon[:-1],  lat[1:],  lon[1:])   # len n-1
    brg_out = _bearing(lat[1:-1], lon[1:-1], lat[2:],  lon[2:])  # len n-2
    delta   = (brg_out - brg_in[:-1] + 180) % 360 - 180
    turn    = np.abs(delta)
    result  = np.full(n, np.nan)
    result[1:n - 1] = turn
    return result


def assign_segments(turns, max_turn_deg):
    """Integer segment ID for each ping (0, 0, …, 1, 1, …, 2, …)."""
    reversal = (~np.isnan(turns)) & (turns > max_turn_deg)
    return np.cumsum(reversal)


# ── Track selection ────────────────────────────────────────────────────────────

def select_tracks(df, id_col, time_col, lat_col, lon_col,
                  max_turn_deg, n_tracks, seed):
    """
    Returns a list of (vessel_id, sorted_group_df, n_reversals) tuples.
    Prefers vessels with the most reversals; falls back to any vessel.
    """
    rng = random.Random(seed)

    with_reversals = []
    without        = []

    for vid, grp in df.groupby(id_col, sort=False):
        grp  = grp.sort_values(time_col).reset_index(drop=True)
        lat  = grp[lat_col].values.astype(float)
        lon  = grp[lon_col].values.astype(float)
        turns = compute_turns(lat, lon)
        n_rev = int(np.sum((~np.isnan(turns)) & (turns > max_turn_deg)))
        if n_rev >= 1:
            with_reversals.append((vid, grp, n_rev))
        else:
            without.append((vid, grp, 0))

    # Sort candidates by reversal count (most interesting first)
    with_reversals.sort(key=lambda x: -x[2])
    rng.shuffle(without)
    pool = with_reversals + without
    return pool[:n_tracks]


# ── Plotting ───────────────────────────────────────────────────────────────────

_SEG_COLORS = [
    "tab:blue", "tab:orange", "tab:green", "tab:red",
    "tab:purple", "tab:brown", "tab:pink", "tab:gray",
    "tab:cyan", "tab:olive",
]


def plot_track_pair(ax_orig, ax_split, vid, grp,
                    lat_col, lon_col, max_turn_deg, n_rev):
    lat   = grp[lat_col].values.astype(float)
    lon   = grp[lon_col].values.astype(float)
    turns = compute_turns(lat, lon)
    seg   = assign_segments(turns, max_turn_deg)
    n_seg = int(seg.max()) + 1

    # ── Left: original track + reversal markers ────────────────────────────────
    ax_orig.plot(lon, lat, color="steelblue", linewidth=0.9, alpha=0.85, zorder=2)
    ax_orig.plot(lon[0],  lat[0],  "^", color="limegreen", ms=7, zorder=5,
                 label="start")
    ax_orig.plot(lon[-1], lat[-1], "s", color="tomato",    ms=7, zorder=5,
                 label="end")

    rev_idx = np.where((~np.isnan(turns)) & (turns > max_turn_deg))[0]
    if len(rev_idx):
        ax_orig.scatter(lon[rev_idx], lat[rev_idx],
                        marker="*", s=100, color="red", zorder=6,
                        label=f"reversal ({len(rev_idx)})")

    ax_orig.set_title(f"{vid}\n{len(lat)} pings  •  {n_rev} reversal(s) detected",
                      fontsize=8)
    ax_orig.set_aspect("auto")
    ax_orig.tick_params(labelsize=6)
    ax_orig.set_xlabel("lon", fontsize=7)
    ax_orig.set_ylabel("lat", fontsize=7)
    ax_orig.legend(fontsize=7, loc="best", markerscale=0.9)

    # ── Right: same track, coloured by segment ─────────────────────────────────
    for s in range(n_seg):
        mask = seg == s
        c    = _SEG_COLORS[s % len(_SEG_COLORS)]
        ax_split.plot(lon[mask], lat[mask], color=c, linewidth=1.1,
                      alpha=0.9, label=f"seg {s}  ({mask.sum()} pings)")
        # Mark segment start point
        ax_split.plot(lon[mask][0], lat[mask][0], "o", color=c, ms=5, zorder=5)

    ax_split.set_title(f"Split → {n_seg} segment(s)", fontsize=8)
    ax_split.set_aspect("auto")
    ax_split.tick_params(labelsize=6)
    ax_split.set_xlabel("lon", fontsize=7)
    ax_split.set_ylabel("lat", fontsize=7)
    # Only show up to 8 legend entries to avoid legend overflow
    handles, labels = ax_split.get_legend_handles_labels()
    if len(handles) > 8:
        handles, labels = handles[:8], labels[:8]
        labels[-1] = f"… ({n_seg} segs total)"
    ax_split.legend(handles, labels, fontsize=7, loc="best", markerscale=0.9)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Preview angle-based track splitting without modifying the CSV",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--csv",          required=True)
    parser.add_argument("--out_dir",      default="runs/")
    parser.add_argument("--id_col",       default="MMSI")
    parser.add_argument("--time_col",     default="BaseDateTime")
    parser.add_argument("--lat_col",      default="LAT")
    parser.add_argument("--lon_col",      default="LON")
    parser.add_argument("--max_turn_deg", type=float, default=150.0,
                        help="Heading change (°) that would trigger a new segment")
    parser.add_argument("--n_tracks",     type=int,   default=5,
                        help="Number of tracks to display")
    parser.add_argument("--seed",         type=int,   default=42)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading {args.csv} ...")
    df = pd.read_csv(
        args.csv,
        usecols=[args.id_col, args.time_col, args.lat_col, args.lon_col],
        low_memory=False)
    df[args.time_col] = pd.to_datetime(df[args.time_col])
    print(f"  {len(df):,} rows, {df[args.id_col].nunique():,} vessels")

    print(f"Selecting {args.n_tracks} tracks "
          f"(prefer those with reversals > {args.max_turn_deg}°) ...")
    selected = select_tracks(df, args.id_col, args.time_col,
                             args.lat_col, args.lon_col,
                             args.max_turn_deg, args.n_tracks, args.seed)

    n = len(selected)
    if n == 0:
        print("No tracks found — nothing to plot.")
        return

    fig, axes = plt.subplots(n, 2, figsize=(13, 4.5 * n),
                             squeeze=False)

    for row, (vid, grp, n_rev) in enumerate(selected):
        plot_track_pair(axes[row][0], axes[row][1],
                        vid, grp, args.lat_col, args.lon_col,
                        args.max_turn_deg, n_rev)

    fig.suptitle(
        f"Angle-Split Preview  (threshold = {args.max_turn_deg}°)\n"
        f"Left: original track  (red ★ = detected reversal)  |  "
        f"Right: result of splitting at each reversal",
        fontsize=12, fontweight="bold")
    fig.subplots_adjust(top=0.93, hspace=0.5, wspace=0.3)

    out_path = os.path.join(args.out_dir, "viz_angle_split_preview.png")
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
