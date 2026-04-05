#!/usr/bin/env python3
"""
audit_tracks.py — Data quality audit: discover anomaly types in processed AIS tracks.

Loads a random sample of tracks, computes 8 per-track metrics, flags outliers in each
dimension, plots representative examples of each anomaly category, and writes a summary.

This is a read-only discovery script — it does NOT modify any data.

Usage:
    python3 scripts/audit_tracks.py \
        --csv data/processed/AIS_2024_sample.csv \
        --n_sample 5000 \
        --seed 42 \
        --out_dir outputs/audit
"""

import argparse
import csv
import math
import os
import random
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.data import load_tracks


# ── Geometry helpers (from preview_angle_split.py) ────────────────────────────

def _bearing(lat1, lon1, lat2, lon2):
    dlon = np.radians(lon2 - lon1)
    r1   = np.radians(lat1)
    r2   = np.radians(lat2)
    x    = np.sin(dlon) * np.cos(r2)
    y    = np.cos(r1) * np.sin(r2) - np.sin(r1) * np.cos(r2) * np.cos(dlon)
    return (np.degrees(np.arctan2(x, y)) + 360) % 360


def compute_turns(lat, lon):
    n = len(lat)
    if n < 3:
        return np.full(n, np.nan)
    brg_in  = _bearing(lat[:-1],  lon[:-1],  lat[1:],  lon[1:])
    brg_out = _bearing(lat[1:-1], lon[1:-1], lat[2:],  lon[2:])
    delta   = (brg_out - brg_in[:-1] + 180) % 360 - 180
    turn    = np.abs(delta)
    result  = np.full(n, np.nan)
    result[1:n - 1] = turn
    return result


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(math.radians, [float(lat1), float(lon1),
                                                  float(lat2), float(lon2)])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ── Per-track metrics ─────────────────────────────────────────────────────────

def track_metrics(vid, pts):
    """
    pts: (T, 3) float32 — [lon, lat, dt_seconds]
    Returns a dict of scalar metrics.
    """
    lon = pts[:, 0].astype(float)
    lat = pts[:, 1].astype(float)
    dt  = pts[:, 2].astype(float)
    T   = len(pts)

    # Step displacements and speeds
    dlons = np.diff(lon)
    dlats = np.diff(lat)
    step_km = np.array([
        haversine_km(lat[i], lon[i], lat[i+1], lon[i+1])
        for i in range(T - 1)
    ])
    valid_dt = np.where(dt[1:] > 0, dt[1:], np.nan)
    speed_kmh = step_km / (valid_dt / 3600.0)

    # 1. Spatial extent: haversine bbox diagonal
    bbox_diag_km = haversine_km(lat.min(), lon.min(), lat.max(), lon.max())

    # 2. Fraction of steps with tiny displacement (< 0.05 km)
    frac_tiny_disp = float(np.mean(step_km < 0.05))

    # 3. Turn angles
    turns = compute_turns(lat, lon)
    valid_turns = turns[~np.isnan(turns)]
    frac_sharp_turns = float(np.mean(valid_turns > 120)) if len(valid_turns) > 0 else 0.0
    max_turn_deg     = float(np.max(valid_turns))        if len(valid_turns) > 0 else 0.0

    # 4. Path length / bbox diagonal  (bbox_fill)
    total_path_km = float(step_km.sum())
    bbox_fill = total_path_km / max(bbox_diag_km, 0.01)

    # 5. Speed coefficient of variation
    valid_speed = speed_kmh[~np.isnan(speed_kmh)]
    if len(valid_speed) > 1 and valid_speed.mean() > 0:
        speed_cv = float(valid_speed.std() / valid_speed.mean())
    else:
        speed_cv = 0.0

    # 6. Ping rate coefficient of variation (inter-ping dt)
    valid_dt_arr = dt[1:][dt[1:] > 0]
    if len(valid_dt_arr) > 1 and valid_dt_arr.mean() > 0:
        ping_rate_cv = float(valid_dt_arr.std() / valid_dt_arr.mean())
    else:
        ping_rate_cv = 0.0

    # 7. 95th percentile speed
    p95_speed_kmh = float(np.nanpercentile(speed_kmh, 95)) if len(valid_speed) > 0 else 0.0

    return {
        "vessel_id":        vid,
        "n_pings":          T,
        "total_path_km":    round(total_path_km, 3),
        "spatial_extent_km": round(bbox_diag_km, 3),
        "frac_tiny_disp":   round(frac_tiny_disp, 4),
        "frac_sharp_turns": round(frac_sharp_turns, 4),
        "max_turn_deg":     round(max_turn_deg, 2),
        "bbox_fill":        round(bbox_fill, 3),
        "speed_cv":         round(speed_cv, 4),
        "ping_rate_cv":     round(ping_rate_cv, 4),
        "p95_speed_kmh":    round(p95_speed_kmh, 3),
    }


# ── Anomaly flagging ──────────────────────────────────────────────────────────

ANOMALY_DEFS = [
    # (flag_name, metric, side, percentile_or_threshold, use_percentile)
    # side "high" = top outliers; "low" = bottom outliers
    ("flag_stationary",    "spatial_extent_km",  "low",  2,   True),
    ("flag_tiny_disp",     "frac_tiny_disp",     "high", 2,   True),
    ("flag_looping",       "bbox_fill",          "high", 2,   True),
    ("flag_sharp_turns",   "frac_sharp_turns",   "high", 2,   True),
    ("flag_reversal",      "max_turn_deg",       "high", 150, False),
    ("flag_erratic_speed", "speed_cv",           "high", 2,   True),
    ("flag_ping_cluster",  "ping_rate_cv",       "high", 2,   True),
    ("flag_extreme_speed", "p95_speed_kmh",      "high", 2,   True),
]

ANOMALY_LABELS = {
    "flag_stationary":    "Near-stationary (low spatial extent)",
    "flag_tiny_disp":     "Drifting anchored (high frac tiny displacement)",
    "flag_looping":       "Looping / survey route (high bbox fill)",
    "flag_sharp_turns":   "Zigzag / frequent sharp turns",
    "flag_reversal":      "Heading reversal (turn > 150°)",
    "flag_erratic_speed": "Erratic speed (high CV)",
    "flag_ping_cluster":  "Ping clustering (bursty dt)",
    "flag_extreme_speed": "Occasional extreme speed (high p95)",
}

VIZ_METRIC = {
    "flag_stationary":    "spatial_extent_km",
    "flag_tiny_disp":     "frac_tiny_disp",
    "flag_looping":       "bbox_fill",
    "flag_sharp_turns":   "frac_sharp_turns",
    "flag_reversal":      "max_turn_deg",
    "flag_erratic_speed": "speed_cv",
    "flag_ping_cluster":  "ping_rate_cv",
    "flag_extreme_speed": "p95_speed_kmh",
}


def flag_anomalies(metrics_list):
    """Add boolean flag columns to each metrics dict. Returns updated list."""
    keys = [m for _, m, _, _, _ in ANOMALY_DEFS]
    arrays = {k: np.array([r[k] for r in metrics_list]) for k in keys}

    for flag_name, metric, side, threshold, use_pct in ANOMALY_DEFS:
        vals = arrays[metric]
        if use_pct:
            if side == "high":
                cutoff = np.percentile(vals, 100 - threshold)
                mask   = vals >= cutoff
            else:
                cutoff = np.percentile(vals, threshold)
                mask   = vals <= cutoff
        else:
            mask = vals >= threshold  # absolute threshold (e.g. max_turn_deg > 150)

        for i, row in enumerate(metrics_list):
            row[flag_name] = bool(mask[i])

    return metrics_list


# ── Plotting ──────────────────────────────────────────────────────────────────

def _draw_panel(ax, pts, vid, metric_val, metric_name):
    lon = pts[:, 0]
    lat = pts[:, 1]
    n   = len(lon)
    cmap = plt.cm.Blues

    for j in range(n - 1):
        t = j / max(n - 2, 1)
        ax.plot(lon[j:j+2], lat[j:j+2],
                color=cmap(0.3 + 0.7 * t), linewidth=0.9, alpha=0.6 + 0.4 * t)

    ax.scatter(lon[0],  lat[0],  marker="s", color="#2e7d32", s=30, zorder=5)
    ax.scatter(lon[-1], lat[-1], marker="o", color="#c62828", s=30, zorder=5)

    ax.set_title(str(vid)[-20:], fontsize=6, pad=2)
    ax.text(0.02, 0.02,
            f"{n} pings\n{metric_name}={metric_val:.3g}",
            transform=ax.transAxes, fontsize=5,
            color="white", bbox=dict(fc="#000000bb", boxstyle="round,pad=0.2"))
    ax.tick_params(labelsize=4)
    ax.set_xlabel("Lon", fontsize=5)
    ax.set_ylabel("Lat", fontsize=5)


def plot_anomaly_grid(flag_name, flagged_rows, tracks, out_path, n_panels=16):
    """Plot up to n_panels example tracks for one anomaly category."""
    if not flagged_rows:
        return

    # Sort by metric value descending (worst first)
    metric = VIZ_METRIC[flag_name]
    flagged_rows = sorted(flagged_rows, key=lambda r: -r[metric])
    chosen = flagged_rows[:n_panels]

    ncols = 4
    nrows = math.ceil(len(chosen) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3.5))
    axes_flat = np.array(axes).reshape(-1)

    for i, row in enumerate(chosen):
        vid = row["vessel_id"]
        pts = tracks.get(vid)
        if pts is None:
            axes_flat[i].set_visible(False)
            continue
        _draw_panel(axes_flat[i], pts, vid, row[metric], metric)

    for ax in axes_flat[len(chosen):]:
        ax.set_visible(False)

    fig.suptitle(
        f"{ANOMALY_LABELS[flag_name]}\n"
        f"({len(flagged_rows)} flagged vessels — showing {len(chosen)} worst by {metric})",
        fontsize=10, y=1.01,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_distributions(metrics_list, out_path):
    metric_names = [
        "spatial_extent_km", "frac_tiny_disp", "bbox_fill", "frac_sharp_turns",
        "max_turn_deg", "speed_cv", "ping_rate_cv", "p95_speed_kmh",
    ]
    titles = [
        "Spatial extent (km) — bbox diagonal",
        "Frac tiny displacement (< 0.05 km)",
        "Bbox fill — path / bbox diagonal",
        "Frac sharp turns (> 120°)",
        "Max turn angle (°)",
        "Speed CV — std/mean",
        "Ping rate CV — std(dt)/mean(dt)",
        "P95 speed (km/h)",
    ]
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    axes_flat = axes.reshape(-1)

    for ax, name, title in zip(axes_flat, metric_names, titles):
        vals = np.array([r[name] for r in metrics_list])
        # Clip top 1% for readability
        cap = np.percentile(vals, 99)
        vals_clipped = np.clip(vals, None, cap)
        ax.hist(vals_clipped, bins=80, color="#1565c0", alpha=0.75, edgecolor="none")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel(name, fontsize=7)
        ax.set_ylabel("count", fontsize=7)
        ax.tick_params(labelsize=6)
        # Mark p2 / p98
        p2  = np.percentile(vals, 2)
        p98 = np.percentile(vals, 98)
        ax.axvline(p98, color="#c62828", linewidth=1.2, linestyle="--",
                   label=f"p98={p98:.3g}")
        ax.axvline(p2,  color="#e65100", linewidth=1.2, linestyle=":",
                   label=f"p2={p2:.3g}")
        ax.legend(fontsize=6)

    plt.suptitle("Per-track metric distributions (dashed=p98, dotted=p2)", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--csv",       required=True,           help="Processed AIS CSV")
    p.add_argument("--n_sample",  type=int, default=5000,  help="Random vessels to audit")
    p.add_argument("--seed",      type=int, default=42)
    p.add_argument("--out_dir",   default="outputs/audit")
    p.add_argument("--n_panels",  type=int, default=16,    help="Example panels per anomaly plot")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = random.Random(args.seed)

    # ── Load tracks ────────────────────────────────────────────────────────────
    print(f"Loading tracks from {args.csv} ...")
    tracks, _ = load_tracks(args.csv)

    all_ids = list(tracks.keys())
    if args.n_sample < len(all_ids):
        rng.shuffle(all_ids)
        sample_ids = all_ids[:args.n_sample]
    else:
        sample_ids = all_ids
    print(f"  Sampled {len(sample_ids):,} / {len(all_ids):,} vessels\n")

    # ── Compute metrics ────────────────────────────────────────────────────────
    print("Computing per-track metrics ...")
    metrics_list = []
    for i, vid in enumerate(sample_ids):
        if (i + 1) % 500 == 0:
            print(f"  {i+1:,}/{len(sample_ids):,}")
        metrics_list.append(track_metrics(vid, tracks[vid]))
    print(f"  Done — {len(metrics_list):,} tracks\n")

    # ── Flag anomalies ─────────────────────────────────────────────────────────
    print("Flagging outliers ...")
    metrics_list = flag_anomalies(metrics_list)

    # ── Summary ────────────────────────────────────────────────────────────────
    summary_lines = ["Anomaly summary", "=" * 60]
    for flag_name, _, _, _, _ in ANOMALY_DEFS:
        n_flagged = sum(1 for r in metrics_list if r[flag_name])
        pct = 100.0 * n_flagged / len(metrics_list)
        metric = VIZ_METRIC[flag_name]
        flagged_vals = [r[metric] for r in metrics_list if r[flag_name]]
        med_val = np.median(flagged_vals) if flagged_vals else 0.0
        line = (f"  {flag_name:<22}  {n_flagged:5d} / {len(metrics_list):5d}"
                f"  ({pct:5.1f}%)  median_{metric}={med_val:.3g}")
        summary_lines.append(line)
        print(line)

    summary_path = os.path.join(args.out_dir, "anomaly_summary.txt")
    with open(summary_path, "w") as f:
        f.write("\n".join(summary_lines) + "\n")
    print(f"\n  Saved: {summary_path}")

    # ── Save metrics CSV ───────────────────────────────────────────────────────
    csv_path = os.path.join(args.out_dir, "audit_metrics.csv")
    fieldnames = list(metrics_list[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics_list)
    print(f"  Saved: {csv_path}\n")

    # ── Distributions plot ────────────────────────────────────────────────────
    print("Plotting metric distributions ...")
    plot_distributions(metrics_list, os.path.join(args.out_dir, "viz_distributions.png"))

    # ── Per-anomaly example plots ──────────────────────────────────────────────
    print("\nPlotting anomaly examples ...")
    flag_to_filename = {
        "flag_stationary":    "viz_stationary.png",
        "flag_tiny_disp":     "viz_tiny_disp.png",
        "flag_looping":       "viz_looping.png",
        "flag_sharp_turns":   "viz_sharp_turns.png",
        "flag_reversal":      "viz_reversal.png",
        "flag_erratic_speed": "viz_erratic_speed.png",
        "flag_ping_cluster":  "viz_ping_cluster.png",
        "flag_extreme_speed": "viz_extreme_speed.png",
    }
    for flag_name, fname in flag_to_filename.items():
        flagged = [r for r in metrics_list if r[flag_name]]
        out_path = os.path.join(args.out_dir, fname)
        plot_anomaly_grid(flag_name, flagged, tracks, out_path, n_panels=args.n_panels)

    print("\nDone.")


if __name__ == "__main__":
    main()
