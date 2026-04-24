#!/usr/bin/env python3
"""
quantize_trajgen.py — v12 data prep + Exp 1 (discretization floor).

Reads `data/processed/trajgen_128.npz` + `data/processed/water_graph_005deg.npz`
and produces:

1. A quantized cell dataset saved to `data/processed/trajgen_128_cells_005deg.npz`:
     cell_ix        (N, 128) int32    column (longitude) grid index
     cell_iy        (N, 128) int32    row    (latitude)  grid index  (0 = maxy)
     offset         (N, 128, 2)       sub-cell offset, in cell-edge units ∈ [-0.5, +0.5]
                                       layout: [Δix_frac, Δiy_frac]
     snapped        (N, 128) bool     True iff the point was snapped off land
     trajectories   (N, 128, 2)       passthrough (lon, lat), for convenience
     vessel_types   (N,)              passthrough
     track_ids      (N,)              passthrough
     bbox, dlon, dlat, grid_shape     copied from the water graph

2. A discretization-floor diagnostic on the v10/v9 val split (seed=42, val_frac=0.15):
     - ADE (centroid-only)    — lower bound w/ no offset head
     - ADE (centroid + offset) — lower bound w/ a perfect ±0.5-cell-edge offset head
     - Per length bucket (short / medium / long)
     - Also reports cell-run-length stats (how often consecutive 128-points live in
       the same cell → informs whether to re-resample for v12 sequence length)

Quantization rule: a trajectory point at (lon, lat) maps to cell
  ix = floor((lon - minx) / dlon)
  iy = floor((maxy - lat) / dlat)              # row 0 = maxy
If the cell is land (water_mask is False), snap to the nearest water cell
inside a 20-cell window (~10 km at this resolution) by L1 cell distance —
equivalent to LandMask.project_to_water but cheaper since we already have
the water mask in hand.

Offset is in cell-edge units, NOT degrees:
  offset_ix = (lon - cell_lon_center) / dlon        ∈ [-0.5, +0.5]
  offset_iy = (lat - cell_lat_center) / dlat        ∈ [-0.5, +0.5]  (sign: +y = north)
Reconstruction:
  cell_lon_center = minx + (ix + 0.5) * dlon
  cell_lat_center = maxy - (iy + 0.5) * dlat
  lon_reconstructed = cell_lon_center + offset_ix * dlon
  lat_reconstructed = cell_lat_center + offset_iy * dlat
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from src.data_gen import train_val_split_gen  # noqa: E402
from src.metrics_gen import evaluate_generation  # noqa: E402


# ─── Quantization ────────────────────────────────────────────────────────────

def quantize_points(lon: np.ndarray, lat: np.ndarray, bbox, dlon, dlat,
                    water_mask: np.ndarray, snap_radius_cells: int = 20):
    """Vectorized quantization of (lon, lat) arrays to (ix, iy) + offset.

    Returns:
        ix, iy       (*shape,)  int32 — grid indices after land-snap
        off_x, off_y (*shape,)  float32 — sub-cell offset in cell-edge units
        snapped      (*shape,)  bool   — True iff the point was snapped off land
    """
    minx, miny, maxx, maxy = bbox
    H, W = water_mask.shape
    orig_shape = lon.shape

    lon_f = lon.ravel().astype(np.float64)
    lat_f = lat.ravel().astype(np.float64)

    col_f = (lon_f - minx) / dlon
    row_f = (maxy - lat_f) / dlat

    ix = np.clip(np.floor(col_f).astype(np.int64), 0, W - 1)
    iy = np.clip(np.floor(row_f).astype(np.int64), 0, H - 1)

    on_water = water_mask[iy, ix]
    snapped = np.zeros_like(on_water, dtype=bool)
    land_idx = np.nonzero(~on_water)[0]

    for k in land_idx:
        cy, cx = int(iy[k]), int(ix[k])
        rlo = max(0, cy - snap_radius_cells); rhi = min(H, cy + snap_radius_cells + 1)
        clo = max(0, cx - snap_radius_cells); chi = min(W, cx + snap_radius_cells + 1)
        win = water_mask[rlo:rhi, clo:chi]
        if not win.any():
            continue
        ry, rx = np.nonzero(win)
        d2 = (ry + rlo - cy) ** 2 + (rx + clo - cx) ** 2
        kk = int(np.argmin(d2))
        iy[k] = ry[kk] + rlo
        ix[k] = rx[kk] + clo
        snapped[k] = True

    # Offsets relative to the (possibly snapped) cell centers
    cell_lon_center = minx + (ix + 0.5) * dlon
    cell_lat_center = maxy - (iy + 0.5) * dlat
    off_x = ((lon_f - cell_lon_center) / dlon).astype(np.float32)
    off_y = ((lat_f - cell_lat_center) / dlat).astype(np.float32)

    # For snapped points, the original offset may be out of range; clamp to [-0.5, 0.5]
    off_x = np.clip(off_x, -0.5, 0.5)
    off_y = np.clip(off_y, -0.5, 0.5)

    return (ix.reshape(orig_shape).astype(np.int32),
            iy.reshape(orig_shape).astype(np.int32),
            off_x.reshape(orig_shape),
            off_y.reshape(orig_shape),
            snapped.reshape(orig_shape))


def reconstruct_lonlat(ix, iy, off_x, off_y, bbox, dlon, dlat):
    """Invert quantize_points: (ix, iy, offset) → (lon, lat)."""
    minx, _miny, _maxx, maxy = bbox
    cell_lon_center = minx + (ix + 0.5) * dlon
    cell_lat_center = maxy - (iy + 0.5) * dlat
    lon = cell_lon_center + off_x * dlon
    lat = cell_lat_center + off_y * dlat
    return lon.astype(np.float32), lat.astype(np.float32)


# ─── Exp 1: discretization floor ─────────────────────────────────────────────

def run_discretization_floor(
    val_traj: np.ndarray,              # (M, 128, 2)
    water_mask: np.ndarray,
    bbox, dlon, dlat,
    n_eval: int = 500,
    seed: int = 42,
) -> dict:
    """Compute the discretization-floor ADE on n_eval val trajectories."""
    rng = np.random.default_rng(seed)
    M = len(val_traj)
    if M > n_eval:
        idx = rng.choice(M, size=n_eval, replace=False)
        idx.sort()
        sub = val_traj[idx]
    else:
        sub = val_traj
    print(f"\n── Exp 1: discretization floor on {len(sub)} val trajectories ──")

    lon = sub[:, :, 0]
    lat = sub[:, :, 1]

    t0 = time.time()
    ix, iy, off_x, off_y, snapped = quantize_points(
        lon, lat, bbox, dlon, dlat, water_mask, snap_radius_cells=20)
    print(f"Quantize: {time.time() - t0:.1f}s  "
          f"(snapped {100 * snapped.mean():.2f}% of points off land)")

    # Variant A: centroid only (off_x = off_y = 0)
    lon_c, lat_c = reconstruct_lonlat(
        ix, iy, np.zeros_like(off_x), np.zeros_like(off_y),
        bbox, dlon, dlat)
    traj_c = np.stack([lon_c, lat_c], axis=-1)

    # Variant B: centroid + perfect clamped offset
    lon_b, lat_b = reconstruct_lonlat(ix, iy, off_x, off_y, bbox, dlon, dlat)
    traj_b = np.stack([lon_b, lat_b], axis=-1)

    print("\n─ Variant A — centroid only (no offset head): ─")
    res_c = evaluate_generation(traj_c, sub, compute_frechet=False)
    _print_bucket(res_c)

    print("\n─ Variant B — centroid + clamped offset (±0.5 cell-edge): ─")
    res_b = evaluate_generation(traj_b, sub, compute_frechet=False)
    _print_bucket(res_b)

    # Cell-run-length stats — how often is cell_t+1 == cell_t ?
    same_cell = (ix[:, 1:] == ix[:, :-1]) & (iy[:, 1:] == iy[:, :-1])
    run_len_est = 128 / (1.0 + (~same_cell).sum(axis=1))   # unique cells per traj
    print(f"\nCell-run stats:")
    print(f"  mean same-cell rate        : {100 * same_cell.mean():.1f}%")
    print(f"  median unique cells / 128  : {int(np.median(128 - same_cell.sum(axis=1)))}")
    print(f"  P10 unique cells / 128     : {int(np.percentile(128 - same_cell.sum(axis=1), 10))}")
    print(f"  P90 unique cells / 128     : {int(np.percentile(128 - same_cell.sum(axis=1), 90))}")
    print(f"  mean unique cells / 128    : {(128 - same_cell.sum(axis=1)).mean():.1f}")

    return {
        "centroid_only": res_c,
        "centroid_plus_offset": res_b,
        "same_cell_rate": float(same_cell.mean()),
        "mean_unique_cells": float((128 - same_cell.sum(axis=1)).mean()),
    }


def _print_bucket(res: dict):
    keys = [("ade_m", "ADE"),
            ("fde_m", "FDE"),
            ("ade_short_m", "ADE short"),
            ("ade_medium_m", "ADE medium"),
            ("ade_long_m", "ADE long"),
            ("path_length_ratio", "path ratio"),
            ("normalized_ade", "norm ADE")]
    for k, label in keys:
        v = res.get(k)
        if v is None:
            continue
        if "ratio" in k or "normalized" in k:
            print(f"  {label:12s}: {v:.4f}")
        else:
            print(f"  {label:12s}: {v:8.1f} m")
    buckets = [k for k in res if k.startswith("n_") and not k.startswith("n_trajectories")]
    for k in buckets:
        print(f"  {k:12s}: {res[k]}")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--data_npz", default="data/processed/trajgen_128.npz")
    ap.add_argument("--water_graph",
                    default="data/processed/water_graph_005deg.npz")
    ap.add_argument("--out",
                    default="data/processed/trajgen_128_cells_005deg.npz")
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--seed",     type=int,   default=42)
    ap.add_argument("--n_eval",   type=int,   default=500)
    ap.add_argument("--skip_quantize_all", action="store_true",
                    help="Only run Exp 1 diagnostic, don't write the full dataset.")
    ap.add_argument("--exp1_only", action="store_true",
                    help="Alias for --skip_quantize_all.")
    args = ap.parse_args()

    if args.exp1_only:
        args.skip_quantize_all = True

    print(f"Loading trajectories : {args.data_npz}")
    z = np.load(args.data_npz)
    traj = z["trajectories"]
    vt   = z["vessel_types"]
    tids = z["track_ids"]
    print(f"  N={len(traj)}, T={traj.shape[1]}")

    print(f"Loading water graph  : {args.water_graph}")
    g = np.load(args.water_graph)
    water_mask = g["water_mask"]
    bbox = tuple(float(x) for x in g["bbox"])
    dlon = float(g["dlon"])
    dlat = float(g["dlat"])
    H, W = water_mask.shape
    print(f"  grid {H} × {W}  ({100 * water_mask.mean():.1f}% water)")

    # Val split (matches v9/v10/v11)
    (train_traj, train_vt, train_ids,
     val_traj,   val_vt,   val_ids) = train_val_split_gen(
        traj, vt, tids, args.val_frac, args.seed)
    print(f"  split: train={len(train_traj)}  val={len(val_traj)}")

    # ── Exp 1 diagnostic ──
    run_discretization_floor(
        val_traj, water_mask, bbox, dlon, dlat,
        n_eval=args.n_eval, seed=args.seed)

    if args.skip_quantize_all:
        print("\nSkipping full-dataset quantization (--skip_quantize_all).")
        return

    # ── Full-dataset quantization ──
    print(f"\n── Quantizing full dataset → {args.out} ──")
    N, T, _ = traj.shape
    ix_all = np.empty((N, T), dtype=np.int32)
    iy_all = np.empty((N, T), dtype=np.int32)
    off_all = np.empty((N, T, 2), dtype=np.float32)
    snapped_all = np.empty((N, T), dtype=bool)

    CHUNK = 4096
    for start in tqdm(range(0, N, CHUNK), desc="quantize", ncols=80):
        stop = min(N, start + CHUNK)
        ix, iy, off_x, off_y, snapped = quantize_points(
            traj[start:stop, :, 0], traj[start:stop, :, 1],
            bbox, dlon, dlat, water_mask, snap_radius_cells=20)
        ix_all[start:stop] = ix
        iy_all[start:stop] = iy
        off_all[start:stop, :, 0] = off_x
        off_all[start:stop, :, 1] = off_y
        snapped_all[start:stop] = snapped

    print(f"  land-snap rate across all N×T points: "
          f"{100 * snapped_all.mean():.3f}%")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(
        args.out,
        cell_ix=ix_all,
        cell_iy=iy_all,
        offset=off_all,
        snapped=snapped_all,
        trajectories=traj,
        vessel_types=vt,
        track_ids=tids,
        bbox=np.array(bbox, dtype=np.float64),
        dlon=np.float64(dlon),
        dlat=np.float64(dlat),
        grid_shape=np.array([H, W], dtype=np.int64),
        val_frac=np.float64(args.val_frac),
        split_seed=np.int64(args.seed),
    )
    print(f"Saved {args.out} ({os.path.getsize(args.out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
