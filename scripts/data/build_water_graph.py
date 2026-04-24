#!/usr/bin/env python3
"""
build_water_graph.py — v12 water-cell graph from the fine 0.005° SDF.

Reads `data/processed/land_sdf_005deg.npz`, produces a dense 8-neighbor
bitmask over the same (H, W) grid. Any cell whose SDF <= -τ_km is water.
A neighbor in one of the 8 Moore directions is a valid transition iff:

  - the neighbor cell is also water;
  - if diagonal, both axis-aligned corner cells are water too — this
    prevents the segment from clipping a land corner.

The segment from one water cell to a water neighbor is ~0.5 km long at
this resolution, so point-water checks on both endpoints + the diagonal
corner rule are sufficient; no extra interior sampling is required.

Output NPZ (`data/processed/water_graph_005deg.npz`):

    water_mask     (H, W) bool       τ-thresholded
    neighbor_bits  (H, W) uint8      bit i = valid transition in direction i,
                                     order: N, NE, E, SE, S, SW, W, NW
                                     (N = row index -1, i.e. toward maxy)
    bbox           (4,)  float64
    dlon, dlat     ()    float64     degrees
    midlat_cos     ()    float64
    tau_km         ()    float64

Run once (~1 minute), load at train/eval time.
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np


# Directions: index i ↔ (dy, dx). Row 0 = maxy, so dy=-1 means "north".
#   0: N    (-1,  0)
#   1: NE   (-1, +1)
#   2: E    ( 0, +1)
#   3: SE   (+1, +1)
#   4: S    (+1,  0)
#   5: SW   (+1, -1)
#   6: W    ( 0, -1)
#   7: NW   (-1, -1)
DIRECTIONS = [
    (-1,  0),
    (-1,  1),
    ( 0,  1),
    ( 1,  1),
    ( 1,  0),
    ( 1, -1),
    ( 0, -1),
    (-1, -1),
]


def shift_bool(mask: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Return out such that out[y, x] = mask[y+dy, x+dx] if in bounds, else False.

    No wraparound — cells outside the grid are treated as land (False).
    """
    H, W = mask.shape
    out = np.zeros_like(mask)
    y_src0 = max(0, dy); y_src1 = min(H, H + dy)
    x_src0 = max(0, dx); x_src1 = min(W, W + dx)
    y_dst0 = max(0, -dy); y_dst1 = y_dst0 + (y_src1 - y_src0)
    x_dst0 = max(0, -dx); x_dst1 = x_dst0 + (x_src1 - x_src0)
    out[y_dst0:y_dst1, x_dst0:x_dst1] = mask[y_src0:y_src1, x_src0:x_src1]
    return out


def build_neighbor_bits(water: np.ndarray) -> np.ndarray:
    """Compute the (H, W) uint8 neighbor bitmask. O(H * W * 8)."""
    H, W = water.shape
    bits = np.zeros((H, W), dtype=np.uint8)

    for i, (dy, dx) in enumerate(DIRECTIONS):
        t0 = time.time()
        neighbor = shift_bool(water, dy, dx)
        valid = water & neighbor
        if dy != 0 and dx != 0:
            # Diagonal: require both corner-adjacent cells also water so the
            # segment can't clip through a land corner.
            corner_row = shift_bool(water, dy, 0)
            corner_col = shift_bool(water, 0, dx)
            valid &= corner_row & corner_col
        bits |= valid.astype(np.uint8) << i
        print(f"  dir {i} (dy={dy:+d}, dx={dx:+d}): "
              f"{int(valid.sum()) / 1e6:.2f}M valid edges  "
              f"({time.time() - t0:.1f}s)")

    return bits


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--sdf",   default="data/processed/land_sdf_005deg.npz")
    ap.add_argument("--out",   default="data/processed/water_graph_005deg.npz")
    ap.add_argument("--tau_km", type=float, default=0.5,
                    help="Water threshold: sdf <= -tau_km counts as water.")
    args = ap.parse_args()

    print(f"Loading SDF: {args.sdf}")
    z = np.load(args.sdf)
    sdf = z["sdf_km"]
    bbox = z["bbox"]
    dlon = float(z["dlon"])
    dlat = float(z["dlat"])
    midlat_cos = float(z["midlat_cos"])
    H, W = sdf.shape
    print(f"  grid {H} × {W}  ({sdf.size / 1e6:.1f} M cells)")
    print(f"  bbox {bbox.tolist()}")
    print(f"  cell size: {dlat * 111.0:.3f} km (lat) × "
          f"{dlon * 111.0 * midlat_cos:.3f} km (lon)")

    water = sdf <= -args.tau_km
    print(f"  water cells @ τ={args.tau_km} km: "
          f"{int(water.sum()) / 1e6:.2f} M "
          f"({100 * water.mean():.1f}%)")

    print("Building 8-neighbor bitmask…")
    t0 = time.time()
    bits = build_neighbor_bits(water)
    print(f"Done in {time.time() - t0:.1f}s")
    print(f"  cells with ≥1 neighbor: "
          f"{int((bits > 0).sum()) / 1e6:.2f} M "
          f"({100 * (bits > 0).mean():.1f}%)")

    # Degree histogram (popcount on uint8)
    popcount = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
    deg = popcount[bits]
    unique, counts = np.unique(deg, return_counts=True)
    print("  degree distribution:")
    for k, c in zip(unique, counts):
        print(f"    deg={int(k)}: {int(c) / 1e6:.2f} M")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    print(f"Saving {args.out}…")
    np.savez_compressed(
        args.out,
        water_mask=water,
        neighbor_bits=bits,
        bbox=bbox,
        dlon=np.float64(dlon),
        dlat=np.float64(dlat),
        midlat_cos=np.float64(midlat_cos),
        tau_km=np.float64(args.tau_km),
        directions=np.array(DIRECTIONS, dtype=np.int64),
    )
    size_mb = os.path.getsize(args.out) / 1e6
    print(f"Saved ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
