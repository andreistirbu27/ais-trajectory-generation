"""Visualise the precomputed land signed-distance field (SDF) raster.

Usage:
    python3 scripts/eval/plot_land_sdf.py \
        --land_sdf data/processed/land_sdf_050deg.npz \
        --out figures/land_sdf.png

Sign convention: positive = on land, negative = on water (see src/land_mask.py).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--land_sdf", default="data/processed/land_sdf_050deg.npz")
    parser.add_argument("--out", default="figures/land_sdf.png")
    parser.add_argument("--vmax_km", type=float, default=100.0,
                        help="Colour scale clamp in km (default 100)")
    args = parser.parse_args()

    sdf_path = Path(args.land_sdf)
    if not sdf_path.exists():
        print(f"ERROR: SDF file not found: {sdf_path}", file=sys.stderr)
        sys.exit(1)

    data = np.load(sdf_path)
    sdf = data["sdf_km"].astype(float)
    bbox = data["bbox"]            # [lon_min, lat_min, lon_max, lat_max]
    lon_min, lat_min, lon_max, lat_max = bbox

    fig, ax = plt.subplots(figsize=(10, 6))
    clipped = np.clip(sdf, -args.vmax_km, args.vmax_km)
    im = ax.imshow(
        clipped,
        origin="upper",
        extent=[lon_min, lon_max, lat_min, lat_max],
        aspect="auto",
        cmap="RdBu_r",
        vmin=-args.vmax_km,
        vmax=args.vmax_km,
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.04)
    cbar.set_label("SDF (km)  |  blue = water, red = land")

    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.set_title(
        f"Land signed-distance field — 0.05° resolution ({sdf.shape[0]}×{sdf.shape[1]} cells)\n"
        "Positive = land, Negative = water  (US coverage bbox)"
    )
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.grid(alpha=0.2, linestyle=":")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
