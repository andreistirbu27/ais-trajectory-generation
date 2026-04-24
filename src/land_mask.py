"""
Land mask + signed distance field (SDF) utilities for trajectory models.

Two phases:

1. BUILD (one-shot, slow):
     python3 -m src.land_mask --shoreline data/gshhg/GSHHS_shp/i/GSHHS_i_L1.shp
   Rasterizes GSHHS L1 coastlines onto a regular lon/lat grid (default 0.05°)
   and computes a signed distance field in kilometres (positive = land,
   negative = water). Caches to data/processed/land_sdf_005deg.npz.
   Requires: rasterio, geopandas, shapely, scipy.

2. RUNTIME (loads cache, fast):
     mask = LandMask.load("data/processed/land_sdf_005deg.npz")
     sdf_km = mask.sample_km_np(lon, lat)              # numpy
     sdf_km = mask.sample_km_torch(lon_t, lat_t)       # torch, differentiable
     on_land = mask.on_land(lon, lat)
     lon_w, lat_w = mask.project_to_water(lon, lat)
     stats = mask.trajectory_stats(traj)               # (N, T, 2) → land metrics
   Runtime needs only numpy/scipy/torch.

SDF sign convention: POSITIVE inside land (penalty target), NEGATIVE in
water. The training-time soft penalty is mean(ReLU(sdf_km)²); this drives
waypoints toward water.
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

EARTH_RADIUS_KM = 6371.0

# Default bbox matches scripts/data/prepare_data.py (US coastal)
DEFAULT_BBOX = (-125.0, 10.0, -60.0, 55.0)   # (minx, miny, maxx, maxy)
DEFAULT_DLON = DEFAULT_DLAT = 0.05           # ~5 km near mid-latitudes


# ─── Build: rasterize + SDF (one-shot, needs rasterio/geopandas/shapely) ────

def build_sdf(shp_path: str, out_path: str,
              bbox: Tuple[float, float, float, float] = DEFAULT_BBOX,
              dlon_deg: float = DEFAULT_DLON,
              dlat_deg: float = DEFAULT_DLAT,
              lakes_path: Optional[str] = None) -> None:
    """Rasterize GSHHS shoreline → binary land mask → signed distance field (km).

    If `lakes_path` is provided (e.g. GSHHS L2), those polygons are carved
    back out of the land mask as water — crucial for the Great Lakes and
    large inland waterways that carry AIS traffic.

    Saves an NPZ with keys: sdf_km, bbox, dlon, dlat, shape, midlat_cos.
    """
    import geopandas as gpd
    from rasterio import features
    from rasterio.transform import from_origin
    from scipy.ndimage import distance_transform_edt

    minx, miny, maxx, maxy = bbox
    W = int(round((maxx - minx) / dlon_deg))
    H = int(round((maxy - miny) / dlat_deg))
    print(f"Grid: {W} lon × {H} lat  ({dlon_deg}° resolution)")

    print(f"Loading land (L1): {shp_path}")
    gdf = gpd.read_file(shp_path, bbox=bbox)
    print(f"  Polygons in bbox: {len(gdf):,}")

    # Raster affine transform: origin at top-left (max lat), rows go south.
    transform = from_origin(minx, maxy, dlon_deg, dlat_deg)
    land = features.rasterize(
        ((geom, 1) for geom in gdf.geometry),
        out_shape=(H, W),
        transform=transform,
        fill=0,
        all_touched=True,
        dtype=np.uint8,
    )   # (H, W) with 1 where land, 0 where water. Row 0 = maxy (north).
    print(f"  Land cells after L1: {int(land.sum()):,} / {H * W:,} "
          f"({100 * land.mean():.1f}%)")

    if lakes_path is not None:
        print(f"Loading lakes (L2): {lakes_path}")
        lakes_gdf = gpd.read_file(lakes_path, bbox=bbox)
        print(f"  Polygons in bbox: {len(lakes_gdf):,}")
        if len(lakes_gdf) > 0:
            lakes = features.rasterize(
                ((geom, 1) for geom in lakes_gdf.geometry),
                out_shape=(H, W),
                transform=transform,
                fill=0,
                all_touched=False,   # stricter for lakes so we don't over-flood
                dtype=np.uint8,
            )
            carved = int((land & lakes).sum())
            land = land & (1 - lakes)
            print(f"  Carved out {carved:,} lake cells; land now "
                  f"{int(land.sum()):,} ({100 * land.mean():.1f}%)")

    # EDT sampling: convert cell spacing to km at the bbox's midlatitude
    midlat = 0.5 * (miny + maxy)
    midlat_cos = float(np.cos(np.deg2rad(midlat)))
    dlat_km = dlat_deg * 111.0
    dlon_km = dlon_deg * 111.0 * midlat_cos

    # Positive distance from water cells to nearest land (0 inside land)
    dist_water_to_land = distance_transform_edt(
        land == 0, sampling=(dlat_km, dlon_km))
    # Positive distance from land cells to nearest water (0 on water)
    dist_land_to_water = distance_transform_edt(
        land == 1, sampling=(dlat_km, dlon_km))

    # Signed distance: +ve inside land, -ve in water
    sdf_km = (dist_land_to_water.astype(np.float32)
              - dist_water_to_land.astype(np.float32))

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.savez(out_path,
             sdf_km=sdf_km,
             bbox=np.array(bbox, dtype=np.float64),
             dlon=np.float64(dlon_deg),
             dlat=np.float64(dlat_deg),
             shape=np.array([H, W], dtype=np.int64),
             midlat_cos=np.float64(midlat_cos))
    print(f"Saved SDF to {out_path}  "
          f"({os.path.getsize(out_path) / 1e6:.1f} MB)")


# ─── Runtime: LandMask class ────────────────────────────────────────────────

@dataclass
class LandMask:
    sdf_km: np.ndarray                        # (H, W), row 0 = maxy
    bbox: Tuple[float, float, float, float]   # (minx, miny, maxx, maxy)
    dlon: float
    dlat: float
    midlat_cos: float

    @classmethod
    def load(cls, path: str) -> "LandMask":
        z = np.load(path)
        bbox = tuple(float(x) for x in z["bbox"])
        return cls(
            sdf_km=z["sdf_km"].astype(np.float32),
            bbox=bbox,
            dlon=float(z["dlon"]),
            dlat=float(z["dlat"]),
            midlat_cos=float(z["midlat_cos"]),
        )

    @property
    def shape(self) -> Tuple[int, int]:
        return self.sdf_km.shape

    # ── Numpy bilinear sampler ──────────────────────────────────────────

    def _to_grid(self, lon, lat):
        """Convert (lon, lat) arrays to continuous (row, col) indices."""
        minx, miny, maxx, maxy = self.bbox
        col = (np.asarray(lon) - minx) / self.dlon
        row = (maxy - np.asarray(lat)) / self.dlat   # row 0 = maxy
        return row, col

    def sample_km_np(self, lon, lat) -> np.ndarray:
        """Bilinear interpolation of SDF at (lon, lat) points. Positive = land.

        Points outside bbox return 0 (we don't train/eval there).
        """
        H, W = self.shape
        row, col = self._to_grid(lon, lat)
        # Clip on integer indices so r1/c1 can't spill past the grid edge
        # even when float32 rounding pushes `row`/`col` up to H-1/W-1.
        r0 = np.clip(np.floor(row).astype(np.int64), 0, H - 2)
        c0 = np.clip(np.floor(col).astype(np.int64), 0, W - 2)
        r1 = r0 + 1
        c1 = c0 + 1
        fr = np.clip(row - r0, 0.0, 1.0)
        fc = np.clip(col - c0, 0.0, 1.0)
        sdf = self.sdf_km
        v00 = sdf[r0, c0]
        v01 = sdf[r0, c1]
        v10 = sdf[r1, c0]
        v11 = sdf[r1, c1]
        return ((1 - fr) * ((1 - fc) * v00 + fc * v01)
                + fr * ((1 - fc) * v10 + fc * v11))

    def on_land(self, lon, lat) -> np.ndarray:
        return self.sample_km_np(lon, lat) > 0.0

    # ── Torch differentiable sampler (grid_sample) ──────────────────────

    def as_torch(self, device=None, dtype=None):
        """Cache the SDF tensor on a device. Call once, then use sample_km_torch."""
        import torch
        if dtype is None:
            dtype = torch.float32
        t = torch.from_numpy(self.sdf_km).to(device=device, dtype=dtype)
        self._sdf_flat = t.reshape(-1).contiguous()
        self._device = device
        return self

    def sample_km_torch(self, lon, lat):
        """Differentiable bilinear sample of SDF (km). Inputs: torch tensors
        of matching shape, in degrees. Returns same shape, positive = land.

        Manual bilinear with integer-indexed gather — avoids
        `F.grid_sample`, whose backward is not implemented on MPS. Gradient
        flows through the float interpolation weights (fr, fc).
        """
        if not hasattr(self, "_sdf_flat"):
            self.as_torch(device=lon.device, dtype=lon.dtype)
        H, W = self.shape
        minx, _miny, _maxx, maxy = self.bbox
        col = (lon - minx) / self.dlon
        row = (maxy - lat) / self.dlat
        # Clip on the integer indices so (r1, c1) stay in bounds even at the
        # grid edge; float32 epsilon isn't enough around magnitude ~1e3.
        r0 = row.detach().floor().long().clamp(0, H - 2)
        c0 = col.detach().floor().long().clamp(0, W - 2)
        r1 = r0 + 1
        c1 = c0 + 1
        fr = (row - r0.to(row.dtype)).clamp(0.0, 1.0)
        fc = (col - c0.to(col.dtype)).clamp(0.0, 1.0)
        flat = self._sdf_flat
        v00 = flat[r0 * W + c0]
        v01 = flat[r0 * W + c1]
        v10 = flat[r1 * W + c0]
        v11 = flat[r1 * W + c1]
        return ((1 - fr) * ((1 - fc) * v00 + fc * v01)
                + fr * ((1 - fc) * v10 + fc * v11))

    # ── Hard projection: on-land → nearest water cell ───────────────────

    def project_to_water(self, lon, lat,
                         search_radius_cells: int = 20) -> Tuple[np.ndarray, np.ndarray]:
        """For any (lon, lat) currently on land, shift to the nearest water
        cell within a square search window. Points already in water are
        returned unchanged.

        search_radius_cells=20 at 0.05° = 100 km window. Fast per-point.
        """
        lon = np.asarray(lon, dtype=np.float64)
        lat = np.asarray(lat, dtype=np.float64)
        out_lon = lon.copy()
        out_lat = lat.copy()
        sdf = self.sdf_km
        H, W = self.shape
        minx, miny, maxx, maxy = self.bbox

        land_flags = self.sample_km_np(lon, lat) > 0.0
        land_idx = np.nonzero(land_flags.ravel())[0]
        if land_idx.size == 0:
            return out_lon, out_lat

        flat_lon = lon.ravel()
        flat_lat = lat.ravel()
        for idx in land_idx:
            row0, col0 = self._to_grid(flat_lon[idx], flat_lat[idx])
            r = int(np.clip(np.round(row0), 0, H - 1))
            c = int(np.clip(np.round(col0), 0, W - 1))
            rlo = max(0, r - search_radius_cells)
            rhi = min(H, r + search_radius_cells + 1)
            clo = max(0, c - search_radius_cells)
            chi = min(W, c + search_radius_cells + 1)
            window = sdf[rlo:rhi, clo:chi]
            # Water = negative SDF; pick the most-negative (deepest) cell
            # that is closest to the original (row, col) — ties broken by
            # absolute distance in cell units.
            water_mask = window < 0
            if not water_mask.any():
                continue   # no water in window (landlocked); leave as is
            ii, jj = np.nonzero(water_mask)
            d2 = (ii + rlo - r) ** 2 + (jj + clo - c) ** 2
            k = int(np.argmin(d2))
            new_r = ii[k] + rlo
            new_c = jj[k] + clo
            out_lon.ravel()[idx] = minx + (new_c + 0.5) * self.dlon
            out_lat.ravel()[idx] = maxy - (new_r + 0.5) * self.dlat
        return out_lon, out_lat

    # ── Diagnostic: land-crossing stats for a batch of trajectories ─────

    def trajectory_stats(self, traj: np.ndarray,
                         threshold_km: float = 0.0) -> dict:
        """Given (N, T, 2) [lon, lat] degrees, report land-crossing metrics.

        `threshold_km` sets what counts as "on land". Use 0.0 for raw
        rasterization; a positive value (e.g. 10.0) discounts coastal/harbor
        rasterization noise and only counts genuinely inland penetration.
        """
        N, T, _ = traj.shape
        sdf = self.sample_km_np(traj[:, :, 0], traj[:, :, 1])   # (N, T)
        on_land = sdf > threshold_km
        return {
            "n_trajectories": int(N),
            "threshold_km": float(threshold_km),
            "land_frac": float(on_land.mean()),
            "traj_crossing_rate": float(on_land.any(axis=1).mean()),
            "mean_on_land_pts": float(on_land.sum(axis=1).mean()),
            "max_on_land_pts": int(on_land.sum(axis=1).max()),
            "mean_max_penetration_km": float(
                np.clip(sdf, 0, None).max(axis=1).mean()),
        }


def default_cache_path(dlon_deg: float = DEFAULT_DLON) -> str:
    return f"data/processed/land_sdf_{int(dlon_deg * 1000):03d}deg.npz"


# ─── CLI entry ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--shoreline",
                   default="data/gshhg/GSHHS_shp/i/GSHHS_i_L1.shp",
                   help="GSHHS L1 shapefile (continental coastlines)")
    p.add_argument("--lakes",
                   default="data/gshhg/GSHHS_shp/i/GSHHS_i_L2.shp",
                   help="GSHHS L2 shapefile (lakes, carved out of land)")
    p.add_argument("--no_lakes", action="store_true",
                   help="Skip the L2 carve-out (land-only SDF)")
    p.add_argument("--out", default=None,
                   help="Output NPZ path (default: data/processed/land_sdf_XXXdeg.npz)")
    p.add_argument("--dlon", type=float, default=DEFAULT_DLON)
    p.add_argument("--dlat", type=float, default=DEFAULT_DLAT)
    p.add_argument("--bbox", type=float, nargs=4, default=DEFAULT_BBOX,
                   metavar=("MINX", "MINY", "MAXX", "MAXY"))
    args = p.parse_args()
    out = args.out or default_cache_path(args.dlon)
    lakes_path = None if args.no_lakes else args.lakes
    build_sdf(args.shoreline, out, tuple(args.bbox), args.dlon, args.dlat,
              lakes_path=lakes_path)
