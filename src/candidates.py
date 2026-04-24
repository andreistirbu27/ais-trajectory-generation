"""
Candidate generation + water-validity filtering for v11 pointer model.

The v11 decoder emits logits over a finite pool of geometric candidates
computed per step, not a continuous delta. Candidates live in RAW degree
space; after filtering they are normalized via the position scaler before
being fed to the model.

Pipeline per (position, destination):
  1. generate_candidates(p_t, end, K, cone_deg, d_min_km, d_max_km)
     → (K, 2) raw-degree points sampled in a forward cone toward the
       destination, with a Halton quasi-random (angle, radius) grid.
  2. point_water_check(candidates, land_mask, tau_km)
     → (K,) bool — candidate itself is in water.
  3. segment_water_check(p_t_rep, candidates, land_mask, n_samples, tau_km)
     → (K,) bool — interpolated points along the segment (p_t → candidate)
       are all in water.
  4. storm_free_check(...) — stub for v11-lite (returns all True).

If no candidate passes, fallback cascade (widen cone → shrink d_min → snap
to nearest-water along bearing ray) runs until one is found. A true dead
end raises InfeasibleQueryError.

Sign convention: land_mask.sdf positive = land, negative = water. A point
is "in water" iff sdf_km <= -tau_km (strictly inside water by tau km).
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np

# ─── Constants ──────────────────────────────────────────────────────────────

KM_PER_DEG_LAT = 111.0   # consistent with land_mask.py's raster conversions


class InfeasibleQueryError(RuntimeError):
    """Raised when no water-valid candidate exists after all fallbacks."""


# ─── Halton quasi-random sequence (1D) ──────────────────────────────────────

def _halton(index: int, base: int) -> float:
    """Halton sequence value at position `index` in base `base`. index >= 0."""
    f = 1.0
    r = 0.0
    i = index + 1
    while i > 0:
        f /= base
        r += f * (i % base)
        i //= base
    return r


def _halton_grid(k: int, base_a: int = 2, base_b: int = 3) -> np.ndarray:
    """Return (k, 2) Halton points in [0, 1)². Deterministic."""
    pts = np.empty((k, 2), dtype=np.float64)
    for i in range(k):
        pts[i, 0] = _halton(i, base_a)
        pts[i, 1] = _halton(i, base_b)
    return pts


# Precomputed once at module import; sized for the largest K we expect.
_HALTON_MAX_K = 128
_HALTON = _halton_grid(_HALTON_MAX_K).astype(np.float32)


def get_halton(k: int) -> np.ndarray:
    """Return first k Halton (angle_u, radius_u) points, each in [0, 1)."""
    if k <= _HALTON_MAX_K:
        return _HALTON[:k]
    return _halton_grid(k).astype(np.float32)


# ─── Geometry helpers ───────────────────────────────────────────────────────

def _bearing_to_end(p_t: np.ndarray, end: np.ndarray) -> np.ndarray:
    """Bearing in radians from each p_t to corresponding end, in planar
    (lon, lat) space with local cos(lat) correction.

    Args:
      p_t: (B, 2) [lon, lat] degrees
      end: (B, 2) [lon, lat] degrees
    Returns:
      (B,) radians
    """
    lat_mid = 0.5 * (p_t[:, 1] + end[:, 1])
    cos_lat = np.cos(np.deg2rad(lat_mid))
    dlon_km = (end[:, 0] - p_t[:, 0]) * KM_PER_DEG_LAT * cos_lat
    dlat_km = (end[:, 1] - p_t[:, 1]) * KM_PER_DEG_LAT
    return np.arctan2(dlat_km, dlon_km)


def sdf_batch_np(land_mask, lonlat: np.ndarray) -> np.ndarray:
    """Sample SDF (km) at (N, 2) [lon, lat] points. Positive = land."""
    return land_mask.sample_km_np(lonlat[..., 0], lonlat[..., 1])


# ─── Core checks ────────────────────────────────────────────────────────────

def point_water_check(p: np.ndarray, land_mask, tau_km: float = 2.0) -> np.ndarray:
    """(N, 2) → (N,) bool. True iff sdf(p) <= -tau_km (strictly in water)."""
    sdf = land_mask.sample_km_np(p[..., 0], p[..., 1])
    return sdf <= -tau_km


def segment_water_check(
    p0: np.ndarray,
    p1: np.ndarray,
    land_mask,
    n_samples: int = 5,
    tau_km: float = 2.0,
) -> np.ndarray:
    """Check that every interior sample of the segment p0→p1 is in water.

    Args:
      p0: (N, 2) degrees
      p1: (N, 2) degrees
      n_samples: number of interior points to sample along the segment.
                 Samples at t = i/(n_samples+1) for i=1..n_samples.
      tau_km: minimum depth below water (sdf_km <= -tau_km)

    Returns:
      (N,) bool — True iff all n_samples interior points are in water.
    """
    N = p0.shape[0]
    if N == 0:
        return np.ones(0, dtype=bool)

    # t values: 1/(n+1), 2/(n+1), ..., n/(n+1)
    ts = (np.arange(1, n_samples + 1, dtype=np.float32)
          / (n_samples + 1)).reshape(1, n_samples, 1)    # (1, S, 1)
    p0_b = p0[:, None, :]                                # (N, 1, 2)
    p1_b = p1[:, None, :]                                # (N, 1, 2)
    samples = p0_b + ts * (p1_b - p0_b)                  # (N, S, 2)

    samples_flat = samples.reshape(-1, 2)
    sdf_flat = land_mask.sample_km_np(samples_flat[:, 0], samples_flat[:, 1])
    sdf = sdf_flat.reshape(N, n_samples)
    return (sdf <= -tau_km).all(axis=1)


def storm_free_check(
    p0: np.ndarray,
    p1: np.ndarray,
    storms: Optional[List[Tuple[float, float, float]]] = None,
    n_samples: int = 5,
) -> np.ndarray:
    """Stub for v11-lite: no storms → all segments pass.

    storms: list of (center_lon, center_lat, radius_km).
    """
    N = p0.shape[0]
    if not storms:
        return np.ones(N, dtype=bool)

    # Simple implementation: reject if any interior sample is inside any storm.
    ts = (np.arange(1, n_samples + 1, dtype=np.float32)
          / (n_samples + 1)).reshape(1, n_samples, 1)
    p0_b = p0[:, None, :]
    p1_b = p1[:, None, :]
    samples = p0_b + ts * (p1_b - p0_b)                  # (N, S, 2)
    ok = np.ones(N, dtype=bool)
    for c_lon, c_lat, r_km in storms:
        dlat_km = (samples[..., 1] - c_lat) * KM_PER_DEG_LAT
        cos_lat = np.cos(np.deg2rad(c_lat))
        dlon_km = (samples[..., 0] - c_lon) * KM_PER_DEG_LAT * cos_lat
        inside = (dlon_km ** 2 + dlat_km ** 2) <= r_km ** 2    # (N, S)
        ok = ok & ~inside.any(axis=1)
    return ok


# ─── Candidate sampling ─────────────────────────────────────────────────────

def generate_candidates(
    p_t: np.ndarray,            # (B, 2) degrees
    end: np.ndarray,             # (B, 2) degrees
    K: int = 32,
    cone_deg: float = 60.0,
    d_min_km: float = 3.0,
    d_max_km: float = 25.0,
) -> np.ndarray:
    """Generate K candidates per batch element in a forward cone toward `end`.

    Halton quasi-random (angle_u, radius_u) in [0,1)² gets mapped to
      angle   = bearing_to_end + (angle_u - 0.5) * 2 * cone_rad
      radius  = d_min_km + radius_u * (d_max_km - d_min_km)
    Then converted to deg offsets via local cos(lat) and added to p_t.

    Returns:
      (B, K, 2) raw-degree candidate positions.
    """
    B = p_t.shape[0]
    cone_rad = math.radians(cone_deg)
    halton = get_halton(K)                               # (K, 2) float32

    bearing = _bearing_to_end(p_t, end)                  # (B,)
    # (B, K) angle and radius
    angles = bearing[:, None] + (halton[None, :, 0] - 0.5) * 2.0 * cone_rad
    radii_km = d_min_km + halton[None, :, 1] * (d_max_km - d_min_km)

    dlat_km = np.sin(angles) * radii_km
    dlon_km = np.cos(angles) * radii_km

    cos_lat = np.cos(np.deg2rad(p_t[:, 1:2]))           # (B, 1)
    dlat_deg = dlat_km / KM_PER_DEG_LAT                  # (B, K)
    dlon_deg = dlon_km / (KM_PER_DEG_LAT * np.clip(cos_lat, 1e-3, None))

    cands = np.stack([p_t[:, 0:1] + dlon_deg,
                      p_t[:, 1:2] + dlat_deg], axis=-1)  # (B, K, 2)
    return cands.astype(np.float32)


def _in_bbox(cands: np.ndarray, bbox: Tuple[float, float, float, float]) -> np.ndarray:
    """(B, K, 2) → (B, K) bool. True inside (minx, miny, maxx, maxy)."""
    minx, miny, maxx, maxy = bbox
    lon = cands[..., 0]
    lat = cands[..., 1]
    return (lon >= minx) & (lon <= maxx) & (lat >= miny) & (lat <= maxy)


# ─── Main filter entry point ────────────────────────────────────────────────

def filter_candidates(
    p_t: np.ndarray,              # (B, 2)
    cands: np.ndarray,            # (B, K, 2)
    land_mask,
    tau_km: float = 2.0,
    n_segment_samples: int = 5,
    storms: Optional[List[Tuple[float, float, float]]] = None,
    use_segment_check: bool = True,
    use_storm_check: bool = False,
) -> np.ndarray:
    """Cascade filter: bbox → point-water → segment-water → storm-free.

    Returns (B, K) bool mask — True = candidate is valid.
    """
    B, K, _ = cands.shape
    bbox = land_mask.bbox

    valid = _in_bbox(cands, bbox)

    flat_cands = cands.reshape(-1, 2)
    pt_water = point_water_check(flat_cands, land_mask, tau_km).reshape(B, K)
    valid = valid & pt_water

    if use_segment_check:
        # Broadcast p_t to match candidates: (B*K, 2)
        p0 = np.broadcast_to(p_t[:, None, :], (B, K, 2)).reshape(-1, 2)
        p1 = flat_cands
        seg = segment_water_check(p0, p1, land_mask,
                                   n_samples=n_segment_samples,
                                   tau_km=tau_km).reshape(B, K)
        valid = valid & seg

    if use_storm_check:
        p0 = np.broadcast_to(p_t[:, None, :], (B, K, 2)).reshape(-1, 2)
        p1 = flat_cands
        sf = storm_free_check(p0, p1, storms, n_samples=n_segment_samples
                               ).reshape(B, K)
        valid = valid & sf

    return valid


# ─── Fallback: snap to nearest water along bearing ray ──────────────────────

def _snap_to_water_along_bearing(
    p_t: np.ndarray,              # (2,) single sample
    end: np.ndarray,               # (2,)
    land_mask,
    tau_km: float = 2.0,
    d_min_km: float = 1.0,
    d_max_km: float = 30.0,
    n_radii: int = 30,
) -> Optional[np.ndarray]:
    """Last-resort: march along the bearing-to-end ray, return the first
    water-valid point. Returns None if none found along the ray.
    """
    brg = _bearing_to_end(p_t[None], end[None])[0]
    radii = np.linspace(d_min_km, d_max_km, n_radii)
    dlat = np.sin(brg) * radii / KM_PER_DEG_LAT
    cos_lat = max(1e-3, math.cos(math.radians(p_t[1])))
    dlon = np.cos(brg) * radii / (KM_PER_DEG_LAT * cos_lat)
    pts = np.stack([p_t[0] + dlon, p_t[1] + dlat], axis=-1)   # (n_radii, 2)

    minx, miny, maxx, maxy = land_mask.bbox
    in_bbox = ((pts[:, 0] >= minx) & (pts[:, 0] <= maxx)
               & (pts[:, 1] >= miny) & (pts[:, 1] <= maxy))
    pts = pts[in_bbox]
    if len(pts) == 0:
        return None

    sdf = land_mask.sample_km_np(pts[:, 0], pts[:, 1])
    water = sdf <= -tau_km
    # Also need segment to be all-water from p_t to picked point
    ok = water.copy()
    if ok.any():
        seg = segment_water_check(
            np.broadcast_to(p_t[None], (len(pts), 2)),
            pts, land_mask, n_samples=5, tau_km=tau_km)
        ok = ok & seg
        if ok.any():
            idx = int(np.argmax(ok))   # first True
            return pts[idx].astype(np.float32)
    return None


# ─── One-shot per-step generation with fallback ─────────────────────────────

def generate_and_filter(
    p_t: np.ndarray,              # (B, 2)
    end: np.ndarray,               # (B, 2)
    land_mask,
    K: int = 32,
    cone_deg: float = 60.0,
    d_min_km: float = 3.0,
    d_max_km: float = 25.0,
    tau_km: float = 2.0,
    n_segment_samples: int = 5,
    storms: Optional[List[Tuple[float, float, float]]] = None,
    use_segment_check: bool = True,
    use_storm_check: bool = False,
    strict: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate + filter with the fallback cascade documented in the plan.

    Fallback for empty pools (per batch element):
      1. widen cone to ±120°
      2. shrink d_min to 1 km, widen d_max to 40 km
      3. snap to nearest-water along bearing-to-end ray
      4. if strict=True, raise InfeasibleQueryError; else mark fallback.

    Returns:
      candidates:    (B, K, 2)
      valid_mask:    (B, K)  — True for valid candidates
      fallback_flag: (B,)    — True for samples that hit the snap fallback.
                              For those, candidates[b, 0] is the snapped
                              point and valid_mask[b, 0] = True.
    """
    B = p_t.shape[0]
    cands = generate_candidates(p_t, end, K, cone_deg, d_min_km, d_max_km)
    valid = filter_candidates(
        p_t, cands, land_mask, tau_km=tau_km,
        n_segment_samples=n_segment_samples, storms=storms,
        use_segment_check=use_segment_check, use_storm_check=use_storm_check,
    )

    fallback = np.zeros(B, dtype=bool)
    empty = ~valid.any(axis=1)
    if not empty.any():
        return cands, valid, fallback

    # Stage 1: widen cone
    idxs = np.nonzero(empty)[0]
    cands_w = generate_candidates(
        p_t[idxs], end[idxs], K, cone_deg=120.0,
        d_min_km=d_min_km, d_max_km=d_max_km)
    valid_w = filter_candidates(
        p_t[idxs], cands_w, land_mask, tau_km=tau_km,
        n_segment_samples=n_segment_samples, storms=storms,
        use_segment_check=use_segment_check, use_storm_check=use_storm_check,
    )
    still_empty = ~valid_w.any(axis=1)
    keep = ~still_empty
    if keep.any():
        cands[idxs[keep]] = cands_w[keep]
        valid[idxs[keep]] = valid_w[keep]

    # Stage 2: shrink d_min + widen d_max on the remaining empties
    idxs2 = idxs[still_empty]
    if len(idxs2) > 0:
        cands_s = generate_candidates(
            p_t[idxs2], end[idxs2], K, cone_deg=120.0,
            d_min_km=1.0, d_max_km=40.0)
        valid_s = filter_candidates(
            p_t[idxs2], cands_s, land_mask, tau_km=tau_km,
            n_segment_samples=n_segment_samples, storms=storms,
            use_segment_check=use_segment_check, use_storm_check=use_storm_check,
        )
        still_empty2 = ~valid_s.any(axis=1)
        keep2 = ~still_empty2
        if keep2.any():
            cands[idxs2[keep2]] = cands_s[keep2]
            valid[idxs2[keep2]] = valid_s[keep2]

        # Stage 3: snap to bearing ray for the truly stuck
        idxs3 = idxs2[still_empty2]
        for b in idxs3:
            snap = _snap_to_water_along_bearing(
                p_t[b], end[b], land_mask, tau_km=tau_km)
            if snap is None:
                if strict:
                    raise InfeasibleQueryError(
                        f"No water-valid candidate for p_t={p_t[b].tolist()}, "
                        f"end={end[b].tolist()}")
                # Mark slot 0 invalid; caller must skip this sample.
                valid[b] = False
                fallback[b] = True
            else:
                cands[b, 0] = snap
                valid[b] = False
                valid[b, 0] = True
                fallback[b] = True

    return cands, valid, fallback


# ─── GT-insertion (training-time only) ──────────────────────────────────────

def insert_gt_into_candidates(
    cands: np.ndarray,            # (B, K, 2)
    valid: np.ndarray,             # (B, K)
    p_t: np.ndarray,              # (B, 2)
    gt_next: np.ndarray,           # (B, 2)
    land_mask,
    tau_km: float = 2.0,
    n_segment_samples: int = 5,
    segment_check: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For each batch element, ensure gt_next is in the candidate pool:
      - If gt_next passes point + segment checks, overwrite candidate 0
        with gt_next and set valid[b, 0] = True.
      - Otherwise flag gt_index as -100 (CE ignore_index).

    Returns:
      cands (B, K, 2) — possibly modified
      valid (B, K)    — possibly modified
      gt_indices (B,) — int64; -100 for samples whose GT is infeasible.
    """
    B = gt_next.shape[0]
    gt_ok = point_water_check(gt_next, land_mask, tau_km=tau_km)
    if segment_check:
        seg_ok = segment_water_check(p_t, gt_next, land_mask,
                                      n_samples=n_segment_samples,
                                      tau_km=tau_km)
        gt_ok = gt_ok & seg_ok

    gt_indices = np.full(B, -100, dtype=np.int64)

    # Insert GT into slot 0 where it passes the filter
    insert_idx = np.nonzero(gt_ok)[0]
    if len(insert_idx) > 0:
        cands[insert_idx, 0] = gt_next[insert_idx]
        valid[insert_idx, 0] = True
        gt_indices[insert_idx] = 0

    # For GT that fails but there are still valid candidates: use nearest
    # valid candidate to GT. This is a training-time approximation — avoids
    # losing the sample entirely when GT crosses a narrow channel.
    leftover = np.nonzero(~gt_ok & valid.any(axis=1))[0]
    for b in leftover:
        vm = valid[b]
        if not vm.any():
            continue
        d2 = ((cands[b] - gt_next[b]) ** 2).sum(axis=-1)
        d2_masked = np.where(vm, d2, np.inf)
        gt_indices[b] = int(np.argmin(d2_masked))

    return cands, valid, gt_indices
