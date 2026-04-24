"""
Evaluation metrics for conditional trajectory generation.

All distance metrics are in metres (haversine on the WGS84 sphere).
Trajectories are (N, T, 2) arrays with columns [lon, lat] in degrees.
"""

import numpy as np
from tqdm import tqdm


EARTH_RADIUS_M = 6_371_000.0


# ─── Distance helpers ────────────────────────────────────────────────────────

def haversine_meters(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Haversine distance in metres between two sets of (lat, lon) in degrees."""
    lat1, lon1, lat2, lon2 = map(np.deg2rad, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    return EARTH_RADIUS_M * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def path_length_m(traj):
    """Total path length in metres for trajectory (T, 2) [lon, lat]."""
    return haversine_meters(
        traj[:-1, 1], traj[:-1, 0], traj[1:, 1], traj[1:, 0]).sum()


# ─── Discrete Frechet distance ──────────────────────────────────────────────

def discrete_frechet(P, Q):
    """Discrete Frechet distance between two trajectories.

    Args:
        P: (N, 2) [lon, lat] in degrees
        Q: (M, 2) [lon, lat] in degrees

    Returns:
        Frechet distance in metres
    """
    n, m = len(P), len(Q)
    # Precompute pairwise distances
    dist = np.zeros((n, m))
    for i in range(n):
        dist[i] = haversine_meters(P[i, 1], P[i, 0], Q[:, 1], Q[:, 0])

    # DP
    ca = np.full((n, m), -1.0)
    ca[0, 0] = dist[0, 0]
    for i in range(1, n):
        ca[i, 0] = max(ca[i - 1, 0], dist[i, 0])
    for j in range(1, m):
        ca[0, j] = max(ca[0, j - 1], dist[0, j])
    for i in range(1, n):
        for j in range(1, m):
            ca[i, j] = max(min(ca[i - 1, j], ca[i, j - 1], ca[i - 1, j - 1]),
                           dist[i, j])
    return ca[n - 1, m - 1]


# ─── Great-circle baseline ──────────────────────────────────────────────────

def great_circle_trajectory(start, end, n_points=128):
    """Generate n_points along the great-circle arc from start to end.

    Args:
        start: (2,) [lon, lat] in degrees
        end:   (2,) [lon, lat] in degrees
        n_points: number of interpolation points

    Returns:
        (n_points, 2) [lon, lat] in degrees
    """
    lon1, lat1 = np.deg2rad(start)
    lon2, lat2 = np.deg2rad(end)

    d = 2 * np.arcsin(np.sqrt(
        np.sin((lat2 - lat1) / 2)**2 +
        np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2)**2))

    if d < 1e-12:
        # Start == end
        return np.tile(start, (n_points, 1))

    fracs = np.linspace(0, 1, n_points)
    a = np.sin((1 - fracs) * d) / np.sin(d)
    b = np.sin(fracs * d) / np.sin(d)

    x = a * np.cos(lat1) * np.cos(lon1) + b * np.cos(lat2) * np.cos(lon2)
    y = a * np.cos(lat1) * np.sin(lon1) + b * np.cos(lat2) * np.sin(lon2)
    z = a * np.sin(lat1) + b * np.sin(lat2)

    lat = np.rad2deg(np.arctan2(z, np.sqrt(x**2 + y**2)))
    lon = np.rad2deg(np.arctan2(y, x))

    return np.stack([lon, lat], axis=1)


# ─── Evaluate full batch of trajectories ────────────────────────────────────

def evaluate_generation(pred_traj, gt_traj, compute_frechet=True):
    """Evaluate generated trajectories against ground truth.

    Args:
        pred_traj: (N, T, 2) [lon, lat] degrees — generated
        gt_traj:   (N, T, 2) [lon, lat] degrees — ground truth
        compute_frechet: if True, compute Frechet distance (slow for large N)

    Returns:
        dict with metric names → values
    """
    N, T, _ = pred_traj.shape

    # Per-point haversine errors (N, T)
    point_errors = haversine_meters(
        gt_traj[:, :, 1], gt_traj[:, :, 0],
        pred_traj[:, :, 1], pred_traj[:, :, 0])

    # Per-trajectory ADE (N,)
    per_traj_ade = point_errors.mean(axis=1)

    # ADE: average displacement error (mean over all points and trajectories)
    ade_m = point_errors.mean()

    # FDE: final displacement error (at last point)
    fde_m = point_errors[:, -1].mean()

    # Endpoint error (same as FDE for fixed-length trajectories)
    endpoint_error_m = fde_m

    # Path length ratio
    pred_lengths = np.array([path_length_m(pred_traj[i]) for i in range(N)])
    gt_lengths = np.array([path_length_m(gt_traj[i]) for i in range(N)])
    # Avoid division by zero
    valid = gt_lengths > 1.0
    if valid.any():
        path_length_ratio = np.median(pred_lengths[valid] / gt_lengths[valid])
    else:
        path_length_ratio = float("nan")

    # Normalized ADE: per-traj ADE / GT path length (unitless ratio)
    # Short routes with small ADE still get large ratio; long routes with large
    # ADE still get small ratio. Mean over valid trajectories.
    if valid.any():
        normalized_ade = float(np.mean(per_traj_ade[valid] / gt_lengths[valid]))
    else:
        normalized_ade = float("nan")

    # Smoothness: mean squared acceleration in degree space
    pred_deltas = np.diff(pred_traj, axis=1)         # (N, T-1, 2)
    pred_accel = np.diff(pred_deltas, axis=1)         # (N, T-2, 2)
    smoothness = (pred_accel ** 2).mean()

    gt_deltas = np.diff(gt_traj, axis=1)
    gt_accel = np.diff(gt_deltas, axis=1)
    gt_smoothness = (gt_accel ** 2).mean()

    result = {
        "ade_m": float(ade_m),
        "fde_m": float(fde_m),
        "endpoint_error_m": float(endpoint_error_m),
        "normalized_ade": normalized_ade,
        "path_length_ratio": float(path_length_ratio),
        "smoothness": float(smoothness),
        "gt_smoothness": float(gt_smoothness),
        "n_trajectories": N,
    }

    # Length-bucketed ADE: short <50km, medium 50–500km, long >500km.
    # A 10 km error on a 20 km route and on a 2000 km route shouldn't be pooled.
    gt_km = gt_lengths / 1000.0
    buckets = [
        ("short",  gt_km < 50.0),
        ("medium", (gt_km >= 50.0) & (gt_km < 500.0)),
        ("long",   gt_km >= 500.0),
    ]
    for name, mask in buckets:
        n_b = int(mask.sum())
        result[f"n_{name}"] = n_b
        if n_b > 0:
            result[f"ade_{name}_m"] = float(per_traj_ade[mask].mean())
            bucket_valid = mask & valid
            if bucket_valid.any():
                result[f"normalized_ade_{name}"] = float(
                    np.mean(per_traj_ade[bucket_valid] / gt_lengths[bucket_valid]))
            else:
                result[f"normalized_ade_{name}"] = float("nan")
        else:
            result[f"ade_{name}_m"] = float("nan")
            result[f"normalized_ade_{name}"] = float("nan")

    # Frechet distance (expensive — subsample if needed)
    if compute_frechet:
        n_frechet = min(N, 200)  # cap for speed
        frechet_dists = []
        for i in tqdm(range(n_frechet), desc="Frechet", leave=False):
            frechet_dists.append(discrete_frechet(pred_traj[i], gt_traj[i]))
        result["frechet_m"] = float(np.mean(frechet_dists))
        result["frechet_median_m"] = float(np.median(frechet_dists))

    return result


def evaluate_great_circle_baseline(gt_traj):
    """Evaluate the great-circle (straight-line) baseline.

    Args:
        gt_traj: (N, T, 2) [lon, lat] degrees

    Returns:
        dict with metrics for the great-circle baseline
    """
    N, T, _ = gt_traj.shape
    gc_traj = np.zeros_like(gt_traj)
    for i in range(N):
        gc_traj[i] = great_circle_trajectory(gt_traj[i, 0], gt_traj[i, -1], T)
    return evaluate_generation(gc_traj, gt_traj, compute_frechet=True)
