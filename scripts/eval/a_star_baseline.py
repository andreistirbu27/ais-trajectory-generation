"""A*-shortest-water-path baseline.

Computes the shortest legal water path between (start, end) for every val/test
route, resamples to 128 equally-spaced waypoints, and scores ADE against GT.
This is the most aggressive water-strict baseline: it ignores shipping
conventions and takes the geodesic water path, which lets us isolate "what
the model adds beyond raw legality".

Outputs:
    results/a_star_baseline.csv   (per-route ADE + path length)
    results/a_star_baseline_summary.txt
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data_gen import train_val_split_gen
from src.land_mask import LandMask
from src.water_router import WaterRouter, _haversine_km


def resample_path_to_T(path_lonlat: np.ndarray, T: int) -> np.ndarray:
    """Resample (n, 2) polyline to T waypoints equally spaced by arc length."""
    if len(path_lonlat) < 2:
        return np.tile(path_lonlat[:1], (T, 1)).astype(np.float64)
    seg = _haversine_km(path_lonlat[:-1, 0], path_lonlat[:-1, 1],
                        path_lonlat[1:, 0],  path_lonlat[1:, 1])
    s = np.concatenate(([0.0], np.cumsum(seg)))
    total = float(s[-1]) if s[-1] > 0 else 1e-9
    targets = np.linspace(0.0, total, T)
    out = np.empty((T, 2), dtype=np.float64)
    out[:, 0] = np.interp(targets, s, path_lonlat[:, 0])
    out[:, 1] = np.interp(targets, s, path_lonlat[:, 1])
    return out


def ade_haversine_m(pred: np.ndarray, gt: np.ndarray) -> float:
    d = _haversine_km(pred[:, 0], pred[:, 1], gt[:, 0], gt[:, 1])
    return float(d.mean() * 1000.0)


def a_star_predict(router: WaterRouter, start: np.ndarray, end: np.ndarray,
                   T: int, window_margin: int, max_window: int) -> tuple[np.ndarray, dict]:
    """Run A* between start and end. Returns (T, 2) predicted trajectory + stats."""
    r0, c0 = router._to_grid(start[0], start[1])
    r1, c1 = router._to_grid(end[0],   end[1])
    nr0, nc0 = router.nearest_water_cell(int(r0), int(c0),
                                          search_radius_cells=200,
                                          require_connected=True)
    nr1, nc1 = router.nearest_water_cell(int(r1), int(c1),
                                          search_radius_cells=200,
                                          require_connected=True)
    path = router.route(int(nr0), int(nc0), int(nr1), int(nc1),
                        window_margin_cells=window_margin,
                        max_window_cells=max_window)
    if path is None:
        # Window too small or unreachable on the fine graph: fall back to
        # great-circle interpolation between snapped endpoints (still water
        # by construction at the endpoints, but the line may not be).
        lon0, lat0 = router._to_lonlat(nr0, nc0)
        lon1, lat1 = router._to_lonlat(nr1, nc1)
        out = np.stack([np.linspace(lon0, lon1, T),
                        np.linspace(lat0, lat1, T)], axis=1)
        return out, {"fallback": 1, "path_nodes": 0}
    rows = np.array([p[0] for p in path], dtype=np.int64)
    cols = np.array([p[1] for p in path], dtype=np.int64)
    lons, lats = router._to_lonlat(rows, cols)
    polyline = np.stack([lons, lats], axis=1)
    return resample_path_to_T(polyline, T), {"fallback": 0, "path_nodes": len(path)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_npz",    default="data/processed/trajgen_128_clean.npz")
    ap.add_argument("--water_graph", default="data/processed/water_graph_005deg.npz")
    ap.add_argument("--land_sdf",    default="data/processed/land_sdf_050deg.npz")
    ap.add_argument("--val_frac",  type=float, default=0.15)
    ap.add_argument("--test_frac", type=float, default=0.05)
    ap.add_argument("--seed",      type=int, default=42)
    ap.add_argument("--n_eval_val",  type=int, default=500,
                    help="Subsample of val (seed=42) for speed; 0 = full val")
    ap.add_argument("--window_margin", type=int, default=400)
    ap.add_argument("--max_window",    type=int, default=1500)
    ap.add_argument("--out_csv",     default="results/a_star_baseline.csv")
    ap.add_argument("--out_summary", default="results/a_star_baseline_summary.txt")
    args = ap.parse_args()

    print(f"Loading land SDF: {args.land_sdf}")
    coarse = LandMask.load(args.land_sdf)
    print(f"Loading water graph: {args.water_graph}")
    router = WaterRouter.load(args.water_graph, coarse_mask=coarse)
    print(f"  shape={router.shape}, tau_km={router.tau_km}")

    data = np.load(args.data_npz, allow_pickle=True)
    trajs = data["trajectories"].astype(np.float64)
    vts   = data["vessel_types"]
    tids  = data["track_ids"]
    T = trajs.shape[1]

    (_, _, _, val_t, _, _, test_t, _, _) = train_val_split_gen(
        trajs, vts, tids, args.val_frac, args.seed, test_frac=args.test_frac
    )
    print(f"val={len(val_t)}  test={len(test_t)}")

    if 0 < args.n_eval_val < len(val_t):
        rng = np.random.RandomState(42)
        idx = rng.choice(len(val_t), args.n_eval_val, replace=False)
        val_eval = val_t[idx]
        print(f"  subsampled val to {args.n_eval_val} (seed=42)")
    else:
        val_eval = val_t

    rows = []
    for split_name, partition in [("val", val_eval), ("test", test_t)]:
        ades, lengths, falls = [], [], 0
        t0 = time.time()
        for i in tqdm(range(len(partition)), desc=f"A* {split_name}"):
            traj = partition[i]
            pred, st = a_star_predict(router, traj[0], traj[-1], T,
                                       args.window_margin, args.max_window)
            ade = ade_haversine_m(pred, traj)
            arc = _haversine_km(traj[:-1, 0], traj[:-1, 1],
                                traj[1:, 0],  traj[1:, 1]).sum()
            pred_arc = _haversine_km(pred[:-1, 0], pred[:-1, 1],
                                     pred[1:, 0],  pred[1:, 1]).sum()
            ades.append(ade)
            lengths.append((arc, pred_arc))
            falls += int(st["fallback"])
            rows.append({"split": split_name, "route_idx": i,
                         "gt_arc_km": float(arc), "pred_arc_km": float(pred_arc),
                         "ade_m": float(ade), "fallback": int(st["fallback"]),
                         "path_nodes": int(st["path_nodes"])})
        dt = time.time() - t0
        ades_arr = np.array(ades)
        print(f"  {split_name}: n={len(ades_arr)}  ADE mean={ades_arr.mean():.1f}  "
              f"median={np.median(ades_arr):.1f}  fallbacks={falls}  "
              f"wall={dt:.1f}s ({1000 * dt / len(ades_arr):.1f} ms/route)")

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with open(args.out_summary, "w") as f:
        for split_name in ("val", "test"):
            xs = [r["ade_m"] for r in rows if r["split"] == split_name]
            fs = sum(r["fallback"] for r in rows if r["split"] == split_name)
            if xs:
                f.write(f"{split_name}\tn={len(xs)}\tmean_ade={np.mean(xs):.1f}\t"
                        f"median_ade={np.median(xs):.1f}\t"
                        f"std_ade={np.std(xs, ddof=0):.1f}\tfallbacks={fs}\n")
    print(f"\nWrote {args.out_csv}, {args.out_summary}")


if __name__ == "__main__":
    main()
