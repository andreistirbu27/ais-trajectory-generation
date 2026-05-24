#!/usr/bin/env python3
"""
region_split.py — Split trajgen_128_clean.npz by US coastal region for the
cross-region transfer experiment (strategic plan §5.1).

Each trajectory is labeled by its *midpoint* (lon, lat). Midpoint is more
stable than start/end as a region tag — many trajectories cross a regional
boundary at one endpoint but spend their bulk in a single basin.

Default 3-way split (US coastal AIS box, midpoint-based):
    Pacific  : lon < -100                            (US west coast)
    Gulf     : -100 <= lon < -82 AND lat < 31        (Gulf of Mexico)
    Atlantic : lon >= -82 OR (-100 <= lon < -82 AND lat >= 31)
                                                       (US east coast + FL cape)

Each region's NPZ has the same schema as the input (trajectories, vessel_types,
track_ids). Per-region KNN caches are also built so cross-region experiments
can use the appropriate retrieval bank.

Usage (one-shot; outputs cached, idempotent):
    paper/.venv/bin/python3 scripts/paper/region_split.py \\
        --in_npz data/processed/trajgen_128_clean.npz \\
        --out_dir data/processed/regions/ \\
        --build_knn

Cross-region eval orchestration (later, outside this script):
  • Train v10/v13 on regions/atlantic.npz with regions/atlantic_knn_k5.npz
  • Eval on regions/gulf.npz with regions/gulf_knn_k5.npz (gulf train as bank)
  → measures transfer of model + retrieval mechanism, NOT memorization.

For the "memorization" variant — train Atlantic, eval Gulf-val but retrieving
from Atlantic train — build a `cross_knn_gulf_from_atlantic.npz` manually
with `build_knn_index(query=gulf_val, corpus=atlantic_train, ...)`.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


REGIONS_3WAY = {
    "pacific":  lambda mid: mid[:, 0] < -100.0,
    "gulf":     lambda mid: (mid[:, 0] >= -100.0) & (mid[:, 0] < -82.0) & (mid[:, 1] < 31.0),
    "atlantic": lambda mid: (mid[:, 0] >= -82.0) | ((mid[:, 0] >= -100.0) & (mid[:, 0] < -82.0) & (mid[:, 1] >= 31.0)),
}


def region_assignments(trajectories: np.ndarray, regions: dict) -> dict:
    """Return {region_name: bool_mask of length N}.

    Asserts the masks partition (each trajectory in exactly one region) — if
    they don't, the rules below have a gap or overlap and need fixing.
    """
    N = len(trajectories)
    mid = trajectories[:, trajectories.shape[1] // 2, :]    # (N, 2)
    masks = {name: fn(mid) for name, fn in regions.items()}

    stacked = np.stack(list(masks.values()), axis=1).astype(np.int32)
    sums = stacked.sum(axis=1)
    if not np.all(sums == 1):
        n_zero = int((sums == 0).sum())
        n_multi = int((sums > 1).sum())
        raise RuntimeError(
            f"Region rules don't partition: {n_zero} trajectories assigned "
            f"to NO region, {n_multi} assigned to multiple. Fix the rules.")
    return masks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_npz", default="data/processed/trajgen_128_clean.npz")
    ap.add_argument("--out_dir", default="data/processed/regions/")
    ap.add_argument("--build_knn", action="store_true",
                    help="Also build per-region KNN caches (CPU only; ~2 min per region).")
    ap.add_argument("--k_retrieval", type=int, default=5)
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--knn_seed", type=int, default=42)
    ap.add_argument("--vtype_weight", type=float, default=0.5)
    args = ap.parse_args()

    print(f"Loading {args.in_npz}")
    z = np.load(args.in_npz, allow_pickle=True)
    trajectories = z["trajectories"].astype(np.float32)
    vessel_types = z["vessel_types"]
    track_ids    = z["track_ids"]
    print(f"  {len(trajectories):,} trajectories")

    masks = region_assignments(trajectories, REGIONS_3WAY)

    os.makedirs(args.out_dir, exist_ok=True)
    region_paths = {}
    for name, m in masks.items():
        out_path = os.path.join(args.out_dir, f"{name}.npz")
        n = int(m.sum())
        print(f"  {name:8s}: {n:6,d} trajectories  -> {out_path}")
        np.savez(out_path,
                 trajectories=trajectories[m],
                 vessel_types=vessel_types[m],
                 track_ids=track_ids[m])
        region_paths[name] = out_path

    if args.build_knn:
        from src.data_gen_retrieval import build_and_cache_knn
        for name, path in region_paths.items():
            cache_path = os.path.join(args.out_dir, f"{name}_knn_k{args.k_retrieval}.npz")
            print(f"\nBuilding KNN cache for {name} -> {cache_path}")
            build_and_cache_knn(
                data_npz_path=path, cache_path=cache_path,
                val_frac=args.val_frac, seed=args.knn_seed,
                k=args.k_retrieval, vtype_weight=args.vtype_weight,
                test_frac=0.0,
            )

    print("\nDone.")
    print("Region splits:")
    for name, path in region_paths.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
