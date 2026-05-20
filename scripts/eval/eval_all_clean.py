#!/usr/bin/env python3
"""eval_all_clean.py — unified 5-seed evaluation of every comparator on the
*cleaned* corpus.

This is the audit-time driver that replaces the ad-hoc mixture of
``eval_gen.py``, ``eval_gen_retrieval.py``, ``eval_gen_v10.py``,
``eval_gen_v10_router.py`` and ``multi_seed_variance.py`` with a single
script that:

* loads ``data/processed/trajgen_128_clean.npz`` once,
* reproduces the cleaned-corpus MMSI-grouped val split
  (``val_frac=0.15``, ``seed=42``) once,
* takes the union of 5 disjoint 500-row subsample seeds
  ``{0, 1, 2, 42, 123}`` of that val split,
* generates predictions for every variant on the union (one inference pass
  per variant — no checkpoint is re-run per seed),
* slices the union back to each seed's 500-row subsample,
* computes ADE, FDE, length-bucketed ADE, smoothness, normalised ADE and
  land/cross-validity at thresholds {10, 25} km for every (variant, seed),
* writes ``results/headline_clean_5seed.csv`` (one row per variant × seed)
  and ``results/headline_clean_5seed_summary.csv`` (mean / std / n per
  variant).

Variants supported (selectable via ``--variants``):

    v2          (TrajectoryGenerator, no retrieval, runs/trajgen_v2/best.pt)
    v3          (TrajectoryGenerator, no retrieval, runs/trajgen_v3/best.pt)
    v4          (TrajectoryGenerator, no retrieval, runs/trajgen_v4/best.pt)
    v5          (TrajectoryGenerator, no retrieval, runs/trajgen_v5/best.pt)
    mvp         (TrajectoryGenerator, no retrieval, runs/trajgen_mvp/best.pt)
    v9          (RetrievalTrajectoryGenerator, mean-pool retrieval)
    v10_raw     (TrajectoryGeneratorV10, no land projection)
    v10_hardproj (TrajectoryGeneratorV10, land projection during AR)
    v10_router  (v10 raw + WaterRouter snap-only post-processor)
    retr_top1   (zero-training retrieval-top-1 from the clean KNN cache)
    great_circle (geodesic between start and end)

For variants whose checkpoint fails to load (e.g. architecture drift since
training), the script prints a warning, records ``ade_m=nan`` for that
variant and continues. The CSV always reflects what was actually measured.

Usage:
    python3 scripts/eval/eval_all_clean.py \\
        --variants v2 v3 v4 v5 v9 v10_raw v10_hardproj v10_router \\
                   retr_top1 great_circle \\
        --out_csv results/headline_clean_5seed.csv
"""

import argparse
import csv
import os
import sys
import time
import traceback
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.data_gen import Scaler, TrajGenScalers, train_val_split_gen
from src.land_mask import LandMask
from src.metrics_gen import evaluate_generation, great_circle_trajectory
from src.model_gen import TrajectoryGenerator
from src.model_gen_retrieval import RetrievalTrajectoryGenerator
from src.model_gen_v10 import TrajectoryGeneratorV10
from src.water_router import WaterRouter


SEEDS = [0, 1, 2, 42, 123]
N_EVAL = 500


# ─────────────────────────────────────────────────────────────────────────
#  Device + checkpoint utilities
# ─────────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _strip_orig_mod(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if any(k.startswith("_orig_mod.") for k in sd):
        return {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    return sd


def _make_scalers(ckpt) -> TrajGenScalers:
    return TrajGenScalers(
        pos=Scaler(mean=ckpt["pos_mean"], std=ckpt["pos_std"]),
        delta=Scaler(mean=ckpt["delta_mean"], std=ckpt["delta_std"]),
    )


def load_v2v5(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = argparse.Namespace(**ckpt["args"])
    model = TrajectoryGenerator(
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=getattr(args, "num_encoder_layers", 0),
        num_decoder_layers=args.num_decoder_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        num_vessel_types=ckpt["num_vessel_types"],
    ).to(device)
    model.load_state_dict(_strip_orig_mod(ckpt["model"]))
    model.eval()
    return model, _make_scalers(ckpt), ckpt["vtype_vocab"], ckpt.get("n_resample", 128)


def load_v9(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = argparse.Namespace(**ckpt["args"])
    model = RetrievalTrajectoryGenerator(
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        num_vessel_types=ckpt["num_vessel_types"],
        max_retrieved=ckpt["k_retrieval"],
        route_encoder=ckpt["route_encoder"],
        n_points=ckpt.get("n_resample", 128),
    ).to(device)
    model.load_state_dict(_strip_orig_mod(ckpt["model"]))
    model.eval()
    return model, _make_scalers(ckpt), ckpt["vtype_vocab"], ckpt.get("n_resample", 128)


def load_v10(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = argparse.Namespace(**ckpt["args"])
    model = TrajectoryGeneratorV10(
        d_model=args.d_model, nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        dim_feedforward=args.dim_feedforward, dropout=args.dropout,
        num_vessel_types=ckpt["num_vessel_types"],
        max_retrieved=ckpt["k_retrieval"],
        t_retr=ckpt["t_retr"],
        n_points=ckpt.get("n_resample", 128),
        k_past=ckpt.get("k_past", None),
    ).to(device)
    model.load_state_dict(_strip_orig_mod(ckpt["model"]))
    model.eval()
    return model, _make_scalers(ckpt), ckpt["vtype_vocab"], ckpt.get("n_resample", 128), ckpt


# ─────────────────────────────────────────────────────────────────────────
#  Per-variant generators (all return a (N, T, 2) ndarray of [lon, lat] degrees)
# ─────────────────────────────────────────────────────────────────────────

def _gen_v2v5(model, scalers, vtype_vocab, val_traj, val_vt, T, device,
              batch_size: int = 32) -> np.ndarray:
    N = len(val_traj)
    pred = np.zeros_like(val_traj)
    for s in tqdm(range(0, N, batch_size), desc="v2-v5 gen", leave=False):
        e = min(s + batch_size, N)
        batch_gt = val_traj[s:e]
        batch_vt = val_vt[s:e]
        starts = scalers.pos.transform(batch_gt[:, 0])
        ends   = scalers.pos.transform(batch_gt[:, -1])
        vtypes = np.array([vtype_vocab.get(int(v), 0) for v in batch_vt])
        st = torch.tensor(starts, dtype=torch.float32, device=device)
        et = torch.tensor(ends,   dtype=torch.float32, device=device)
        vt = torch.tensor(vtypes, dtype=torch.long,    device=device)
        with torch.no_grad():
            gen = model.generate(st, et, vt, n_steps=T,
                                 pos_scaler=scalers.pos,
                                 delta_scaler=scalers.delta)
        gen = gen.permute(1, 0, 2).cpu().numpy()
        for i in range(e - s):
            pred[s + i] = scalers.pos.inverse(gen[i])
    return pred


def _gen_v9(model, scalers, vtype_vocab, val_traj, val_vt, train_traj, val_knn,
            T, device, batch_size: int = 32) -> np.ndarray:
    N = len(val_traj)
    pred = np.zeros_like(val_traj)
    for s in tqdm(range(0, N, batch_size), desc="v9 gen", leave=False):
        e = min(s + batch_size, N)
        batch_gt = val_traj[s:e]
        batch_vt = val_vt[s:e]
        batch_knn = val_knn[s:e]
        B = e - s
        starts = scalers.pos.transform(batch_gt[:, 0])
        ends   = scalers.pos.transform(batch_gt[:, -1])
        vtypes = np.array([vtype_vocab.get(int(v), 0) for v in batch_vt])
        retrieved = train_traj[batch_knn]
        K = retrieved.shape[1]
        retrieved_norm = scalers.pos.transform(
            retrieved.reshape(-1, 2)).reshape(B, K, T, 2)
        retrieval_mask = np.zeros((B, K), dtype=bool)
        st = torch.tensor(starts, dtype=torch.float32, device=device)
        et = torch.tensor(ends,   dtype=torch.float32, device=device)
        vt = torch.tensor(vtypes, dtype=torch.long,    device=device)
        rt = torch.tensor(retrieved_norm, dtype=torch.float32, device=device)
        rm = torch.tensor(retrieval_mask, dtype=torch.bool,    device=device)
        with torch.no_grad():
            gen = model.generate(st, et, vt, rt, rm, n_steps=T,
                                 pos_scaler=scalers.pos,
                                 delta_scaler=scalers.delta)
        gen = gen.permute(1, 0, 2).cpu().numpy()
        for i in range(B):
            pred[s + i] = scalers.pos.inverse(gen[i])
    return pred


def _gen_v10(model, scalers, vtype_vocab, val_traj, val_vt, train_traj, val_knn,
             T, device, land_mask: Optional[LandMask] = None,
             hard_threshold_km: float = 10.0,
             batch_size: int = 32) -> np.ndarray:
    N = len(val_traj)
    pred = np.zeros_like(val_traj)
    for s in tqdm(range(0, N, batch_size), desc="v10 gen", leave=False):
        e = min(s + batch_size, N)
        batch_gt = val_traj[s:e]
        batch_vt = val_vt[s:e]
        batch_knn = val_knn[s:e]
        B = e - s
        starts = scalers.pos.transform(batch_gt[:, 0])
        ends   = scalers.pos.transform(batch_gt[:, -1])
        vtypes = np.array([vtype_vocab.get(int(v), 0) for v in batch_vt])
        retrieved = train_traj[batch_knn]
        K = retrieved.shape[1]
        retrieved_norm = scalers.pos.transform(
            retrieved.reshape(-1, 2)).reshape(B, K, T, 2)
        retrieval_mask = np.zeros((B, K), dtype=bool)
        st = torch.tensor(starts, dtype=torch.float32, device=device)
        et = torch.tensor(ends,   dtype=torch.float32, device=device)
        vt = torch.tensor(vtypes, dtype=torch.long,    device=device)
        rt = torch.tensor(retrieved_norm, dtype=torch.float32, device=device)
        rm = torch.tensor(retrieval_mask, dtype=torch.bool,    device=device)
        with torch.no_grad():
            gen = model.generate(st, et, vt, rt, rm, n_steps=T,
                                 pos_scaler=scalers.pos,
                                 delta_scaler=scalers.delta,
                                 land_mask=land_mask,
                                 hard_threshold_km=hard_threshold_km)
        gen = gen.permute(1, 0, 2).cpu().numpy()
        for i in range(B):
            pred[s + i] = scalers.pos.inverse(gen[i])
    return pred


def _apply_router(pred: np.ndarray, router: WaterRouter,
                  threshold_km: float) -> np.ndarray:
    out = np.empty_like(pred)
    for i in tqdm(range(len(pred)), desc="router snap", leave=False):
        repaired, _ = router.repair_trajectory(
            pred[i].astype(np.float64),
            threshold_km=threshold_km,
            do_segment_bridge=False,
            window_margin_cells=50,
            max_window_cells=400,
        )
        out[i] = repaired.astype(np.float32)
    return out


# ─────────────────────────────────────────────────────────────────────────
#  Variant registry
# ─────────────────────────────────────────────────────────────────────────

VARIANTS_DEFAULT = [
    "v2", "v3", "v4", "v5",
    "v9", "v10_raw", "v10_hardproj", "v10_router",
    "retr_top1", "great_circle",
]

CHECKPOINT_PATHS = {
    "mvp": "runs/trajgen_mvp/best.pt",
    "v2":  "runs/trajgen_v2/best.pt",
    "v3":  "runs/trajgen_v3/best.pt",
    "v4":  "runs/trajgen_v4/best.pt",
    "v5":  "runs/trajgen_v5/best.pt",
    "v9":  "runs/trajgen_v9_retrieval/best.pt",
    "v10_raw":      "runs/trajgen_v10/best.pt",
    "v10_hardproj": "runs/trajgen_v10/best.pt",
    "v10_router":   "runs/trajgen_v10/best.pt",
}


def run_variant(variant: str,
                union_traj: np.ndarray, union_vt: np.ndarray,
                union_knn: Optional[np.ndarray],
                train_traj: np.ndarray,
                T: int, device: torch.device,
                land_mask: LandMask, router: Optional[WaterRouter],
                threshold_km: float) -> Optional[np.ndarray]:
    """Return (N, T, 2) predictions for `variant`, or None on load failure."""
    t0 = time.time()
    try:
        if variant in ("v2", "v3", "v4", "v5", "mvp"):
            model, scalers, vocab, T_ck = load_v2v5(CHECKPOINT_PATHS[variant], device)
            assert T_ck == T, f"checkpoint T={T_ck} != data T={T}"
            pred = _gen_v2v5(model, scalers, vocab, union_traj, union_vt, T, device)
        elif variant == "v9":
            model, scalers, vocab, T_ck = load_v9(CHECKPOINT_PATHS["v9"], device)
            assert T_ck == T
            pred = _gen_v9(model, scalers, vocab, union_traj, union_vt,
                           train_traj, union_knn, T, device)
        elif variant in ("v10_raw", "v10_hardproj", "v10_router"):
            model, scalers, vocab, T_ck, _ = load_v10(CHECKPOINT_PATHS["v10_raw"], device)
            assert T_ck == T
            lm = land_mask if variant == "v10_hardproj" else None
            pred = _gen_v10(model, scalers, vocab, union_traj, union_vt,
                            train_traj, union_knn, T, device,
                            land_mask=lm, hard_threshold_km=threshold_km)
            if variant == "v10_router":
                pred = _apply_router(pred, router, threshold_km)
        elif variant == "retr_top1":
            pred = train_traj[union_knn[:, 0]].astype(np.float32)
        elif variant == "great_circle":
            pred = np.zeros_like(union_traj)
            for i in range(len(union_traj)):
                pred[i] = great_circle_trajectory(union_traj[i, 0],
                                                  union_traj[i, -1], T)
        else:
            print(f"[WARN] unknown variant {variant!r}; skipping")
            return None
        print(f"  [{variant}] generated {len(pred)} traj in {time.time()-t0:.1f}s")
        return pred
    except Exception as exc:
        print(f"[WARN] {variant} failed: {exc.__class__.__name__}: {exc}")
        traceback.print_exc()
        return None


# ─────────────────────────────────────────────────────────────────────────
#  Subsampling
# ─────────────────────────────────────────────────────────────────────────

def subsample_indices(n_val: int, n_eval: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return np.sort(rng.choice(n_val, n_eval, replace=False))


# ─────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data_npz",    default="data/processed/trajgen_128_clean.npz")
    p.add_argument("--knn_cache",   default="data/processed/trajgen_128_clean_knn_k5.npz")
    p.add_argument("--land_sdf",    default="data/processed/land_sdf_050deg.npz")
    p.add_argument("--water_graph", default="data/processed/water_graph_005deg.npz")
    p.add_argument("--val_frac",    type=float, default=0.15)
    p.add_argument("--split_seed",  type=int,   default=42)
    p.add_argument("--seeds",       type=int,   nargs="+", default=SEEDS)
    p.add_argument("--n_eval",      type=int,   default=N_EVAL)
    p.add_argument("--threshold_km", type=float, default=10.0)
    p.add_argument("--variants",    nargs="+", default=VARIANTS_DEFAULT)
    p.add_argument("--out_csv",     default="results/headline_clean_5seed.csv")
    p.add_argument("--out_summary", default="results/headline_clean_5seed_summary.csv")
    args = p.parse_args()

    device = get_device()
    print(f"Device: {device}")
    print(f"Seeds: {args.seeds}    n_eval={args.n_eval}    "
          f"threshold_km={args.threshold_km}")
    print(f"Variants: {args.variants}\n")

    # ─── Load data and reproduce val split ─────────────────────────────
    print(f"Loading {args.data_npz}")
    data = np.load(args.data_npz, allow_pickle=True)
    trajectories = data["trajectories"].astype(np.float32)
    vessel_types = data["vessel_types"].astype(np.int32)
    track_ids    = data["track_ids"]
    train_traj, train_vt, _, val_traj, val_vt, _ = train_val_split_gen(
        trajectories, vessel_types, track_ids,
        args.val_frac, args.split_seed)
    T = val_traj.shape[1]
    print(f"  Train: {len(train_traj):,}    Val: {len(val_traj):,}    T={T}")

    # ─── Union of seed-subsamples ──────────────────────────────────────
    per_seed_idx = {s: subsample_indices(len(val_traj), args.n_eval, s)
                    for s in args.seeds}
    union = np.unique(np.concatenate([per_seed_idx[s] for s in args.seeds]))
    pos_in_union = {int(o): i for i, o in enumerate(union)}
    print(f"  Union of {len(args.seeds)} subsamples: {len(union)} unique val rows\n")
    union_traj = val_traj[union]
    union_vt   = val_vt[union]

    # ─── KNN cache (cleaned-corpus, val-queries-vs-train-corpus) ───────
    print(f"Loading KNN cache {args.knn_cache}")
    knn = np.load(args.knn_cache)
    val_knn = knn["val_knn"]
    union_knn = val_knn[union]

    # ─── SDF + water graph ─────────────────────────────────────────────
    print(f"Loading SDF {args.land_sdf}")
    land_mask = LandMask.load(args.land_sdf)
    router = None
    if "v10_router" in args.variants:
        print(f"Loading water graph {args.water_graph}")
        router = WaterRouter.load(args.water_graph, coarse_mask=land_mask)

    # ─── Generate predictions for every variant on the union ───────────
    preds: Dict[str, np.ndarray] = {}
    for variant in args.variants:
        print(f"\n─── {variant} ───")
        out = run_variant(variant, union_traj, union_vt, union_knn, train_traj,
                          T, device, land_mask, router, args.threshold_km)
        if out is not None:
            preds[variant] = out

    # ─── GT land/cross diagnostic (for the report) ─────────────────────
    gt_stats_10 = land_mask.trajectory_stats(union_traj, args.threshold_km)
    gt_stats_25 = land_mask.trajectory_stats(union_traj, 25.0)
    print(f"\nGT (union) land%@10km = {gt_stats_10['land_frac']*100:.4f}, "
          f"cross%@10km = {gt_stats_10['traj_crossing_rate']*100:.2f}")
    print(f"GT (union) land%@25km = {gt_stats_25['land_frac']*100:.4f}, "
          f"cross%@25km = {gt_stats_25['traj_crossing_rate']*100:.2f}")

    # ─── Per-(variant, seed) metrics ───────────────────────────────────
    fieldnames = [
        "variant", "seed", "n", "dataset", "split_seed",
        "ade_m", "fde_m", "normalized_ade", "path_length_ratio",
        "ade_short_m", "ade_medium_m", "ade_long_m",
        "n_short", "n_medium", "n_long",
        "land_pct_10", "cross_pct_10",
        "land_pct_25", "cross_pct_25",
        "max_pen_km_mean",
        "smoothness", "gt_smoothness",
    ]
    rows: List[Dict] = []
    print(f"\n{'='*78}\n  PER-SEED METRICS\n{'='*78}")
    print(f"{'variant':<14} {'seed':>5} {'ADE':>9} {'FDE':>9} "
          f"{'land%10':>8} {'cross%10':>9} {'land%25':>8}")

    for variant, pred_union in preds.items():
        for seed in args.seeds:
            idx_full = per_seed_idx[seed]
            local = np.array([pos_in_union[int(i)] for i in idx_full],
                             dtype=np.int64)
            gt    = union_traj[local]
            pred  = pred_union[local]
            m = evaluate_generation(pred, gt, compute_frechet=False)
            ls10 = land_mask.trajectory_stats(pred, args.threshold_km)
            ls25 = land_mask.trajectory_stats(pred, 25.0)
            row = {
                "variant": variant,
                "seed": seed,
                "n": int(m["n_trajectories"]),
                "dataset": os.path.basename(args.data_npz),
                "split_seed": args.split_seed,
                "ade_m": float(m["ade_m"]),
                "fde_m": float(m["fde_m"]),
                "normalized_ade": float(m["normalized_ade"]),
                "path_length_ratio": float(m["path_length_ratio"]),
                "ade_short_m":  float(m.get("ade_short_m",  float("nan"))),
                "ade_medium_m": float(m.get("ade_medium_m", float("nan"))),
                "ade_long_m":   float(m.get("ade_long_m",   float("nan"))),
                "n_short":  int(m.get("n_short",  0)),
                "n_medium": int(m.get("n_medium", 0)),
                "n_long":   int(m.get("n_long",   0)),
                "land_pct_10":  float(ls10["land_frac"] * 100.0),
                "cross_pct_10": float(ls10["traj_crossing_rate"] * 100.0),
                "land_pct_25":  float(ls25["land_frac"] * 100.0),
                "cross_pct_25": float(ls25["traj_crossing_rate"] * 100.0),
                "max_pen_km_mean": float(ls10["mean_max_penetration_km"]),
                "smoothness":    float(m["smoothness"]),
                "gt_smoothness": float(m["gt_smoothness"]),
            }
            rows.append(row)
            print(f"{variant:<14} {seed:>5d} "
                  f"{row['ade_m']:>9.1f} {row['fde_m']:>9.1f} "
                  f"{row['land_pct_10']:>7.2f}% {row['cross_pct_10']:>8.2f}% "
                  f"{row['land_pct_25']:>7.2f}%")

    # ─── Write per-seed CSV ────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {args.out_csv}  ({len(rows)} rows)")

    # ─── Aggregate summary ─────────────────────────────────────────────
    print(f"\n{'='*78}\n  AGGREGATE (mean ± std over seeds, population std ddof=0)\n{'='*78}")
    summary_rows = []
    cols_to_aggregate = [
        "ade_m", "fde_m", "normalized_ade", "path_length_ratio",
        "ade_short_m", "ade_medium_m", "ade_long_m",
        "land_pct_10", "cross_pct_10", "land_pct_25", "cross_pct_25",
        "max_pen_km_mean",
    ]
    summary_fields = ["variant", "n_seeds", "n_per_seed", "dataset"]
    for c in cols_to_aggregate:
        summary_fields += [f"{c}_mean", f"{c}_std"]

    print(f"{'variant':<14} {'ADE':>14} {'FDE':>12} "
          f"{'land%10':>10} {'cross%10':>10}")

    for variant in preds:
        seed_rows = [r for r in rows if r["variant"] == variant]
        summary = {
            "variant": variant,
            "n_seeds": len(seed_rows),
            "n_per_seed": args.n_eval,
            "dataset": os.path.basename(args.data_npz),
        }
        for c in cols_to_aggregate:
            vals = np.array([r[c] for r in seed_rows], dtype=np.float64)
            summary[f"{c}_mean"] = float(np.nanmean(vals))
            summary[f"{c}_std"]  = float(np.nanstd(vals, ddof=0))
        summary_rows.append(summary)
        print(f"{variant:<14} "
              f"{summary['ade_m_mean']:>7.1f} ± {summary['ade_m_std']:>4.1f} "
              f"{summary['fde_m_mean']:>6.1f} ± {summary['fde_m_std']:>3.1f}  "
              f"{summary['land_pct_10_mean']:>5.2f} ± {summary['land_pct_10_std']:>4.2f}  "
              f"{summary['cross_pct_10_mean']:>5.2f} ± {summary['cross_pct_10_std']:>4.2f}")

    with open(args.out_summary, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary_fields)
        w.writeheader()
        w.writerows(summary_rows)
    print(f"\nWrote {args.out_summary}  ({len(summary_rows)} variants)")


if __name__ == "__main__":
    main()
