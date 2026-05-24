#!/usr/bin/env python3
"""
eval_shared_protocol.py — Paper-track K-anchor unified benchmark.

Every method is scored on the **same K anchor trajectories** drawn from
the same MMSI-grouped val split, with identical (start, end, vessel_type,
retrieved_K) inputs. The output table is what the paper reports — one row
per (method × seed), aggregated with mean ± std across seeds.

Methods supported:
    v10_router    — v10 raw + WaterRouter snap (production baseline)
    v10_raw       — v10 raw, no projection
    v10_hardproj  — v10 raw + per-step BFS projection (older variant)
    retr_top1     — zero-training retrieval-top-1
    great_circle  — geodesic between start and end
    v13           — diffusion Transformer; spec via --v13_checkpoint
                    Each v13 entry can be parameterised with (guidance,
                    land_strength, n_steps) via --v13_configs.

The "anchor" idea: we deterministically subsample K=500 val rows per seed,
take the *union* of those rows, run each method **once** on the union, then
slice the union back per seed for scoring. Each method therefore sees
the identical inputs and we don't pay for re-inference per seed.

Usage (typical paper run):
    paper/.venv/bin/python3 scripts/paper/eval_shared_protocol.py \\
        --data_npz   data/processed/trajgen_128_clean.npz \\
        --knn_cache  data/processed/trajgen_128_clean_knn_k5.npz \\
        --land_sdf   data/processed/land_sdf_050deg.npz \\
        --water_graph data/processed/water_graph_005deg.npz \\
        --v10_checkpoint runs/trajgen_v10/best.pt \\
        --v13_checkpoint runs/paper/v13_trajdiff_base/best.pt \\
        --v13_configs "g=0,l=0;g=2,l=0;g=2,l=0.5" \\
        --methods v10_router v13 retr_top1 great_circle \\
        --seeds 0 1 2 42 123 \\
        --n_eval 500 \\
        --out_csv     results/paper/shared_protocol.csv \\
        --out_summary results/paper/shared_protocol_summary.csv
"""
from __future__ import annotations

import argparse
import copy
import csv
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# v2-v10 + GC + retrieval machinery from the audit-time evaluator. These
# are private helpers but stable enough to depend on for the paper-track.
from scripts.eval.eval_all_clean import (
    _apply_router,
    _gen_v10,
    get_device,
    load_v10,
    subsample_indices,
)
from src.data_gen import train_val_split_gen
from src.land_mask import LandMask
from src.metrics_gen import evaluate_generation, great_circle_trajectory
from src.water_router import WaterRouter

# v13 helpers — local to the paper track.
from src.data_gen import Scaler, TrajGenScalers
from src.data_gen_retrieval import RetrievalConfig, get_retrieval_trajgen_loader
from src.diffusion import EDMSchedule, edm_sample, make_endpoint_anchor
from src.metrics_gen import haversine_meters, path_length_m
from src.model_gen_v13 import TrajDiff, TrajDiffConfig, make_uncond


# ─────────────────────────────────────────────────────────────────────────
# K-anchor protocol helpers (see paper/notes/k_anchor_protocol.md)
# ─────────────────────────────────────────────────────────────────────────
def compute_anchor_indices(T: int, K: int) -> List[int]:
    """K evenly spaced indices in [0, T-1] inclusive. K=2 → [0, T-1]."""
    if K < 2:
        raise ValueError(f"K must be >= 2 (got {K})")
    if K > T:
        raise ValueError(f"K={K} > T={T}; cannot place that many anchors")
    if K == 2:
        return [0, T - 1]
    return [int(round(k * (T - 1) / (K - 1))) for k in range(K)]


def make_k_anchor_mask_from_gt(
    gt_norm: torch.Tensor, anchor_idx: List[int],
) -> "tuple[torch.Tensor, torch.Tensor]":
    """Build (anchor_mask: (B, T) bool, anchor_values: (B, T, 2)) from GT.

    Generalizes src.diffusion.make_endpoint_anchor — that helper only fixes
    positions 0 and T-1; this one fixes any set of indices.
    """
    B, T, D = gt_norm.shape
    mask = torch.zeros(B, T, device=gt_norm.device, dtype=torch.bool)
    values = torch.zeros_like(gt_norm)
    mask[:, anchor_idx] = True
    values[:, anchor_idx, :] = gt_norm[:, anchor_idx, :]
    return mask, values


def piecewise_great_circle(traj_gt: np.ndarray, anchor_idx: List[int], T: int) -> np.ndarray:
    """Piecewise GC interpolation through anchor positions of one trajectory.

    Each anchored segment from anchor_idx[i] to anchor_idx[i+1] is filled with
    a great-circle arc of (anchor_idx[i+1] - anchor_idx[i] + 1) points; the
    arc endpoints overwrite the previous segment's endpoint so no duplicates.
    """
    from src.metrics_gen import great_circle_trajectory
    out = np.zeros_like(traj_gt)
    for i in range(len(anchor_idx) - 1):
        a, b = anchor_idx[i], anchor_idx[i + 1]
        seg_len = b - a + 1
        arc = great_circle_trajectory(traj_gt[a], traj_gt[b], seg_len)
        out[a:b + 1] = arc
    return out


def score_k_anchor(pred: np.ndarray, gt: np.ndarray, anchor_idx: List[int]) -> dict:
    """ADE/FDE/normalized-ADE on non-anchor indices only.

    Land/cross statistics are computed on the FULL trajectory by the caller
    (they care about geographic validity along the whole path; anchors are
    GT and trivially water-valid for in-corpus eval).
    """
    T = pred.shape[1]
    non_anchor = [t for t in range(T) if t not in anchor_idx]
    if len(non_anchor) == 0:
        return {"ade_m": 0.0, "fde_m": 0.0, "normalized_ade": 0.0,
                "n_trajectories": len(pred), "path_length_ratio": 1.0,
                "smoothness": 0.0, "gt_smoothness": 0.0}
    err = haversine_meters(
        gt[:, non_anchor, 1], gt[:, non_anchor, 0],
        pred[:, non_anchor, 1], pred[:, non_anchor, 0],
    )                                            # (N, len(non_anchor))
    per_traj_ade = err.mean(axis=1)
    ade_m = float(err.mean())
    fde_m = float(err[:, -1].mean())             # last non-anchor index

    N = pred.shape[0]
    pred_lens = np.array([path_length_m(pred[i]) for i in range(N)])
    gt_lens   = np.array([path_length_m(gt[i])   for i in range(N)])
    valid = gt_lens > 1.0
    plr = float(np.median(pred_lens[valid] / gt_lens[valid])) if valid.any() else float("nan")
    norm_ade = float(np.mean(per_traj_ade[valid] / gt_lens[valid])) if valid.any() else float("nan")

    pred_accel = np.diff(np.diff(pred, axis=1), axis=1)
    gt_accel   = np.diff(np.diff(gt,   axis=1), axis=1)
    return {
        "ade_m": ade_m, "fde_m": fde_m, "normalized_ade": norm_ade,
        "n_trajectories": N, "path_length_ratio": plr,
        "smoothness":    float((pred_accel ** 2).mean()),
        "gt_smoothness": float((gt_accel   ** 2).mean()),
    }


# ─────────────────────────────────────────────────────────────────────────
# v13 inference (parallel to _gen_v10 in eval_all_clean.py)
# ─────────────────────────────────────────────────────────────────────────
def _load_v13(ckpt_path: str, device: torch.device):
    ck = torch.load(ckpt_path, map_location=device)
    cfg = ck["config"]
    mdl_cfg = TrajDiffConfig(
        seq_len=cfg["seq_len"], in_dim=cfg["in_dim"],
        n_vessel_types=cfg["n_vessel_types"], vessel_emb_dim=cfg["vessel_emb_dim"],
        d_model=cfg["d_model"], n_heads=cfg["n_heads"], ffn_mult=cfg["ffn_mult"],
        n_layer=cfg["n_layer"], dropout=cfg["dropout"], p_uncond=cfg["p_uncond"],
        max_retrieved=cfg["k_retrieval"], t_retr=cfg["t_retr"],
        time_emb_dim=cfg["time_emb_dim"],
    )
    model = TrajDiff(mdl_cfg).to(device).eval()
    # Prefer EMA weights for paper numbers.
    if "ema" in ck and ck["ema"]:
        sd = model.state_dict()
        for n, v in ck["ema"].items():
            if n in sd:
                sd[n].copy_(v)
        model.load_state_dict(sd)
    else:
        model.load_state_dict(ck["model"])

    schedule = EDMSchedule(
        sigma_min=cfg["sigma_min"], sigma_max=cfg["sigma_max"],
        sigma_data=cfg["sigma_data"], rho=cfg["rho"],
    )
    sc = ck["scalers"]
    scalers = TrajGenScalers(
        pos=Scaler(
            mean=np.asarray(sc["pos_mean"],   dtype=np.float32),
            std =np.asarray(sc["pos_std"],    dtype=np.float32),
        ),
        delta=Scaler(
            mean=np.asarray(sc["delta_mean"], dtype=np.float32),
            std =np.asarray(sc["delta_std"],  dtype=np.float32),
        ),
    )
    return model, schedule, scalers, ck.get("vtype_vocab"), cfg


def _v13_anchor_loader(union_traj, union_vt, train_traj, union_knn,
                       scalers, vtype_vocab, k_retrieval, vtype_weight, t_retr,
                       batch_size):
    """Build a non-shuffled retrieval loader over the union of anchors.

    Mimics what train_gen_v13 builds — necessary so v13 sees the same
    retrieved-route tokens it was trained with.
    """
    retr_cfg = RetrievalConfig(k_retrieval=k_retrieval, vtype_weight=vtype_weight,
                               exclude_self=False)
    return get_retrieval_trajgen_loader(
        trajectories=union_traj, vessel_types=union_vt,
        corpus_trajectories=train_traj, knn_indices=union_knn,
        vtype_vocab=vtype_vocab, scalers=scalers, config=retr_cfg,
        batch_size=batch_size, shuffle=False, drop_last=False,
        num_workers=0, pin_memory=True, persistent_workers=False,
    )


def _make_land_guidance(land_mask: LandMask, pos_mean_t, pos_std_t,
                        strength: float, threshold_km: float):
    """Closure returning the optional sampling-time score correction."""
    if strength == 0.0:
        return None

    def guidance(x_norm: torch.Tensor, sigma):
        # Re-enable autograd locally; edm_sample is @torch.no_grad().
        # Also normalize the gradient to a unit direction so `strength`
        # has the interpretable units of km-of-water-movement per step
        # (see eval_gen_v13.py for full justification).
        with torch.enable_grad():
            x_deg = (x_norm.detach() * pos_std_t + pos_mean_t).detach().clone().requires_grad_(True)
            sdf = land_mask.sample_km_torch(x_deg[..., 0], x_deg[..., 1])
            penalty = torch.relu(sdf - threshold_km).pow(2).sum()
            (grad,) = torch.autograd.grad(penalty, x_deg)
        grad_mag = grad.pow(2).sum(dim=-1, keepdim=True).clamp_min(1e-8).sqrt()
        direction_deg = grad / grad_mag
        active = (grad_mag > 0).to(direction_deg.dtype)
        km_per_deg = 111.0
        corr_deg = direction_deg * (strength / km_per_deg) * active
        return x_norm - corr_deg / pos_std_t

    return guidance


@torch.no_grad()
def _gen_v13(model, schedule, scalers, loader, *,
             n_steps: int, guidance: float, land_strength: float,
             land_mask: Optional[LandMask], land_threshold_km: float,
             device: torch.device,
             anchor_idx: Optional[List[int]] = None) -> np.ndarray:
    """Sample full trajectories for every batch in `loader`. Returns (N, T, 2) deg.

    anchor_idx: K indices in [0, T-1] that are clamped to GT throughout
    denoising. None or [0, T-1] = default v10-style endpoint conditioning.
    For general K (paper/notes/k_anchor_protocol.md), pass the indices
    produced by `compute_anchor_indices(T, K)`.
    """
    pos_mean_t = torch.tensor(scalers.pos.mean, device=device)
    pos_std_t  = torch.tensor(scalers.pos.std,  device=device)

    land_guidance = (
        _make_land_guidance(land_mask, pos_mean_t, pos_std_t,
                            land_strength, land_threshold_km)
        if (land_mask is not None and land_strength != 0.0) else None
    )

    preds = []
    for batch in loader:
        start = batch["start_norm"].to(device)
        end   = batch["end_norm"].to(device)
        vtype = batch["vessel_type"].to(device)
        retr  = batch["retrieved_norm"].to(device)
        x_gt  = batch["traj_norm"].permute(1, 0, 2).to(device)   # (B, T, 2) norm
        B = start.shape[0]
        T = x_gt.shape[1]

        cond = {"start": start, "end": end, "vessel_type": vtype, "retrieved": retr}
        if anchor_idx is None or anchor_idx == [0, T - 1]:
            anchor_mask, anchor_values = make_endpoint_anchor(start, end, T, device=device)
        else:
            anchor_mask, anchor_values = make_k_anchor_mask_from_gt(x_gt, anchor_idx)
        uncond = make_uncond(cond, device) if guidance != 0.0 else None

        if land_guidance is not None:
            with torch.enable_grad():
                x_norm = edm_sample(
                    model, (B, T, 2), cond, schedule,
                    n_steps=n_steps, device=device,
                    anchor_mask=anchor_mask, anchor_values=anchor_values,
                    guidance_scale=guidance, uncond_cond=uncond,
                    score_guidance=land_guidance,
                )
        else:
            x_norm = edm_sample(
                model, (B, T, 2), cond, schedule,
                n_steps=n_steps, device=device,
                anchor_mask=anchor_mask, anchor_values=anchor_values,
                guidance_scale=guidance, uncond_cond=uncond,
            )
        x_deg = (x_norm * pos_std_t + pos_mean_t).cpu().numpy()
        preds.append(x_deg)

    return np.concatenate(preds, 0)


# ─────────────────────────────────────────────────────────────────────────
# Method dispatch
# ─────────────────────────────────────────────────────────────────────────
def parse_v13_configs(spec: Optional[str]) -> List[dict]:
    """Parse "g=0,l=0,n=50;g=2,l=0.5,n=50" into a list of dicts."""
    if not spec:
        return [{"guidance": 0.0, "land_strength": 0.0, "n_steps": 50}]
    out = []
    for chunk in spec.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        d = {"guidance": 0.0, "land_strength": 0.0, "n_steps": 50}
        for kv in chunk.split(","):
            k, v = kv.split("=")
            k = k.strip()
            v = float(v.strip())
            if k == "g":   d["guidance"]      = v
            elif k == "l": d["land_strength"] = v
            elif k == "n": d["n_steps"]       = int(v)
            else:          raise ValueError(f"unknown v13 config key {k!r}")
        out.append(d)
    return out


def run_method(name: str,
               union_traj, union_vt, union_knn, train_traj, train_vt,
               T: int, device: torch.device,
               land_mask: LandMask, router: Optional[WaterRouter],
               threshold_km: float,
               v10_ckpt: Optional[str],
               v13_state: Optional[dict],
               v13_cfg: Optional[dict],
               anchor_idx: Optional[List[int]] = None) -> Optional[np.ndarray]:
    """Return (N, T, 2) predictions, or None on load failure.

    anchor_idx, if provided, places K anchors at those GT positions:
    - v13:          all K anchors clamped during EDM sampling.
    - great_circle: piecewise GC between consecutive anchors.
    - v10_*, retr_top1: the model/retrieval does NOT see the middle anchors
                        (would require retraining). Predictions are full
                        trajectories; scoring at non-anchor indices reveals
                        the upper bound of what these K=2-trained methods can
                        deliver at K>2. Honest baseline.
    """
    t0 = time.time()
    try:
        if name in ("v10_raw", "v10_hardproj", "v10_router"):
            if v10_ckpt is None:
                print(f"  [{name}] no --v10_checkpoint provided; skipping")
                return None
            model, scalers, vocab, T_ck, _ = load_v10(v10_ckpt, device)
            assert T_ck == T, f"v10 checkpoint T={T_ck} != data T={T}"
            lm = land_mask if name == "v10_hardproj" else None
            pred = _gen_v10(model, scalers, vocab, union_traj, union_vt,
                            train_traj, union_knn, T, device,
                            land_mask=lm, hard_threshold_km=threshold_km)
            if name == "v10_router":
                pred = _apply_router(pred, router, threshold_km)
        elif name == "retr_top1":
            pred = train_traj[union_knn[:, 0]].astype(np.float32)
        elif name == "great_circle":
            pred = np.zeros_like(union_traj, dtype=np.float32)
            if anchor_idx is None or anchor_idx == [0, T - 1]:
                for i in range(len(union_traj)):
                    pred[i] = great_circle_trajectory(
                        union_traj[i, 0], union_traj[i, -1], T)
            else:
                for i in range(len(union_traj)):
                    pred[i] = piecewise_great_circle(union_traj[i], anchor_idx, T)
        elif name == "v13":
            assert v13_state is not None and v13_cfg is not None
            model, schedule, scalers, vtype_vocab, train_cfg = v13_state
            loader = _v13_anchor_loader(
                union_traj, union_vt, train_traj, union_knn,
                scalers, vtype_vocab,
                k_retrieval=train_cfg["k_retrieval"],
                vtype_weight=train_cfg["retrieval_vtype_weight"],
                t_retr=train_cfg["t_retr"],
                batch_size=32,
            )
            pred = _gen_v13(model, schedule, scalers, loader,
                            n_steps=v13_cfg["n_steps"],
                            guidance=v13_cfg["guidance"],
                            land_strength=v13_cfg["land_strength"],
                            land_mask=land_mask,
                            land_threshold_km=threshold_km,
                            device=device,
                            anchor_idx=anchor_idx)
        else:
            print(f"[WARN] unknown method {name!r}; skipping")
            return None
        print(f"  [{name}] generated {len(pred)} traj in {time.time()-t0:.1f}s")
        return pred.astype(np.float32)
    except Exception as exc:
        print(f"[WARN] {name} failed: {exc.__class__.__name__}: {exc}")
        traceback.print_exc()
        return None


# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--data_npz",    default="data/processed/trajgen_128_clean.npz")
    ap.add_argument("--knn_cache",   default="data/processed/trajgen_128_clean_knn_k5.npz")
    ap.add_argument("--land_sdf",    default="data/processed/land_sdf_050deg.npz")
    ap.add_argument("--water_graph", default="data/processed/water_graph_005deg.npz")
    ap.add_argument("--val_frac",    type=float, default=0.15)
    ap.add_argument("--split_seed",  type=int,   default=42)
    ap.add_argument("--seeds",       type=int,   nargs="+", default=[0, 1, 2, 42, 123])
    ap.add_argument("--n_eval",      type=int,   default=500)
    ap.add_argument("--threshold_km", type=float, default=10.0)
    ap.add_argument("--methods", nargs="+",
                    default=["v10_router", "v13", "retr_top1", "great_circle"])
    ap.add_argument("--v10_checkpoint", default="runs/trajgen_v10/best.pt")
    ap.add_argument("--v13_checkpoint", default=None,
                    help="Path to v13 best.pt. Required if 'v13' is in --methods.")
    ap.add_argument("--v13_configs", default=None,
                    help="Semicolon-separated v13 configs: "
                         "'g=0,l=0,n=50;g=2,l=0.5,n=50'. Each becomes a "
                         "separate row labelled v13_g0_l0_n50, etc.")
    ap.add_argument("--out_csv",     default="results/paper/shared_protocol.csv")
    ap.add_argument("--out_summary", default="results/paper/shared_protocol_summary.csv")
    ap.add_argument("--k_anchor", type=int, default=2,
                    help="K-anchor protocol (paper/notes/k_anchor_protocol.md). "
                         "K=2 (default) is endpoint-conditioned. K=18 approximates "
                         "TrAISformer's setting. ADE/FDE are scored on non-anchor "
                         "indices only; land/cross stats use the full trajectory.")
    args = ap.parse_args()

    device = get_device()
    print(f"Device: {device}")
    print(f"Seeds: {args.seeds}    n_eval={args.n_eval}    "
          f"threshold_km={args.threshold_km}")
    print(f"Methods: {args.methods}\n")

    # ─── Data + val split ──────────────────────────────────────────────
    data = np.load(args.data_npz, allow_pickle=True)
    trajectories = data["trajectories"].astype(np.float32)
    vessel_types = data["vessel_types"].astype(np.int32)
    track_ids    = data["track_ids"]
    train_traj, train_vt, _, val_traj, val_vt, _ = train_val_split_gen(
        trajectories, vessel_types, track_ids, args.val_frac, args.split_seed)
    T = val_traj.shape[1]
    anchor_idx = compute_anchor_indices(T, args.k_anchor)
    print(f"Train: {len(train_traj):,}    Val: {len(val_traj):,}    T={T}")
    print(f"K-anchor protocol: K={args.k_anchor}, indices={anchor_idx}")

    # ─── Anchor union ──────────────────────────────────────────────────
    per_seed_idx = {s: subsample_indices(len(val_traj), args.n_eval, s)
                    for s in args.seeds}
    union = np.unique(np.concatenate([per_seed_idx[s] for s in args.seeds]))
    pos_in_union = {int(o): i for i, o in enumerate(union)}
    print(f"Union of {len(args.seeds)} subsamples: {len(union)} unique anchors\n")
    union_traj = val_traj[union]
    union_vt   = val_vt[union]

    knn = np.load(args.knn_cache)
    val_knn = knn["val_knn"]
    union_knn = val_knn[union]

    # ─── SDF + router ──────────────────────────────────────────────────
    land_mask = LandMask.load(args.land_sdf)
    router = None
    if "v10_router" in args.methods:
        router = WaterRouter.load(args.water_graph, coarse_mask=land_mask)
    # v13 land guidance uses torch; warm up the cache on the right device.
    if any(m == "v13" for m in args.methods):
        land_mask.as_torch(device=device)

    # ─── v13 setup ─────────────────────────────────────────────────────
    v13_state = None
    v13_configs = parse_v13_configs(args.v13_configs)
    if "v13" in args.methods:
        if not args.v13_checkpoint:
            print("[ERROR] --v13_checkpoint required when 'v13' is in --methods.")
            sys.exit(2)
        v13_state = _load_v13(args.v13_checkpoint, device)
        print(f"Loaded v13 from {args.v13_checkpoint}")
        print(f"v13 sweep configs ({len(v13_configs)}):")
        for c in v13_configs:
            print(f"  g={c['guidance']:.2f} l={c['land_strength']:.3f} n={c['n_steps']}")

    # ─── Generate predictions for every method on the union ────────────
    preds: Dict[str, np.ndarray] = {}
    for method in args.methods:
        print(f"\n─── {method} ───")
        if method == "v13":
            for cfg in v13_configs:
                label = f"v13_g{cfg['guidance']:g}_l{cfg['land_strength']:g}_n{cfg['n_steps']}"
                out = run_method(method, union_traj, union_vt, union_knn,
                                 train_traj, train_vt, T, device,
                                 land_mask, router, args.threshold_km,
                                 v10_ckpt=args.v10_checkpoint,
                                 v13_state=v13_state, v13_cfg=cfg,
                                 anchor_idx=anchor_idx)
                if out is not None:
                    preds[label] = out
        else:
            out = run_method(method, union_traj, union_vt, union_knn,
                             train_traj, train_vt, T, device,
                             land_mask, router, args.threshold_km,
                             v10_ckpt=args.v10_checkpoint,
                             v13_state=None, v13_cfg=None,
                             anchor_idx=anchor_idx)
            if out is not None:
                preds[method] = out

    # ─── GT diagnostic ─────────────────────────────────────────────────
    gt_stats = land_mask.trajectory_stats(union_traj, args.threshold_km)
    print(f"\nGT (union, n={len(union_traj)}): "
          f"land%@{args.threshold_km:g}km = {gt_stats['land_frac']*100:.4f}, "
          f"cross%@{args.threshold_km:g}km = {gt_stats['traj_crossing_rate']*100:.2f}")

    # ─── Per-(method, seed) metrics ────────────────────────────────────
    fieldnames = [
        "method", "seed", "n", "dataset", "split_seed",
        "ade_m", "fde_m", "normalized_ade", "path_length_ratio",
        "ade_short_m", "ade_medium_m", "ade_long_m",
        "n_short", "n_medium", "n_long",
        "land_pct_10", "cross_pct_10",
        "max_pen_km_mean",
        "smoothness", "gt_smoothness",
    ]
    rows: List[Dict] = []
    print(f"\n{'='*84}\n  PER-SEED METRICS  (n_eval={args.n_eval}, K={args.k_anchor}, "
          f"threshold={args.threshold_km:g} km)\n{'='*84}")
    print(f"{'method':<28} {'seed':>5} {'ADE':>9} {'FDE':>9} {'land%':>7} {'cross%':>7}")

    for label, pred_union in preds.items():
        for seed in args.seeds:
            idx_full = per_seed_idx[seed]
            local = np.array([pos_in_union[int(i)] for i in idx_full], dtype=np.int64)
            gt = union_traj[local]
            pr = pred_union[local]
            if args.k_anchor == 2:
                m  = evaluate_generation(pr, gt, compute_frechet=False)
            else:
                # For K > 2, use the K-anchor scorer (ADE on non-anchor indices only).
                # Length-bucketed ADE is not meaningful per-bucket here so we drop it
                # and the downstream CSV will get NaN for those columns.
                m = score_k_anchor(pr, gt, anchor_idx)
            ls = land_mask.trajectory_stats(pr, args.threshold_km)
            row = {
                "method": label,
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
                "land_pct_10":  float(ls["land_frac"] * 100.0),
                "cross_pct_10": float(ls["traj_crossing_rate"] * 100.0),
                "max_pen_km_mean": float(ls["mean_max_penetration_km"]),
                "smoothness":    float(m["smoothness"]),
                "gt_smoothness": float(m["gt_smoothness"]),
            }
            rows.append(row)
            print(f"{label:<28} {seed:>5d} "
                  f"{row['ade_m']:>9.1f} {row['fde_m']:>9.1f} "
                  f"{row['land_pct_10']:>6.2f}% {row['cross_pct_10']:>6.2f}%")

    # ─── Write per-seed CSV ────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {args.out_csv}  ({len(rows)} rows)")

    # ─── Aggregate (mean ± std across seeds, ddof=0) ───────────────────
    print(f"\n{'='*84}\n  AGGREGATE (mean ± std over seeds)\n{'='*84}")
    cols = ["ade_m", "fde_m", "normalized_ade",
            "ade_short_m", "ade_medium_m", "ade_long_m",
            "land_pct_10", "cross_pct_10", "max_pen_km_mean"]
    summary_rows = []
    summary_fields = ["method", "n_seeds"] + sum(
        [[f"{c}_mean", f"{c}_std"] for c in cols], [])
    print(f"{'method':<28} {'n_seeds':>7} {'ADE_mean':>9} {'±std':>7} "
          f"{'land%_mean':>10} {'cross%_mean':>11}")
    for label in preds.keys():
        method_rows = [r for r in rows if r["method"] == label]
        agg = {"method": label, "n_seeds": len(method_rows)}
        for c in cols:
            vals = np.asarray([r[c] for r in method_rows], dtype=np.float64)
            agg[f"{c}_mean"] = float(np.nanmean(vals))
            agg[f"{c}_std"]  = float(np.nanstd(vals, ddof=0))
        summary_rows.append(agg)
        print(f"{label:<28} {agg['n_seeds']:>7d} "
              f"{agg['ade_m_mean']:>9.1f} {agg['ade_m_std']:>7.1f} "
              f"{agg['land_pct_10_mean']:>9.2f}% {agg['cross_pct_10_mean']:>10.2f}%")

    os.makedirs(os.path.dirname(args.out_summary) or ".", exist_ok=True)
    with open(args.out_summary, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary_fields)
        w.writeheader()
        w.writerows(summary_rows)
    print(f"\nWrote {args.out_summary}  ({len(summary_rows)} rows)")


if __name__ == "__main__":
    main()
