#!/usr/bin/env python3
"""
eval_shared_protocol_dma.py — K-anchor sweep on TrAISformer's Kattegat DMA.

This is the DMA-specific sibling of eval_shared_protocol.py. It scores every
method on the **same** TrAISformer test trajectories (1,593 tracks from
`ct_dma_test.pkl`, converted via `dma_pkl_to_npz.py`), at varying K.

The differences from the US version:
  - Eval queries = `data/processed/dma_128_test.npz` (TrAISformer's test
    split). No land mask, no water router (Kattegat ROI isn't covered by
    the US-bbox SDF).
  - Retrieval corpus = train split of `dma_128_traineval.npz` (same data
    v10-DMA was trained on, matching its inference-time behaviour).
  - Adds method: `traisformer` — wraps the upstream sampler. Uses each
    track's first `init_seqlen=18` GT points as history (TrAISformer's
    native conditioning) and rolls out the remaining 110 steps. The
    K-anchor scorer evaluates only at non-anchor indices, so intermediate
    anchors that TrAISformer doesn't see don't penalise it.

Usage:
    paper/.venv/bin/python3 scripts/paper/eval_shared_protocol_dma.py \\
        --v10_dma_checkpoint runs/paper/v10_dma/best.pt \\
        --traisformer_checkpoint paper/external/TrAISformer/results/<name>/model.pt \\
        --methods v10_dma retr_top1 great_circle traisformer \\
        --k_anchor 2 \\
        --out_csv     results/paper/dma_k_anchor_K2.csv \\
        --out_summary results/paper/dma_k_anchor_K2_summary.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# Helpers shared with the US evaluator.
from scripts.paper.eval_shared_protocol import (
    compute_anchor_indices,
    score_k_anchor,
    piecewise_great_circle,
)
from scripts.eval.eval_all_clean import (
    _gen_v10, get_device, load_v10, subsample_indices,
)
from src.metrics_gen import (
    evaluate_generation, great_circle_trajectory, haversine_meters, path_length_m,
)


# ─────────────────────────────────────────────────────────────────────────
# TrAISformer dispatch (Kattegat-trained checkpoint)
# ─────────────────────────────────────────────────────────────────────────
def _arc_length_resample(lon: np.ndarray, lat: np.ndarray, n_points: int) -> np.ndarray:
    """Arc-length resample a (lon, lat) polyline to n_points. Returns (n_points, 2)."""
    dlon = np.diff(lon); dlat = np.diff(lat)
    step = np.sqrt(dlon ** 2 + dlat ** 2)
    cum = np.concatenate([[0.0], np.cumsum(step)])
    total = cum[-1]
    if total < 1e-10:
        return np.stack([np.full(n_points, lon[0]), np.full(n_points, lat[0])], axis=-1)
    target = np.linspace(0.0, total, n_points)
    return np.stack([np.interp(target, cum, lon), np.interp(target, cum, lat)], axis=-1)


def _gen_traisformer(
    test_pickle_entries: list,    # list of dicts {"mmsi": int, "traj": (L, 6) float64}
    ckpt_path: str,
    device: torch.device,
    T: int,                       # target output length (128, arc-length)
    batch_size: int = 64,         # GPU memory headroom on shared hosts
    n_samples: int = 1,           # n_samples > 1 → per-timestep best-of-N
) -> np.ndarray:
    """Roll out TrAISformer in its native time-uniform parameterisation, then
    arc-length-resample the predicted polyline to T points for comparison
    against the arc-length GT.

    Per track:
        history          = first init_seqlen=18 time-uniform points (REAL
                           SOG/COG/timestamps from the original pickle).
        future_len       = original_length - init_seqlen  (the actual remaining
                           time-uniform steps the track lasted).
        full prediction  = first-18 prefix + predicted future_len positions
                           = (original_length, 2) time-uniform polyline.
        output           = arc-length-resampled to T points to match GT
                           parameterisation.

    n_samples > 1 replicates TrAISformer's published best-of-N protocol
    (trAISformer.py:140-156): draw N independent samples per track, then
    for each (track, time-uniform timestep) pick the sample with minimum
    haversine distance to the pickle's time-uniform GT at that timestep.
    The resulting "envelope" trajectory is what their Fig. 4 reports.

    This avoids the time-vs-arc-length parameterisation mismatch that
    catastrophically inflates ADE if TrAISformer's raw output is compared
    position-by-position against arc-length GT.
    """
    trais_dir = REPO_ROOT / "paper" / "external" / "TrAISformer"
    ckpt_abs = str(Path(ckpt_path).resolve())
    sys.path.insert(0, str(trais_dir))
    cwd_before = os.getcwd()
    os.chdir(str(trais_dir))
    try:
        from config_trAISformer import Config
        import models
        import trainers

        cf = Config()
        model = models.TrAISformer(cf, partition_model=None).to(device)
        model.load_state_dict(torch.load(ckpt_abs, map_location=device))
        model.eval()

        init_seqlen = cf.init_seqlen
        N = len(test_pickle_entries)
        # Each track has a different rollout length; bucket-process by
        # future_len to amortise model forwarding while keeping it simple.
        per_track_future = np.array([
            max(1, len(e["traj"]) - init_seqlen) for e in test_pickle_entries
        ], dtype=np.int32)
        max_future = int(per_track_future.max())

        # Build histories in TrAISformer's native normalised format. The pickle
        # entries are already normalised to the DMA bbox, so we feed columns
        # [lat_norm, lon_norm, sog_norm, cog_norm] directly.
        hist = np.zeros((N, init_seqlen, 4), dtype=np.float32)
        for n, entry in enumerate(test_pickle_entries):
            arr = entry["traj"][:init_seqlen, :4].astype(np.float32)
            # Pad if shorter than init_seqlen (rare; DMA min is 24).
            L = arr.shape[0]
            hist[n, :L] = arr
            if L < init_seqlen:
                hist[n, L:] = arr[-1]  # repeat last known
        np.clip(hist, 0, 0.9999, out=hist)

        # Batch the AR rollout — the full N x init_seqlen x model activations
        # OOMs on a shared GPU. Predict batch_size tracks at a time, n_samples
        # times. Effective batch per call = batch_size / n_samples to keep
        # peak memory comparable across n_samples values.
        eff_bs = max(1, batch_size // max(1, n_samples))
        # samples_norm[s] holds the s-th sample for every track:
        # shape (N, init_seqlen + max_future, 4)
        samples_norm = []
        with torch.no_grad():
            for sample_i in range(n_samples):
                pn_chunks = []
                for s in range(0, N, eff_bs):
                    e = min(s + eff_bs, N)
                    init_t = torch.tensor(hist[s:e], device=device)
                    preds_norm = trainers.sample(
                        model, init_t, max_future, temperature=1.0,
                        sample=True,
                        sample_mode=cf.sample_mode, r_vicinity=cf.r_vicinity,
                        top_k=cf.top_k,
                    )                              # (B, init_seqlen + max_future, 4)
                    pn_chunks.append(preds_norm.cpu().numpy())
                    del preds_norm
                    torch.cuda.empty_cache()
                samples_norm.append(np.concatenate(pn_chunks, axis=0))
        # Stack: (n_samples, N, init_seqlen + max_future, 4)
        samples_norm = np.stack(samples_norm, axis=0)
    finally:
        os.chdir(cwd_before)

    # De-normalise lat/lon to degrees.
    lat_deg = samples_norm[..., 0] * (cf.lat_max - cf.lat_min) + cf.lat_min
    lon_deg = samples_norm[..., 1] * (cf.lon_max - cf.lon_min) + cf.lon_min
    # Shape: (n_samples, N, init_seqlen + max_future)

    out = np.zeros((N, T, 2), dtype=np.float32)
    if n_samples == 1:
        # Fast path: one sample per track, just arc-length resample.
        for n in range(N):
            L = init_seqlen + int(per_track_future[n])
            out[n] = _arc_length_resample(lon_deg[0, n, :L], lat_deg[0, n, :L], T)
    else:
        # Per-timestep best-of-N (TrAISformer's published Fig. 4 protocol).
        # For each (track, time-uniform timestep), keep the sample whose
        # predicted (lat, lon) at that timestep is closest to the pickle GT
        # at the same timestep.
        from src.metrics_gen import haversine_meters
        for n in range(N):
            L = init_seqlen + int(per_track_future[n])
            entry = test_pickle_entries[n]
            gt_arr = entry["traj"][:L, :2].astype(np.float64)
            gt_lat = gt_arr[:, 0] * (cf.lat_max - cf.lat_min) + cf.lat_min
            gt_lon = gt_arr[:, 1] * (cf.lon_max - cf.lon_min) + cf.lon_min
            # (n_samples, L) distance from each sample to GT at each timestep
            d = haversine_meters(
                np.broadcast_to(gt_lat[None, :], (n_samples, L)),
                np.broadcast_to(gt_lon[None, :], (n_samples, L)),
                lat_deg[:, n, :L], lon_deg[:, n, :L],
            )
            best_sample_per_t = d.argmin(axis=0)                 # (L,)
            # Build envelope trajectory: best sample's (lon, lat) at each t.
            sel_lon = lon_deg[best_sample_per_t, n, np.arange(L)]
            sel_lat = lat_deg[best_sample_per_t, n, np.arange(L)]
            # The first init_seqlen positions are deterministic (history copy)
            # but for sanity, overwrite with GT to remove float noise.
            sel_lon[:init_seqlen] = gt_lon[:init_seqlen]
            sel_lat[:init_seqlen] = gt_lat[:init_seqlen]
            out[n] = _arc_length_resample(sel_lon, sel_lat, T)
    return out


def run_method(name, union_traj, union_vt, union_knn, train_traj, train_vt,
               T, device, anchor_idx, v10_dma_ckpt, traisformer_ckpt,
               test_pickle_subset=None, traisformer_n_samples=1):
    t0 = time.time()
    try:
        if name == "v10_dma":
            if v10_dma_ckpt is None:
                print(f"  [{name}] no --v10_dma_checkpoint; skipping")
                return None
            model, scalers, vocab, T_ck, _ = load_v10(v10_dma_ckpt, device)
            assert T_ck == T, f"v10 ckpt T={T_ck} != data T={T}"
            pred = _gen_v10(model, scalers, vocab, union_traj, union_vt,
                            train_traj, union_knn, T, device,
                            land_mask=None, hard_threshold_km=10.0)
        elif name == "retr_top1":
            pred = train_traj[union_knn[:, 0]].astype(np.float32)
        elif name == "great_circle":
            pred = np.zeros_like(union_traj, dtype=np.float32)
            if anchor_idx == [0, T - 1]:
                for i in range(len(union_traj)):
                    pred[i] = great_circle_trajectory(
                        union_traj[i, 0], union_traj[i, -1], T)
            else:
                for i in range(len(union_traj)):
                    pred[i] = piecewise_great_circle(union_traj[i], anchor_idx, T)
        elif name == "traisformer":
            if traisformer_ckpt is None:
                print(f"  [{name}] no --traisformer_checkpoint; skipping")
                return None
            if test_pickle_subset is None:
                print(f"  [{name}] requires test_pickle_subset (time-uniform "
                      f"input); skipping")
                return None
            pred = _gen_traisformer(test_pickle_subset, traisformer_ckpt,
                                    device, T, n_samples=traisformer_n_samples)
        else:
            print(f"[WARN] unknown method {name!r}")
            return None
        print(f"  [{name}] generated {len(pred)} traj in {time.time()-t0:.1f}s")
        return pred.astype(np.float32)
    except Exception as exc:
        print(f"[WARN] {name} failed: {exc.__class__.__name__}: {exc}")
        traceback.print_exc()
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_npz",      default="data/processed/dma_128_test.npz")
    ap.add_argument("--traineval_npz", default="data/processed/dma_128_traineval.npz")
    ap.add_argument("--test_knn",      default="data/processed/dma_128_test_knn_to_traineval.npz")
    ap.add_argument("--split_seed",  type=int, default=42,
                    help="MMSI-grouped split inside traineval (extracts the "
                         "train corpus used for retrieval).")
    ap.add_argument("--val_frac",    type=float, default=0.15)
    ap.add_argument("--seeds",       type=int, nargs="+", default=[0, 1, 2, 42, 123])
    ap.add_argument("--n_eval",      type=int, default=500)
    ap.add_argument("--methods", nargs="+",
                    default=["v10_dma", "retr_top1", "great_circle", "traisformer"])
    ap.add_argument("--v10_dma_checkpoint", default="runs/paper/v10_dma/best.pt")
    ap.add_argument("--traisformer_checkpoint", default=None)
    ap.add_argument("--traisformer_n_samples", type=int, default=1,
                    help="N > 1 triggers TrAISformer's published best-of-N "
                         "per-timestep-min protocol (Fig. 4 uses N=16).")
    ap.add_argument("--k_anchor", type=int, default=2)
    ap.add_argument("--out_csv",     required=True)
    ap.add_argument("--out_summary", required=True)
    args = ap.parse_args()

    device = get_device()
    print(f"Device: {device}")
    print(f"K-anchor: K={args.k_anchor}, seeds={args.seeds}, n_eval={args.n_eval}")
    print(f"Methods: {args.methods}\n")

    # ── Test set (queries) ─────────────────────────────────────────────
    test = np.load(args.test_npz, allow_pickle=True)
    test_traj = test["trajectories"].astype(np.float32)
    test_vt   = test["vessel_types"].astype(np.int32)
    T = test_traj.shape[1]
    anchor_idx = compute_anchor_indices(T, args.k_anchor)
    print(f"Test set: {len(test_traj)} tracks, T={T}, anchor_idx={anchor_idx[:5]}"
          f"{'...' if len(anchor_idx) > 5 else ''}")

    # ── Train corpus for retrieval = the same train split v10-DMA saw ──
    from src.data_gen import train_val_split_gen
    te = np.load(args.traineval_npz, allow_pickle=True)
    train_traj, train_vt, _, _, _, _ = train_val_split_gen(
        te["trajectories"].astype(np.float32),
        te["vessel_types"].astype(np.int32),
        te["track_ids"], args.val_frac, args.split_seed)
    print(f"Train corpus: {len(train_traj)} tracks (matches v10-DMA train split)")

    # ── KNN cache (test → train_corpus) ────────────────────────────────
    knn = np.load(args.test_knn)
    test_knn = knn["val_knn"]                                 # key name reused
    assert test_knn.shape[0] == len(test_traj), \
        f"KNN cache rows {test_knn.shape[0]} != test set {len(test_traj)}"
    assert test_knn.max() < len(train_traj), \
        f"KNN index {test_knn.max()} >= corpus {len(train_traj)}"

    # ── Anchor union over seeds ────────────────────────────────────────
    per_seed_idx = {s: subsample_indices(len(test_traj), args.n_eval, s)
                    for s in args.seeds}
    union = np.unique(np.concatenate([per_seed_idx[s] for s in args.seeds]))
    pos_in_union = {int(o): i for i, o in enumerate(union)}
    union_traj = test_traj[union]
    union_vt   = test_vt[union]
    union_knn  = test_knn[union]
    print(f"Union of {len(args.seeds)} subsamples: {len(union)} unique tracks\n")

    # ── For TrAISformer: load native time-uniform pickle entries ───────
    test_pickle_subset = None
    if "traisformer" in args.methods:
        import pickle
        pkl_path = (REPO_ROOT / "paper" / "external" / "TrAISformer"
                    / "data" / "ct_dma" / "ct_dma_test.pkl")
        with open(pkl_path, "rb") as f:
            full_test_pickle = pickle.load(f)
        assert len(full_test_pickle) == len(test_traj), (
            f"test pickle {len(full_test_pickle)} != test NPZ {len(test_traj)} "
            "— the seg_idx ordering in dma_pkl_to_npz.py must match the pickle.")
        test_pickle_subset = [full_test_pickle[int(i)] for i in union]
        print(f"Loaded {len(test_pickle_subset)} test pickle entries for "
              f"TrAISformer (time-uniform input).")

    # ── Generate ───────────────────────────────────────────────────────
    preds: Dict[str, np.ndarray] = {}
    for method in args.methods:
        print(f"─── {method} ───")
        out = run_method(method, union_traj, union_vt, union_knn,
                         train_traj, train_vt, T, device, anchor_idx,
                         v10_dma_ckpt=args.v10_dma_checkpoint,
                         traisformer_ckpt=args.traisformer_checkpoint,
                         test_pickle_subset=test_pickle_subset,
                         traisformer_n_samples=args.traisformer_n_samples)
        if out is not None:
            preds[method] = out

    # ── Per-seed metrics ────────────────────────────────────────────────
    fieldnames = ["method", "seed", "n", "dataset", "k_anchor",
                  "ade_m", "fde_m", "normalized_ade", "path_length_ratio"]
    rows: List[Dict] = []
    print(f"\n{'='*72}\n  PER-SEED METRICS (K={args.k_anchor})\n{'='*72}")
    print(f"{'method':<14} {'seed':>5} {'ADE':>9} {'FDE':>9}")
    for label, pred_union in preds.items():
        for seed in args.seeds:
            idx_full = per_seed_idx[seed]
            local = np.array([pos_in_union[int(i)] for i in idx_full], dtype=np.int64)
            gt = union_traj[local]
            pr = pred_union[local]
            if args.k_anchor == 2:
                m = evaluate_generation(pr, gt, compute_frechet=False)
            else:
                m = score_k_anchor(pr, gt, anchor_idx)
            row = {
                "method": label, "seed": seed,
                "n": int(m["n_trajectories"]),
                "dataset": os.path.basename(args.test_npz),
                "k_anchor": args.k_anchor,
                "ade_m": float(m["ade_m"]), "fde_m": float(m["fde_m"]),
                "normalized_ade": float(m["normalized_ade"]),
                "path_length_ratio": float(m["path_length_ratio"]),
            }
            rows.append(row)
            print(f"{label:<14} {seed:>5d} {row['ade_m']:>9.1f} {row['fde_m']:>9.1f}")

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames); w.writeheader(); w.writerows(rows)
    print(f"\nWrote {args.out_csv} ({len(rows)} rows)")

    # ── Aggregate ──────────────────────────────────────────────────────
    print(f"\n{'='*72}\n  AGGREGATE (mean ± std)\n{'='*72}")
    cols = ["ade_m", "fde_m", "normalized_ade"]
    summary_fields = ["method", "n_seeds"] + sum(
        [[f"{c}_mean", f"{c}_std"] for c in cols], [])
    summary_rows = []
    print(f"{'method':<14} {'n':>4} {'ADE_mean':>10} {'±std':>7}")
    for label in preds:
        mr = [r for r in rows if r["method"] == label]
        agg = {"method": label, "n_seeds": len(mr)}
        for c in cols:
            vals = np.asarray([r[c] for r in mr], dtype=np.float64)
            agg[f"{c}_mean"] = float(np.nanmean(vals))
            agg[f"{c}_std"]  = float(np.nanstd(vals, ddof=0))
        summary_rows.append(agg)
        print(f"{label:<14} {agg['n_seeds']:>4d} "
              f"{agg['ade_m_mean']:>10.1f} {agg['ade_m_std']:>7.1f}")
    os.makedirs(os.path.dirname(args.out_summary) or ".", exist_ok=True)
    with open(args.out_summary, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary_fields); w.writeheader(); w.writerows(summary_rows)
    print(f"\nWrote {args.out_summary}")


if __name__ == "__main__":
    main()
