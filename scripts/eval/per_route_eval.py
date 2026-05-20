"""Per-route evaluation of the production v10 checkpoint on val (subsample)
and test (full held-out partition).

Saves one row per (split, route_idx) with: vessel_type, gt_arc_km,
knn_dist_top1_km, ade_m for each of {v10_raw, v10_router, retr_top1, gc}.
Used by:
    vessel_type_breakdown.py     (per-type ADE)
    retrieval_quality_analysis.py (KNN-dist vs ADE scatter)
    paired_bootstrap_ci.py       (paired CI of v10 minus baselines)
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data_gen import Scaler, TrajGenScalers, train_val_split_gen
from src.land_mask import LandMask
from src.metrics_gen import great_circle_trajectory
from src.model_gen_v10 import TrajectoryGeneratorV10
from src.water_router import WaterRouter, _haversine_km


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_checkpoint(path: str, device: torch.device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    targs = argparse.Namespace(**ckpt["args"])
    scalers = TrajGenScalers(
        pos=Scaler(mean=ckpt["pos_mean"], std=ckpt["pos_std"]),
        delta=Scaler(mean=ckpt["delta_mean"], std=ckpt["delta_std"]),
    )
    model = TrajectoryGeneratorV10(
        d_model=targs.d_model, nhead=targs.nhead,
        num_encoder_layers=targs.num_encoder_layers,
        num_decoder_layers=targs.num_decoder_layers,
        dim_feedforward=targs.dim_feedforward, dropout=targs.dropout,
        num_vessel_types=ckpt["num_vessel_types"],
        max_retrieved=ckpt["k_retrieval"],
        t_retr=ckpt["t_retr"],
        n_points=ckpt.get("n_resample", 128),
        k_past=ckpt.get("k_past", None),
    ).to(device)
    state_dict = ckpt["model"]
    if any(k.startswith("_orig_mod.") for k in state_dict):
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    return model, scalers, ckpt["vtype_vocab"], ckpt.get("n_resample", 128), targs


def gen_raw_batch(model, val_traj, val_vt, train_traj, val_knn, vtype_vocab,
                  scalers, n_resample, device, batch_size=32, land_mask=None,
                  hard_threshold_km=10.0) -> np.ndarray:
    """Run autoregressive generation; return (N, T, 2) lon/lat predictions."""
    N = len(val_traj)
    pred = np.zeros_like(val_traj, dtype=np.float64)
    for s in tqdm(range(0, N, batch_size), desc="Gen"):
        e = min(s + batch_size, N)
        batch_gt = val_traj[s:e]
        batch_vt = val_vt[s:e]
        batch_kn = val_knn[s:e]
        B = e - s
        starts = scalers.pos.transform(batch_gt[:, 0])
        ends   = scalers.pos.transform(batch_gt[:, -1])
        vtypes = np.array([vtype_vocab.get(int(v), 0) for v in batch_vt])
        retr   = train_traj[batch_kn]
        retr_n = scalers.pos.transform(retr.reshape(-1, 2)).reshape(B, retr.shape[1], n_resample, 2)
        rmask  = np.zeros((B, retr.shape[1]), dtype=bool)
        with torch.no_grad():
            gen = model.generate(
                torch.tensor(starts, dtype=torch.float32, device=device),
                torch.tensor(ends,   dtype=torch.float32, device=device),
                torch.tensor(vtypes, dtype=torch.long,    device=device),
                torch.tensor(retr_n, dtype=torch.float32, device=device),
                torch.tensor(rmask,  dtype=torch.bool,    device=device),
                n_steps=n_resample,
                pos_scaler=scalers.pos, delta_scaler=scalers.delta,
                land_mask=land_mask, hard_threshold_km=hard_threshold_km,
            )
        gen_np = gen.permute(1, 0, 2).cpu().numpy()
        for i in range(B):
            pred[s + i] = scalers.pos.inverse(gen_np[i])
    return pred


def apply_snap(router: WaterRouter, trajs: np.ndarray, threshold_km: float = 10.0) -> np.ndarray:
    """Per-trajectory WaterRouter snap-only."""
    out = np.empty_like(trajs, dtype=np.float64)
    for i in tqdm(range(len(trajs)), desc="Snap"):
        rep, _ = router.repair_trajectory(
            trajs[i], threshold_km=threshold_km, do_segment_bridge=False,
            window_margin_cells=100, max_window_cells=600)
        out[i] = rep
    return out


def per_route_metrics(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """(N, T, 2), (N, T, 2) -> (N,) per-route ADE in metres."""
    N, T, _ = gt.shape
    out = np.zeros(N)
    for i in range(N):
        d = _haversine_km(pred[i, :, 0], pred[i, :, 1],
                          gt[i, :, 0],  gt[i, :, 1])
        out[i] = float(d.mean() * 1000.0)
    return out


def knn_dist_km_top1(query_traj: np.ndarray, neighbour_traj: np.ndarray) -> float:
    """Average of haversine(start->start) and haversine(end->end) in km."""
    qs, qe = query_traj[0], query_traj[-1]
    ns, ne = neighbour_traj[0], neighbour_traj[-1]
    d_s = float(_haversine_km(qs[0], qs[1], ns[0], ns[1]))
    d_e = float(_haversine_km(qe[0], qe[1], ne[0], ne[1]))
    return 0.5 * (d_s + d_e)


def arc_length_km(traj: np.ndarray) -> float:
    return float(_haversine_km(traj[:-1, 0], traj[:-1, 1],
                                traj[1:, 0],  traj[1:, 1]).sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_npz",   default="data/processed/trajgen_128_clean.npz")
    ap.add_argument("--checkpoint", default="runs/trajgen_v10_clean_train_lambdaland05/best.pt")
    ap.add_argument("--knn_cache",  default="data/processed/trajgen_128_clean_knn_k5_test005.npz")
    ap.add_argument("--water_graph", default="data/processed/water_graph_005deg.npz")
    ap.add_argument("--land_sdf",   default="data/processed/land_sdf_050deg.npz")
    ap.add_argument("--val_frac",  type=float, default=0.15)
    ap.add_argument("--test_frac", type=float, default=0.05)
    ap.add_argument("--seed",      type=int, default=42)
    ap.add_argument("--n_eval_val", type=int, default=2000,
                    help="Subsample of val (seed=42) for inference speed.")
    ap.add_argument("--threshold_km", type=float, default=10.0)
    ap.add_argument("--out_csv", default="results/per_route_lambdaland05.csv")
    args = ap.parse_args()

    device = get_device()
    print(f"Device: {device}")

    model, scalers, vtype_vocab, n_resample, targs = load_checkpoint(args.checkpoint, device)
    print(f"Loaded {args.checkpoint}  (n_resample={n_resample})")

    coarse = LandMask.load(args.land_sdf)
    router = WaterRouter.load(args.water_graph, coarse_mask=coarse)

    data = np.load(args.data_npz, allow_pickle=True)
    trajs = data["trajectories"].astype(np.float32)
    vts   = data["vessel_types"].astype(np.int32)
    tids  = data["track_ids"]

    (train_t, _, _, val_t, val_v, _, test_t, test_v, _) = train_val_split_gen(
        trajs, vts, tids, args.val_frac, args.seed, test_frac=args.test_frac
    )
    print(f"train={len(train_t)}  val={len(val_t)}  test={len(test_t)}")

    knn = np.load(args.knn_cache)
    val_knn  = knn["val_knn"]
    test_knn = knn["test_knn"]

    # Subsample val
    if 0 < args.n_eval_val < len(val_t):
        rng = np.random.RandomState(42)
        idx_val = rng.choice(len(val_t), args.n_eval_val, replace=False)
        val_t   = val_t[idx_val]
        val_v   = val_v[idx_val]
        val_knn = val_knn[idx_val]
        print(f"  subsampled val to {args.n_eval_val} (seed=42)")

    rows = []
    for split_name, partition_t, partition_v, partition_knn in [
        ("val",  val_t,  val_v,  val_knn),
        ("test", test_t, test_v, test_knn),
    ]:
        print(f"\n=== {split_name}  n={len(partition_t)} ===")
        t0 = time.time()
        pred_raw = gen_raw_batch(model, partition_t, partition_v, train_t,
                                  partition_knn, vtype_vocab, scalers,
                                  n_resample, device, land_mask=None)
        pred_router = apply_snap(router, pred_raw, threshold_km=args.threshold_km)
        retr_top1 = train_t[partition_knn[:, 0]].astype(np.float64)
        gc = np.zeros_like(partition_t, dtype=np.float64)
        for i in range(len(partition_t)):
            gc[i] = great_circle_trajectory(partition_t[i, 0], partition_t[i, -1],
                                             partition_t.shape[1])

        ade_raw    = per_route_metrics(pred_raw,    partition_t.astype(np.float64))
        ade_router = per_route_metrics(pred_router, partition_t.astype(np.float64))
        ade_retr   = per_route_metrics(retr_top1,   partition_t.astype(np.float64))
        ade_gc     = per_route_metrics(gc,          partition_t.astype(np.float64))

        for i in range(len(partition_t)):
            gt = partition_t[i]
            nbr = train_t[partition_knn[i, 0]]
            rows.append({
                "split": split_name,
                "route_idx": i,
                "vessel_type": int(partition_v[i]),
                "gt_arc_km": arc_length_km(gt),
                "gt_start_lon": float(gt[0, 0]),  "gt_start_lat": float(gt[0, 1]),
                "gt_end_lon":   float(gt[-1, 0]), "gt_end_lat":   float(gt[-1, 1]),
                "knn_dist_top1_km": knn_dist_km_top1(gt, nbr),
                "ade_raw_m":    float(ade_raw[i]),
                "ade_router_m": float(ade_router[i]),
                "ade_retr_top1_m": float(ade_retr[i]),
                "ade_gc_m":     float(ade_gc[i]),
            })
        print(f"  wall={time.time() - t0:.1f}s   "
              f"ADE mean router={ade_router.mean():.1f}  "
              f"raw={ade_raw.mean():.1f}  "
              f"retr={ade_retr.mean():.1f}  "
              f"GC={ade_gc.mean():.1f}")

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {args.out_csv}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
