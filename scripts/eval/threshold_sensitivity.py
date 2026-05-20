"""threshold_sensitivity.py -- sweep router threshold (5, 10, 25 km) on clean
data with the production v10 checkpoint, seed=42, n_eval=500.

The report (\S 5.3) asserts theta_train = 10 km is "above the raster-noise
floor" without showing evidence. This script computes ADE / FDE / land% /
cross% at 5, 10, and 25 km using the existing eval_gen_v10_router pipeline,
so the report can cite a real sensitivity table.

Output: results/threshold_sensitivity.csv
"""
import sys
import csv
import argparse
import os
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data_gen import TrajGenScalers, Scaler, train_val_split_gen  # noqa: E402
from src.land_mask import LandMask  # noqa: E402
from src.metrics_gen import evaluate_generation  # noqa: E402
from src.model_gen_v10 import TrajectoryGeneratorV10  # noqa: E402
from src.water_router import WaterRouter  # noqa: E402


SEED = 42
N_EVAL = 500
THRESHOLDS = [5.0, 10.0, 25.0]


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    args = argparse.Namespace(**ckpt["args"])
    scalers = TrajGenScalers(
        pos=Scaler(mean=ckpt["pos_mean"], std=ckpt["pos_std"]),
        delta=Scaler(mean=ckpt["delta_mean"], std=ckpt["delta_std"]),
    )
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
    sd = ckpt["model"]
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.eval()
    return model, scalers, ckpt["vtype_vocab"], ckpt.get("n_resample", 128), args


def generate(model, val_traj, val_vt, train_traj, val_knn, vtype_vocab,
             scalers, n_resample, device):
    N = len(val_traj)
    pred = np.zeros_like(val_traj)
    B_size = 32
    for s in tqdm(range(0, N, B_size), desc="v10 raw"):
        e = min(s + B_size, N)
        B = e - s
        starts_norm = scalers.pos.transform(val_traj[s:e, 0])
        ends_norm   = scalers.pos.transform(val_traj[s:e, -1])
        vtypes = np.array([vtype_vocab.get(int(v), 0) for v in val_vt[s:e]])
        retrieved_raw = train_traj[val_knn[s:e]]
        K = retrieved_raw.shape[1]
        retrieved_norm = scalers.pos.transform(
            retrieved_raw.reshape(-1, 2)).reshape(B, K, n_resample, 2)
        retrieval_mask = np.zeros((B, K), dtype=bool)

        st = torch.tensor(starts_norm, dtype=torch.float32, device=device)
        et = torch.tensor(ends_norm,   dtype=torch.float32, device=device)
        vt = torch.tensor(vtypes,      dtype=torch.long,    device=device)
        rt = torch.tensor(retrieved_norm, dtype=torch.float32, device=device)
        rm = torch.tensor(retrieval_mask, dtype=torch.bool,    device=device)
        with torch.no_grad():
            g = model.generate(st, et, vt, rt, rm,
                               n_steps=n_resample,
                               pos_scaler=scalers.pos,
                               delta_scaler=scalers.delta,
                               land_mask=None)
        g = g.permute(1, 0, 2).cpu().numpy()
        for i in range(B):
            pred[s + i] = scalers.pos.inverse(g[i])
    return pred


def apply_router(trajs, router, threshold_km):
    out = np.empty_like(trajs)
    for i in tqdm(range(len(trajs)), desc=f"snap @ {threshold_km:.0f}km"):
        rep, _ = router.repair_trajectory(
            trajs[i], threshold_km=threshold_km,
            do_segment_bridge=False, window_margin_cells=50, max_window_cells=400)
        out[i] = rep
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_npz",    default="data/processed/trajgen_128_clean.npz")
    p.add_argument("--checkpoint",  default="runs/trajgen_v10/best.pt")
    p.add_argument("--knn_cache",   default="data/processed/trajgen_128_clean_knn_k5.npz")
    p.add_argument("--water_graph", default="data/processed/water_graph_005deg.npz")
    p.add_argument("--land_sdf",    default="data/processed/land_sdf_050deg.npz")
    p.add_argument("--out_csv",     default="results/threshold_sensitivity.csv")
    args = p.parse_args()

    device = get_device()
    print(f"Device: {device}")

    model, scalers, vtype_vocab, n_resample, train_args = load_checkpoint(
        args.checkpoint, device)
    land_mask = LandMask.load(args.land_sdf)
    router = WaterRouter.load(args.water_graph, coarse_mask=land_mask)

    data = np.load(args.data_npz, allow_pickle=True)
    traj = data["trajectories"].astype(np.float32)
    vt   = data["vessel_types"].astype(np.int32)
    tid  = data["track_ids"]
    train_traj, _, _, val_traj, val_vt, _ = train_val_split_gen(
        traj, vt, tid, train_args.val_frac, train_args.seed)
    knn = np.load(args.knn_cache)
    val_knn = knn["val_knn"]

    # Subsample 500 with seed=42 (the canonical eval seed).
    rng = np.random.RandomState(SEED)
    idx = np.sort(rng.choice(len(val_traj), N_EVAL, replace=False))
    val_traj_s = val_traj[idx]
    val_vt_s   = val_vt[idx]
    val_knn_s  = val_knn[idx]
    print(f"Evaluating {N_EVAL} routes (seed={SEED})")

    # Generate once; snap multiple times.
    pred_raw = generate(model, val_traj_s, val_vt_s, train_traj, val_knn_s,
                        vtype_vocab, scalers, n_resample, device)
    pred_raw = pred_raw.astype(np.float64)

    rows = []
    for theta in THRESHOLDS:
        pred_snap = apply_router(pred_raw, router, theta).astype(np.float32)
        m_raw  = evaluate_generation(pred_raw.astype(np.float32), val_traj_s,
                                     compute_frechet=False)
        m_snap = evaluate_generation(pred_snap, val_traj_s, compute_frechet=False)
        ls_raw  = land_mask.trajectory_stats(pred_raw.astype(np.float32),  theta)
        ls_snap = land_mask.trajectory_stats(pred_snap, theta)
        for name, m, ls in [("v10 raw", m_raw, ls_raw),
                             ("v10+router", m_snap, ls_snap)]:
            rows.append({
                "threshold_km": theta,
                "method": name,
                "ade_m": round(m["ade_m"], 1),
                "fde_m": round(m["fde_m"], 1),
                "n": m["n_trajectories"],
                "land_frac_pct": round(ls["land_frac"] * 100.0, 3),
                "cross_pct": round(ls["traj_crossing_rate"] * 100.0, 2),
            })

    # Also report GT stats at each threshold for context.
    gt_rows = []
    for theta in THRESHOLDS:
        ls_gt = land_mask.trajectory_stats(val_traj_s, theta)
        gt_rows.append({
            "threshold_km": theta,
            "method": "ground truth",
            "ade_m": 0.0,
            "fde_m": 0.0,
            "n": N_EVAL,
            "land_frac_pct": round(ls_gt["land_frac"] * 100.0, 3),
            "cross_pct": round(ls_gt["traj_crossing_rate"] * 100.0, 2),
        })

    all_rows = rows + gt_rows
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nWrote {args.out_csv}")

    # Print table
    print()
    print(f"{'theta(km)':>10} {'method':<14} "
          f"{'ADE(m)':>9} {'FDE(m)':>9} {'land%':>8} {'cross%':>8}")
    for r in all_rows:
        print(f"{r['threshold_km']:>10.0f} {r['method']:<14} "
              f"{r['ade_m']:>9.1f} {r['fde_m']:>9.1f} "
              f"{r['land_frac_pct']:>7.2f}% {r['cross_pct']:>7.2f}%")


if __name__ == "__main__":
    main()
