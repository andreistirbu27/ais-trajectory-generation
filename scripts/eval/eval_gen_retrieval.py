#!/usr/bin/env python3
"""
eval_gen_retrieval.py — Evaluate a v9 retrieval-augmented trajectory generator.

Loads a RetrievalTrajectoryGenerator checkpoint and generates trajectories on
the val split with K retrieved routes per query (from the cached KNN index).
Reports ADE/FDE/Frechet vs great-circle baseline.

Usage:
    python3 scripts/eval/eval_gen_retrieval.py \
        --data_npz data/processed/trajgen_128.npz \
        --checkpoint runs/trajgen_v9_retrieval/best.pt \
        --knn_cache data/processed/trajgen_128_knn_k5.npz

    # Quick eval (500 samples, skip Frechet)
    python3 scripts/eval/eval_gen_retrieval.py \
        --data_npz data/processed/trajgen_128.npz \
        --checkpoint runs/trajgen_v9_retrieval/best.pt \
        --knn_cache data/processed/trajgen_128_knn_k5.npz \
        --n_eval 500 --no_frechet
"""

import argparse
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.data_gen import TrajGenScalers, Scaler, train_val_split_gen
from src.model_gen_retrieval import RetrievalTrajectoryGenerator
from src.metrics_gen import evaluate_generation, evaluate_great_circle_baseline


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_checkpoint(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = argparse.Namespace(**ckpt["args"])

    scalers = TrajGenScalers(
        pos=Scaler(mean=ckpt["pos_mean"], std=ckpt["pos_std"]),
        delta=Scaler(mean=ckpt["delta_mean"], std=ckpt["delta_std"]),
    )

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

    state_dict = ckpt["model"]
    if any(k.startswith("_orig_mod.") for k in state_dict):
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()

    return model, scalers, ckpt["vtype_vocab"], ckpt.get("n_resample", 128), args


def generate_with_retrieval(model, val_traj, val_vt, train_traj, val_knn,
                             vtype_vocab, scalers, n_resample, device):
    N = len(val_traj)
    gt_traj = val_traj.copy()
    pred_traj = np.zeros_like(gt_traj)

    batch_size = 32
    for start_idx in tqdm(range(0, N, batch_size), desc="Generating"):
        end_idx = min(start_idx + batch_size, N)
        batch_gt = gt_traj[start_idx:end_idx]
        batch_vt = val_vt[start_idx:end_idx]
        batch_knn = val_knn[start_idx:end_idx]      # (B, K)
        B = len(batch_gt)

        starts_raw = batch_gt[:, 0]
        ends_raw = batch_gt[:, -1]
        starts_norm = scalers.pos.transform(starts_raw)
        ends_norm = scalers.pos.transform(ends_raw)
        vtypes = np.array([vtype_vocab.get(int(v), 0) for v in batch_vt])

        # Retrieved routes: (B, K, T, 2) normalized
        retrieved_raw = train_traj[batch_knn]        # (B, K, T, 2)
        K = retrieved_raw.shape[1]
        retrieved_flat = retrieved_raw.reshape(-1, 2)
        retrieved_norm = scalers.pos.transform(retrieved_flat).reshape(B, K, n_resample, 2)
        retrieval_mask = np.zeros((B, K), dtype=bool)

        start_t = torch.tensor(starts_norm, dtype=torch.float32, device=device)
        end_t   = torch.tensor(ends_norm,   dtype=torch.float32, device=device)
        vtype_t = torch.tensor(vtypes,      dtype=torch.long,    device=device)
        retr_t  = torch.tensor(retrieved_norm, dtype=torch.float32, device=device)
        rmask_t = torch.tensor(retrieval_mask, dtype=torch.bool,    device=device)

        with torch.no_grad():
            gen_norm = model.generate(
                start_t, end_t, vtype_t, retr_t, rmask_t,
                n_steps=n_resample,
                pos_scaler=scalers.pos, delta_scaler=scalers.delta)

        gen_norm_np = gen_norm.permute(1, 0, 2).cpu().numpy()
        for i in range(B):
            pred_traj[start_idx + i] = scalers.pos.inverse(gen_norm_np[i])

    return pred_traj, gt_traj


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_npz",   required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--knn_cache",  required=True,
        help="Path to the KNN cache NPZ written by training")
    parser.add_argument("--n_eval",     type=int, default=None)
    parser.add_argument("--no_frechet", action="store_true")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}\n")

    model, scalers, vtype_vocab, n_resample, train_args = load_checkpoint(
        args.checkpoint, device)
    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"  epochs={train_args.epochs}  d_model={train_args.d_model}  "
          f"k_retrieval={train_args.k_retrieval}  "
          f"route_encoder={train_args.route_encoder}\n")

    data = np.load(args.data_npz, allow_pickle=True)
    trajectories = data["trajectories"].astype(np.float32)
    vessel_types = data["vessel_types"].astype(np.int32)
    track_ids    = data["track_ids"]

    train_traj, _, _, val_traj, val_vt, _ = train_val_split_gen(
        trajectories, vessel_types, track_ids,
        train_args.val_frac, train_args.seed)
    print(f"Train corpus: {len(train_traj):,}  |  Val: {len(val_traj):,}\n")

    knn = np.load(args.knn_cache)
    val_knn = knn["val_knn"]
    print(f"Loaded KNN cache: val_knn {val_knn.shape}\n")

    # Subsample val set once (shared across model / retrieval-top-1 / GC)
    if args.n_eval is not None and args.n_eval < len(val_traj):
        rng = np.random.RandomState(42)
        idx = rng.choice(len(val_traj), args.n_eval, replace=False)
        val_traj = val_traj[idx]
        val_vt = val_vt[idx]
        val_knn = val_knn[idx]
        print(f"Subsampled to {args.n_eval} val samples (seed=42)\n")

    pred_traj, gt_traj = generate_with_retrieval(
        model, val_traj, val_vt, train_traj, val_knn,
        vtype_vocab, scalers, n_resample, device)

    compute_frechet = not args.no_frechet

    print("=" * 65)
    print("  MODEL EVALUATION (v9 retrieval)")
    print("=" * 65)
    model_metrics = evaluate_generation(pred_traj, gt_traj,
                                         compute_frechet=compute_frechet)
    _print_metrics(model_metrics)

    print(f"\n{'='*65}")
    print("  RETRIEVAL-TOP-1 BASELINE (zero training)")
    print("=" * 65)
    retr_pred = train_traj[val_knn[:, 0]]   # top-1 neighbor as prediction
    retr_metrics = evaluate_generation(retr_pred, gt_traj,
                                        compute_frechet=compute_frechet)
    _print_metrics(retr_metrics)

    print(f"\n{'='*65}")
    print("  GREAT-CIRCLE BASELINE")
    print("=" * 65)
    gc_metrics = evaluate_great_circle_baseline(gt_traj)
    _print_metrics(gc_metrics)

    print(f"\n{'='*65}")
    print("  COMPARISON  (lower is better)")
    print("=" * 65)
    compare_metrics = ["ade_m", "normalized_ade", "fde_m", "endpoint_error_m",
                       "ade_short_m", "ade_medium_m", "ade_long_m"]
    if compute_frechet:
        compare_metrics.append("frechet_m")
    header = f"  {'metric':<22} {'model':>10} {'retr-top1':>12} {'GC':>10}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for metric in compare_metrics:
        m = model_metrics.get(metric, float("nan"))
        r = retr_metrics.get(metric,  float("nan"))
        g = gc_metrics.get(metric,    float("nan"))
        if metric.startswith("normalized_"):
            print(f"  {metric:.<22} {m:>10.4f} {r:>12.4f} {g:>10.4f}")
        else:
            print(f"  {metric:.<22} {m:>10.1f} {r:>12.1f} {g:>10.1f}")
    print("=" * 65)


def _print_metrics(metrics: dict):
    """Print a metrics dict in a grouped, readable order."""
    order_core = ["ade_m", "fde_m", "endpoint_error_m", "normalized_ade",
                  "path_length_ratio", "smoothness", "gt_smoothness",
                  "frechet_m", "frechet_median_m", "n_trajectories"]
    order_buckets = ["n_short", "ade_short_m", "normalized_ade_short",
                     "n_medium", "ade_medium_m", "normalized_ade_medium",
                     "n_long", "ade_long_m", "normalized_ade_long"]
    for k in order_core:
        if k in metrics:
            _print_one(k, metrics[k])
    for k in order_buckets:
        if k in metrics:
            _print_one(k, metrics[k])


def _print_one(k: str, v):
    if isinstance(v, float):
        if k.startswith("normalized_"):
            print(f"  {k:.<30} {v:>10.4f}")
        else:
            print(f"  {k:.<30} {v:>10.2f}")
    else:
        print(f"  {k:.<30} {v}")


if __name__ == "__main__":
    main()
