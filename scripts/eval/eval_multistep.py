#!/usr/bin/env python3
"""
eval_multistep.py — Autoregressive multi-step rollout evaluation.

Compares the transformer model against the constant-velocity (CV) baseline
at multiple prediction horizons (H = 1, 2, 5, 10, 20 steps).

For each horizon H:
  - Model: feeds its own predictions back as input for H steps
  - CV:    extrapolates the last displacement forward H times unchanged

Reports ADE@H (average error across H steps) and FDE@H (error at step H).

Usage:
    python3 scripts/eval/eval_multistep.py \
        --csv data/processed/AIS_2024_gt80_opensea.csv \
        --checkpoint runs/12mo_seq120_opensea/best.pt

    # Custom horizons and number of evaluation windows
    python3 scripts/eval/eval_multistep.py \
        --csv data/processed/AIS_2024_gt80_opensea.csv \
        --checkpoint runs/12mo_seq120_opensea/best.pt \
        --horizons 1 2 5 10 20 \
        --n_windows 500
"""

import argparse
import os
import random
import sys

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.data import Scaler, Scalers, load_tracks, make_input, train_val_split
from src.metrics import haversine_meters
from src.model import AISTransformer


def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    a = ckpt["args"]
    scalers = Scalers(
        pos=Scaler(mean=ckpt["pos_mean"], std=ckpt["pos_std"]),
        logdt=Scaler(mean=ckpt["logdt_mean"], std=ckpt["logdt_std"]),
        disp=Scaler(mean=ckpt["disp_mean"], std=ckpt["disp_std"]),
    )
    vtype_vocab = ckpt["vtype_vocab"]
    num_vessel_types = ckpt["num_vessel_types"]

    model = AISTransformer(
        input_dim=5 if a.get("use_velocity", True) else 3,
        d_model=a.get("d_model", 128),
        nhead=a.get("nhead", 8),
        num_layers=a.get("num_layers", 3),
        dim_feedforward=a.get("dim_feedforward", 512),
        dropout=0.0,
        pred_mode=a.get("pred_mode", "causal"),
        num_vessel_types=num_vessel_types,
        vessel_type_embed_dim=8,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, scalers, vtype_vocab, a


@torch.no_grad()
def rollout_model(model, context_pts, scalers, vtype_idx, horizon,
                  use_velocity, max_gap_sec, device):
    """Autoregressive model rollout for `horizon` steps.

    Args:
        model: trained AISTransformer (causal mode)
        context_pts: (seq_len, 3) raw [lon, lat, dt] array
        scalers: Scalers with pos/logdt/disp
        vtype_idx: int — vessel type vocab index
        horizon: number of steps to predict
        use_velocity: bool
        max_gap_sec: float — gap mask threshold
        device: torch device

    Returns:
        predicted_positions: (horizon, 2) array of [lon, lat] in degrees
    """
    seq_len = len(context_pts)
    # Work on a copy so we can append predictions
    pts = context_pts.copy()  # (seq_len, 3)
    median_dt = float(np.median(pts[1:, 2]))  # typical dt for synthetic steps

    predicted = []

    for _ in range(horizon):
        # Build input features from current context
        x_feat = make_input(pts, scalers, use_velocity)  # (seq_len, input_dim)
        x_t = torch.tensor(x_feat, dtype=torch.float32).unsqueeze(1).to(device)
        gap_mask = torch.tensor(
            pts[:, 2] > max_gap_sec, dtype=torch.bool
        ).unsqueeze(0).to(device)
        vt = torch.tensor([vtype_idx], dtype=torch.long).to(device)

        # Forward pass — take last timestep prediction
        out = model(x_t, gap_mask=gap_mask, vessel_type=vt)  # (seq_len, 1, 2)
        disp_norm = out[-1, 0].cpu().numpy()  # (2,)
        disp_geo = scalers.disp.inverse(disp_norm.reshape(1, 2))[0]  # (2,)

        # Predicted next position
        last_pos = pts[-1, :2]
        next_pos = last_pos + disp_geo
        predicted.append(next_pos.copy())

        # Shift window: drop oldest, append predicted position
        new_row = np.array([next_pos[0], next_pos[1], median_dt], dtype=np.float32)
        pts = np.concatenate([pts[1:], new_row.reshape(1, 3)], axis=0)

    return np.array(predicted)  # (horizon, 2)


def rollout_cv(context_pts, horizon):
    """Constant-velocity rollout: extrapolate last displacement H times.

    Args:
        context_pts: (seq_len, 3) raw [lon, lat, dt]
        horizon: number of steps

    Returns:
        predicted_positions: (horizon, 2) array of [lon, lat] in degrees
    """
    last_disp = context_pts[-1, :2] - context_pts[-2, :2]  # (2,)
    last_pos = context_pts[-1, :2]

    predicted = []
    pos = last_pos.copy()
    for _ in range(horizon):
        pos = pos + last_disp
        predicted.append(pos.copy())

    return np.array(predicted)  # (horizon, 2)


def main():
    p = argparse.ArgumentParser(
        description="Multi-step autoregressive rollout evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--csv", required=True, help="Processed AIS CSV")
    p.add_argument("--checkpoint", required=True, help="Path to best.pt")
    p.add_argument("--horizons", type=int, nargs="+", default=[1, 2, 5, 10, 20],
                   help="Prediction horizons to evaluate")
    p.add_argument("--n_windows", type=int, default=500,
                   help="Number of evaluation windows to sample")
    p.add_argument("--seed", type=int, default=None,
                   help="RNG seed for window sampling (default: use training seed)")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    model, scalers, vtype_vocab, a = load_checkpoint(args.checkpoint, device)

    seq_len = a.get("seq_len", 120)
    val_frac = a.get("val_frac", 0.15)
    train_seed = a.get("seed", 42)
    use_vel = a.get("use_velocity", True)
    max_gap = a.get("max_gap_sec", 600.0)

    sample_seed = args.seed if args.seed is not None else train_seed

    print(f"Checkpoint : {args.checkpoint}")
    print(f"seq_len={seq_len}  horizons={args.horizons}  n_windows={args.n_windows}")

    # Load data and reproduce val split
    tracks, vtypes = load_tracks(args.csv)
    _, _, val_tracks, val_vtypes = train_val_split(tracks, vtypes, val_frac, train_seed)
    print(f"Val vessels: {len(val_tracks):,}")

    max_h = max(args.horizons)
    min_len = seq_len + max_h

    # Collect eligible (vessel_id, start_idx) windows
    eligible = []
    for vid, pts in val_tracks.items():
        T = len(pts)
        if T < min_len:
            continue
        # Collect all valid start positions
        for s in range(0, T - min_len + 1, seq_len // 2):
            eligible.append((vid, s))

    print(f"Eligible windows: {len(eligible):,} (from vessels with >= {min_len} pings)")
    if not eligible:
        print("No eligible windows. Try a smaller max horizon or more data.")
        return

    rng = random.Random(sample_seed)
    rng.shuffle(eligible)
    selected = eligible[:args.n_windows]
    print(f"Evaluating {len(selected)} windows\n")

    # Per-horizon accumulators: {H: {"model_ade": [], "model_fde": [], "cv_ade": [], "cv_fde": []}}
    results = {h: {"model_ade": [], "model_fde": [], "cv_ade": [], "cv_fde": []}
               for h in args.horizons}

    for vid, start in tqdm(selected, desc="Rollout eval", unit="window"):
        pts = val_tracks[vid]
        raw_code = val_vtypes.get(vid, 0)
        vtype_idx = vtype_vocab.get(raw_code, 0)

        context = pts[start: start + seq_len]  # (seq_len, 3)
        future = pts[start + seq_len: start + seq_len + max_h]  # (max_h, 3)

        # Model rollout
        model_pred = rollout_model(
            model, context, scalers, vtype_idx, max_h,
            use_vel, max_gap, device,
        )  # (max_h, 2)

        # CV rollout
        cv_pred = rollout_cv(context, max_h)  # (max_h, 2)

        # True future positions
        true_pos = future[:, :2]  # (max_h, 2) [lon, lat]

        # Compute errors at each horizon step
        model_dists = haversine_meters(
            true_pos[:, 1], true_pos[:, 0],
            model_pred[:, 1], model_pred[:, 0],
        )  # (max_h,)
        cv_dists = haversine_meters(
            true_pos[:, 1], true_pos[:, 0],
            cv_pred[:, 1], cv_pred[:, 0],
        )  # (max_h,)

        for h in args.horizons:
            results[h]["model_ade"].append(float(model_dists[:h].mean()))
            results[h]["model_fde"].append(float(model_dists[h - 1]))
            results[h]["cv_ade"].append(float(cv_dists[:h].mean()))
            results[h]["cv_fde"].append(float(cv_dists[h - 1]))

    # ── Print results table ────────────────────────────────────────────────
    print(f"\n{'MULTI-STEP ROLLOUT EVALUATION':^72}")
    print(f"{'(' + str(len(selected)) + ' windows)':^72}")
    print("=" * 72)
    print(f"{'H':>4}  {'CV ADE':>10}  {'Model ADE':>10}  {'Diff':>8}  "
          f"{'CV FDE':>10}  {'Model FDE':>10}  {'Diff':>8}")
    print("-" * 72)

    for h in args.horizons:
        cv_ade = np.mean(results[h]["cv_ade"])
        m_ade = np.mean(results[h]["model_ade"])
        cv_fde = np.mean(results[h]["cv_fde"])
        m_fde = np.mean(results[h]["model_fde"])
        ade_diff = m_ade - cv_ade
        fde_diff = m_fde - cv_fde
        ade_sign = "+" if ade_diff >= 0 else ""
        fde_sign = "+" if fde_diff >= 0 else ""
        print(f"{h:>4}  {cv_ade:>9.1f}m  {m_ade:>9.1f}m  {ade_sign}{ade_diff:>6.1f}m  "
              f"{cv_fde:>9.1f}m  {m_fde:>9.1f}m  {fde_sign}{fde_diff:>6.1f}m")

    print("=" * 72)
    print("Negative diff = model is better than CV")


if __name__ == "__main__":
    main()
