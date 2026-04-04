#!/usr/bin/env python3
"""
Compute average true displacement (step size) on the val windows actually used
during evaluation — same split, same seq_len, same stride as training.

Usage:
    python3 scripts/avg_step_size.py \
        --csv  data/processed/AIS_2024_clean_v2.csv \
        --checkpoint runs/ais_transformer_v3/best.pt
"""

import argparse, sys, os
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data import load_tracks, train_val_split, get_loader, Scalers, Scaler
from src.metrics import haversine_meters


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv",        required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--val_frac",   type=float, default=0.15)
    p.add_argument("--seed",       type=int,   default=42)
    args = p.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    ckpt_args = ckpt["args"]
    scalers = Scalers(
        pos=Scaler(mean=ckpt["pos_mean"],   std=ckpt["pos_std"]),
        logdt=Scaler(mean=ckpt["logdt_mean"], std=ckpt["logdt_std"]),
        disp=Scaler(mean=ckpt["disp_mean"],  std=ckpt["disp_std"]),
    )
    vtype_vocab = ckpt["vtype_vocab"]

    seq_len  = ckpt_args.get("seq_len",  80)
    stride   = ckpt_args.get("stride",   50)
    batch    = ckpt_args.get("batch_size", 256)
    use_vel  = ckpt_args.get("use_velocity", True)
    max_gap  = ckpt_args.get("max_gap_sec",  600)
    pred_mode = ckpt_args.get("pred_mode",  "causal")

    tracks, vtypes = load_tracks(args.csv, "MMSI", "BaseDateTime", "LAT", "LON")
    _, _, val_tracks, val_vtypes = train_val_split(
        tracks, vtypes, args.val_frac, args.seed)

    loader = get_loader(
        val_tracks, val_vtypes, vtype_vocab, scalers,
        seq_len, batch, pred_mode=pred_mode, stride=stride,
        shuffle=False, drop_last=False, use_velocity=use_vel,
        num_workers=0, max_gap_sec=max_gap,
    )

    all_dists = []
    for x, y, _gm, _vt in loader:
        x_np = x.cpu().numpy()              # (S, B, D)
        y_np = y.cpu().numpy()              # (S, B, 2) normalised displacement

        S, B = y_np.shape[:2]
        input_pos_geo = scalers.pos.inverse(
            x_np[:, :, :2].reshape(-1, 2)).reshape(S, B, 2)   # lon, lat
        true_disp_geo = scalers.disp.inverse(
            y_np.reshape(-1, 2)).reshape(S, B, 2)
        true_pos_geo  = input_pos_geo + true_disp_geo          # predicted next pos

        # haversine between input pos and true next pos = actual step size
        # exclude t=0 (velocity feature artifact)
        for t in range(1, S):
            d = haversine_meters(
                input_pos_geo[t, :, 1], input_pos_geo[t, :, 0],
                true_pos_geo [t, :, 1], true_pos_geo [t, :, 0])
            all_dists.extend(d.tolist())

    all_dists = np.array(all_dists)
    print(f"\nTrue step size on val windows (t>0, {len(all_dists):,} steps):")
    print(f"  mean   : {all_dists.mean():.1f} m")
    print(f"  median : {np.median(all_dists):.1f} m")
    print(f"  p25    : {np.percentile(all_dists, 25):.1f} m")
    print(f"  p75    : {np.percentile(all_dists, 75):.1f} m")
    print(f"  p95    : {np.percentile(all_dists, 95):.1f} m")


if __name__ == "__main__":
    main()
