#!/usr/bin/env python3
"""
Evaluate a checkpoint: model ADE/FDE/MSE and CV baseline,
using the exact val split from training.

Usage:
    python3 scripts/eval_checkpoint.py \
        --csv  data/processed/AIS_2024_clean_v2.csv \
        --checkpoint runs/ais_transformer_v2/best.pt

    python3 scripts/eval_checkpoint.py \
        --csv  data/processed/AIS_2024_clean_v2.csv \
        --checkpoint runs/ais_transformer_v3/best.pt
"""

import argparse, sys, os
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.data import load_tracks, train_val_split, get_loader, Scalers, Scaler
from src.model import AISTransformer
from src.metrics import evaluate, evaluate_constant_velocity_baseline, sanity_check


def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    a = ckpt["args"]
    scalers = Scalers(
        pos=Scaler(mean=ckpt["pos_mean"],   std=ckpt["pos_std"]),
        logdt=Scaler(mean=ckpt["logdt_mean"], std=ckpt["logdt_std"]),
        disp=Scaler(mean=ckpt["disp_mean"],  std=ckpt["disp_std"]),
    )
    vtype_vocab      = ckpt["vtype_vocab"]
    num_vessel_types = ckpt["num_vessel_types"]

    model = AISTransformer(
        input_dim        = 5 if a.get("use_velocity", True) else 3,
        d_model          = a.get("d_model", 128),
        nhead            = a.get("nhead", 8),
        num_layers       = a.get("num_layers", 3),
        dim_feedforward  = a.get("dim_feedforward", 512),
        dropout          = a.get("dropout", 0.1),
        pred_mode        = a.get("pred_mode", "causal"),
        num_vessel_types = num_vessel_types,
        vessel_type_embed_dim = 8,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    return model, scalers, vtype_vocab, a


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv",        required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--device",     default="cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    model, scalers, vtype_vocab, a = load_checkpoint(args.checkpoint, device)

    seq_len   = a.get("seq_len",   120)
    stride    = a.get("stride",    50)
    batch     = a.get("batch_size", 256)
    val_frac  = a.get("val_frac",  0.15)
    seed      = a.get("seed",      42)
    use_vel   = a.get("use_velocity", True)
    max_gap   = a.get("max_gap_sec",  600)
    pred_mode = a.get("pred_mode",  "causal")

    print(f"Checkpoint : {args.checkpoint}")
    print(f"seq_len={seq_len}  stride={stride}  pred_mode={pred_mode}")

    tracks, vtypes = load_tracks(args.csv, "MMSI", "BaseDateTime", "LAT", "LON")
    _, _, val_tracks, val_vtypes = train_val_split(tracks, vtypes, val_frac, seed)
    print(f"Val vessels: {len(val_tracks):,}")

    val_loader = get_loader(
        val_tracks, val_vtypes, vtype_vocab, scalers,
        seq_len, batch, pred_mode=pred_mode, stride=stride,
        shuffle=False, drop_last=False, use_velocity=use_vel,
        num_workers=0, max_gap_sec=max_gap,
    )
    print(f"Val batches: {len(val_loader):,}\n")

    bl = evaluate_constant_velocity_baseline(val_loader, device, scalers, pred_mode)
    print(f"CV baseline  -- MSE {bl['mse']:.6f}  ADE {bl['ade_m']:.1f}m  FDE {bl['fde_m']:.1f}m")

    m = evaluate(model, val_loader, device, scalers, pred_mode)
    print(f"Model        -- MSE {m['mse']:.6f}  ADE {m['ade_m']:.1f}m  FDE {m['fde_m']:.1f}m")

    print(f"\nModel vs baseline:")
    print(f"  ADE improvement: {bl['ade_m'] - m['ade_m']:.1f}m  "
          f"({(bl['ade_m'] - m['ade_m']) / bl['ade_m'] * 100:.1f}% better)")


if __name__ == "__main__":
    main()
