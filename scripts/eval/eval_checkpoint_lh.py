#!/usr/bin/env python3
"""
Evaluate a lighthouse-feature checkpoint: model ADE/FDE/MSE and CV baseline,
using the exact val split from training.

Usage:
    python3 scripts/eval/eval_checkpoint_lh.py \
        --csv  data/processed/AIS_2024_gt80_lh.csv \
        --checkpoint runs/12mo_seq120_lh/best.pt
"""

import argparse, sys, os
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.data import Scaler, train_val_split
from src.data_lh import LHScalers, load_tracks_lh, get_loader_lh
from src.model import AISTransformer
from src.metrics import evaluate, evaluate_constant_velocity_baseline, sanity_check


def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    a = ckpt["args"]
    scalers = LHScalers(
        pos   = Scaler(mean=ckpt["pos_mean"],   std=ckpt["pos_std"]),
        logdt = Scaler(mean=ckpt["logdt_mean"], std=ckpt["logdt_std"]),
        disp  = Scaler(mean=ckpt["disp_mean"],  std=ckpt["disp_std"]),
        lh    = Scaler(mean=ckpt["lh_mean"],    std=ckpt["lh_std"]),
    )
    vtype_vocab      = ckpt["vtype_vocab"]
    num_vessel_types = ckpt["num_vessel_types"]

    input_dim = 3 + (2 if a.get("use_velocity", True) else 0) + 3

    model = AISTransformer(
        input_dim        = input_dim,
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
    lh_cols   = a.get("lighthouse_cols",
                      ["lh_dist_1_km", "lh_dist_2_km", "lh_dist_3_km"])

    print(f"Checkpoint : {args.checkpoint}")
    print(f"seq_len={seq_len}  stride={stride}  pred_mode={pred_mode}")
    print(f"lighthouse_cols={lh_cols}")

    tracks, vtypes = load_tracks_lh(args.csv, lh_cols,
                                    "MMSI", "BaseDateTime", "LAT", "LON")
    _, _, val_tracks, val_vtypes = train_val_split(tracks, vtypes, val_frac, seed)
    print(f"Val vessels: {len(val_tracks):,}")

    val_loader = get_loader_lh(
        val_tracks, val_vtypes, vtype_vocab, scalers,
        seq_len, batch, stride=stride,
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
