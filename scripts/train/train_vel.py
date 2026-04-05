#!/usr/bin/env python3
"""
train_vel.py — AIS Vessel Trajectory Prediction (Velocity Target)

Same as train.py but predicts velocity [dlon/dt, dlat/dt] (deg/s) instead of
raw displacement [dlon, dlat]. This removes ping-interval dependence from the
target: the same vessel at the same speed always has the same velocity target
regardless of how frequently it reports position.

Target per timestep: [dlon/dt_norm, dlat/dt_norm]  (normalised velocity in deg/s)
Recovery at eval:    pred_pos = input_pos + vel_scaler.inverse(pred) * dt_next

Compare with train_disp.py which predicts displacement and recovers:
  pred_pos = input_pos + disp_scaler.inverse(pred)

Usage:
    python3 scripts/train_vel.py --csv data/processed/AIS_combined_processed.csv \\
        --epochs 40 --val_frac 0.15 \\
        --seq_len 120 --stride 50 \\
        --num_layers 3 --lambda_smooth 5.0
"""

import argparse
import math
import os
import random

import numpy as np
import torch
import torch.nn as nn

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.data    import (VelScalers, load_tracks, train_val_split, get_vel_loader)
from src.model   import AISTransformer
from src.metrics import (evaluate_vel, evaluate_constvel_baseline_vel, sanity_check_vel)


# ── Training loop ─────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_one_epoch(model, loader, optimizer, scheduler, device, grad_clip,
                    epoch, log_every=50, lambda_smooth=0.0, loss_fn="mse"):
    model.train()
    base_loss_fn = nn.HuberLoss(delta=1.0) if loss_fn == "huber" else nn.MSELoss()
    running, running_smooth, total, n = 0.0, 0.0, 0.0, 0

    for i, (x, y, gap_mask, vtype, dt_next) in enumerate(loader):
        x, y     = x.to(device), y.to(device)
        gap_mask = gap_mask.to(device)
        vtype    = vtype.to(device)
        optimizer.zero_grad()
        pred = model(x, gap_mask=gap_mask, vessel_type=vtype)

        # Skip t=0: input velocity feature is 0 there (same reason as train.py)
        mse_loss    = base_loss_fn(pred[1:], y[1:])
        smooth_loss = torch.tensor(0.0, device=device)
        if lambda_smooth > 0 and pred.size(0) > 2:
            accel       = pred[2:] - pred[1:-1]
            smooth_loss = (accel ** 2).mean()
        loss = mse_loss + lambda_smooth * smooth_loss

        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()

        running        += mse_loss.item()
        running_smooth += smooth_loss.item()
        total          += mse_loss.item()
        n              += 1

        if (i + 1) % log_every == 0:
            smooth_str = (f" | smooth {running_smooth/log_every:.4f}"
                          if lambda_smooth > 0 else "")
            print(f"  epoch {epoch:3d} | step {i+1:5d}/{len(loader)} "
                  f"| mse {running/log_every:.6f}{smooth_str} "
                  f"| grad {grad_norm:.3f} "
                  f"| lr {scheduler.get_last_lr()[0]:.2e}")
            running = 0.0
            running_smooth = 0.0

    return total / max(n, 1)


def build_scheduler(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        t = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(1e-2, 0.5 * (1.0 + math.cos(math.pi * t)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--config",    default=None,
                        help="Path to YAML config file. CLI args override config values.")
    parser.add_argument("--csv",       default=None)
    parser.add_argument("--out_dir",   default="runs/ais_transformer_vel")
    parser.add_argument("--id_col",    default="MMSI")
    parser.add_argument("--time_col",  default="BaseDateTime")
    parser.add_argument("--lat_col",   default="LAT")
    parser.add_argument("--lon_col",   default="LON")

    parser.add_argument("--seq_len",               type=int,   default=120)
    parser.add_argument("--stride",                type=int,   default=50)
    parser.add_argument("--max_windows_per_track", type=int,   default=None)
    parser.add_argument("--max_gap_sec",           type=float, default=600.0)
    parser.add_argument("--use_velocity",          action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--val_frac",              type=float, default=0.1)
    parser.add_argument("--seed",                  type=int,   default=42)

    parser.add_argument("--d_model",         type=int,   default=128)
    parser.add_argument("--nhead",           type=int,   default=8)
    parser.add_argument("--num_layers",      type=int,   default=2)
    parser.add_argument("--dim_feedforward", type=int,   default=None)
    parser.add_argument("--dropout",         type=float, default=0.1)

    parser.add_argument("--batch_size",     type=int,   default=256)
    parser.add_argument("--epochs",         type=int,   default=20)
    parser.add_argument("--lr",             type=float, default=3e-4)
    parser.add_argument("--weight_decay",   type=float, default=1e-4)
    parser.add_argument("--grad_clip",      type=float, default=1.0)
    parser.add_argument("--warmup_frac",    type=float, default=0.05)
    parser.add_argument("--num_workers",    type=int,   default=0)
    parser.add_argument("--lambda_smooth",  type=float, default=0.1,
                        help="Smoothness regularisation weight (0 = off)")
    parser.add_argument("--loss_fn",        default="mse", choices=["mse", "huber"])

    # Load config file first, then re-parse so CLI args override config values
    _pre = parser.parse_known_args()[0]
    if _pre.config:
        import yaml
        with open(_pre.config) as _f:
            _cfg = yaml.safe_load(_f)
        parser.set_defaults(**{k: v for k, v in _cfg.items() if k != "config"})

    args = parser.parse_args()

    if not args.csv:
        parser.error("--csv is required (either via CLI or --config)")

    set_seed(args.seed)
    device = get_device()
    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 65)
    print("  AIS TRANSFORMER -- VELOCITY PREDICTION TRAINING")
    print("=" * 65)
    print(f"  Device  : {device}")
    print(f"  Out dir : {args.out_dir}\n")

    tracks, vessel_types = load_tracks(args.csv, args.id_col, args.time_col,
                                       args.lat_col, args.lon_col)
    if not tracks:
        raise RuntimeError("No valid tracks loaded.")

    train_tracks, train_vtypes, val_tracks, val_vtypes = train_val_split(
        tracks, vessel_types, args.val_frac, args.seed)
    print(f"  Train: {len(train_tracks):,} vessels  |  Val: {len(val_tracks):,}\n")

    scalers = VelScalers.fit(train_tracks)
    print(f"  Position scaler -- mean: {scalers.pos.mean}  std: {scalers.pos.std}")
    print(f"  log(dt) scaler  -- mean: {scalers.logdt.mean.item():.4f}  "
          f"std: {scalers.logdt.std.item():.4f}")
    print(f"  Disp scaler     -- mean: {scalers.disp.mean}  std: {scalers.disp.std}")
    print(f"  Vel scaler      -- mean: {scalers.vel.mean}  std: {scalers.vel.std}\n")

    all_codes   = sorted(set(train_vtypes.values()))
    vtype_vocab = {code: idx + 1 for idx, code in enumerate(all_codes)}
    vtype_vocab[0] = 0
    num_vessel_types = len(vtype_vocab)
    print(f"  Vessel types : {num_vessel_types - 1} unique codes  "
          f"(vocab size {num_vessel_types})\n")

    input_dim = 5 if args.use_velocity else 3
    print(f"  Input  : {input_dim}D "
          f"{'[lon,lat,log_dt,vel_lon,vel_lat]' if args.use_velocity else '[lon,lat,log_dt]'}"
          f" + vessel-type embedding ({num_vessel_types} types → 8-dim)")
    print(f"  Target : velocity [dlon/dt_norm, dlat/dt_norm] (deg/s, vel-scaler)")
    print(f"  Loss   : {args.loss_fn.upper()} + smooth*{args.lambda_smooth} "
          f"(t=0 excluded)\n")

    train_loader = get_vel_loader(
        train_tracks, train_vtypes, vtype_vocab, scalers, args.seq_len, args.batch_size,
        stride=args.stride, max_windows_per_track=args.max_windows_per_track,
        shuffle=True, drop_last=True, use_velocity=args.use_velocity,
        num_workers=args.num_workers, max_gap_sec=args.max_gap_sec,
    )
    val_loader = get_vel_loader(
        val_tracks, val_vtypes, vtype_vocab, scalers, args.seq_len, args.batch_size,
        stride=args.stride, max_windows_per_track=args.max_windows_per_track,
        shuffle=False, drop_last=False, use_velocity=args.use_velocity,
        num_workers=args.num_workers, max_gap_sec=args.max_gap_sec,
    ) if val_tracks else None

    print(f"  Train batches : {len(train_loader):,}")
    if val_loader:
        print(f"  Val batches   : {len(val_loader):,}")

    model = AISTransformer(
        input_dim=input_dim, d_model=args.d_model, nhead=args.nhead,
        num_layers=args.num_layers, dim_feedforward=args.dim_feedforward,
        dropout=args.dropout, pred_mode="causal",
        num_vessel_types=num_vessel_types, vessel_type_embed_dim=8,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Params  : {n_params:,}")
    print(f"  d_model={args.d_model}  nhead={args.nhead}  "
          f"layers={args.num_layers}  ffn={args.dim_feedforward or 4*args.d_model}")

    optimizer    = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                     weight_decay=args.weight_decay)
    total_steps  = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_frac)
    scheduler    = build_scheduler(optimizer, warmup_steps, total_steps)
    print(f"  Steps   : {total_steps:,} total | {warmup_steps:,} warmup\n")

    if val_loader:
        print("─" * 65)
        bl = evaluate_constvel_baseline_vel(val_loader, device, scalers)
        print(f"  Const-vel baseline -- "
              f"MSE {bl['mse']:.6f}  ADE {bl['ade_m']:.1f}m  FDE {bl['fde_m']:.1f}m")
        print(f"  Beat this or something is wrong.")
        print("─" * 65 + "\n")

    best_val_mse = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_mse = train_one_epoch(
            model, train_loader, optimizer, scheduler,
            device, args.grad_clip, epoch,
            lambda_smooth=args.lambda_smooth, loss_fn=args.loss_fn,
        )
        log = f"Epoch {epoch:3d}/{args.epochs} | train mse {train_mse:.6f}"

        if val_loader:
            metrics = evaluate_vel(model, val_loader, device, scalers)
            log += (f" | val mse {metrics['mse']:.6f} | "
                    f"ADE {metrics['ade_m']:.1f}m | FDE {metrics['fde_m']:.1f}m")

            if metrics["mse"] < best_val_mse:
                best_val_mse = metrics["mse"]
                torch.save({
                    "epoch": epoch, "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "pos_mean": scalers.pos.mean, "pos_std": scalers.pos.std,
                    "logdt_mean": scalers.logdt.mean, "logdt_std": scalers.logdt.std,
                    "disp_mean": scalers.disp.mean, "disp_std": scalers.disp.std,
                    "vel_mean": scalers.vel.mean, "vel_std": scalers.vel.std,
                    "vtype_vocab": vtype_vocab, "num_vessel_types": num_vessel_types,
                    "metrics": metrics, "args": vars(args),
                }, os.path.join(args.out_dir, "best.pt"))
                log += "  * saved"

            if epoch % 5 == 0 or epoch == 1:
                print(f"  [sanity check epoch {epoch}]")
                sanity_check_vel(model, val_loader, device, scalers)

        print(log)

    print(f"\n  Best val MSE : {best_val_mse:.6f}")
    print(f"  Checkpoint   : {args.out_dir}/best.pt")
    print("=" * 65)


if __name__ == "__main__":
    main()
