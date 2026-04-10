#!/usr/bin/env python3
"""
train_disp_lh.py — Train AIS displacement prediction with lighthouse distance features.

Identical to train_disp.py but adds 3 lighthouse distance features (log1p(dist_km))
to the input, increasing input_dim from 5 to 8 (with velocity).

Requires a pre-processed CSV with lighthouse distance columns (see add_lighthouse_dist.py).

Usage:
    python3 scripts/train/train_disp_lh.py --config configs/12mo_seq120_lh.yaml

    python3 scripts/train/train_disp_lh.py \
        --csv data/processed/AIS_2024_gt80_lh.csv \
        --epochs 40 --val_frac 0.15 \
        --seq_len 120 --stride 50 \
        --num_layers 3 --lambda_smooth 5.0 \
        --out_dir runs/12mo_seq120_lh
"""

import argparse
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.data import train_val_split
from src.data_lh import LHScalers, load_tracks_lh, get_loader_lh
from src.model import AISTransformer
from src.metrics import evaluate, evaluate_constant_velocity_baseline, sanity_check


# ── Reproducibility + device ──────────────────────────────────────────────────

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


# ── Training loop ─────────────────────────────────────────────────────────────

def _val_subsample_mse(model, val_loader, val_eval_batches, device):
    """Run a partial val pass (first val_eval_batches batches) and return MSE."""
    model.eval()
    running, n = 0.0, 0
    with torch.no_grad():
        for j, (xv, yv, gm, vt) in enumerate(val_loader):
            if j >= val_eval_batches:
                break
            pv = model(xv.to(device), gap_mask=gm.to(device), vessel_type=vt.to(device))
            running += F.mse_loss(pv[1:], yv[1:].to(device)).item()
            n += 1
    model.train()
    return running / max(n, 1)


def train_one_epoch(model, loader, optimizer, scheduler,
                    device, grad_clip, epoch, pred_mode, log_every=50,
                    lambda_smooth=0.0, loss_fn="mse",
                    val_loader=None, val_eval_batches=30, global_step_offset=0,
                    metrics_path=None):
    model.train()
    base_loss_fn = nn.HuberLoss(delta=1.0) if loss_fn == "huber" else nn.MSELoss()
    running, running_smooth, total, n = 0.0, 0.0, 0.0, 0

    pbar = tqdm(enumerate(loader), total=len(loader),
                desc=f"  Epoch {epoch}", unit="batch", leave=False)
    for i, (x, y, gap_mask, vtype) in pbar:
        x, y = x.to(device), y.to(device)
        gap_mask = gap_mask.to(device)
        vtype    = vtype.to(device)
        optimizer.zero_grad()
        pred = model(x, gap_mask=gap_mask, vessel_type=vtype)

        if pred_mode == "causal":
            mse_loss = base_loss_fn(pred[1:], y[1:])
            smooth_loss = torch.tensor(0.0, device=device)
            if lambda_smooth > 0 and pred.size(0) > 2:
                accel = pred[2:] - pred[1:-1]
                smooth_loss = (accel ** 2).mean()
            loss = mse_loss + lambda_smooth * smooth_loss
        else:
            mse_loss = base_loss_fn(pred, y)
            smooth_loss = torch.tensor(0.0, device=device)
            loss = mse_loss

        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()

        running        += mse_loss.item()
        running_smooth += smooth_loss.item()
        total          += mse_loss.item()
        n              += 1

        if (i + 1) % log_every == 0:
            train_mse_avg = running / log_every
            smooth_str = (f" | smooth {running_smooth/log_every:.4f}"
                          if lambda_smooth > 0 else "")

            val_mse_approx = None
            if val_loader is not None and val_eval_batches > 0:
                val_mse_approx = _val_subsample_mse(model, val_loader,
                                                    val_eval_batches, device)
                val_str = f" | val_mse(~) {val_mse_approx:.6f}"
            else:
                val_str = ""

            tqdm.write(f"  epoch {epoch:3d} | step {i+1:5d}/{len(loader)} "
                       f"| mse {train_mse_avg:.6f}{smooth_str}{val_str} "
                       f"| grad {grad_norm:.3f} "
                       f"| lr {scheduler.get_last_lr()[0]:.2e}")

            global_step = global_step_offset + i + 1
            if metrics_path is not None:
                val_mse_str = f"{val_mse_approx:.6f}" if val_mse_approx is not None else ""
                with open(metrics_path, "a") as _f:
                    _f.write(f"{global_step},{epoch},{train_mse_avg:.6f},{val_mse_str},,\n")
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
    parser.add_argument("--out_dir",   default="runs/ais_transformer_lh")
    parser.add_argument("--id_col",    default="MMSI")
    parser.add_argument("--time_col",  default="BaseDateTime")
    parser.add_argument("--lat_col",   default="LAT")
    parser.add_argument("--lon_col",   default="LON")

    parser.add_argument("--lighthouse_cols", nargs=3,
                        default=["lh_dist_1_km", "lh_dist_2_km", "lh_dist_3_km"],
                        help="Column names for lighthouse distances in the CSV")

    parser.add_argument("--pred_mode",  default="causal",
                        choices=["causal", "single"])
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
    parser.add_argument("--lambda_smooth",  type=float, default=0.1)
    parser.add_argument("--loss_fn",        default="mse", choices=["mse", "huber"])
    parser.add_argument("--val_eval_batches", type=int, default=30)

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
    print("  AIS TRANSFORMER -- TRAINING (with lighthouse features)")
    print("=" * 65)
    print(f"  Device    : {device}")
    print(f"  Pred mode : {args.pred_mode}")
    print(f"  Out dir   : {args.out_dir}")
    print(f"  Lighthouse cols : {args.lighthouse_cols}\n")

    tracks, vessel_types = load_tracks_lh(
        args.csv, args.lighthouse_cols,
        args.id_col, args.time_col, args.lat_col, args.lon_col)
    if not tracks:
        raise RuntimeError("No valid tracks loaded.")

    train_tracks, train_vtypes, val_tracks, val_vtypes = train_val_split(
        tracks, vessel_types, args.val_frac, args.seed)
    print(f"  Train: {len(train_tracks):,} vessels  |  Val: {len(val_tracks):,}\n")

    scalers = LHScalers.fit(train_tracks)
    print(f"  Position scaler -- mean: {scalers.pos.mean}  std: {scalers.pos.std}")
    print(f"  log(dt) scaler  -- mean: {scalers.logdt.mean.item():.4f}  "
          f"std: {scalers.logdt.std.item():.4f}")
    print(f"  Disp scaler     -- mean: {scalers.disp.mean}  std: {scalers.disp.std}")
    print(f"  LH scaler       -- mean: {scalers.lh.mean}  std: {scalers.lh.std}\n")

    # Build vessel type vocab from training vessels only
    all_codes   = sorted(set(train_vtypes.values()))
    vtype_vocab = {code: idx + 1 for idx, code in enumerate(all_codes)}
    vtype_vocab[0] = 0
    num_vessel_types = len(vtype_vocab)
    print(f"  Vessel types    : {num_vessel_types - 1} unique codes  "
          f"(vocab size {num_vessel_types}  codes: {all_codes})\n")

    # input_dim = base(3) + velocity(2) + lighthouse(3) = 8 max
    input_dim = 3 + (2 if args.use_velocity else 0) + 3
    print(f"  Input  : {input_dim}D "
          f"[lon,lat,log_dt"
          f"{',' + 'vel_lon,vel_lat' if args.use_velocity else ''}"
          f",lh1,lh2,lh3]"
          f" + vessel-type embedding ({num_vessel_types} types → 8-dim)")
    print(f"  Target : displacement [dlon_norm, dlat_norm]")
    print(f"  Mode   : {args.pred_mode}")
    print(f"  Loss   : {args.loss_fn.upper()} + smoothness*{args.lambda_smooth}\n")

    train_loader = get_loader_lh(
        train_tracks, train_vtypes, vtype_vocab, scalers, args.seq_len, args.batch_size,
        stride=args.stride,
        max_windows_per_track=args.max_windows_per_track,
        shuffle=True, drop_last=True, use_velocity=args.use_velocity,
        num_workers=args.num_workers, max_gap_sec=args.max_gap_sec,
    )
    val_loader = get_loader_lh(
        val_tracks, val_vtypes, vtype_vocab, scalers, args.seq_len, args.batch_size,
        stride=args.stride,
        max_windows_per_track=args.max_windows_per_track,
        shuffle=False, drop_last=False, use_velocity=args.use_velocity,
        num_workers=args.num_workers, max_gap_sec=args.max_gap_sec,
    ) if val_tracks else None

    print(f"  Train batches : {len(train_loader):,}")
    if val_loader:
        print(f"  Val batches   : {len(val_loader):,}")

    model = AISTransformer(
        input_dim=input_dim, d_model=args.d_model, nhead=args.nhead,
        num_layers=args.num_layers, dim_feedforward=args.dim_feedforward,
        dropout=args.dropout, pred_mode=args.pred_mode,
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
        bl = evaluate_constant_velocity_baseline(
            val_loader, device, scalers, args.pred_mode)
        print(f"  Const-vel baseline -- "
              f"MSE {bl['mse']:.6f}  ADE {bl['ade_m']:.1f}m  FDE {bl['fde_m']:.1f}m")
        print(f"  Beat this or something is wrong.")
        print("─" * 65 + "\n")
        import json
        with open(os.path.join(args.out_dir, "baseline.json"), "w") as _f:
            json.dump(bl, _f)

    best_val_mse = float("inf")
    metrics_path = os.path.join(args.out_dir, "metrics.csv")

    with open(metrics_path, "w") as f:
        f.write("global_step,epoch,train_mse,val_mse,val_ade_m,val_fde_m\n")

    training_start = time.time()
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        global_step_offset = (epoch - 1) * len(train_loader)
        train_mse = train_one_epoch(
            model, train_loader, optimizer, scheduler,
            device, args.grad_clip, epoch, args.pred_mode,
            lambda_smooth=args.lambda_smooth, loss_fn=args.loss_fn,
            val_loader=val_loader if val_loader else None,
            val_eval_batches=args.val_eval_batches,
            global_step_offset=global_step_offset,
            metrics_path=metrics_path,
        )

        log = f"Epoch {epoch:3d}/{args.epochs} | train mse {train_mse:.6f}"

        val_mse, val_ade, val_fde = None, None, None
        if val_loader:
            metrics = evaluate(model, val_loader, device, scalers, args.pred_mode)
            val_mse, val_ade, val_fde = metrics["mse"], metrics["ade_m"], metrics["fde_m"]
            log += (f" | val mse {val_mse:.6f} | "
                    f"ADE {val_ade:.1f}m | FDE {val_fde:.1f}m")

            if val_mse < best_val_mse:
                best_val_mse = val_mse
                torch.save({
                    "epoch": epoch, "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "pos_mean": scalers.pos.mean, "pos_std": scalers.pos.std,
                    "logdt_mean": scalers.logdt.mean, "logdt_std": scalers.logdt.std,
                    "disp_mean": scalers.disp.mean, "disp_std": scalers.disp.std,
                    "lh_mean": scalers.lh.mean, "lh_std": scalers.lh.std,
                    "vtype_vocab": vtype_vocab, "num_vessel_types": num_vessel_types,
                    "metrics": metrics, "args": vars(args),
                }, os.path.join(args.out_dir, "best.pt"))
                log += "  * saved"

            if epoch % 5 == 0 or epoch == 1:
                print(f"  [sanity check epoch {epoch}]")
                sanity_check(model, val_loader, device, scalers, args.pred_mode)
        else:
            torch.save({
                "epoch": epoch, "model": model.state_dict(),
                "pos_mean": scalers.pos.mean, "pos_std": scalers.pos.std,
                "logdt_mean": scalers.logdt.mean, "logdt_std": scalers.logdt.std,
                "disp_mean": scalers.disp.mean, "disp_std": scalers.disp.std,
                "lh_mean": scalers.lh.mean, "lh_std": scalers.lh.std,
                "args": vars(args),
            }, os.path.join(args.out_dir, "last.pt"))

        # Append epoch-end row
        epoch_step = epoch * len(train_loader)
        mse_str = f"{val_mse:.6f}" if val_mse is not None else ""
        ade_str = f"{val_ade:.2f}" if val_ade is not None else ""
        fde_str = f"{val_fde:.2f}" if val_fde is not None else ""
        with open(metrics_path, "a") as f:
            f.write(f"{epoch_step},{epoch},,{mse_str},{ade_str},{fde_str}\n")

        epoch_secs = time.time() - epoch_start
        elapsed    = time.time() - training_start
        remaining  = elapsed / epoch * (args.epochs - epoch)
        def _fmt(s):
            h, m = divmod(int(s), 3600)
            m, s = divmod(m, 60)
            return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"
        log += (f"  |  epoch {_fmt(epoch_secs)}  elapsed {_fmt(elapsed)}"
                f"  ETA {_fmt(remaining)}")
        print(log)

    print(f"\n  Best val MSE : {best_val_mse:.6f}")
    print(f"  Checkpoint   : {args.out_dir}/best.pt")
    print("=" * 65)


if __name__ == "__main__":
    main()
