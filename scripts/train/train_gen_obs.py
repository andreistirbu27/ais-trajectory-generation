#!/usr/bin/env python3
"""
train_gen_obs.py — Obstacle-Conditioned Trajectory Generation Training

Extends train_gen.py with:
  - Synthetic obstacle augmentation during training
  - Obstacle penetration loss (L_obstacle) that forces avoidance
  - Obstacle tokens in the encoder memory

Loss: L_delta + λ_end * L_endpoint + λ_smooth * L_smooth + λ_obstacle * L_obstacle

Usage:
    python3 scripts/train/train_gen_obs.py --config configs/trajgen_v6_obstacle.yaml
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
from src.data_gen import TrajGenScalers, train_val_split_gen, build_vtype_vocab
from src.data_gen_obs import (
    ObstacleConfig, get_obstacle_trajgen_loader,
)
from src.model_gen_obs import ObstacleTrajectoryGenerator


# ── Reproducibility + device ─────────────────────────────────────────────────

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


# ── Obstacle penetration loss ────────────────────────────────────────────────

def compute_obstacle_loss(pred_traj_raw, obstacles_raw, obstacle_mask,
                          margin_km=5.0):
    """Compute obstacle penetration + soft repulsion loss.

    Args:
        pred_traj_raw: (B, T, 2) predicted trajectory in raw [lon, lat] degrees
        obstacles_raw: (B, K, 3) [center_lon, center_lat, radius_deg] in raw space
        obstacle_mask: (B, K) bool — True for padding (ignore)
        margin_km: soft repulsion margin in km

    Returns:
        loss: scalar — mean penalty over all waypoints and active obstacles
    """
    B, T, _ = pred_traj_raw.shape
    K = obstacles_raw.shape[1]

    if obstacle_mask.all():
        # No obstacles in this batch
        return torch.tensor(0.0, device=pred_traj_raw.device)

    # margin in approximate degrees
    margin_deg = margin_km / 111.0

    # Expand for broadcasting: (B, T, 1, 2) vs (B, 1, K, 2)
    traj_exp = pred_traj_raw.unsqueeze(2)                    # (B, T, 1, 2)
    centers = obstacles_raw[:, :, :2].unsqueeze(1)            # (B, 1, K, 2)
    radii = obstacles_raw[:, :, 2].unsqueeze(1)               # (B, 1, K)

    # Approximate Euclidean distance in degrees (fine for nearby points)
    diff = traj_exp - centers                                  # (B, T, K, 2)
    dist = torch.sqrt((diff ** 2).sum(dim=-1) + 1e-8)         # (B, T, K)

    # Penetration + soft repulsion: penalty when dist < radius + margin
    effective_radius = radii + margin_deg                      # (B, 1, K)
    violation = torch.clamp(effective_radius - dist, min=0)    # (B, T, K)
    penalty = violation ** 2                                   # quadratic

    # Mask out padded obstacles: (B, K) → (B, 1, K)
    active_mask = (~obstacle_mask).unsqueeze(1).float()        # (B, 1, K)
    penalty = penalty * active_mask                            # zero out padding

    # Average over waypoints and active obstacles
    n_active = active_mask.sum() * T
    if n_active > 0:
        return penalty.sum() / n_active
    return torch.tensor(0.0, device=pred_traj_raw.device)


# ── Loss computation ─────────────────────────────────────────────────────────

def compute_loss(pred_deltas, batch, scalers, lambda_end, lambda_smooth,
                 lambda_obstacle, obstacle_margin_km, input_traj=None):
    """Compute full loss including obstacle penetration penalty.

    Args:
        pred_deltas: (T-1, B, 2) predicted normalized deltas
        batch: dict with trajectory data + obstacles
        scalers: TrajGenScalers
        lambda_end: endpoint loss weight
        lambda_smooth: smoothness loss weight
        lambda_obstacle: obstacle penetration loss weight
        obstacle_margin_km: soft repulsion margin in km
        input_traj: (T, B, 2) decoder input (None or target_traj for TF).
                    In scheduled sampling, this differs from target_traj
                    and target deltas are recomputed relative to it.

    Returns:
        loss, loss_dict
    """
    start_norm = batch["start_norm"]
    end_norm   = batch["end_norm"]

    device = pred_deltas.device
    delta_mean = torch.tensor(scalers.delta.mean, device=device)
    delta_std  = torch.tensor(scalers.delta.std,  device=device)
    pos_mean   = torch.tensor(scalers.pos.mean,   device=device)
    pos_std    = torch.tensor(scalers.pos.std,    device=device)

    target_traj = batch["traj_norm"]
    if input_traj is None or input_traj is target_traj:
        gt_deltas = batch["deltas_norm"]
    else:
        target_raw = target_traj * pos_std + pos_mean
        input_raw  = input_traj  * pos_std + pos_mean
        gt_deltas_raw = target_raw[1:] - input_raw[:-1]
        gt_deltas = (gt_deltas_raw - delta_mean) / delta_std

    # L_delta: Huber
    l_delta = F.huber_loss(pred_deltas, gt_deltas, delta=1.0)

    # L_endpoint: predicted path must reach destination

    pred_deltas_b = pred_deltas.permute(1, 0, 2)               # (B, T-1, 2)
    pred_deltas_raw = pred_deltas_b * delta_std + delta_mean
    pred_cumsum_raw = torch.cumsum(pred_deltas_raw, dim=1)

    start_raw = start_norm * pos_std + pos_mean
    pred_endpoint_raw = start_raw + pred_cumsum_raw[:, -1]
    end_raw = end_norm * pos_std + pos_mean
    l_endpoint = F.mse_loss(pred_endpoint_raw, end_raw)

    # L_smooth: acceleration penalty
    if lambda_smooth > 0 and pred_deltas.shape[0] > 1:
        accel = pred_deltas[1:] - pred_deltas[:-1]
        l_smooth = (accel ** 2).mean()
    else:
        l_smooth = torch.tensor(0.0, device=device)

    # L_obstacle: penetration + soft repulsion
    l_obstacle = torch.tensor(0.0, device=device)
    if lambda_obstacle > 0 and "obstacles" in batch:
        obstacles = batch["obstacles"]        # (B, K, 3) normalized centers + radius_deg
        obstacle_mask = batch["obstacle_mask"] # (B, K) bool

        if not obstacle_mask.all():
            # Reconstruct predicted trajectory in raw space
            # pred_cumsum_raw: (B, T-1, 2) cumulative raw deltas from start
            pred_traj_raw = start_raw.unsqueeze(1) + torch.cat([
                torch.zeros(start_raw.shape[0], 1, 2, device=device),
                pred_cumsum_raw,
            ], dim=1)  # (B, T, 2) in raw lon/lat

            # De-normalize obstacle centers to raw space
            obs_raw = obstacles.clone()
            # Centers are normalized, radius is in degrees
            obs_raw[:, :, :2] = obstacles[:, :, :2] * pos_std + pos_mean

            l_obstacle = compute_obstacle_loss(
                pred_traj_raw, obs_raw, obstacle_mask, obstacle_margin_km)

    loss = (l_delta
            + lambda_end * l_endpoint
            + lambda_smooth * l_smooth
            + lambda_obstacle * l_obstacle)

    return loss, {
        "delta":    l_delta.item(),
        "endpoint": l_endpoint.item(),
        "smooth":   l_smooth.item(),
        "obstacle": l_obstacle.item(),
        "total":    loss.item(),
    }


# ── Training loop ────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, scheduler, device, grad_clip,
                    epoch, scalers, lambda_end, lambda_smooth,
                    lambda_obstacle, obstacle_margin_km,
                    val_loader=None, val_eval_batches=30,
                    global_step_offset=0, metrics_path=None, log_every=50,
                    ss_prob=0.0):
    model.train()
    running = {"delta": 0, "endpoint": 0, "smooth": 0, "obstacle": 0, "total": 0}
    n = 0

    pbar = tqdm(enumerate(loader), total=len(loader),
                desc=f"  Epoch {epoch}", unit="batch", leave=False)
    for i, batch in pbar:
        batch = {k: v.to(device) for k, v in batch.items()}

        optimizer.zero_grad()
        pred_deltas, input_traj = model(
            start=batch["start_norm"],
            end=batch["end_norm"],
            vessel_type=batch["vessel_type"],
            target_traj=batch["traj_norm"],
            obstacles=batch["obstacles"],
            obstacle_mask=batch["obstacle_mask"],
            ss_prob=ss_prob,
            delta_scaler=scalers.delta if ss_prob > 0 else None,
            pos_scaler=scalers.pos if ss_prob > 0 else None,
        )

        loss, loss_dict = compute_loss(
            pred_deltas, batch, scalers,
            lambda_end, lambda_smooth,
            lambda_obstacle, obstacle_margin_km,
            input_traj=input_traj)

        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()

        for k in running:
            running[k] += loss_dict[k]
        n += 1

        if (i + 1) % log_every == 0:
            avg = {k: running[k] / n for k in running}

            val_str = ""
            if val_loader is not None and val_eval_batches > 0:
                val_loss = _val_subsample(
                    model, val_loader, val_eval_batches, device, scalers,
                    lambda_end, lambda_smooth, lambda_obstacle, obstacle_margin_km)
                val_str = f" | val {val_loss:.6f}"

            tqdm.write(
                f"  epoch {epoch:3d} | step {i+1:5d}/{len(loader)} "
                f"| loss {avg['total']:.6f} (delta {avg['delta']:.6f} "
                f"end {avg['endpoint']:.6f} obs {avg['obstacle']:.6f})"
                f"{val_str} | grad {grad_norm:.3f} "
                f"| lr {scheduler.get_last_lr()[0]:.2e}"
            )

            global_step = global_step_offset + i + 1
            if metrics_path is not None:
                val_str_csv = f"{val_loss:.6f}" if val_loader else ""
                with open(metrics_path, "a") as f:
                    f.write(f"{global_step},{epoch},{avg['total']:.6f},"
                            f"{val_str_csv},,,{avg['obstacle']:.6f}\n")

    return {k: running[k] / max(n, 1) for k in running}


def _val_subsample(model, val_loader, n_batches, device, scalers,
                   lambda_end, lambda_smooth, lambda_obstacle, obstacle_margin_km):
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for j, batch in enumerate(val_loader):
            if j >= n_batches:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            pred_deltas, _ = model(
                start=batch["start_norm"], end=batch["end_norm"],
                vessel_type=batch["vessel_type"], target_traj=batch["traj_norm"],
                obstacles=batch["obstacles"], obstacle_mask=batch["obstacle_mask"])
            loss, _ = compute_loss(
                pred_deltas, batch, scalers,
                lambda_end, lambda_smooth,
                lambda_obstacle, obstacle_margin_km)
            total += loss.item()
            n += 1
    model.train()
    return total / max(n, 1)


def evaluate_full(model, loader, device, scalers,
                  lambda_end, lambda_smooth, lambda_obstacle, obstacle_margin_km):
    model.eval()
    running = {"delta": 0, "endpoint": 0, "smooth": 0, "obstacle": 0, "total": 0}
    n = 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            pred_deltas, _ = model(
                start=batch["start_norm"], end=batch["end_norm"],
                vessel_type=batch["vessel_type"], target_traj=batch["traj_norm"],
                obstacles=batch["obstacles"], obstacle_mask=batch["obstacle_mask"])
            _, loss_dict = compute_loss(
                pred_deltas, batch, scalers,
                lambda_end, lambda_smooth,
                lambda_obstacle, obstacle_margin_km)
            for k in running:
                running[k] += loss_dict[k]
            n += 1
    model.train()
    return {k: running[k] / max(n, 1) for k in running}


def build_scheduler(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        t = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(1e-2, 0.5 * (1.0 + math.cos(math.pi * t)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--config", default=None)
    parser.add_argument("--data_npz", default=None)
    parser.add_argument("--out_dir", default="runs/trajgen_v6_obstacle")

    # Dataset
    parser.add_argument("--val_frac", type=float, default=0.15)
    parser.add_argument("--seed",     type=int,   default=42)

    # Model
    parser.add_argument("--d_model",            type=int,   default=256)
    parser.add_argument("--nhead",              type=int,   default=8)
    parser.add_argument("--num_encoder_layers", type=int,   default=2)
    parser.add_argument("--num_decoder_layers", type=int,   default=4)
    parser.add_argument("--dim_feedforward",    type=int,   default=1024)
    parser.add_argument("--dropout",            type=float, default=0.1)

    # Training
    parser.add_argument("--batch_size",    type=int,   default=64)
    parser.add_argument("--epochs",        type=int,   default=60)
    parser.add_argument("--lr",            type=float, default=2e-4)
    parser.add_argument("--weight_decay",  type=float, default=1e-4)
    parser.add_argument("--grad_clip",     type=float, default=1.0)
    parser.add_argument("--warmup_frac",   type=float, default=0.05)
    parser.add_argument("--num_workers",   type=int,   default=0)

    # Loss weights
    parser.add_argument("--lambda_end",      type=float, default=10.0)
    parser.add_argument("--lambda_smooth",   type=float, default=1.0)
    parser.add_argument("--lambda_obstacle", type=float, default=50.0)
    parser.add_argument("--obstacle_margin_km", type=float, default=5.0)

    # Scheduled sampling
    parser.add_argument("--ss_warmup_epochs", type=int, default=20)
    parser.add_argument("--ss_max_prob", type=float, default=0.5)

    # Obstacles
    parser.add_argument("--max_obstacles",         type=int,   default=3)
    parser.add_argument("--p_obstacle",            type=float, default=0.5)
    parser.add_argument("--obstacle_radius_min_km", type=float, default=10.0)
    parser.add_argument("--obstacle_radius_max_km", type=float, default=100.0)

    # Resume
    parser.add_argument("--resume", default=None,
        help="Path to checkpoint to resume training from")

    # Logging
    parser.add_argument("--val_eval_batches", type=int, default=30)
    parser.add_argument("--log_every",        type=int, default=100)

    # Load config
    _pre = parser.parse_known_args()[0]
    if _pre.config:
        import yaml
        with open(_pre.config) as f:
            cfg = yaml.safe_load(f)
        parser.set_defaults(**{k: v for k, v in cfg.items() if k != "config"})

    args = parser.parse_args()

    if not args.data_npz:
        parser.error("--data_npz is required (either via CLI or --config)")

    set_seed(args.seed)
    device = get_device()
    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 65)
    print("  OBSTACLE-CONDITIONED TRAJECTORY GENERATOR — TRAINING")
    print("=" * 65)
    print(f"  Device  : {device}")
    print(f"  Out dir : {args.out_dir}\n")

    # ── Load NPZ ─────────────────────────────────────────────────────────────
    print(f"Loading data: {args.data_npz}")
    data = np.load(args.data_npz, allow_pickle=True)
    trajectories = data["trajectories"]
    vessel_types = data["vessel_types"]
    track_ids    = data["track_ids"]
    n_resample = trajectories.shape[1]
    print(f"  {len(trajectories):,} trajectories, {n_resample} points each\n")

    # ── Train/val split ──────────────────────────────────────────────────────
    (train_traj, train_vt, train_ids,
     val_traj, val_vt, val_ids) = train_val_split_gen(
        trajectories, vessel_types, track_ids, args.val_frac, args.seed)
    print(f"  Train: {len(train_traj):,}  |  Val: {len(val_traj):,}\n")

    # ── Scalers ──────────────────────────────────────────────────────────────
    scalers = TrajGenScalers.fit(train_traj)
    print(f"  Pos scaler   — mean: {scalers.pos.mean}  std: {scalers.pos.std}")
    print(f"  Delta scaler — mean: {scalers.delta.mean}  std: {scalers.delta.std}\n")

    # ── Vessel type vocab ────────────────────────────────────────────────────
    vtype_vocab = build_vtype_vocab(train_vt)
    num_vessel_types = len(vtype_vocab)
    print(f"  Vessel types : {num_vessel_types} codes\n")

    # ── Obstacle config ──────────────────────────────────────────────────────
    obs_cfg = ObstacleConfig(
        max_obstacles=args.max_obstacles,
        p_obstacle=args.p_obstacle,
        radius_min_km=args.obstacle_radius_min_km,
        radius_max_km=args.obstacle_radius_max_km,
    )
    print(f"  Obstacles    : max={obs_cfg.max_obstacles}  p={obs_cfg.p_obstacle}"
          f"  r={obs_cfg.radius_min_km}-{obs_cfg.radius_max_km}km"
          f"  λ_obs={args.lambda_obstacle}  margin={args.obstacle_margin_km}km\n")

    # ── Data loaders ─────────────────────────────────────────────────────────
    train_loader = get_obstacle_trajgen_loader(
        train_traj, train_vt, vtype_vocab, scalers, obs_cfg,
        args.batch_size, shuffle=True, drop_last=True,
        num_workers=args.num_workers, seed=args.seed, deterministic=False)
    val_loader = get_obstacle_trajgen_loader(
        val_traj, val_vt, vtype_vocab, scalers, obs_cfg,
        args.batch_size, shuffle=False, drop_last=False,
        num_workers=args.num_workers, seed=args.seed + 1, deterministic=True) if len(val_traj) > 0 else None

    print(f"  Train batches : {len(train_loader):,}")
    if val_loader:
        print(f"  Val batches   : {len(val_loader):,}")

    # ── Model ────────────────────────────────────────────────────────────────
    model = ObstacleTrajectoryGenerator(
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        num_vessel_types=num_vessel_types,
        max_obstacles=args.max_obstacles,
    ).to(device)

    # torch.compile
    if hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
            print("  torch.compile enabled")
        except Exception as e:
            print(f"  torch.compile skipped: {e}")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Params  : {n_params:,}")
    print(f"  d_model={args.d_model}  nhead={args.nhead}  "
          f"enc_layers={args.num_encoder_layers}  dec_layers={args.num_decoder_layers}"
          f"  ffn={args.dim_feedforward}")
    print(f"  Loss    : Huber + {args.lambda_end}*endpoint + "
          f"{args.lambda_smooth}*smooth + {args.lambda_obstacle}*obstacle\n")

    # ── Optimizer + scheduler ────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_frac)
    scheduler = build_scheduler(optimizer, warmup_steps, total_steps)
    print(f"  Steps   : {total_steps:,} total | {warmup_steps:,} warmup\n")

    # ── Resume from checkpoint ───────────────────────────────────────────────
    start_epoch = 1
    best_val_loss = float("inf")

    if args.resume:
        print(f"\n  Resuming from: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)

        state_dict = ckpt["model"]
        if any(k.startswith("_orig_mod.") for k in state_dict):
            state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
        raw_model.load_state_dict(state_dict)

        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])

        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("metrics", {}).get("total", float("inf"))

        resume_step = ckpt["epoch"] * len(train_loader)
        for _ in range(resume_step):
            scheduler.step()

        print(f"  Resumed at epoch {start_epoch}, best_val_loss={best_val_loss:.6f}")
        print(f"  Scheduler advanced to step {resume_step}\n")

    # ── Metrics CSV ──────────────────────────────────────────────────────────
    metrics_path = os.path.join(args.out_dir, "metrics.csv")
    if args.resume and os.path.exists(metrics_path):
        print(f"  Appending to existing {metrics_path}")
    else:
        with open(metrics_path, "w") as f:
            f.write("global_step,epoch,train_loss,val_loss,val_delta,"
                    "val_endpoint,val_obstacle\n")

    training_start = time.time()

    # ── Training ─────────────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()
        global_step_offset = (epoch - 1) * len(train_loader)

        # Scheduled sampling ramp
        ss_prob = 0.0
        if args.ss_max_prob > 0 and epoch > args.ss_warmup_epochs:
            ramp_epochs = args.epochs - args.ss_warmup_epochs
            ss_prob = args.ss_max_prob * min(
                1.0, (epoch - args.ss_warmup_epochs) / max(ramp_epochs, 1))

        train_losses = train_one_epoch(
            model, train_loader, optimizer, scheduler, device,
            args.grad_clip, epoch, scalers,
            args.lambda_end, args.lambda_smooth,
            args.lambda_obstacle, args.obstacle_margin_km,
            val_loader=val_loader,
            val_eval_batches=args.val_eval_batches,
            global_step_offset=global_step_offset,
            metrics_path=metrics_path,
            log_every=args.log_every,
            ss_prob=ss_prob,
        )

        log = (f"Epoch {epoch:3d}/{args.epochs} "
               f"| train {train_losses['total']:.6f} "
               f"(delta {train_losses['delta']:.6f} "
               f"end {train_losses['endpoint']:.6f} "
               f"obs {train_losses['obstacle']:.6f})")

        val_losses = None
        if val_loader:
            val_losses = evaluate_full(
                model, val_loader, device, scalers,
                args.lambda_end, args.lambda_smooth,
                args.lambda_obstacle, args.obstacle_margin_km)
            log += (f" | val {val_losses['total']:.6f} "
                    f"(delta {val_losses['delta']:.6f} "
                    f"end {val_losses['endpoint']:.6f} "
                    f"obs {val_losses['obstacle']:.6f})")

            if val_losses["total"] < best_val_loss:
                best_val_loss = val_losses["total"]
                torch.save({
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "pos_mean": scalers.pos.mean,
                    "pos_std": scalers.pos.std,
                    "delta_mean": scalers.delta.mean,
                    "delta_std": scalers.delta.std,
                    "vtype_vocab": vtype_vocab,
                    "num_vessel_types": num_vessel_types,
                    "n_resample": n_resample,
                    "max_obstacles": args.max_obstacles,
                    "metrics": val_losses,
                    "args": vars(args),
                }, os.path.join(args.out_dir, "best.pt"))
                log += "  * saved"

        # Epoch-end row
        epoch_step = epoch * len(train_loader)
        if val_losses:
            with open(metrics_path, "a") as f:
                f.write(f"{epoch_step},{epoch},,"
                        f"{val_losses['total']:.6f},"
                        f"{val_losses['delta']:.6f},"
                        f"{val_losses['endpoint']:.6f},"
                        f"{val_losses['obstacle']:.6f}\n")

        epoch_secs = time.time() - epoch_start
        elapsed = time.time() - training_start
        remaining = elapsed / epoch * (args.epochs - epoch)

        def _fmt(s):
            h, m = divmod(int(s), 3600)
            m, s = divmod(m, 60)
            return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"

        log += (f"  |  epoch {_fmt(epoch_secs)}  elapsed {_fmt(elapsed)}"
                f"  ETA {_fmt(remaining)}")
        print(log)

    print(f"\n  Best val loss : {best_val_loss:.6f}")
    print(f"  Checkpoint    : {args.out_dir}/best.pt")
    print("=" * 65)


if __name__ == "__main__":
    main()
