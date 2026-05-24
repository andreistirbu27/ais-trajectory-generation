#!/usr/bin/env python3
"""
train_gen_v13.py — v13 (TrajDiff) diffusion-Transformer training.

Paper-track training loop for the EDM-style diffusion model defined in
`src/model_gen_v13.py` and `src/diffusion.py`. Reuses the v9/v10 retrieval
dataset and KNN cache, so v13 is directly comparable with v10 — only the
backbone and objective change.

Usage:
    paper/.venv/bin/python3 scripts/paper/train_gen_v13.py \\
        --config configs/paper/v13/trajdiff_base.yaml
    # Resume:
    paper/.venv/bin/python3 scripts/paper/train_gen_v13.py \\
        --config configs/paper/v13/trajdiff_base.yaml \\
        --resume runs/paper/v13_trajdiff_base/last.pt

Outputs (under `out_dir`):
    config.yaml         — copy of the config used for this run
    best.pt             — EMA weights with the best val loss so far
    last.pt             — most recent checkpoint (for --resume)
    metrics.csv         — per-step training + per-epoch val (loss only;
                          full ADE eval is in eval_gen_v13.py)
"""
from __future__ import annotations

import argparse
import copy
import csv
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.optim import AdamW

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.data_gen import TrajGenScalers, build_vtype_vocab, train_val_split_gen
from src.data_gen_retrieval import (
    RetrievalConfig, build_and_cache_knn, get_retrieval_trajgen_loader,
)
from src.diffusion import EDMSchedule, edm_training_loss, make_endpoint_anchor
from src.model_gen_v13 import TrajDiff, TrajDiffConfig, make_cfg_conds


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
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


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def cosine_lr_schedule(step: int, total: int, warmup: int, min_ratio: float = 1e-2):
    if step < warmup:
        return step / max(warmup, 1)
    t = (step - warmup) / max(total - warmup, 1)
    return max(min_ratio, 0.5 * (1.0 + math.cos(math.pi * t)))


class EMA:
    """Exponential moving average of model parameters for diffusion stability."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            self.shadow[n].mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)

    def state_dict(self):
        return {n: v.clone() for n, v in self.shadow.items()}

    def load_state_dict(self, sd):
        for n, v in sd.items():
            self.shadow[n] = v.clone()

    @torch.no_grad()
    def copy_to(self, model: torch.nn.Module):
        for n, p in model.named_parameters():
            if n in self.shadow:
                p.copy_(self.shadow[n])


# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------
def build_loaders(cfg: dict, device: torch.device):
    # KNN cache must be built/loaded before splitting trajectories — cache
    # rows are aligned to the same MMSI-grouped split (val_frac, seed).
    knn = build_and_cache_knn(
        data_npz_path=cfg["data_npz"],
        cache_path=cfg["knn_cache"],
        val_frac=cfg["val_frac"],
        seed=cfg["seed"],
        k=cfg["k_retrieval"],
        vtype_weight=cfg["retrieval_vtype_weight"],
        test_frac=cfg.get("test_frac", 0.0),
    )
    train_knn = knn["train_knn"]
    val_knn   = knn["val_knn"]
    print(f"KNN: train {train_knn.shape}, val {val_knn.shape}")

    print(f"Loading {cfg['data_npz']}...")
    data = np.load(cfg["data_npz"], allow_pickle=True)
    trajectories = data["trajectories"].astype(np.float32)
    vessel_types = data["vessel_types"].astype(np.int32)
    track_ids    = data["track_ids"]
    print(f"  {len(trajectories):,} trajectories, T={trajectories.shape[1]}")

    if cfg.get("test_frac", 0.0) > 0:
        (train_traj, train_vt, _,
         val_traj,   val_vt,   _,
         _t_traj, _t_vt, _) = train_val_split_gen(
            trajectories, vessel_types, track_ids,
            cfg["val_frac"], cfg["seed"], test_frac=cfg["test_frac"])
    else:
        (train_traj, train_vt, _,
         val_traj,   val_vt,   _) = train_val_split_gen(
            trajectories, vessel_types, track_ids,
            cfg["val_frac"], cfg["seed"])
    print(f"  Train: {len(train_traj):,}  Val: {len(val_traj):,}")

    scalers = TrajGenScalers.fit(train_traj)
    vtype_vocab = build_vtype_vocab(train_vt)

    retr_cfg = RetrievalConfig(
        k_retrieval=cfg["k_retrieval"],
        vtype_weight=cfg["retrieval_vtype_weight"],
        exclude_self=cfg["exclude_self"],
    )

    pin = device.type == "cuda"
    train_loader = get_retrieval_trajgen_loader(
        trajectories=train_traj, vessel_types=train_vt,
        corpus_trajectories=train_traj, knn_indices=train_knn,
        vtype_vocab=vtype_vocab, scalers=scalers, config=retr_cfg,
        batch_size=cfg["batch_size"], shuffle=True, drop_last=True,
        num_workers=2, pin_memory=pin, persistent_workers=True,
    )
    val_loader = get_retrieval_trajgen_loader(
        trajectories=val_traj, vessel_types=val_vt,
        corpus_trajectories=train_traj, knn_indices=val_knn,
        vtype_vocab=vtype_vocab, scalers=scalers, config=retr_cfg,
        batch_size=cfg["batch_size"], shuffle=False, drop_last=False,
        num_workers=2, pin_memory=pin, persistent_workers=True,
    )
    return train_loader, val_loader, scalers, vtype_vocab


def cond_from_batch(batch, device):
    """Convert a retrieval-loader batch into the cond dict TrajDiff expects."""
    return {
        "start": batch["start_norm"].to(device),
        "end":   batch["end_norm"].to(device),
        "vessel_type": batch["vessel_type"].to(device),
        "retrieved":   batch["retrieved_norm"].to(device),
    }


# -----------------------------------------------------------------------------
# Train / val one epoch
# -----------------------------------------------------------------------------
def train_one_epoch(model, ema, loader, opt, schedule, cfg, device, scaler, epoch,
                    csv_writer, step_offset):
    model.train()
    losses = []
    total_steps = cfg["epochs"] * len(loader)
    warmup_steps = int(cfg["warmup_frac"] * total_steps)

    for batch_idx, batch in enumerate(loader):
        step = step_offset + batch_idx

        # LR schedule
        lr_mult = cosine_lr_schedule(step, total_steps, warmup_steps)
        for pg in opt.param_groups:
            pg["lr"] = cfg["lr"] * lr_mult

        x = batch["traj_norm"].permute(1, 0, 2).to(device)   # (T, B, 2) → (B, T, 2)
        cond = cond_from_batch(batch, device)
        cond = make_cfg_conds(cond, p_uncond=cfg["p_uncond"], device=device)

        anchor_mask, _ = make_endpoint_anchor(cond["start"], cond["end"], x.shape[1], device=device)

        opt.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss = edm_training_loss(model, x, cond, schedule, anchor_mask=anchor_mask)
        else:
            loss = edm_training_loss(model, x, cond, schedule, anchor_mask=anchor_mask)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        opt.step()
        ema.update(model)

        losses.append(loss.item())
        if step % cfg["log_every"] == 0:
            print(f"  ep {epoch} step {step:6d} loss {loss.item():.4f} lr {cfg['lr']*lr_mult:.2e}")
            csv_writer.writerow({"epoch": epoch, "global_step": step,
                                 "train_loss": f"{loss.item():.6f}",
                                 "val_loss": ""})

    return float(np.mean(losses)), step_offset + len(loader)


@torch.no_grad()
def validate(model_eval, loader, schedule, cfg, device):
    """Validation loss in EDM training-loss form (cheap; sampling-based ADE
    is in eval_gen_v13.py)."""
    model_eval.eval()
    losses = []
    for batch in loader:
        x = batch["traj_norm"].permute(1, 0, 2).to(device)
        cond = cond_from_batch(batch, device)
        # No CFG dropout at val.
        anchor_mask, _ = make_endpoint_anchor(cond["start"], cond["end"], x.shape[1], device=device)
        loss = edm_training_loss(model_eval, x, cond, schedule, anchor_mask=anchor_mask)
        losses.append(loss.item())
    return float(np.mean(losses))


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", default=None,
                    help="Path to a checkpoint (.pt) to resume from.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("train_seed", 0))

    device = get_device()
    print(f"Using device {device}")

    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, out_dir / "config.yaml")

    # ---- Data ----
    train_loader, val_loader, scalers, vtype_vocab = build_loaders(cfg, device)

    # ---- Model ----
    mdl_cfg = TrajDiffConfig(
        seq_len=cfg["seq_len"], in_dim=cfg["in_dim"],
        n_vessel_types=cfg["n_vessel_types"], vessel_emb_dim=cfg["vessel_emb_dim"],
        d_model=cfg["d_model"], n_heads=cfg["n_heads"], ffn_mult=cfg["ffn_mult"],
        n_layer=cfg["n_layer"], dropout=cfg["dropout"], p_uncond=cfg["p_uncond"],
        max_retrieved=cfg["k_retrieval"], t_retr=cfg["t_retr"],
        time_emb_dim=cfg["time_emb_dim"],
    )
    model = TrajDiff(mdl_cfg).to(device)
    n_params = model.count_parameters()
    print(f"TrajDiff: {n_params:,} parameters")

    ema = EMA(model, decay=cfg["ema_decay"])
    opt = AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"],
                betas=(0.9, 0.95))
    schedule = EDMSchedule(
        sigma_min=cfg["sigma_min"], sigma_max=cfg["sigma_max"],
        sigma_data=cfg["sigma_data"], rho=cfg["rho"],
    )
    use_amp = (cfg.get("amp", "off") == "bf16" and device.type == "cuda")
    grad_scaler = None  # bf16 doesn't need a GradScaler

    # ---- Resume ----
    start_epoch = 0
    global_step = 0
    best_val = float("inf")
    if args.resume:
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        ema.load_state_dict(ckpt["ema"])
        opt.load_state_dict(ckpt["opt"])
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt["global_step"]
        best_val = ckpt.get("best_val", float("inf"))

    # ---- Metrics CSV ----
    metrics_path = out_dir / "metrics.csv"
    is_new = not metrics_path.exists()
    metrics_file = open(metrics_path, "a", newline="")
    csv_writer = csv.DictWriter(metrics_file,
                                fieldnames=["epoch", "global_step", "train_loss", "val_loss"])
    if is_new:
        csv_writer.writeheader()
        metrics_file.flush()

    # ---- Training loop ----
    for epoch in range(start_epoch, cfg["epochs"]):
        t0 = time.time()
        train_loss, global_step = train_one_epoch(
            model, ema, train_loader, opt, schedule, cfg, device, grad_scaler,
            epoch, csv_writer, global_step,
        )

        if (epoch + 1) % cfg["val_every_epochs"] == 0 or epoch == cfg["epochs"] - 1:
            # Val with EMA weights.
            model_eval = copy.deepcopy(model)
            ema.copy_to(model_eval)
            val_loss = validate(model_eval, val_loader, schedule, cfg, device)
            del model_eval

            dt = time.time() - t0
            print(f"[epoch {epoch}] train {train_loss:.4f}  val {val_loss:.4f}  ({dt:.1f}s)")
            csv_writer.writerow({"epoch": epoch, "global_step": global_step,
                                 "train_loss": f"{train_loss:.6f}",
                                 "val_loss":   f"{val_loss:.6f}"})
            metrics_file.flush()

            if val_loss < best_val:
                best_val = val_loss
                torch.save({"model": model.state_dict(),
                            "ema": ema.state_dict(),
                            "opt": opt.state_dict(),
                            "epoch": epoch, "global_step": global_step,
                            "best_val": best_val,
                            "scalers": {"pos_mean": scalers.pos.mean.tolist(),
                                        "pos_std": scalers.pos.std.tolist(),
                                        "delta_mean": scalers.delta.mean.tolist(),
                                        "delta_std": scalers.delta.std.tolist()},
                            "vtype_vocab": vtype_vocab,
                            "config": cfg},
                           out_dir / "best.pt")

        if (epoch + 1) % cfg["ckpt_every_epochs"] == 0 or epoch == cfg["epochs"] - 1:
            torch.save({"model": model.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "epoch": epoch, "global_step": global_step,
                        "best_val": best_val,
                        "scalers": {"pos_mean": scalers.pos.mean.tolist(),
                                    "pos_std": scalers.pos.std.tolist(),
                                    "delta_mean": scalers.delta.mean.tolist(),
                                    "delta_std": scalers.delta.std.tolist()},
                        "vtype_vocab": vtype_vocab,
                        "config": cfg},
                       out_dir / "last.pt")

    metrics_file.close()
    print(f"Done. best val loss = {best_val:.4f}")


if __name__ == "__main__":
    main()
