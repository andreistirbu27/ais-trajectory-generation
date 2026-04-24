#!/usr/bin/env python3
"""
train_gen_v12.py — v12 pointer training (K-ring over water-valid cell graph).

Usage:
    python3 scripts/train/train_gen_v12.py --config configs/trajgen/trajgen_v12.yaml
"""
import argparse
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.data_gen import TrajGenScalers, build_vtype_vocab
from src.data_gen_retrieval import RetrievalConfig, build_and_cache_knn
from src.data_gen_v12 import get_v12_loader, load_v12_splits
from src.model_gen_v12 import TrajectoryGeneratorV12, V12Config, v12_loss


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ─── Forward + loss ──────────────────────────────────────────────────────────

def _forward(model, batch):
    return model(
        start_norm=batch["start_norm"],
        end_norm=batch["end_norm"],
        vessel_type=batch["vessel_type"],
        retrieved_norm=batch["retrieved_norm"],
        retrieval_mask=batch["retrieval_mask"],
        cell_ix=batch["cell_ix"],
        cell_iy=batch["cell_iy"],
        offset=batch["offset"],
        end_ix=batch["end_ix"],
        end_iy=batch["end_iy"],
    )


def compute_loss(logits, offset_pred, batch, K, lambda_offset, lambda_smooth):
    losses = v12_loss(
        logits, offset_pred,
        target_cell_ix=batch["cell_ix"],
        target_cell_iy=batch["cell_iy"],
        target_offset=batch["offset"],
        K=K,
        lambda_offset=lambda_offset,
        lambda_smooth=lambda_smooth,
    )
    as_floats = {k: losses[k].detach().item() for k in ["ce", "offset", "smooth", "top1"]}
    as_floats["total"] = losses["total"].detach().item()
    return losses["total"], as_floats


def _val_subsample(model, val_loader, n_batches, device,
                    K, lambda_offset, lambda_smooth):
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for j, batch in enumerate(val_loader):
            if j >= n_batches: break
            batch = {k: v.to(device) for k, v in batch.items()}
            logits, off_pred = _forward(model, batch)
            loss, _ = compute_loss(logits, off_pred, batch,
                                    K, lambda_offset, lambda_smooth)
            total += loss.item()
            n += 1
    model.train()
    return total / max(n, 1)


def train_one_epoch(model, loader, optimizer, scheduler, device, grad_clip,
                    epoch, K, lambda_offset, lambda_smooth,
                    val_loader=None, val_eval_batches=30,
                    global_step_offset=0, metrics_path=None, log_every=100):
    model.train()
    running = {"ce": 0, "offset": 0, "smooth": 0, "top1": 0, "total": 0}
    n = 0

    pbar = tqdm(enumerate(loader), total=len(loader),
                desc=f"  Epoch {epoch}", unit="batch", leave=False)
    for i, batch in pbar:
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        logits, off_pred = _forward(model, batch)
        loss, loss_dict = compute_loss(logits, off_pred, batch,
                                        K, lambda_offset, lambda_smooth)
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()

        for k in running: running[k] += loss_dict[k]
        n += 1

        if (i + 1) % log_every == 0:
            avg = {k: running[k] / n for k in running}
            val_str = ""
            val_loss = None
            if val_loader is not None and val_eval_batches > 0:
                val_loss = _val_subsample(model, val_loader, val_eval_batches,
                                           device, K, lambda_offset, lambda_smooth)
                val_str = f" | val {val_loss:.6f}"

            tqdm.write(
                f"  epoch {epoch:3d} | step {i+1:5d}/{len(loader)} "
                f"| loss {avg['total']:.6f} "
                f"(ce {avg['ce']:.4f} off {avg['offset']:.4f} "
                f"top1 {avg['top1']:.3f})"
                f"{val_str} | grad {grad_norm:.3f} "
                f"| lr {scheduler.get_last_lr()[0]:.2e}"
            )

            if metrics_path is not None:
                global_step = global_step_offset + i + 1
                val_csv = f"{val_loss:.6f}" if val_loss is not None else ""
                with open(metrics_path, "a") as f:
                    f.write(f"{global_step},{epoch},{avg['total']:.6f},"
                            f"{val_csv},,,,\n")

    return {k: running[k] / max(n, 1) for k in running}


def evaluate_full(model, loader, device, K, lambda_offset, lambda_smooth):
    model.eval()
    running = {"ce": 0, "offset": 0, "smooth": 0, "top1": 0, "total": 0}
    n = 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits, off_pred = _forward(model, batch)
            _, loss_dict = compute_loss(logits, off_pred, batch,
                                         K, lambda_offset, lambda_smooth)
            for k in running: running[k] += loss_dict[k]
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


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--config", default=None)
    parser.add_argument("--data_npz",
        default="data/processed/trajgen_128.npz",
        help="Raw trajgen NPZ (used for KNN build; should match the NPZ "
             "`quantize_trajgen.py` ingested).")
    parser.add_argument("--quantized_npz",
        default="data/processed/trajgen_128_cells_005deg.npz")
    parser.add_argument("--water_graph",
        default="data/processed/water_graph_005deg.npz")
    parser.add_argument("--knn_cache",
        default="data/processed/trajgen_128_knn_k5.npz")
    parser.add_argument("--out_dir", default="runs/trajgen_v12_lite")

    # Split
    parser.add_argument("--val_frac", type=float, default=0.15)
    parser.add_argument("--seed",     type=int,   default=42)

    # Retrieval
    parser.add_argument("--k_retrieval",            type=int,   default=5)
    parser.add_argument("--retrieval_vtype_weight", type=float, default=0.5)
    parser.add_argument("--t_retr",                 type=int,   default=32)

    # Pointer
    parser.add_argument("--K",                      type=int,   default=10,
        help="K-ring Chebyshev radius over the water grid.")

    # Model
    parser.add_argument("--d_model",            type=int,   default=256)
    parser.add_argument("--nhead",              type=int,   default=8)
    parser.add_argument("--num_encoder_layers", type=int,   default=2)
    parser.add_argument("--num_decoder_layers", type=int,   default=3)
    parser.add_argument("--dim_feedforward",    type=int,   default=1024)
    parser.add_argument("--dropout",            type=float, default=0.1)
    parser.add_argument("--k_past",             type=int,   default=None)

    # Training
    parser.add_argument("--batch_size",    type=int,   default=32)
    parser.add_argument("--epochs",        type=int,   default=20)
    parser.add_argument("--lr",            type=float, default=3e-4)
    parser.add_argument("--weight_decay",  type=float, default=1e-4)
    parser.add_argument("--grad_clip",     type=float, default=1.0)
    parser.add_argument("--warmup_frac",   type=float, default=0.05)
    parser.add_argument("--num_workers",   type=int,   default=0)

    # Loss
    parser.add_argument("--lambda_offset", type=float, default=1.0)
    parser.add_argument("--lambda_smooth", type=float, default=0.5)

    # Resume + compile
    parser.add_argument("--resume", default=None)
    parser.add_argument("--no_compile", action="store_true", default=False)

    # Logging
    parser.add_argument("--val_eval_batches", type=int, default=30)
    parser.add_argument("--log_every",        type=int, default=100)

    _pre = parser.parse_known_args()[0]
    if _pre.config:
        import yaml
        with open(_pre.config) as f:
            cfg = yaml.safe_load(f)
        parser.set_defaults(**{k: v for k, v in cfg.items() if k != "config"})

    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 70)
    print("  TRAJECTORY GENERATOR v12 — pointer over water-cell K-ring")
    print("=" * 70)
    print(f"  Device  : {device}")
    print(f"  Out dir : {args.out_dir}\n")

    # ── Load water graph, quantized NPZ, KNN cache ──
    print(f"Loading water graph: {args.water_graph}")
    g = np.load(args.water_graph, allow_pickle=True)
    water_mask_np = g["water_mask"]
    bbox = g["bbox"]; dlon = float(g["dlon"]); dlat = float(g["dlat"])
    H, W = water_mask_np.shape
    print(f"  grid {H} × {W}  ({100 * water_mask_np.mean():.1f}% water)")

    # Build KNN on the raw trajgen NPZ (v9/v10 path)
    build_and_cache_knn(
        data_npz_path=args.data_npz,
        cache_path=args.knn_cache,
        val_frac=args.val_frac,
        seed=args.seed,
        k=args.k_retrieval,
        vtype_weight=args.retrieval_vtype_weight,
    )

    print(f"Loading quantized splits: {args.quantized_npz}")
    splits = load_v12_splits(
        quantized_npz=args.quantized_npz,
        knn_cache_npz=args.knn_cache,
        val_frac=args.val_frac,
        split_seed=args.seed,
    )
    train = splits["train"]; val = splits["val"]; corpus = splits["train_corpus_traj"]
    print(f"  train {len(train['traj']):,} | val {len(val['traj']):,}")

    scalers = TrajGenScalers.fit(train["traj"])
    vtype_vocab = build_vtype_vocab(train["vt"])
    num_vessel_types = len(vtype_vocab)
    print(f"  pos mean {scalers.pos.mean}  std {scalers.pos.std}")
    print(f"  vessel_types {num_vessel_types}")

    retr_cfg = RetrievalConfig(
        k_retrieval=args.k_retrieval,
        vtype_weight=args.retrieval_vtype_weight,
        exclude_self=True,
    )
    train_loader = get_v12_loader(
        trajectories=train["traj"], vessel_types=train["vt"],
        cell_ix=train["cix"], cell_iy=train["ciy"], offset=train["off"],
        corpus_trajectories=corpus, knn_indices=train["knn"],
        vtype_vocab=vtype_vocab, scalers=scalers, config=retr_cfg,
        batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=args.num_workers)
    val_loader = get_v12_loader(
        trajectories=val["traj"], vessel_types=val["vt"],
        cell_ix=val["cix"], cell_iy=val["ciy"], offset=val["off"],
        corpus_trajectories=corpus, knn_indices=val["knn"],
        vtype_vocab=vtype_vocab, scalers=scalers, config=retr_cfg,
        batch_size=args.batch_size, shuffle=False, drop_last=False,
        num_workers=args.num_workers) if len(val["traj"]) > 0 else None

    print(f"  train batches {len(train_loader):,}"
          + (f"  val batches {len(val_loader):,}" if val_loader else ""))

    # ── Model ──
    mcfg = V12Config(
        N_x=W, N_y=H, K=args.K,
        max_retrieved=args.k_retrieval, t_retr=args.t_retr,
        d_model=args.d_model, nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        dim_feedforward=args.dim_feedforward, dropout=args.dropout,
        k_past=args.k_past, num_vessel_types=num_vessel_types,
    )
    model = TrajectoryGeneratorV12(mcfg).to(device)
    water_mask_tensor = torch.from_numpy(water_mask_np).bool().to(device)
    model.attach_water_mask(water_mask_tensor)
    print(f"  water_mask on device  {water_mask_tensor.dtype}  "
          f"{water_mask_tensor.element_size() * water_mask_tensor.numel() / 1e6:.0f} MB")

    if hasattr(torch, "compile") and not args.no_compile:
        try:
            model = torch.compile(model)
            print("  torch.compile enabled")
        except Exception as e:
            print(f"  torch.compile skipped: {e}")
    else:
        print("  torch.compile disabled")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Params  : {n_params:,}")
    print(f"  K={args.K}  n_cand={mcfg.n_cand}"
          f"  d_model={args.d_model}  enc={args.num_encoder_layers}"
          f"  dec={args.num_decoder_layers}  ffn={args.dim_feedforward}\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                   weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_frac)
    scheduler = build_scheduler(optimizer, warmup_steps, total_steps)

    # ── Resume ──
    start_epoch = 1
    best_val_loss = float("inf")
    if args.resume:
        print(f"  Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        state_dict = ckpt["model"]
        if any(k.startswith("_orig_mod.") for k in state_dict):
            state_dict = {k.replace("_orig_mod.", ""): v
                          for k, v in state_dict.items()}
        raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
        raw_model.load_state_dict(state_dict)
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("metrics", {}).get("total", float("inf"))
        resume_step = ckpt["epoch"] * len(train_loader)
        for _ in range(resume_step): scheduler.step()
        print(f"  Resumed at epoch {start_epoch}, best={best_val_loss:.6f}\n")

    metrics_path = os.path.join(args.out_dir, "metrics.csv")
    if args.resume and os.path.exists(metrics_path):
        print(f"  Appending to existing {metrics_path}")
    else:
        with open(metrics_path, "w") as f:
            f.write("global_step,epoch,train_loss,val_loss,val_ce,"
                    "val_offset,val_smooth,val_top1\n")

    training_start = time.time()

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()
        global_step_offset = (epoch - 1) * len(train_loader)

        train_losses = train_one_epoch(
            model, train_loader, optimizer, scheduler, device,
            args.grad_clip, epoch, args.K,
            args.lambda_offset, args.lambda_smooth,
            val_loader=val_loader,
            val_eval_batches=args.val_eval_batches,
            global_step_offset=global_step_offset,
            metrics_path=metrics_path,
            log_every=args.log_every,
        )

        log = (f"Epoch {epoch:3d}/{args.epochs} "
               f"| train {train_losses['total']:.6f} "
               f"(ce {train_losses['ce']:.4f} "
               f"off {train_losses['offset']:.4f} "
               f"top1 {train_losses['top1']:.3f})")

        val_losses = None
        if val_loader:
            val_losses = evaluate_full(
                model, val_loader, device, args.K,
                args.lambda_offset, args.lambda_smooth)
            log += (f" | val {val_losses['total']:.6f} "
                    f"(ce {val_losses['ce']:.4f} "
                    f"off {val_losses['offset']:.4f} "
                    f"top1 {val_losses['top1']:.3f})")

            if val_losses["total"] < best_val_loss:
                best_val_loss = val_losses["total"]
                torch.save({
                    "epoch": epoch,
                    "model": (model._orig_mod if hasattr(model, "_orig_mod") else model).state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "pos_mean": scalers.pos.mean,
                    "pos_std": scalers.pos.std,
                    "delta_mean": scalers.delta.mean,
                    "delta_std": scalers.delta.std,
                    "vtype_vocab": vtype_vocab,
                    "num_vessel_types": num_vessel_types,
                    "bbox": bbox.tolist(),
                    "dlon": dlon, "dlat": dlat,
                    "grid_shape": [H, W],
                    "K": args.K,
                    "k_retrieval": args.k_retrieval,
                    "t_retr": args.t_retr,
                    "k_past": args.k_past,
                    "metrics": val_losses,
                    "args": vars(args),
                }, os.path.join(args.out_dir, "best.pt"))
                log += "  * saved"

        epoch_step = epoch * len(train_loader)
        if val_losses:
            with open(metrics_path, "a") as f:
                f.write(f"{epoch_step},{epoch},,"
                        f"{val_losses['total']:.6f},"
                        f"{val_losses['ce']:.6f},"
                        f"{val_losses['offset']:.6f},"
                        f"{val_losses['smooth']:.6f},"
                        f"{val_losses['top1']:.6f}\n")

        epoch_secs = time.time() - epoch_start
        elapsed = time.time() - training_start
        remaining = elapsed / epoch * (args.epochs - epoch)

        def _fmt(s):
            h, m = divmod(int(s), 3600); m, s = divmod(m, 60)
            return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"

        log += (f"  | epoch {_fmt(epoch_secs)} elapsed {_fmt(elapsed)}"
                f" ETA {_fmt(remaining)}")
        print(log)

    print(f"\n  Best val loss : {best_val_loss:.6f}")
    print(f"  Checkpoint    : {args.out_dir}/best.pt")


if __name__ == "__main__":
    main()
