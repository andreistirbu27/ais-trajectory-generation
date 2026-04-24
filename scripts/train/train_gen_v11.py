#!/usr/bin/env python3
"""
train_gen_v11.py — Trajectory generation v11 training (pointer head).

v11 replaces v10's regression head with a pointer over a water-valid
candidate pool. Loss = cross-entropy on the pointer + (optional) auxiliary
MSE on a candidate-refinement offset + smoothness.

Usage:
    python3 scripts/train/train_gen_v11.py --config configs/trajgen/trajgen_v11.yaml
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
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.data_gen import TrajGenScalers, build_vtype_vocab, train_val_split_gen
from src.data_gen_retrieval import RetrievalConfig, build_and_cache_knn
from src.data_gen_v11 import (CandidateConfig, get_retrieval_v11_loader)
from src.land_mask import LandMask
from src.model_gen_v11 import TrajectoryGeneratorV11


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


def compute_loss(logits, offset, batch, lambda_offset, lambda_smooth):
    """L = L_ce + λ_offset · L_offset + λ_smooth · L_smooth."""
    gt_indices = batch["gt_indices"]                   # (T-1, B)
    Tm1, B, K = logits.shape

    l_ce = F.cross_entropy(
        logits.reshape(-1, K),
        gt_indices.reshape(-1),
        ignore_index=-100,
    )

    # Auxiliary offset loss: predicted offset (in normalized pos space) should
    # match (gt_next_norm - chosen_candidate_norm). We only apply this to
    # valid GT indices.
    device = logits.device
    valid_gt = (gt_indices != -100)                    # (T-1, B)
    if lambda_offset > 0 and valid_gt.any():
        cands_norm = batch["candidates_norm"]          # (T-1, B, K, 2)
        traj_norm  = batch["traj_norm"]                # (T, B, 2)
        gt_next = traj_norm[1:]                        # (T-1, B, 2)

        safe_idx = gt_indices.clamp(min=0)             # invalid → 0 (then masked)
        picked_cand = cands_norm.gather(
            2, safe_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, 2)
        ).squeeze(2)                                    # (T-1, B, 2)

        target_offset = gt_next - picked_cand           # (T-1, B, 2)
        diff = (offset - target_offset)[valid_gt]       # (N, 2)
        l_offset = (diff ** 2).mean()
    else:
        l_offset = torch.tensor(0.0, device=device)

    # Smoothness on the chosen-candidate trajectory deltas.
    if lambda_smooth > 0 and valid_gt.any():
        cands_norm = batch["candidates_norm"]
        safe_idx = gt_indices.clamp(min=0)
        picked_cand = cands_norm.gather(
            2, safe_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, 2)
        ).squeeze(2)                                    # (T-1, B, 2)
        traj_norm = batch["traj_norm"]                  # (T, B, 2)
        p_t = traj_norm[:-1]                            # (T-1, B, 2)
        deltas = picked_cand - p_t                      # (T-1, B, 2)
        if deltas.shape[0] > 1:
            accel = deltas[1:] - deltas[:-1]
            l_smooth = (accel ** 2).mean()
        else:
            l_smooth = torch.tensor(0.0, device=device)
    else:
        l_smooth = torch.tensor(0.0, device=device)

    # Top-1 accuracy diagnostic
    with torch.no_grad():
        pred_idx = logits.argmax(dim=-1)                # (T-1, B)
        correct = ((pred_idx == gt_indices) & valid_gt).sum().item()
        total = valid_gt.sum().item()
        topk_acc = correct / max(total, 1)

    loss = l_ce + lambda_offset * l_offset + lambda_smooth * l_smooth
    return loss, {
        "ce":     l_ce.item(),
        "offset": l_offset.item(),
        "smooth": l_smooth.item(),
        "top1":   float(topk_acc),
        "total":  loss.item(),
    }


def _forward(model, batch):
    return model(
        start=batch["start_norm"],
        end=batch["end_norm"],
        vessel_type=batch["vessel_type"],
        target_traj=batch["traj_norm"],
        retrieved_norm=batch["retrieved_norm"],
        retrieval_mask=batch["retrieval_mask"],
        candidates_norm=batch["candidates_norm"],
        candidate_feats=batch["candidate_feats"],
        candidate_mask=batch["candidate_mask"],
    )


def _val_subsample(model, val_loader, n_batches, device,
                    lambda_offset, lambda_smooth):
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for j, batch in enumerate(val_loader):
            if j >= n_batches:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            logits, offset = _forward(model, batch)
            loss, _ = compute_loss(logits, offset, batch,
                                    lambda_offset, lambda_smooth)
            total += loss.item()
            n += 1
    model.train()
    return total / max(n, 1)


def train_one_epoch(model, loader, optimizer, scheduler, device, grad_clip,
                    epoch, lambda_offset, lambda_smooth,
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
        logits, offset = _forward(model, batch)
        loss, loss_dict = compute_loss(logits, offset, batch,
                                        lambda_offset, lambda_smooth)
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
                    model, val_loader, val_eval_batches, device,
                    lambda_offset, lambda_smooth)
                val_str = f" | val {val_loss:.6f}"

            tqdm.write(
                f"  epoch {epoch:3d} | step {i+1:5d}/{len(loader)} "
                f"| loss {avg['total']:.6f} "
                f"(ce {avg['ce']:.4f} off {avg['offset']:.4f} "
                f"top1 {avg['top1']:.3f})"
                f"{val_str} | grad {grad_norm:.3f} "
                f"| lr {scheduler.get_last_lr()[0]:.2e}"
            )

            global_step = global_step_offset + i + 1
            if metrics_path is not None:
                val_str_csv = f"{val_loss:.6f}" if val_loader else ""
                with open(metrics_path, "a") as f:
                    f.write(f"{global_step},{epoch},{avg['total']:.6f},"
                            f"{val_str_csv},,,,\n")

    return {k: running[k] / max(n, 1) for k in running}


def evaluate_full(model, loader, device, lambda_offset, lambda_smooth):
    model.eval()
    running = {"ce": 0, "offset": 0, "smooth": 0, "top1": 0, "total": 0}
    n = 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits, offset = _forward(model, batch)
            _, loss_dict = compute_loss(logits, offset, batch,
                                         lambda_offset, lambda_smooth)
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


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--config", default=None)
    parser.add_argument("--data_npz", default=None)
    parser.add_argument("--out_dir", default="runs/trajgen_v11")

    # Dataset
    parser.add_argument("--val_frac", type=float, default=0.15)
    parser.add_argument("--seed",     type=int,   default=42)

    # Retrieval
    parser.add_argument("--k_retrieval",            type=int,   default=5)
    parser.add_argument("--retrieval_vtype_weight", type=float, default=0.5)
    parser.add_argument("--exclude_self",           action="store_true", default=True)
    parser.add_argument("--knn_cache",              default=None)
    parser.add_argument("--t_retr",                 type=int,   default=32)

    # Candidates
    parser.add_argument("--n_candidates",     type=int,   default=32)
    parser.add_argument("--cand_cone_deg",    type=float, default=60.0)
    parser.add_argument("--cand_d_min_km",    type=float, default=3.0)
    parser.add_argument("--cand_d_max_km",    type=float, default=25.0)
    parser.add_argument("--water_threshold_km", type=float, default=2.0)
    parser.add_argument("--n_segment_samples", type=int,  default=5)
    parser.add_argument("--use_segment_check", action="store_true", default=True)
    parser.add_argument("--no_segment_check",  dest="use_segment_check", action="store_false")
    parser.add_argument("--use_storm_check",   action="store_true", default=False)

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
    parser.add_argument("--epochs",        type=int,   default=40)
    parser.add_argument("--lr",            type=float, default=3e-4)
    parser.add_argument("--weight_decay",  type=float, default=1e-4)
    parser.add_argument("--grad_clip",     type=float, default=1.0)
    parser.add_argument("--warmup_frac",   type=float, default=0.05)
    parser.add_argument("--num_workers",   type=int,   default=0)

    # Loss weights
    parser.add_argument("--lambda_offset", type=float, default=1.0)
    parser.add_argument("--lambda_smooth", type=float, default=0.5)

    # Scheduled sampling (v11-full; unused in v11-lite)
    parser.add_argument("--use_scheduled_sampling", action="store_true", default=False)
    parser.add_argument("--ss_warmup_epochs", type=int,   default=10)
    parser.add_argument("--ss_max_prob",      type=float, default=0.3)

    # Land mask
    parser.add_argument("--land_sdf",
        default="data/processed/land_sdf_005deg.npz")

    # Resume
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

    if not args.data_npz:
        parser.error("--data_npz is required")
    if not args.knn_cache:
        parser.error("--knn_cache is required")

    set_seed(args.seed)
    device = get_device()
    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 70)
    print("  TRAJECTORY GENERATOR v11 — pointer over water-valid candidates")
    print("=" * 70)
    print(f"  Device  : {device}")
    print(f"  Out dir : {args.out_dir}\n")

    # Land mask (numpy-only for candidate filtering; no torch sampling needed)
    print(f"Loading land SDF: {args.land_sdf}")
    land_mask = LandMask.load(args.land_sdf)
    print(f"  Grid {land_mask.shape}  tau={args.water_threshold_km} km\n")

    # KNN index
    knn = build_and_cache_knn(
        data_npz_path=args.data_npz,
        cache_path=args.knn_cache,
        val_frac=args.val_frac,
        seed=args.seed,
        k=args.k_retrieval,
        vtype_weight=args.retrieval_vtype_weight,
    )
    train_knn = knn["train_knn"]
    val_knn   = knn["val_knn"]
    print(f"  KNN     : train {train_knn.shape}, val {val_knn.shape}\n")

    # Data
    print(f"Loading data: {args.data_npz}")
    data = np.load(args.data_npz, allow_pickle=True)
    trajectories = data["trajectories"].astype(np.float32)
    vessel_types = data["vessel_types"].astype(np.int32)
    track_ids    = data["track_ids"]
    n_resample = trajectories.shape[1]
    print(f"  {len(trajectories):,} trajectories, {n_resample} points each\n")

    (train_traj, train_vt, _,
     val_traj,   val_vt,   _) = train_val_split_gen(
        trajectories, vessel_types, track_ids, args.val_frac, args.seed)
    print(f"  Train: {len(train_traj):,}  |  Val: {len(val_traj):,}\n")

    scalers = TrajGenScalers.fit(train_traj)
    print(f"  Pos scaler   — mean: {scalers.pos.mean}  std: {scalers.pos.std}")
    print(f"  Delta scaler — mean: {scalers.delta.mean}  std: {scalers.delta.std}\n")

    vtype_vocab = build_vtype_vocab(train_vt)
    num_vessel_types = len(vtype_vocab)
    print(f"  Vessel types : {num_vessel_types} codes\n")

    retr_cfg = RetrievalConfig(
        k_retrieval=args.k_retrieval,
        vtype_weight=args.retrieval_vtype_weight,
        exclude_self=args.exclude_self,
    )
    cand_cfg = CandidateConfig(
        K=args.n_candidates,
        cone_deg=args.cand_cone_deg,
        d_min_km=args.cand_d_min_km,
        d_max_km=args.cand_d_max_km,
        tau_km=args.water_threshold_km,
        n_segment_samples=args.n_segment_samples,
        use_segment_check=args.use_segment_check,
        use_storm_check=args.use_storm_check,
    )
    print(f"  Retrieval : k={retr_cfg.k_retrieval}  t_retr={args.t_retr}  "
          f"vtype_w={retr_cfg.vtype_weight}")
    print(f"  Candidates: K={cand_cfg.K}  cone={cand_cfg.cone_deg}°  "
          f"d=[{cand_cfg.d_min_km},{cand_cfg.d_max_km}] km  "
          f"τ={cand_cfg.tau_km}km  seg={cand_cfg.use_segment_check}\n")

    train_loader = get_retrieval_v11_loader(
        trajectories=train_traj, vessel_types=train_vt,
        corpus_trajectories=train_traj, knn_indices=train_knn,
        vtype_vocab=vtype_vocab, scalers=scalers,
        retr_config=retr_cfg, cand_config=cand_cfg,
        land_mask=land_mask,
        batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=args.num_workers)
    val_loader = get_retrieval_v11_loader(
        trajectories=val_traj, vessel_types=val_vt,
        corpus_trajectories=train_traj, knn_indices=val_knn,
        vtype_vocab=vtype_vocab, scalers=scalers,
        retr_config=retr_cfg, cand_config=cand_cfg,
        land_mask=land_mask,
        batch_size=args.batch_size, shuffle=False, drop_last=False,
        num_workers=args.num_workers) if len(val_traj) > 0 else None

    print(f"  Train batches : {len(train_loader):,}")
    if val_loader:
        print(f"  Val batches   : {len(val_loader):,}")

    # Model
    model = TrajectoryGeneratorV11(
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        num_vessel_types=num_vessel_types,
        max_retrieved=args.k_retrieval,
        t_retr=args.t_retr,
        n_points=n_resample,
        k_past=args.k_past,
    ).to(device)

    if hasattr(torch, "compile") and not args.no_compile:
        try:
            model = torch.compile(model)
            print("  torch.compile enabled")
        except Exception as e:
            print(f"  torch.compile skipped: {e}")
    else:
        print("  torch.compile disabled (--no_compile)")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Params  : {n_params:,}")
    print(f"  d_model={args.d_model}  nhead={args.nhead}  "
          f"enc={args.num_encoder_layers}  dec={args.num_decoder_layers}"
          f"  ffn={args.dim_feedforward}")
    print(f"  k_past  : {args.k_past} (None = full causal)")
    print(f"  Loss    : CE + {args.lambda_offset}*offset + "
          f"{args.lambda_smooth}*smooth\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_frac)
    scheduler = build_scheduler(optimizer, warmup_steps, total_steps)
    print(f"  Steps   : {total_steps:,} total | {warmup_steps:,} warmup\n")

    start_epoch = 1
    best_val_loss = float("inf")

    if args.resume:
        print(f"\n  Resuming from: {args.resume}")
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
        for _ in range(resume_step):
            scheduler.step()
        print(f"  Resumed at epoch {start_epoch}, best_val_loss={best_val_loss:.6f}\n")

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
            args.grad_clip, epoch,
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
                model, val_loader, device,
                args.lambda_offset, args.lambda_smooth)
            log += (f" | val {val_losses['total']:.6f} "
                    f"(ce {val_losses['ce']:.4f} "
                    f"off {val_losses['offset']:.4f} "
                    f"top1 {val_losses['top1']:.3f})")

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
                    "k_retrieval": args.k_retrieval,
                    "t_retr": args.t_retr,
                    "k_past": args.k_past,
                    "n_candidates": args.n_candidates,
                    "cand_cone_deg": args.cand_cone_deg,
                    "cand_d_min_km": args.cand_d_min_km,
                    "cand_d_max_km": args.cand_d_max_km,
                    "water_threshold_km": args.water_threshold_km,
                    "n_segment_samples": args.n_segment_samples,
                    "use_segment_check": args.use_segment_check,
                    "land_sdf_path": args.land_sdf,
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
            h, m = divmod(int(s), 3600)
            m, s = divmod(m, 60)
            return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"

        log += (f"  |  epoch {_fmt(epoch_secs)}  elapsed {_fmt(elapsed)}"
                f"  ETA {_fmt(remaining)}")
        print(log)

    print(f"\n  Best val loss : {best_val_loss:.6f}")
    print(f"  Checkpoint    : {args.out_dir}/best.pt")
    print("=" * 70)


if __name__ == "__main__":
    main()
