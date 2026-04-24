#!/usr/bin/env python3
"""
eval_gen_v12.py — Evaluate v12 pointer-over-water-graph trajectory generator.

Reports ADE/FDE/Frechet + length-bucketed metrics + structural land-crossing
guarantee verification for:
  - v12 model output (water-by-construction via cell graph + adjacency mask)
  - Retrieval-top-1 baseline
  - Great-circle baseline

Usage:
    python3 scripts/eval/eval_gen_v12.py \\
        --checkpoint runs/trajgen_v12_lite/best.pt \\
        --water_graph data/processed/water_graph_005deg.npz \\
        --quantized_npz data/processed/trajgen_128_cells_005deg.npz \\
        --knn_cache data/processed/trajgen_128_knn_k5.npz \\
        --n_eval 500 --no_frechet
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.data_gen import Scaler, TrajGenScalers
from src.data_gen_v12 import load_v12_splits
from src.land_mask import LandMask
from src.metrics_gen import evaluate_generation, evaluate_great_circle_baseline
from src.model_gen_v12 import TrajectoryGeneratorV12, V12Config


def get_device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def cells_to_lonlat(cell_ix, cell_iy, offset, bbox, dlon, dlat):
    """Invert quantization back to raw lon/lat arrays."""
    minx, _miny, _maxx, maxy = bbox
    lon = minx + (cell_ix.astype(np.float64) + 0.5) * dlon + offset[..., 0] * dlon
    lat = maxy - (cell_iy.astype(np.float64) + 0.5) * dlat + offset[..., 1] * dlat
    return np.stack([lon, lat], axis=-1).astype(np.float32)


def load_checkpoint(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = argparse.Namespace(**ckpt["args"])
    scalers = TrajGenScalers(
        pos=Scaler(mean=ckpt["pos_mean"], std=ckpt["pos_std"]),
        delta=Scaler(mean=ckpt["delta_mean"], std=ckpt["delta_std"]),
    )
    H, W = ckpt["grid_shape"]
    cfg = V12Config(
        N_x=W, N_y=H, K=ckpt["K"],
        max_retrieved=ckpt["k_retrieval"], t_retr=ckpt["t_retr"],
        d_model=args.d_model, nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        dim_feedforward=args.dim_feedforward, dropout=args.dropout,
        k_past=ckpt.get("k_past"), num_vessel_types=ckpt["num_vessel_types"],
    )
    model = TrajectoryGeneratorV12(cfg).to(device)
    sd = ckpt["model"]
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.eval()
    return model, scalers, ckpt, cfg


def generate_v12(model, splits, scalers, cfg, device, n_eval, seed,
                 batch_size=16, self_bias: float = 0.0):
    """Run v12 autoregressive rollout on n_eval val samples.

    Returns (pred_lonlat, gt_lonlat, retrieved_top1_lonlat, cell_stats).
    """
    val = splits["val"]
    corpus = splits["train_corpus_traj"]
    rng = np.random.default_rng(seed)
    M = len(val["traj"])
    idx = rng.choice(M, size=min(n_eval, M), replace=False); idx.sort()

    meta = splits["meta"]
    bbox = meta["bbox"]; dlon = meta["dlon"]; dlat = meta["dlat"]

    gt_traj    = val["traj"][idx]                       # (N, T, 2)
    gt_vt      = val["vt"][idx]
    gt_cix     = val["cix"][idx]                        # (N, T) int32
    gt_ciy     = val["ciy"][idx]
    gt_off     = val["off"][idx]                        # (N, T, 2)
    gt_knn     = val["knn"][idx]                        # (N, K)
    top1 = corpus[gt_knn[:, 0]]                         # (N, T, 2) nearest-neighbor raw lon/lat

    # Model forward in batches
    N, T, _ = gt_traj.shape
    pred_lonlat = np.zeros_like(gt_traj)
    end_on_land_after = 0
    total_finite_points = 0
    total_points = 0

    vtype_vocab = {int(v): i for i, v in enumerate(np.unique(gt_vt))}  # dummy; replaced below
    # We need the checkpoint's vtype_vocab — load from ckpt in caller
    for start in tqdm(range(0, N, batch_size), desc="v12 rollout"):
        stop = min(N, start + batch_size)
        B = stop - start
        subtraj = gt_traj[start:stop]                   # (B, T, 2)

        start_raw = subtraj[:, 0, :]
        end_raw   = subtraj[:, -1, :]
        start_norm = torch.tensor(
            scalers.pos.transform(start_raw), dtype=torch.float32, device=device)
        end_norm = torch.tensor(
            scalers.pos.transform(end_raw),   dtype=torch.float32, device=device)

        # Build retrieval tokens (K retrieved routes normalized)
        retr_raw = corpus[gt_knn[start:stop]]           # (B, K, T, 2)
        Br, K_r, T_r, _ = retr_raw.shape
        retr_norm = scalers.pos.transform(retr_raw.reshape(-1, 2)).reshape(Br, K_r, T_r, 2)
        retr_norm = torch.tensor(retr_norm, dtype=torch.float32, device=device)
        retr_mask = torch.zeros(Br, K_r, dtype=torch.bool, device=device)

        vtype_vocab_ckpt = getattr(generate_v12, "_vtype_vocab", vtype_vocab)
        vt_idx = [vtype_vocab_ckpt.get(int(v), 0) for v in gt_vt[start:stop]]
        vt_tensor = torch.tensor(vt_idx, dtype=torch.long, device=device)

        start_ix = torch.tensor(gt_cix[start:stop, 0].astype(np.int64), device=device)
        start_iy = torch.tensor(gt_ciy[start:stop, 0].astype(np.int64), device=device)
        start_off = torch.tensor(gt_off[start:stop, 0], dtype=torch.float32, device=device)
        end_ix = torch.tensor(gt_cix[start:stop, -1].astype(np.int64), device=device)
        end_iy = torch.tensor(gt_ciy[start:stop, -1].astype(np.int64), device=device)

        out = model.generate(
            start_norm=start_norm, end_norm=end_norm, vessel_type=vt_tensor,
            retrieved_norm=retr_norm, retrieval_mask=retr_mask,
            start_ix=start_ix, start_iy=start_iy, start_offset=start_off,
            end_ix=end_ix, end_iy=end_iy, n_steps=T,
            self_bias=self_bias,
        )
        ix_out  = out["cell_ix"].cpu().numpy()          # (L, B)
        iy_out  = out["cell_iy"].cpu().numpy()
        off_out = out["offset"].cpu().numpy()            # (L, B, 2)
        L = ix_out.shape[0]

        # Pad to T cells if rollout stopped early (END)
        if L < T:
            ix_pad = np.repeat(ix_out[-1:], T - L, axis=0)
            iy_pad = np.repeat(iy_out[-1:], T - L, axis=0)
            off_pad = np.repeat(off_out[-1:], T - L, axis=0)
            ix_out  = np.concatenate([ix_out,  ix_pad])
            iy_out  = np.concatenate([iy_out,  iy_pad])
            off_out = np.concatenate([off_out, off_pad])

        # (T, B) → (B, T) → reconstruct lon/lat
        ix_bt = ix_out.transpose(1, 0)                  # (B, T)
        iy_bt = iy_out.transpose(1, 0)
        off_bt = off_out.transpose(1, 0, 2)             # (B, T, 2)
        pred_lonlat[start:stop] = cells_to_lonlat(
            ix_bt, iy_bt, off_bt, bbox, dlon, dlat)

    return pred_lonlat, gt_traj, top1


def print_comparison(v12_res, top1_res, gc_res):
    print("\n  COMPARISON  (lower is better)\n")
    print(f"  {'metric':<32s}  {'v12':>10s}  {'retr-top1':>10s}  {'GC':>10s}")
    print(f"  {'-'*32}  {'-'*10}  {'-'*10}  {'-'*10}")
    keys = [("ade_m", "ade_m"), ("normalized_ade", "normalized_ade"),
            ("fde_m", "fde_m"), ("endpoint_error_m", "endpoint_error_m"),
            ("path_length_ratio", "path_length_ratio"),
            ("ade_short_m", "ade_short_m"), ("ade_medium_m", "ade_medium_m"),
            ("ade_long_m", "ade_long_m")]
    for k, label in keys:
        v = v12_res.get(k, float('nan'))
        t = top1_res.get(k, float('nan'))
        g = gc_res.get(k, float('nan'))
        fmt = "{:>10.1f}" if "ratio" not in k and "normalized" not in k else "{:>10.4f}"
        print(f"  {label:<32s}  {fmt.format(v)}  {fmt.format(t)}  {fmt.format(g)}")


def print_land_crossing(land_mask, name_traj_pairs, threshold_km):
    print(f"\n  LAND-CROSSING  (threshold = {threshold_km} km)\n")
    for name, traj in name_traj_pairs:
        stats = land_mask.trajectory_stats(traj, threshold_km=threshold_km)
        print(f"  {name:<28s}"
              f"  land%={stats['land_frac']*100:5.2f}"
              f"  cross%={stats['traj_crossing_rate']*100:5.1f}"
              f"  mean_pts={stats['mean_on_land_pts']:5.2f}"
              f"  max_pen={stats['mean_max_penetration_km']:5.1f} km")


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--checkpoint",    required=True)
    ap.add_argument("--water_graph",   default="data/processed/water_graph_005deg.npz")
    ap.add_argument("--quantized_npz", default="data/processed/trajgen_128_cells_005deg.npz")
    ap.add_argument("--knn_cache",     default="data/processed/trajgen_128_knn_k5.npz")
    ap.add_argument("--land_sdf",      default="data/processed/land_sdf_005deg.npz")
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--seed",     type=int,   default=42)
    ap.add_argument("--n_eval",   type=int,   default=500)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--self_bias", type=float, default=0.0,
                    help="Subtract this from the self-loop logit at inference "
                         "to counter the model's over-prediction of 'stay'. "
                         "Typical sweep values: 0, 1, 2, 3, 5.")
    ap.add_argument("--no_frechet", action="store_true")
    args = ap.parse_args()

    device = get_device()
    print(f"  device: {device}")

    model, scalers, ckpt, cfg = load_checkpoint(args.checkpoint, device)
    generate_v12._vtype_vocab = ckpt["vtype_vocab"]    # closure hack
    print(f"  model  : {cfg}")
    print(f"  params : {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    g = np.load(args.water_graph, allow_pickle=True)
    water_mask = torch.from_numpy(g["water_mask"]).bool().to(device)
    model.attach_water_mask(water_mask)
    print(f"  water_mask attached ({water_mask.shape})")

    splits = load_v12_splits(args.quantized_npz, args.knn_cache,
                             args.val_frac, args.seed)

    pred, gt, top1 = generate_v12(
        model, splits, scalers, cfg, device,
        n_eval=args.n_eval, seed=args.seed, batch_size=args.batch_size,
        self_bias=args.self_bias)

    print(f"\n  v12 trajectories generated: {pred.shape}")

    v12_res  = evaluate_generation(pred,  gt, compute_frechet=not args.no_frechet)
    top1_res = evaluate_generation(top1, gt, compute_frechet=not args.no_frechet)
    gc_res   = evaluate_great_circle_baseline(gt) if not args.no_frechet \
               else _gc_no_frechet(gt)
    print_comparison(v12_res, top1_res, gc_res)

    land_mask = LandMask.load(args.land_sdf)
    print_land_crossing(
        land_mask,
        [("Ground truth",   gt),
         ("v12",             pred),
         ("Retrieval-top-1", top1),
         ("Great-circle",    _gc_array(gt))],
        threshold_km=10.0,
    )
    print_land_crossing(
        land_mask,
        [("Ground truth",   gt),
         ("v12 (strict 0)",  pred)],
        threshold_km=0.0,
    )


def _gc_array(gt_traj):
    from src.metrics_gen import great_circle_trajectory
    out = np.zeros_like(gt_traj)
    for i in range(len(gt_traj)):
        out[i] = great_circle_trajectory(gt_traj[i, 0], gt_traj[i, -1], gt_traj.shape[1])
    return out


def _gc_no_frechet(gt_traj):
    gc = _gc_array(gt_traj)
    return evaluate_generation(gc, gt_traj, compute_frechet=False)


if __name__ == "__main__":
    main()
