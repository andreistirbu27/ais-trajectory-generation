#!/usr/bin/env python3
"""
visualize_gen_retrieval.py — Visualize v9 retrieval-augmented generation.

For a sample of val trajectories, plots a grid showing:
  - ground truth (green)
  - K retrieved neighbor routes from train (thin blue)
  - generated route (red dashed)
  - great-circle baseline (dotted orange)

Usage:
    python3 scripts/eval/visualize_gen_retrieval.py \
        --data_npz data/processed/trajgen_128.npz \
        --checkpoint runs/trajgen_v9_retrieval/best.pt \
        --knn_cache data/processed/trajgen_128_knn_k5.npz \
        --out_dir outputs/viz_v9

    # Add map tiles
    python3 scripts/eval/visualize_gen_retrieval.py \
        --data_npz data/processed/trajgen_128.npz \
        --checkpoint runs/trajgen_v9_retrieval/best.pt \
        --knn_cache data/processed/trajgen_128_knn_k5.npz \
        --out_dir outputs/viz_v9 --tiles
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.data_gen import TrajGenScalers, Scaler, train_val_split_gen
from src.model_gen_retrieval import RetrievalTrajectoryGenerator
from src.metrics_gen import (haversine_meters, great_circle_trajectory,
                              path_length_m)

VTYPE_NAME = {
    **{c: "Passenger" for c in range(60, 70)},
    **{c: "Cargo"     for c in range(70, 80)},
    **{c: "Tanker"    for c in range(80, 90)},
}


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_checkpoint(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = argparse.Namespace(**ckpt["args"])
    scalers = TrajGenScalers(
        pos=Scaler(mean=ckpt["pos_mean"], std=ckpt["pos_std"]),
        delta=Scaler(mean=ckpt["delta_mean"], std=ckpt["delta_std"]),
    )
    model = RetrievalTrajectoryGenerator(
        d_model=args.d_model, nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        dim_feedforward=args.dim_feedforward, dropout=args.dropout,
        num_vessel_types=ckpt["num_vessel_types"],
        max_retrieved=ckpt["k_retrieval"],
        route_encoder=ckpt["route_encoder"],
        n_points=ckpt.get("n_resample", 128),
    ).to(device)
    state_dict = ckpt["model"]
    if any(k.startswith("_orig_mod.") for k in state_dict):
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    return model, scalers, ckpt["vtype_vocab"], ckpt.get("n_resample", 128), args


def generate_one(model, start_raw, end_raw, vtype_code, retrieved_raw,
                  vtype_vocab, scalers, n_resample, device):
    """Generate a single route. retrieved_raw: (K, T, 2) raw [lon, lat]."""
    start_norm = scalers.pos.transform(start_raw.reshape(1, 2))
    end_norm   = scalers.pos.transform(end_raw.reshape(1, 2))
    vtype_idx  = vtype_vocab.get(int(vtype_code), 0)

    K, T, _ = retrieved_raw.shape
    retrieved_norm = scalers.pos.transform(
        retrieved_raw.reshape(-1, 2)).reshape(1, K, T, 2)
    retrieval_mask = np.zeros((1, K), dtype=bool)

    start_t = torch.tensor(start_norm, dtype=torch.float32, device=device)
    end_t   = torch.tensor(end_norm,   dtype=torch.float32, device=device)
    vtype_t = torch.tensor([vtype_idx], dtype=torch.long,    device=device)
    retr_t  = torch.tensor(retrieved_norm, dtype=torch.float32, device=device)
    rmask_t = torch.tensor(retrieval_mask, dtype=torch.bool,    device=device)

    with torch.no_grad():
        gen_norm = model.generate(
            start_t, end_t, vtype_t, retr_t, rmask_t, n_steps=n_resample,
            pos_scaler=scalers.pos, delta_scaler=scalers.delta)
    gen_norm_np = gen_norm[:, 0, :].cpu().numpy()
    return scalers.pos.inverse(gen_norm_np)


def plot_grid(gt_trajs, pred_trajs, gc_trajs, retrieved_trajs, vtypes,
              out_path, ncols=4, use_tiles=False):
    n = len(gt_trajs)
    nrows = max(1, (n + ncols - 1) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes[None, :]
    elif ncols == 1:
        axes = axes[:, None]

    for idx in range(nrows * ncols):
        ax = axes[idx // ncols, idx % ncols]
        if idx >= n:
            ax.set_visible(False)
            continue

        gt   = gt_trajs[idx]
        pred = pred_trajs[idx]
        gc   = gc_trajs[idx]
        retr = retrieved_trajs[idx]   # (K, T, 2)
        vt = int(vtypes[idx])

        # Retrieved routes — thin blue, drawn first so GT/pred sit on top
        for k in range(retr.shape[0]):
            ax.plot(retr[k, :, 0], retr[k, :, 1], '-', color='steelblue',
                    linewidth=0.8, alpha=0.5,
                    label='Retrieved (K)' if k == 0 else None)

        ax.plot(gt[:, 0],   gt[:, 1],   'g-',  linewidth=1.5, alpha=0.9,
                label='Ground truth')
        ax.plot(pred[:, 0], pred[:, 1], 'r--', linewidth=1.2, alpha=0.9,
                label='Generated')
        ax.plot(gc[:, 0],   gc[:, 1],   ':',   color='orange', linewidth=1.0,
                alpha=0.6, label='Great circle')
        ax.plot(gt[0, 0],   gt[0, 1],   'gs', markersize=8, zorder=5)
        ax.plot(gt[-1, 0],  gt[-1, 1],  'ro', markersize=8, zorder=5)

        ade = haversine_meters(gt[:, 1], gt[:, 0],
                               pred[:, 1], pred[:, 0]).mean()
        endpoint_err = haversine_meters(gt[-1, 1], gt[-1, 0],
                                        pred[-1, 1], pred[-1, 0])
        gt_len   = path_length_m(gt)   / 1000
        pred_len = path_length_m(pred) / 1000

        vname = VTYPE_NAME.get(vt, f"type_{vt}")
        ax.set_title(f"{vname} | ADE {ade/1000:.1f}km | end {endpoint_err/1000:.1f}km\n"
                     f"GT {gt_len:.0f}km  pred {pred_len:.0f}km",
                     fontsize=9)
        ax.tick_params(labelsize=7)
        ax.set_aspect('equal')

        if idx == 0:
            ax.legend(fontsize=7, loc='best')

        if use_tiles:
            try:
                import contextily as ctx
                ax.set_xlim(ax.get_xlim())
                ax.set_ylim(ax.get_ylim())
                ctx.add_basemap(ax, crs="EPSG:4326",
                                source=ctx.providers.CartoDB.Positron,
                                zoom='auto', alpha=0.5)
            except Exception:
                pass

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved → {out_path}")


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_npz",   required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--knn_cache",  required=True)
    parser.add_argument("--out_dir",    default="outputs/viz_v9")
    parser.add_argument("--n_plots",    type=int, default=16)
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--tiles",      action="store_true")
    args = parser.parse_args()

    device = get_device()
    os.makedirs(args.out_dir, exist_ok=True)

    model, scalers, vtype_vocab, n_resample, train_args = load_checkpoint(
        args.checkpoint, device)
    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"  k_retrieval={train_args.k_retrieval}  "
          f"route_encoder={train_args.route_encoder}\n")

    data = np.load(args.data_npz, allow_pickle=True)
    trajectories = data["trajectories"].astype(np.float32)
    vessel_types = data["vessel_types"].astype(np.int32)
    track_ids    = data["track_ids"]

    train_traj, _, _, val_traj, val_vt, _ = train_val_split_gen(
        trajectories, vessel_types, track_ids,
        train_args.val_frac, train_args.seed)
    print(f"Train corpus: {len(train_traj):,}  |  Val: {len(val_traj):,}\n")

    knn = np.load(args.knn_cache)
    val_knn = knn["val_knn"]

    rng = np.random.RandomState(args.seed)
    n_plots = min(args.n_plots, len(val_traj))
    sample_idx = rng.choice(len(val_traj), n_plots, replace=False)

    gt_list, pred_list, gc_list, retr_list, vt_list = [], [], [], [], []
    for i in sample_idx:
        gt = val_traj[i]
        vt = val_vt[i]
        retrieved = train_traj[val_knn[i]]    # (K, T, 2) raw
        pred = generate_one(model, gt[0], gt[-1], vt, retrieved,
                            vtype_vocab, scalers, n_resample, device)
        gc = great_circle_trajectory(gt[0], gt[-1], n_resample)
        gt_list.append(gt)
        pred_list.append(pred)
        gc_list.append(gc)
        retr_list.append(retrieved)
        vt_list.append(vt)

    plot_grid(
        np.array(gt_list), np.array(pred_list), np.array(gc_list),
        np.array(retr_list), np.array(vt_list),
        os.path.join(args.out_dir, "trajectories.png"),
        use_tiles=args.tiles)

    print(f"\nAll plots saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
