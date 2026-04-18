#!/usr/bin/env python3
"""
visualize_gen.py — Visualize generated vs ground-truth trajectories on a map.

Loads a trajectory generation checkpoint, generates routes for a sample of
validation trajectories, and plots them on a map with coastlines.

Usage:
    python3 scripts/eval/visualize_gen.py \
        --data_npz data/processed/trajgen_128.npz \
        --checkpoint runs/trajgen_mvp/best.pt \
        --out_dir outputs/viz_gen

    # Show more/fewer examples
    python3 scripts/eval/visualize_gen.py \
        --data_npz data/processed/trajgen_128.npz \
        --checkpoint runs/trajgen_mvp/best.pt \
        --n_plots 12 --out_dir outputs/viz_gen
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.data_gen import (TrajGenScalers, Scaler, train_val_split_gen,
                          build_vtype_vocab)
from src.model_gen import TrajectoryGenerator
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
    model = TrajectoryGenerator(
        d_model=args.d_model, nhead=args.nhead,
        num_decoder_layers=args.num_decoder_layers,
        dim_feedforward=args.dim_feedforward, dropout=args.dropout,
        num_vessel_types=ckpt["num_vessel_types"],
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, scalers, ckpt["vtype_vocab"], ckpt.get("n_resample", 128), args


def generate_one(model, start_raw, end_raw, vtype_code, vtype_vocab,
                 scalers, n_resample, device):
    """Generate a single trajectory given raw start/end coords."""
    start_norm = scalers.pos.transform(start_raw.reshape(1, 2))
    end_norm = scalers.pos.transform(end_raw.reshape(1, 2))
    vtype_idx = vtype_vocab.get(int(vtype_code), 0)

    start_t = torch.tensor(start_norm, dtype=torch.float32, device=device)
    end_t = torch.tensor(end_norm, dtype=torch.float32, device=device)
    vtype_t = torch.tensor([vtype_idx], dtype=torch.long, device=device)

    with torch.no_grad():
        gen_norm = model.generate(start_t, end_t, vtype_t, n_steps=n_resample,
                                  pos_scaler=scalers.pos,
                                  delta_scaler=scalers.delta)
    # (T, 1, 2) → (T, 2)
    gen_norm_np = gen_norm[:, 0, :].cpu().numpy()
    return scalers.pos.inverse(gen_norm_np)  # (T, 2) [lon, lat]


def plot_trajectory_grid(gt_trajs, pred_trajs, gc_trajs, vtypes, out_path,
                         ncols=4, use_tiles=False):
    """Plot a grid of generated vs ground-truth trajectories."""
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

        gt = gt_trajs[idx]       # (T, 2) [lon, lat]
        pred = pred_trajs[idx]   # (T, 2) [lon, lat]
        gc = gc_trajs[idx]       # (T, 2) [lon, lat]
        vt = int(vtypes[idx])

        # Plot
        ax.plot(gt[:, 0], gt[:, 1], 'g-', linewidth=1.5, alpha=0.8,
                label='Ground truth')
        ax.plot(pred[:, 0], pred[:, 1], 'r--', linewidth=1.2, alpha=0.8,
                label='Generated')
        ax.plot(gc[:, 0], gc[:, 1], ':', color='orange', linewidth=1.0,
                alpha=0.6, label='Great circle')

        # Start/end markers
        ax.plot(gt[0, 0], gt[0, 1], 'gs', markersize=8, zorder=5)
        ax.plot(gt[-1, 0], gt[-1, 1], 'ro', markersize=8, zorder=5)

        # Metrics
        ade = haversine_meters(gt[:, 1], gt[:, 0],
                               pred[:, 1], pred[:, 0]).mean()
        endpoint_err = haversine_meters(gt[-1, 1], gt[-1, 0],
                                        pred[-1, 1], pred[-1, 0])
        gt_len = path_length_m(gt) / 1000
        pred_len = path_length_m(pred) / 1000

        vname = VTYPE_NAME.get(vt, f"type_{vt}")
        ax.set_title(f"{vname} | ADE {ade/1000:.1f}km | end err {endpoint_err/1000:.1f}km\n"
                     f"GT {gt_len:.0f}km  pred {pred_len:.0f}km",
                     fontsize=9)
        ax.tick_params(labelsize=7)
        ax.set_aspect('equal')

        if idx == 0:
            ax.legend(fontsize=7, loc='best')

        # Add map tiles if available
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


def plot_overview_map(gt_trajs, pred_trajs, out_path):
    """Plot all trajectories on a single overview map."""
    fig, ax = plt.subplots(figsize=(14, 10))

    for i in range(len(gt_trajs)):
        gt = gt_trajs[i]
        pred = pred_trajs[i]
        alpha = min(0.5, 10.0 / len(gt_trajs))
        ax.plot(gt[:, 0], gt[:, 1], 'g-', linewidth=0.5, alpha=alpha)
        ax.plot(pred[:, 0], pred[:, 1], 'r-', linewidth=0.5, alpha=alpha)

    ax.plot([], [], 'g-', linewidth=2, label='Ground truth')
    ax.plot([], [], 'r-', linewidth=2, label='Generated')
    ax.legend(fontsize=12)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(f'Trajectory Generation Overview ({len(gt_trajs)} routes)')
    ax.set_aspect('equal')

    try:
        import contextily as ctx
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
    parser.add_argument("--data_npz", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out_dir", default="outputs/viz_gen")
    parser.add_argument("--n_plots", type=int, default=16,
        help="Number of individual trajectory plots")
    parser.add_argument("--n_overview", type=int, default=100,
        help="Number of trajectories on the overview map")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tiles", action="store_true",
        help="Add map tiles (requires contextily)")
    args = parser.parse_args()

    device = get_device()
    os.makedirs(args.out_dir, exist_ok=True)

    # Load
    model, scalers, vtype_vocab, n_resample, train_args = load_checkpoint(
        args.checkpoint, device)

    data = np.load(args.data_npz, allow_pickle=True)
    trajectories = data["trajectories"]
    vessel_types = data["vessel_types"]
    track_ids = data["track_ids"]

    _, _, _, val_traj, val_vt, _ = train_val_split_gen(
        trajectories, vessel_types, track_ids,
        train_args.val_frac, train_args.seed)
    print(f"Val set: {len(val_traj):,} trajectories\n")

    # Sample for visualization
    rng = np.random.RandomState(args.seed)
    n_total = max(args.n_plots, args.n_overview)
    n_total = min(n_total, len(val_traj))
    sample_idx = rng.choice(len(val_traj), n_total, replace=False)

    # Generate
    gt_list, pred_list, gc_list, vt_list = [], [], [], []
    for i in sample_idx:
        gt = val_traj[i]
        vt = val_vt[i]
        pred = generate_one(model, gt[0], gt[-1], vt, vtype_vocab,
                            scalers, n_resample, device)
        gc = great_circle_trajectory(gt[0], gt[-1], n_resample)
        gt_list.append(gt)
        pred_list.append(pred)
        gc_list.append(gc)
        vt_list.append(vt)

    gt_arr = np.array(gt_list)
    pred_arr = np.array(pred_list)
    gc_arr = np.array(gc_list)
    vt_arr = np.array(vt_list)

    # Plot individual trajectories
    plot_trajectory_grid(
        gt_arr[:args.n_plots], pred_arr[:args.n_plots],
        gc_arr[:args.n_plots], vt_arr[:args.n_plots],
        os.path.join(args.out_dir, "trajectories.png"),
        use_tiles=args.tiles)

    # Plot overview map
    plot_overview_map(
        gt_arr[:args.n_overview], pred_arr[:args.n_overview],
        os.path.join(args.out_dir, "overview.png"))

    print(f"\nAll plots saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
