#!/usr/bin/env python3
"""
extract_motivation_figs.py — produce small "show, don't tell" figures that
illustrate failure modes which motivated each design pivot in the report.

Outputs (under figures/):
  motiv_dirty_gt_on_river.png       — GT trajectories from trajgen_128.npz
                                       that sit on inland water (motivates E1)
  motiv_v5_lands_on_coast.png        — v5 prediction that crosses a peninsula
                                       (motivates v9 retrieval + v10 land-aware)
  motiv_v9_long_route_collapse.png   — v9 prediction on a long route, drifting
                                       wildly; retrieval-top-1 next to it goes
                                       the right way (motivates v10 per-step retr)
  motiv_v10_raw_crosses_land.png     — v10 raw output crosses small island;
                                       same prediction after WaterRouter snap-only
                                       (motivates the post-processor)
  motiv_v12_zigzag.png               — v12 prediction showing Manhattan zigzag
                                       artefact of K-ring cell quantization
                                       (motivates the v12 abandonment)

All figures share the same plotting style: land shaded grey (from the 0.05°
SDF), GT in solid green, model output in red dashed, retrieval/baselines in
thin steelblue.  Each figure is a small two-panel layout so it fits inline in
the report at half-page width.

Usage:
    python3 scripts/eval/extract_motivation_figs.py
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.data_gen import TrajGenScalers, Scaler, train_val_split_gen
from src.land_mask import LandMask
from src.metrics_gen import (haversine_meters, great_circle_trajectory,
                              path_length_m)
from src.model_gen import TrajectoryGenerator
from src.model_gen_retrieval import RetrievalTrajectoryGenerator
from src.model_gen_v10 import TrajectoryGeneratorV10
from src.model_gen_v12 import TrajectoryGeneratorV12
from src.water_router import WaterRouter


# ─── Plotting helpers ───────────────────────────────────────────────────────


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def shade_land(ax, land_mask: LandMask, lon_min, lon_max, lat_min, lat_max,
               alpha: float = 0.45):
    """Shade land cells from the 0.05° SDF inside the given lon/lat box.

    LandMask stores sdf_km as (H, W) with row 0 = maxy (north).
    The bbox is (minx, miny, maxx, maxy).
    """
    minx, miny, maxx, maxy = land_mask.bbox
    H, W = land_mask.sdf_km.shape
    # Lon centres: minx + (col + 0.5) * dlon
    # Lat centres: maxy - (row + 0.5) * dlat  (row 0 at top)
    col_min = int(max(0, np.floor((lon_min - minx) / land_mask.dlon) - 1))
    col_max = int(min(W, np.ceil((lon_max - minx) / land_mask.dlon) + 1))
    row_min = int(max(0, np.floor((maxy - lat_max) / land_mask.dlat) - 1))
    row_max = int(min(H, np.ceil((maxy - lat_min) / land_mask.dlat) + 1))
    if col_max <= col_min or row_max <= row_min:
        return
    sub_sdf = land_mask.sdf_km[row_min:row_max, col_min:col_max]
    sub_lons = minx + (np.arange(col_min, col_max) + 0.5) * land_mask.dlon
    sub_lats = maxy - (np.arange(row_min, row_max) + 0.5) * land_mask.dlat
    # Flip lats so they ascend (matplotlib pcolormesh wants increasing y).
    sub_lats = sub_lats[::-1]
    sub_sdf = sub_sdf[::-1, :]
    land = (sub_sdf > 0).astype(np.float32)
    if land.any():
        ax.pcolormesh(sub_lons, sub_lats, land,
                      cmap=plt.cm.Greys, alpha=alpha, shading="auto",
                      vmin=0, vmax=1, zorder=0)


def plot_traj(ax, traj, color, style="-", label=None, lw=1.6, alpha=0.95,
              zorder=4):
    ax.plot(traj[:, 0], traj[:, 1], style, color=color,
            linewidth=lw, alpha=alpha, label=label, zorder=zorder)


def mark_endpoints(ax, traj, ms=7):
    ax.plot(traj[0, 0],  traj[0, 1],  marker="s", color="black",
            markersize=ms, zorder=8, linestyle="None")
    ax.plot(traj[-1, 0], traj[-1, 1], marker="o", color="black",
            markersize=ms, zorder=8, linestyle="None")


def autobox(trajs, pad_frac=0.15):
    """Compute a tight lon/lat box around a list of trajectories."""
    all_pts = np.concatenate(trajs, axis=0)
    lon_min, lon_max = all_pts[:, 0].min(), all_pts[:, 0].max()
    lat_min, lat_max = all_pts[:, 1].min(), all_pts[:, 1].max()
    dlon = max(lon_max - lon_min, 0.2)
    dlat = max(lat_max - lat_min, 0.2)
    pad_lon = dlon * pad_frac
    pad_lat = dlat * pad_frac
    return (lon_min - pad_lon, lon_max + pad_lon,
            lat_min - pad_lat, lat_max + pad_lat)


def style_axis(ax, lon_min, lon_max, lat_min, lat_max, title=None):
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(labelsize=8)
    if title is not None:
        ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.25)


# ─── Checkpoint loaders ─────────────────────────────────────────────────────


def load_v5(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    a = argparse.Namespace(**ckpt["args"])
    scalers = TrajGenScalers(
        pos=Scaler(mean=ckpt["pos_mean"], std=ckpt["pos_std"]),
        delta=Scaler(mean=ckpt["delta_mean"], std=ckpt["delta_std"]),
    )
    model = TrajectoryGenerator(
        d_model=a.d_model, nhead=a.nhead,
        num_encoder_layers=getattr(a, "num_encoder_layers", 0),
        num_decoder_layers=a.num_decoder_layers,
        dim_feedforward=a.dim_feedforward, dropout=a.dropout,
        num_vessel_types=ckpt["num_vessel_types"],
    ).to(device)
    sd = ckpt["model"]
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.eval()
    return model, scalers, ckpt["vtype_vocab"], ckpt.get("n_resample", 128)


def load_v9(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    a = argparse.Namespace(**ckpt["args"])
    scalers = TrajGenScalers(
        pos=Scaler(mean=ckpt["pos_mean"], std=ckpt["pos_std"]),
        delta=Scaler(mean=ckpt["delta_mean"], std=ckpt["delta_std"]),
    )
    model = RetrievalTrajectoryGenerator(
        d_model=a.d_model, nhead=a.nhead,
        num_encoder_layers=a.num_encoder_layers,
        num_decoder_layers=a.num_decoder_layers,
        dim_feedforward=a.dim_feedforward, dropout=a.dropout,
        num_vessel_types=ckpt["num_vessel_types"],
        max_retrieved=ckpt["k_retrieval"],
        route_encoder=ckpt["route_encoder"],
        n_points=ckpt.get("n_resample", 128),
    ).to(device)
    sd = ckpt["model"]
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.eval()
    return model, scalers, ckpt["vtype_vocab"], ckpt.get("n_resample", 128), a


def load_v10(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    a = argparse.Namespace(**ckpt["args"])
    scalers = TrajGenScalers(
        pos=Scaler(mean=ckpt["pos_mean"], std=ckpt["pos_std"]),
        delta=Scaler(mean=ckpt["delta_mean"], std=ckpt["delta_std"]),
    )
    model = TrajectoryGeneratorV10(
        d_model=a.d_model, nhead=a.nhead,
        num_encoder_layers=a.num_encoder_layers,
        num_decoder_layers=a.num_decoder_layers,
        dim_feedforward=a.dim_feedforward, dropout=a.dropout,
        num_vessel_types=ckpt["num_vessel_types"],
        max_retrieved=ckpt["k_retrieval"],
        t_retr=ckpt["t_retr"],
        n_points=ckpt.get("n_resample", 128),
        k_past=ckpt.get("k_past", None),
    ).to(device)
    sd = ckpt["model"]
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.eval()
    return model, scalers, ckpt["vtype_vocab"], ckpt.get("n_resample", 128), a


# ─── Single-trajectory generators ───────────────────────────────────────────


def gen_v5(model, start_raw, end_raw, vtype_code, vtype_vocab,
           scalers, n_resample, device):
    start_norm = scalers.pos.transform(start_raw.reshape(1, 2))
    end_norm   = scalers.pos.transform(end_raw.reshape(1, 2))
    vt_idx = vtype_vocab.get(int(vtype_code), 0)
    st = torch.tensor(start_norm, dtype=torch.float32, device=device)
    et = torch.tensor(end_norm,   dtype=torch.float32, device=device)
    vt = torch.tensor([vt_idx],   dtype=torch.long,    device=device)
    with torch.no_grad():
        g = model.generate(st, et, vt, n_steps=n_resample,
                           pos_scaler=scalers.pos, delta_scaler=scalers.delta)
    return scalers.pos.inverse(g[:, 0, :].cpu().numpy())


def gen_v9(model, start_raw, end_raw, vtype_code, retrieved_raw,
           vtype_vocab, scalers, n_resample, device):
    start_norm = scalers.pos.transform(start_raw.reshape(1, 2))
    end_norm   = scalers.pos.transform(end_raw.reshape(1, 2))
    vt_idx = vtype_vocab.get(int(vtype_code), 0)
    K, T, _ = retrieved_raw.shape
    retrieved_norm = scalers.pos.transform(
        retrieved_raw.reshape(-1, 2)).reshape(1, K, T, 2)
    rmask = np.zeros((1, K), dtype=bool)
    st = torch.tensor(start_norm, dtype=torch.float32, device=device)
    et = torch.tensor(end_norm,   dtype=torch.float32, device=device)
    vt = torch.tensor([vt_idx],   dtype=torch.long,    device=device)
    rt = torch.tensor(retrieved_norm, dtype=torch.float32, device=device)
    rm = torch.tensor(rmask, dtype=torch.bool, device=device)
    with torch.no_grad():
        g = model.generate(st, et, vt, rt, rm, n_steps=n_resample,
                           pos_scaler=scalers.pos, delta_scaler=scalers.delta)
    return scalers.pos.inverse(g[:, 0, :].cpu().numpy())


def gen_v10(model, start_raw, end_raw, vtype_code, retrieved_raw,
            vtype_vocab, scalers, n_resample, device, land_mask=None):
    start_norm = scalers.pos.transform(start_raw.reshape(1, 2))
    end_norm   = scalers.pos.transform(end_raw.reshape(1, 2))
    vt_idx = vtype_vocab.get(int(vtype_code), 0)
    K, T, _ = retrieved_raw.shape
    retrieved_norm = scalers.pos.transform(
        retrieved_raw.reshape(-1, 2)).reshape(1, K, T, 2)
    rmask = np.zeros((1, K), dtype=bool)
    st = torch.tensor(start_norm, dtype=torch.float32, device=device)
    et = torch.tensor(end_norm,   dtype=torch.float32, device=device)
    vt = torch.tensor([vt_idx],   dtype=torch.long,    device=device)
    rt = torch.tensor(retrieved_norm, dtype=torch.float32, device=device)
    rm = torch.tensor(rmask, dtype=torch.bool, device=device)
    with torch.no_grad():
        g = model.generate(st, et, vt, rt, rm, n_steps=n_resample,
                           pos_scaler=scalers.pos, delta_scaler=scalers.delta,
                           land_mask=land_mask)
    return scalers.pos.inverse(g[:, 0, :].cpu().numpy())


# ─── Figure 1: Dirty GT on river (no model) ─────────────────────────────────


def fig_dirty_gt(dirty_npz, land_mask, out_path, n_panels=2):
    """Pick a few dirty GT trajectories with high land penetration and plot them.

    These are the trajectories that filter_trajgen_npz.py drops with the
    --land5_max_frac 0.02 --max_pen_km 10.0 filter.
    """
    print(f"\n── motiv_dirty_gt_on_river ──")
    data = np.load(dirty_npz, allow_pickle=True)
    traj = data["trajectories"].astype(np.float32)        # (N, 128, 2)
    # land-fraction per trajectory at threshold 10 km
    print(f"  Scoring {len(traj)} trajectories for land penetration...")
    # subsample to keep it fast — pick from first 20k randomly
    rng = np.random.RandomState(7)
    pool = rng.choice(len(traj), min(20000, len(traj)), replace=False)
    stats = land_mask.trajectory_stats(traj[pool], threshold_km=10.0)
    # need per-traj fractions, so call the helper that returns per-traj:
    # (the stats above is batch-level — we score each individually below)
    on_land_frac = np.zeros(len(pool), dtype=np.float32)
    for i in range(len(pool)):
        sdf_vals = land_mask.sample_km_np(traj[pool[i], :, 0], traj[pool[i], :, 1])
        on_land_frac[i] = (sdf_vals > 0.0).mean()
    # we want trajectories that are mostly on land but visible (river-like)
    candidates = pool[(on_land_frac > 0.30) & (on_land_frac < 0.95)]
    if len(candidates) < n_panels:
        candidates = pool[np.argsort(-on_land_frac)[:n_panels * 2]]
    rng2 = np.random.RandomState(13)
    pick = rng2.choice(candidates, n_panels, replace=False)
    print(f"  Picked indices: {pick.tolist()}")

    fig, axes = plt.subplots(1, n_panels, figsize=(4.5 * n_panels, 4.4))
    if n_panels == 1:
        axes = [axes]
    for ax, idx in zip(axes, pick):
        t = traj[idx]
        lon_min, lon_max, lat_min, lat_max = autobox([t], pad_frac=0.25)
        shade_land(ax, land_mask, lon_min, lon_max, lat_min, lat_max, alpha=0.5)
        plot_traj(ax, t, color="forestgreen", style="-",
                  label="GT trajectory", lw=2.0)
        mark_endpoints(ax, t)
        # Mark points that are on land
        sdf_vals = land_mask.sample_km_np(t[:, 0], t[:, 1])
        on_land = sdf_vals > 0.0
        if on_land.any():
            ax.scatter(t[on_land, 0], t[on_land, 1], c="red", s=8,
                        zorder=6, label=f"On-land pings ({on_land.sum()}/128)")
        ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
        style_axis(ax, lon_min, lon_max, lat_min, lat_max,
                   title=f"GT idx {idx} — {(on_land.mean()*100):.0f}% of pings on land")
        ax.set_xlabel("lon", fontsize=8)
        ax.set_ylabel("lat", fontsize=8)
    fig.suptitle("Dirty GT trajectories on inland water — motivates the E1 NPZ filter",
                  fontsize=11, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


# ─── Figure 2: v5 lands on coast ────────────────────────────────────────────


def fig_v5_on_land(dirty_npz, v5_ckpt, land_mask, out_path, device):
    print(f"\n── motiv_v5_lands_on_coast ──")
    data = np.load(dirty_npz, allow_pickle=True)
    traj = data["trajectories"].astype(np.float32)
    vts  = data["vessel_types"].astype(np.int32)
    track_ids = data["track_ids"]

    # mirror the v5 training split so we sample val indices
    model, scalers, vtype_vocab, n_resample = load_v5(v5_ckpt, device)
    _, _, _, val_traj, val_vt, _ = train_val_split_gen(
        traj, vts, track_ids, val_frac=0.15, seed=42)
    print(f"  val: {len(val_traj)}  generating sample candidates...")

    # Generate predictions for a larger set of medium-length val trajectories
    # and pick those whose FINAL waypoint sits well inland (SDF > 5 km).
    gt_km = np.array([path_length_m(val_traj[i]) / 1000 for i in range(len(val_traj))])
    medium_mask = (gt_km > 60) & (gt_km < 250)
    candidate_idx = np.where(medium_mask)[0]
    rng = np.random.RandomState(3)
    sample_idx = rng.choice(candidate_idx, min(200, len(candidate_idx)), replace=False)

    preds = []
    final_sdf = []
    on_land_count = []
    for i in tqdm(sample_idx, desc="v5 gen"):
        gt = val_traj[i]
        pred = gen_v5(model, gt[0], gt[-1], val_vt[i],
                      vtype_vocab, scalers, n_resample, device)
        sdf_vals = land_mask.sample_km_np(pred[:, 0], pred[:, 1])
        final_sdf.append(float(sdf_vals[-1]))
        on_land_count.append(int((sdf_vals > 0).sum()))
        preds.append(pred)

    final_sdf = np.array(final_sdf)
    on_land_count = np.array(on_land_count)
    # We want: final waypoint just-inland (3 km <= sdf <= 30 km) AND most of the
    # route on water (on_land_count <= half of T).  This gives visually-coherent
    # failures: a believable trajectory that drifts inland at the end.
    T = preds[0].shape[0]
    coastal_landing = np.where(
        (final_sdf > 3.0) & (final_sdf < 30.0) &
        (on_land_count < T * 0.5)
    )[0]
    if len(coastal_landing) >= 2:
        # rank by how cleanly the route fails: shallow inland + few mid-route land points
        score = final_sdf[coastal_landing] - 0.5 * on_land_count[coastal_landing]
        order = coastal_landing[np.argsort(-score)]
    else:
        print(f"  WARNING: only {len(coastal_landing)} clean coastal-landing "
              f"cases; relaxing constraints.")
        # Relax: any final-on-land with on_land_count <= 80% of T
        relaxed = np.where(
            (final_sdf > 1.0) & (on_land_count < T * 0.8)
        )[0]
        if len(relaxed) >= 2:
            order = relaxed[np.argsort(-final_sdf[relaxed])]
        else:
            order = np.argsort(-on_land_count)
    pick = sample_idx[order[:2]]
    pred_pick = [preds[order[k]] for k in range(2)]
    final_sdf_pick = [final_sdf[order[k]] for k in range(2)]

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.4))
    for ax, vidx, pred, fsd in zip(axes, pick, pred_pick, final_sdf_pick):
        gt = val_traj[vidx]
        gc = great_circle_trajectory(gt[0], gt[-1], len(gt))
        lon_min, lon_max, lat_min, lat_max = autobox([gt, pred, gc], pad_frac=0.20)
        shade_land(ax, land_mask, lon_min, lon_max, lat_min, lat_max)
        plot_traj(ax, gt,   "forestgreen", "-",  "Ground truth")
        plot_traj(ax, gc,   "darkorange",  ":",  "Great circle", lw=1.0,
                  alpha=0.6, zorder=3)
        plot_traj(ax, pred, "crimson",    "--", "v5 generated", lw=1.6)
        sdf_vals = land_mask.sample_km_np(pred[:, 0], pred[:, 1])
        on_land = sdf_vals > 0.0
        if on_land.any():
            ax.scatter(pred[on_land, 0], pred[on_land, 1], c="red",
                       edgecolors="black", linewidths=0.4, s=22,
                       zorder=7, label=f"v5 on land ({on_land.sum()})")
        # Highlight the final waypoint
        ax.scatter([pred[-1, 0]], [pred[-1, 1]], marker="X", c="black",
                   s=120, zorder=8, label=f"v5 final (SDF {fsd:.1f} km inland)")
        mark_endpoints(ax, gt)
        ade = haversine_meters(gt[:, 1], gt[:, 0],
                                pred[:, 1], pred[:, 0]).mean()
        ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
        style_axis(ax, lon_min, lon_max, lat_min, lat_max,
                   title=f"val {vidx} — v5 ADE {ade/1000:.1f} km, final on land")
        ax.set_xlabel("lon", fontsize=8)
        ax.set_ylabel("lat", fontsize=8)
    fig.suptitle("v5 generates routes whose final waypoint terminates on land "
                  "--- motivates v9/v10",
                  fontsize=11, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


# ─── Figure 3: v9 long-route collapse ───────────────────────────────────────


def fig_v9_long(dirty_npz, v9_ckpt, knn_cache, land_mask, out_path, device):
    print(f"\n── motiv_v9_long_route_collapse ──")
    data = np.load(dirty_npz, allow_pickle=True)
    traj = data["trajectories"].astype(np.float32)
    vts  = data["vessel_types"].astype(np.int32)
    track_ids = data["track_ids"]

    model, scalers, vtype_vocab, n_resample, a = load_v9(v9_ckpt, device)
    train_traj, _, _, val_traj, val_vt, _ = train_val_split_gen(
        traj, vts, track_ids, val_frac=a.val_frac, seed=a.seed)
    knn = np.load(knn_cache)
    val_knn = knn["val_knn"]

    gt_km = np.array([path_length_m(val_traj[i]) / 1000 for i in range(len(val_traj))])
    long_idx = np.where(gt_km > 500)[0]
    print(f"  long (>500 km) val tracks: {len(long_idx)}")
    if len(long_idx) == 0:
        long_idx = np.argsort(-gt_km)[:20]
    rng = np.random.RandomState(11)
    sample_idx = rng.choice(long_idx, min(15, len(long_idx)), replace=False)

    rows = []
    for i in tqdm(sample_idx, desc="v9 gen long"):
        gt = val_traj[i]
        retrieved = train_traj[val_knn[i]]  # (K, 128, 2)
        pred = gen_v9(model, gt[0], gt[-1], val_vt[i], retrieved,
                      vtype_vocab, scalers, n_resample, device)
        retr_top1 = retrieved[0]
        v9_ade = haversine_meters(gt[:, 1], gt[:, 0],
                                   pred[:, 1], pred[:, 0]).mean()
        retr_ade = haversine_meters(gt[:, 1], gt[:, 0],
                                     retr_top1[:, 1], retr_top1[:, 0]).mean()
        # We want v9 much worse than retrieval-top-1
        rows.append((i, v9_ade, retr_ade, pred, retr_top1, gt))

    # Sort by (v9_ade - retr_ade) descending — the worst-case for v9 vs retr.
    rows.sort(key=lambda r: -(r[1] - r[2]))
    pick = rows[:2]

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.4))
    for ax, (vidx, v9_ade, retr_ade, pred, retr_top1, gt) in zip(axes, pick):
        lon_min, lon_max, lat_min, lat_max = autobox(
            [gt, pred, retr_top1], pad_frac=0.18)
        shade_land(ax, land_mask, lon_min, lon_max, lat_min, lat_max)
        plot_traj(ax, gt,        "forestgreen", "-",  "Ground truth")
        plot_traj(ax, retr_top1, "steelblue",   "-",  "Retrieval-top-1",
                  lw=1.6, alpha=0.85)
        plot_traj(ax, pred,      "crimson",     "--", "v9 generated", lw=1.6)
        mark_endpoints(ax, gt)
        ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
        style_axis(ax, lon_min, lon_max, lat_min, lat_max,
                   title=f"val {vidx} — v9 {v9_ade/1000:.1f} km "
                         f"vs retr-top-1 {retr_ade/1000:.1f} km")
        ax.set_xlabel("lon", fontsize=8)
        ax.set_ylabel("lat", fontsize=8)
    fig.suptitle("v9 mean-pool collapses on long routes (retrieval-top-1 outperforms it) "
                  "— motivates v10 per-step retrieval", fontsize=10.5, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


# ─── Figure 4: v10 raw vs v10+router ────────────────────────────────────────


def fig_v10_router(dirty_npz, v10_ckpt, knn_cache, water_graph, land_mask,
                    out_path, device):
    print(f"\n── motiv_v10_raw_crosses_land ──")
    data = np.load(dirty_npz, allow_pickle=True)
    traj = data["trajectories"].astype(np.float32)
    vts  = data["vessel_types"].astype(np.int32)
    track_ids = data["track_ids"]

    model, scalers, vtype_vocab, n_resample, a = load_v10(v10_ckpt, device)
    train_traj, _, _, val_traj, val_vt, _ = train_val_split_gen(
        traj, vts, track_ids, val_frac=a.val_frac, seed=a.seed)
    knn = np.load(knn_cache)
    val_knn = knn["val_knn"]

    # Need a router on the 0.005° water graph.
    print("  loading WaterRouter...")
    router = WaterRouter.load(water_graph, coarse_mask=land_mask)

    # Sample medium-length tracks
    gt_km = np.array([path_length_m(val_traj[i]) / 1000 for i in range(len(val_traj))])
    pool = np.where((gt_km > 30) & (gt_km < 400))[0]
    rng = np.random.RandomState(7)
    sample_idx = rng.choice(pool, min(60, len(pool)), replace=False)

    candidates = []
    print("  generating sample predictions (no land-projection)...")
    for i in tqdm(sample_idx, desc="v10 gen"):
        gt = val_traj[i]
        retrieved = train_traj[val_knn[i]]
        pred = gen_v10(model, gt[0], gt[-1], val_vt[i], retrieved,
                       vtype_vocab, scalers, n_resample, device,
                       land_mask=None)  # explicitly raw
        sdf_vals = land_mask.sample_km_np(pred[:, 0], pred[:, 1])
        n_on_land = int((sdf_vals > 0).sum())
        candidates.append((i, n_on_land, pred, gt))

    candidates.sort(key=lambda r: -r[1])
    pick = candidates[:2]
    print(f"  picked val indices: {[p[0] for p in pick]} "
          f"(land pings: {[p[1] for p in pick]})")

    fig, axes = plt.subplots(2, 2, figsize=(9.0, 8.0))
    for col, (vidx, n_on_land, pred_raw, gt) in enumerate(pick):
        pred_repaired, _ = router.repair_trajectory(
            pred_raw.astype(np.float64), threshold_km=10.0,
            do_segment_bridge=False,
            window_margin_cells=50, max_window_cells=400)

        for row, (label, pred) in enumerate(
                [("v10 RAW (crosses land)", pred_raw),
                 ("v10 + WaterRouter snap", pred_repaired.astype(np.float32))]):
            ax = axes[row, col]
            lon_min, lon_max, lat_min, lat_max = autobox(
                [gt, pred_raw, pred_repaired.astype(np.float32)], pad_frac=0.15)
            shade_land(ax, land_mask, lon_min, lon_max, lat_min, lat_max)
            plot_traj(ax, gt,   "forestgreen", "-",  "Ground truth")
            plot_traj(ax, pred, "crimson" if row == 0 else "navy",
                       "--" if row == 0 else "-",
                       label, lw=1.6)
            mark_endpoints(ax, gt)
            sdf_vals = land_mask.sample_km_np(pred[:, 0], pred[:, 1])
            on_land = sdf_vals > 0.0
            if on_land.any():
                ax.scatter(pred[on_land, 0], pred[on_land, 1], c="red",
                           edgecolors="black", linewidths=0.3, s=18,
                           zorder=7, label=f"on land ({on_land.sum()})")
            ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
            ade = haversine_meters(gt[:, 1], gt[:, 0],
                                    pred[:, 1], pred[:, 0]).mean()
            style_axis(ax, lon_min, lon_max, lat_min, lat_max,
                       title=f"val {vidx} — {label}\nADE {ade/1000:.1f} km")
            ax.set_xlabel("lon", fontsize=8)
            ax.set_ylabel("lat", fontsize=8)
    fig.suptitle("v10 raw still cuts across small land masses; WaterRouter snaps "
                  "the offending points back to water — motivates the post-processor",
                  fontsize=10.5, y=1.00)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


# ─── Figure 5: v12 Manhattan zigzag ─────────────────────────────────────────


def fig_v12_zigzag(out_path):
    """Re-use the existing viz_v12.png if present; else skip."""
    print(f"\n── motiv_v12_zigzag ──")
    src = "figures/viz_v12.png"
    if not os.path.exists(src):
        print(f"  WARN: {src} not found; skipping (v12 motivation figure)")
        return
    # Copy with a more descriptive caption in the report; nothing to render here.
    # We just symlink/copy.
    import shutil
    shutil.copyfile(src, out_path)
    print(f"  copied {src} -> {out_path}")


# ─── Main ───────────────────────────────────────────────────────────────────


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--dirty_npz",   default="data/processed/trajgen_128.npz")
    p.add_argument("--knn_cache",   default="data/processed/trajgen_128_knn_k5.npz")
    p.add_argument("--water_graph", default="data/processed/water_graph_005deg.npz")
    p.add_argument("--land_sdf",    default="data/processed/land_sdf_050deg.npz")
    p.add_argument("--v5_ckpt",     default="runs/trajgen_v5/best.pt")
    p.add_argument("--v9_ckpt",     default="runs/trajgen_v9_retrieval/best.pt")
    p.add_argument("--v10_ckpt",    default="runs/trajgen_v10/best.pt")
    p.add_argument("--out_dir",     default="figures")
    p.add_argument("--skip", default="", help="Comma-separated list of figures to skip "
                                                "(e.g. 'v5,v9,v10,v12,gt')")
    args = p.parse_args()

    skip = set(s.strip() for s in args.skip.split(",") if s.strip())
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading land SDF: {args.land_sdf}")
    land_mask = LandMask.load(args.land_sdf)
    device = get_device()
    print(f"Device: {device}")

    if "gt" not in skip:
        fig_dirty_gt(args.dirty_npz, land_mask,
                     os.path.join(args.out_dir, "motiv_dirty_gt_on_river.png"))
    if "v5" not in skip and os.path.exists(args.v5_ckpt):
        fig_v5_on_land(args.dirty_npz, args.v5_ckpt, land_mask,
                        os.path.join(args.out_dir, "motiv_v5_lands_on_coast.png"),
                        device)
    if "v9" not in skip and os.path.exists(args.v9_ckpt):
        fig_v9_long(args.dirty_npz, args.v9_ckpt, args.knn_cache, land_mask,
                     os.path.join(args.out_dir, "motiv_v9_long_route_collapse.png"),
                     device)
    if "v10" not in skip and os.path.exists(args.v10_ckpt):
        fig_v10_router(args.dirty_npz, args.v10_ckpt, args.knn_cache,
                        args.water_graph, land_mask,
                        os.path.join(args.out_dir, "motiv_v10_raw_crosses_land.png"),
                        device)
    if "v12" not in skip:
        fig_v12_zigzag(os.path.join(args.out_dir, "motiv_v12_zigzag.png"))

    print("\nDone.")


if __name__ == "__main__":
    main()
