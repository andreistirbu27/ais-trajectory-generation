#!/usr/bin/env python3
"""
multi_seed_variance.py — measure ADE variance across 5 subsample seeds for the
production v10+router configuration on the clean dataset.

The headline production number (v10 + WaterRouter snap-only on
trajgen_128_clean.npz, n_eval=500, seed=42) was reported as ADE 6282 m.
This script re-runs the same evaluation under 4 additional subsample seeds
{0, 1, 2, 123} (plus the canonical 42) so the report can quote a mean ± std
instead of a single point.

To avoid running the v10 decoder 5x, we:
  1. Take the union of 5 seed-specific 500-index subsamples (~1700-2200 unique
     val indices total).
  2. Run v10 inference + WaterRouter snap-only once on the union.
  3. For each seed, slice the union to that seed's 500 indices and compute
     ADE / FDE / land% for v10+router, retrieval-top-1, and great-circle.
  4. Aggregate to mean ± std.

Writes:
  results/seed_variance.csv         — raw per-seed per-method numbers
  figures/seed_variance.png         — grouped bar chart with error bars

Usage:
    python3 scripts/eval/multi_seed_variance.py \\
        --data_npz   data/processed/trajgen_128_clean.npz \\
        --checkpoint runs/trajgen_v10/best.pt \\
        --knn_cache  data/processed/trajgen_128_clean_knn_k5.npz \\
        --water_graph data/processed/water_graph_005deg.npz \\
        --land_sdf   data/processed/land_sdf_050deg.npz
"""

import argparse
import csv
import os
import sys
import time

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.data_gen import TrajGenScalers, Scaler, train_val_split_gen
from src.land_mask import LandMask
from src.metrics_gen import evaluate_generation, great_circle_trajectory
from src.model_gen_v10 import TrajectoryGeneratorV10
from src.water_router import WaterRouter


SEEDS = [0, 1, 2, 42, 123]
N_EVAL = 500
METHODS = ["v10+router", "retrieval-top-1", "great-circle", "great-circle+snap"]


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
    model = TrajectoryGeneratorV10(
        d_model=args.d_model, nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        dim_feedforward=args.dim_feedforward, dropout=args.dropout,
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
    return model, scalers, ckpt["vtype_vocab"], ckpt.get("n_resample", 128), args, ckpt


def generate(model, val_traj, val_vt, train_traj, val_knn, vtype_vocab,
             scalers, n_resample, device):
    N = len(val_traj)
    pred = np.zeros_like(val_traj)
    B_size = 32
    for s in tqdm(range(0, N, B_size), desc="v10 raw gen"):
        e = min(s + B_size, N)
        batch_gt = val_traj[s:e]
        batch_vt = val_vt[s:e]
        batch_knn = val_knn[s:e]
        B = len(batch_gt)
        starts_norm = scalers.pos.transform(batch_gt[:, 0])
        ends_norm   = scalers.pos.transform(batch_gt[:, -1])
        vtypes = np.array([vtype_vocab.get(int(v), 0) for v in batch_vt])
        retrieved_raw = train_traj[batch_knn]
        K = retrieved_raw.shape[1]
        retrieved_norm = scalers.pos.transform(
            retrieved_raw.reshape(-1, 2)).reshape(B, K, n_resample, 2)
        retrieval_mask = np.zeros((B, K), dtype=bool)

        st = torch.tensor(starts_norm, dtype=torch.float32, device=device)
        et = torch.tensor(ends_norm,   dtype=torch.float32, device=device)
        vt = torch.tensor(vtypes,      dtype=torch.long,    device=device)
        rt = torch.tensor(retrieved_norm, dtype=torch.float32, device=device)
        rm = torch.tensor(retrieval_mask, dtype=torch.bool,    device=device)

        with torch.no_grad():
            g = model.generate(st, et, vt, rt, rm,
                               n_steps=n_resample,
                               pos_scaler=scalers.pos, delta_scaler=scalers.delta,
                               land_mask=None)
        g = g.permute(1, 0, 2).cpu().numpy()
        for i in range(B):
            pred[s + i] = scalers.pos.inverse(g[i])
    return pred


def apply_router(trajs, router, threshold_km):
    out = np.empty_like(trajs)
    for i in tqdm(range(len(trajs)), desc="router snap"):
        rep, _ = router.repair_trajectory(
            trajs[i],
            threshold_km=threshold_km,
            do_segment_bridge=False,
            window_margin_cells=50,
            max_window_cells=400,
        )
        out[i] = rep
    return out


def subsample_indices(n_val: int, n_eval: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return np.sort(rng.choice(n_val, n_eval, replace=False))


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data_npz",    default="data/processed/trajgen_128_clean.npz")
    p.add_argument("--checkpoint",  default="runs/trajgen_v10/best.pt")
    p.add_argument("--knn_cache",   default="data/processed/trajgen_128_clean_knn_k5.npz")
    p.add_argument("--water_graph", default="data/processed/water_graph_005deg.npz")
    p.add_argument("--land_sdf",    default="data/processed/land_sdf_050deg.npz")
    p.add_argument("--threshold_km", type=float, default=10.0)
    p.add_argument("--out_csv",     default="results/seed_variance.csv")
    p.add_argument("--out_png",     default="figures/seed_variance.png")
    args = p.parse_args()

    device = get_device()
    print(f"Device: {device}")
    print(f"Seeds: {SEEDS}, n_eval={N_EVAL} per seed\n")

    print(f"Loading {args.checkpoint}")
    model, scalers, vtype_vocab, n_resample, train_args, ckpt = load_checkpoint(
        args.checkpoint, device)
    print(f"  v10 ep={ckpt['epoch']}, k_retrieval={train_args.k_retrieval}, "
          f"t_retr={ckpt['t_retr']}")

    print(f"Loading SDF + water graph")
    land_mask = LandMask.load(args.land_sdf)
    router = WaterRouter.load(args.water_graph, coarse_mask=land_mask)

    print(f"Loading {args.data_npz}")
    data = np.load(args.data_npz, allow_pickle=True)
    trajectories = data["trajectories"].astype(np.float32)
    vessel_types = data["vessel_types"].astype(np.int32)
    track_ids    = data["track_ids"]

    train_traj, _, _, val_traj, val_vt, _ = train_val_split_gen(
        trajectories, vessel_types, track_ids,
        train_args.val_frac, train_args.seed)
    print(f"  Train: {len(train_traj):,}  Val: {len(val_traj):,}")

    knn = np.load(args.knn_cache)
    val_knn = knn["val_knn"]

    # Build union of all seed-subsamples → unique indices we actually need.
    per_seed_idx = {s: subsample_indices(len(val_traj), N_EVAL, s) for s in SEEDS}
    union = np.unique(np.concatenate([per_seed_idx[s] for s in SEEDS]))
    print(f"  Union of 5 seed-subsamples: {len(union)} unique val indices "
          f"(of {len(val_traj)} total)")

    union_traj = val_traj[union]
    union_vt   = val_vt[union]
    union_knn  = val_knn[union]
    # Map original-val-index → position in `union` so we can slice per seed.
    pos_in_union = {int(orig): i for i, orig in enumerate(union)}

    # Generate v10 raw on the union, once.
    t0 = time.time()
    pred_raw = generate(model, union_traj, union_vt, train_traj, union_knn,
                        vtype_vocab, scalers, n_resample, device)
    print(f"  v10 raw generation: {time.time()-t0:.1f}s for {len(union)} traj")

    # Apply WaterRouter snap-only.
    t0 = time.time()
    pred_router = apply_router(pred_raw.astype(np.float64), router, args.threshold_km)
    pred_router = pred_router.astype(np.float32)
    print(f"  router snap:        {time.time()-t0:.1f}s")

    # Retrieval-top-1: train_traj indexed by KNN.
    retr_top1 = train_traj[union_knn[:, 0]].astype(np.float32)

    # Great-circle on the union.
    T = union_traj.shape[1]
    gc_traj = np.zeros_like(union_traj)
    for i in range(len(union_traj)):
        gc_traj[i] = great_circle_trajectory(union_traj[i, 0], union_traj[i, -1], T)

    # Great-circle + WaterRouter snap (baseline to isolate the post-processor
    # contribution from the model contribution).
    t0 = time.time()
    gc_snap = apply_router(gc_traj.astype(np.float64), router, args.threshold_km)
    gc_snap = gc_snap.astype(np.float32)
    print(f"  GC+snap router:     {time.time()-t0:.1f}s")

    # ─── Per-seed metrics ──────────────────────────────────────────────
    rows = []
    print("\n─── Per-seed ADE / FDE / land% ─────────────────────────────")
    print(f"{'seed':>5} {'method':<18} "
          f"{'ADE(m)':>9} {'FDE(m)':>9} {'land%':>8} {'cross%':>8}")
    for seed in SEEDS:
        idx = per_seed_idx[seed]
        local = np.array([pos_in_union[int(i)] for i in idx], dtype=np.int64)
        gt    = union_traj[local]
        sub_router = pred_router[local]
        sub_retr   = retr_top1[local]
        sub_gc     = gc_traj[local]
        sub_gc_snap = gc_snap[local]

        for name, pred in [("v10+router", sub_router),
                            ("retrieval-top-1", sub_retr),
                            ("great-circle", sub_gc),
                            ("great-circle+snap", sub_gc_snap)]:
            m = evaluate_generation(pred, gt, compute_frechet=False)
            ls = land_mask.trajectory_stats(pred, args.threshold_km)
            row = {
                "seed": seed,
                "method": name,
                "ade_m": m["ade_m"],
                "fde_m": m["fde_m"],
                "n": m["n_trajectories"],
                "land_frac_pct": ls["land_frac"] * 100.0,
                "cross_pct": ls["traj_crossing_rate"] * 100.0,
            }
            rows.append(row)
            print(f"{seed:>5d} {name:<18} "
                  f"{m['ade_m']:>9.1f} {m['fde_m']:>9.1f} "
                  f"{ls['land_frac']*100:>7.2f}% {ls['traj_crossing_rate']*100:>7.1f}%")

    # ─── Write CSV ─────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {args.out_csv}")

    # ─── Aggregate + plot ──────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    agg = {m: {"ade": [], "fde": [], "cross": []} for m in METHODS}
    for r in rows:
        agg[r["method"]]["ade"].append(r["ade_m"])
        agg[r["method"]]["fde"].append(r["fde_m"])
        agg[r["method"]]["cross"].append(r["cross_pct"])

    print("\n─── Aggregate (mean ± std over 5 seeds) ────────────────────")
    summary = {}
    for m in METHODS:
        a = np.array(agg[m]["ade"])
        f = np.array(agg[m]["fde"])
        c = np.array(agg[m]["cross"])
        summary[m] = {"ade_mean": a.mean(), "ade_std": a.std(ddof=0),
                      "fde_mean": f.mean(), "fde_std": f.std(ddof=0),
                      "cross_mean": c.mean(), "cross_std": c.std(ddof=0)}
        print(f"  {m:<18}  ADE = {a.mean():7.1f} ± {a.std(ddof=0):5.1f} m   "
              f"FDE = {f.mean():7.1f} ± {f.std(ddof=0):5.1f} m   "
              f"cross%(10km) = {c.mean():.2f} ± {c.std(ddof=0):.2f}")

    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    x = np.arange(len(METHODS))
    means = [summary[m]["ade_mean"] for m in METHODS]
    stds  = [summary[m]["ade_std"]  for m in METHODS]
    colors = ["#1f77b4", "#ff7f0e", "#7f7f7f", "#2ca02c"]
    bars = ax.bar(x, means, yerr=stds, capsize=6,
                   color=colors, edgecolor="black", linewidth=0.7,
                   error_kw={"linewidth": 1.4})

    # Overlay individual seed dots for transparency.
    for i, m in enumerate(METHODS):
        ax.scatter(np.full(len(SEEDS), i), agg[m]["ade"],
                   color="black", s=22, zorder=3, label="seed" if i == 0 else None)

    for i, (mean, std) in enumerate(zip(means, stds)):
        ax.text(i, mean + std + 200, f"{mean:.0f}±{std:.0f} m",
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(METHODS, fontsize=11)
    ax.set_ylabel("ADE (m)", fontsize=11)
    ax.set_title(f"ADE variance across {len(SEEDS)} subsample seeds "
                 f"(n={N_EVAL} per seed, clean dataset)", fontsize=11)
    ax.set_ylim(0, max(means) * 1.6)
    ax.grid(True, axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", frameon=False, fontsize=9)

    os.makedirs(os.path.dirname(args.out_png) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.out_png, dpi=150, bbox_inches="tight")
    print(f"Wrote {args.out_png}")

    # Also dump a small text summary that can be \input{}ed into the report.
    summary_path = os.path.splitext(args.out_csv)[0] + "_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"Seeds: {SEEDS}, n_eval={N_EVAL}, threshold_km={args.threshold_km}\n")
        for m in METHODS:
            s = summary[m]
            f.write(f"{m}: ADE = {s['ade_mean']:.1f} ± {s['ade_std']:.1f} m, "
                    f"FDE = {s['fde_mean']:.1f} ± {s['fde_std']:.1f} m, "
                    f"cross% = {s['cross_mean']:.2f} ± {s['cross_std']:.2f}\n")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
