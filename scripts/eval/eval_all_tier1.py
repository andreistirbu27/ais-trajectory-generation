"""eval_all_tier1.py -- unified evaluation of every Tier-1 checkpoint.

Iterates over every runs/trajgen_v{4,5,9,10}_clean*/best.pt produced by
run_all_tier1_parallel.sh and run_all_tier1b_parallel.sh, evaluates each
under a matched protocol on:
  - cleaned val split (5 seeds x 500-route subsamples)
  - cleaned held-out test split (one shot, n=2326)

Emits results/tier1_eval.csv with columns:
    run, split, seed, method, ade_m, fde_m, n,
    land_frac_pct, cross_pct

Run after both tier1 launchers complete. Uses the existing infrastructure
in scripts/eval/multi_seed_variance.py for the v10-family checkpoints and
falls back to a thin v4/v5/v9 inference path for the ladder variants
(which use different model classes).
"""
import argparse
import csv
import glob
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data_gen import (TrajGenScalers, Scaler, train_val_split_gen,
                          build_vtype_vocab)  # noqa: E402
from src.land_mask import LandMask  # noqa: E402
from src.metrics_gen import evaluate_generation, great_circle_trajectory  # noqa: E402
from src.water_router import WaterRouter  # noqa: E402

SEEDS = [0, 1, 2, 42, 123]
N_EVAL = 500
THRESHOLD_KM = 10.0
DATA_NPZ = ROOT / "data/processed/trajgen_128_clean.npz"
KNN_TEST005 = ROOT / "data/processed/trajgen_128_clean_knn_k5_test005.npz"
LAND_SDF = ROOT / "data/processed/land_sdf_050deg.npz"
WATER_GRAPH = ROOT / "data/processed/water_graph_005deg.npz"


def get_device():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def load_v10(path, device):
    from src.model_gen_v10 import TrajectoryGeneratorV10
    ckpt = torch.load(path, map_location=device, weights_only=False)
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
    return ("v10", model, scalers, ckpt["vtype_vocab"],
            ckpt.get("n_resample", 128), args, ckpt)


def load_v9(path, device):
    from src.model_gen_retrieval import RetrievalTrajectoryGenerator
    ckpt = torch.load(path, map_location=device, weights_only=False)
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
        route_encoder=args.route_encoder,
        n_points=ckpt.get("n_resample", 128),
    ).to(device)
    sd = ckpt["model"]
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.eval()
    return ("v9", model, scalers, ckpt["vtype_vocab"],
            ckpt.get("n_resample", 128), args, ckpt)


def load_v4v5(path, device):
    from src.model_gen import TrajectoryGenerator
    ckpt = torch.load(path, map_location=device, weights_only=False)
    args = argparse.Namespace(**ckpt["args"])
    scalers = TrajGenScalers(
        pos=Scaler(mean=ckpt["pos_mean"], std=ckpt["pos_std"]),
        delta=Scaler(mean=ckpt["delta_mean"], std=ckpt["delta_std"]),
    )
    model = TrajectoryGenerator(
        d_model=args.d_model, nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        dim_feedforward=args.dim_feedforward, dropout=args.dropout,
        num_vessel_types=ckpt["num_vessel_types"],
    ).to(device)
    sd = ckpt["model"]
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.eval()
    return ("v4v5", model, scalers, ckpt["vtype_vocab"],
            ckpt.get("n_resample", 128), args, ckpt)


def detect_family(run_path: Path) -> str:
    name = run_path.name
    if "v9" in name:
        return "v9"
    if "v4" in name:
        return "v4v5"
    if "v5" in name:
        return "v4v5"
    return "v10"


def generate(family, model, val_traj, val_vt, train_traj, val_knn,
             vtype_vocab, scalers, n_resample, device):
    """Family-aware autoregressive generation."""
    N = len(val_traj)
    pred = np.zeros_like(val_traj)
    B = 32
    for s in tqdm(range(0, N, B), desc=f"{family} gen"):
        e = min(s + B, N)
        b = e - s
        starts_norm = scalers.pos.transform(val_traj[s:e, 0])
        ends_norm   = scalers.pos.transform(val_traj[s:e, -1])
        vtypes = np.array([vtype_vocab.get(int(v), 0) for v in val_vt[s:e]])

        st = torch.tensor(starts_norm, dtype=torch.float32, device=device)
        et = torch.tensor(ends_norm,   dtype=torch.float32, device=device)
        vt = torch.tensor(vtypes,      dtype=torch.long,    device=device)

        with torch.no_grad():
            if family == "v10":
                retrieved_raw = train_traj[val_knn[s:e]]
                K = retrieved_raw.shape[1]
                rn = scalers.pos.transform(
                    retrieved_raw.reshape(-1, 2)).reshape(b, K, n_resample, 2)
                rt = torch.tensor(rn, dtype=torch.float32, device=device)
                rm = torch.zeros((b, K), dtype=torch.bool, device=device)
                g = model.generate(st, et, vt, rt, rm,
                                   n_steps=n_resample,
                                   pos_scaler=scalers.pos,
                                   delta_scaler=scalers.delta,
                                   land_mask=None)
            elif family == "v9":
                retrieved_raw = train_traj[val_knn[s:e]]
                K = retrieved_raw.shape[1]
                rn = scalers.pos.transform(
                    retrieved_raw.reshape(-1, 2)).reshape(b, K, n_resample, 2)
                rt = torch.tensor(rn, dtype=torch.float32, device=device)
                rm = torch.zeros((b, K), dtype=torch.bool, device=device)
                g = model.generate(st, et, vt, rt, rm,
                                   n_steps=n_resample,
                                   pos_scaler=scalers.pos,
                                   delta_scaler=scalers.delta)
            else:  # v4/v5
                g = model.generate(st, et, vt,
                                   n_steps=n_resample,
                                   pos_scaler=scalers.pos,
                                   delta_scaler=scalers.delta)
        g = g.permute(1, 0, 2).cpu().numpy()
        for i in range(b):
            pred[s + i] = scalers.pos.inverse(g[i])
    return pred


def apply_router(trajs, router, threshold_km):
    out = np.empty_like(trajs)
    for i in tqdm(range(len(trajs)), desc="snap"):
        rep, _ = router.repair_trajectory(
            trajs[i], threshold_km=threshold_km,
            do_segment_bridge=False, window_margin_cells=50, max_window_cells=400)
        out[i] = rep
    return out


def subsample(n_val, n_eval, seed):
    rng = np.random.RandomState(seed)
    return np.sort(rng.choice(n_val, min(n_eval, n_val), replace=False))


def find_runs():
    runs = []
    patterns = [
        "runs/trajgen_v10_clean_train_*",
        "runs/trajgen_v9_retrieval_clean*",
        "runs/trajgen_v4_clean*",
        "runs/trajgen_v5_clean*",
    ]
    for pat in patterns:
        for p in glob.glob(str(ROOT / pat)):
            best = Path(p) / "best.pt"
            if best.exists():
                runs.append(Path(p))
    return sorted(runs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_csv", default="results/tier1_eval.csv")
    args = ap.parse_args()

    device = get_device()
    print(f"Device: {device}")

    runs = find_runs()
    if not runs:
        print("No Tier-1 best.pt found yet. Run the launchers first.")
        return
    print(f"Found {len(runs)} Tier-1 checkpoints:")
    for r in runs:
        print(f"  - {r.name}")

    land_mask = LandMask.load(str(LAND_SDF))
    router = WaterRouter.load(str(WATER_GRAPH), coarse_mask=land_mask)

    data = np.load(DATA_NPZ, allow_pickle=True)
    traj = data["trajectories"].astype(np.float32)
    vt   = data["vessel_types"].astype(np.int32)
    tid  = data["track_ids"]

    (train_traj, _, _,
     val_traj,   val_vt,   _,
     test_traj,  test_vt,  _) = train_val_split_gen(
        traj, vt, tid, val_frac=0.15, seed=42, test_frac=0.05)
    print(f"Train: {len(train_traj):,}  Val: {len(val_traj):,}  Test: {len(test_traj):,}")

    knn = np.load(KNN_TEST005)
    val_knn  = knn["val_knn"]
    test_knn = knn["test_knn"] if "test_knn" in knn.files else None

    rows = []
    for run_path in runs:
        family = detect_family(run_path)
        print(f"\n=== {run_path.name}  (family={family}) ===")
        loader = {"v10": load_v10, "v9": load_v9, "v4v5": load_v4v5}[family]
        _, model, scalers, vtype_vocab, n_resample, _, _ = loader(
            str(run_path / "best.pt"), device)

        # ── Val: 5 seeds x 500 subsamples ─────────────────────────────────
        union_idx = np.unique(np.concatenate(
            [subsample(len(val_traj), N_EVAL, s) for s in SEEDS]))
        u_traj = val_traj[union_idx]
        u_vt   = val_vt[union_idx]
        u_knn  = val_knn[union_idx] if val_knn is not None else None
        pos_in_union = {int(o): i for i, o in enumerate(union_idx)}

        pred_raw = generate(family, model, u_traj, u_vt, train_traj, u_knn,
                            vtype_vocab, scalers, n_resample, device)
        pred_raw64 = pred_raw.astype(np.float64)
        pred_snap = apply_router(pred_raw64, router, THRESHOLD_KM).astype(np.float32)

        for seed in SEEDS:
            idx = subsample(len(val_traj), N_EVAL, seed)
            local = np.array([pos_in_union[int(i)] for i in idx], dtype=np.int64)
            gt = u_traj[local]
            for variant, pred in [("raw", pred_raw[local]),
                                   ("router", pred_snap[local])]:
                m = evaluate_generation(pred, gt, compute_frechet=False)
                ls = land_mask.trajectory_stats(pred, THRESHOLD_KM)
                rows.append({
                    "run": run_path.name, "split": "val", "seed": seed,
                    "method": f"{family}_{variant}",
                    "ade_m": round(m["ade_m"], 1),
                    "fde_m": round(m["fde_m"], 1),
                    "n": m["n_trajectories"],
                    "land_frac_pct": round(ls["land_frac"] * 100.0, 3),
                    "cross_pct": round(ls["traj_crossing_rate"] * 100.0, 2),
                })

        # ── Test: one shot on full 2326-route partition ──────────────────
        t_knn = test_knn if test_knn is not None else None
        if family in ("v10", "v9") and t_knn is None:
            print("  WARN: no test_knn in cache; skipping test eval for retrieval-family run")
        else:
            test_pred_raw = generate(family, model, test_traj, test_vt,
                                     train_traj, t_knn, vtype_vocab,
                                     scalers, n_resample, device)
            test_pred_snap = apply_router(
                test_pred_raw.astype(np.float64), router, THRESHOLD_KM).astype(np.float32)
            for variant, pred in [("raw", test_pred_raw),
                                   ("router", test_pred_snap)]:
                m = evaluate_generation(pred, test_traj, compute_frechet=False)
                ls = land_mask.trajectory_stats(pred, THRESHOLD_KM)
                rows.append({
                    "run": run_path.name, "split": "test", "seed": -1,
                    "method": f"{family}_{variant}",
                    "ade_m": round(m["ade_m"], 1),
                    "fde_m": round(m["fde_m"], 1),
                    "n": m["n_trajectories"],
                    "land_frac_pct": round(ls["land_frac"] * 100.0, 3),
                    "cross_pct": round(ls["traj_crossing_rate"] * 100.0, 2),
                })

        del model
        torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {args.out_csv}")


if __name__ == "__main__":
    main()
