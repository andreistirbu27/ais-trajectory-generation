#!/usr/bin/env python3
"""
eval_traisformer.py — Evaluate a trained TrAISformer checkpoint against our
metric suite (and optionally feed it into the K-anchor harness).

This is the wrapper around upstream TrAISformer for the paper-track
comparison. It does three things:

1. Loads a TrAISformer checkpoint produced by `scripts/paper/train_traisformer.py`
   (or upstream `paper/external/TrAISformer/trAISformer.py`).
2. Builds a "history" tensor from each of our val trajectories. For the
   K-anchor protocol (paper/notes/k_anchor_protocol.md):
     - K = 2  → replicate y_0 init_seqlen times with zero velocity.
                This is Protocol B from the strategic plan §3 — the
                degeneration case that shows TrAISformer can't do
                endpoint-conditioned generation without modification.
     - K = 18 → use the first 18 GT positions as the history (their
                native setting).
     - other K → re-condition every ⌈(T-1)/(K-1)⌉ steps on the next
                 anchor (teacher-forcing in their token space, then resume
                 free-running). NOT implemented yet — would need to modify
                 the autoregressive sampling loop.
3. Rolls out (T - init_seqlen) steps and converts predictions back to
   (lon, lat) degrees so they slot into the shared eval format.

This script intentionally has NO GPU side effects on import — all TrAISformer
imports are deferred to main() so smoke-testing doesn't load CUDA.

Usage:
    paper/.venv/bin/python3 scripts/paper/eval_traisformer.py \\
        --data_npz   data/processed/trajgen_128_clean.npz \\
        --ckpt       paper/external/TrAISformer/results/<auto-name>/model.pt \\
        --n_eval     500 \\
        --k_anchor   2 \\
        --out_csv    results/paper/traisformer_eval.csv

Limitations
-----------
- TrAISformer takes 4-dim input [lat, lon, SOG, COG]. Our `trajgen_128_clean`
  has only (lon, lat). We synthesize SOG and COG from finite differences of
  the GT positions for the "history" tensor. This is honest but
  approximate; document in the paper.
- TrAISformer's native region is Kattegat. To evaluate on US data, the
  checkpoint must be a US-retrained variant (Protocol A in §3). A
  Kattegat-trained checkpoint will produce nonsense on US queries — the
  script warns if the checkpoint's lat/lon ROI doesn't overlap our data.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIS_DIR = REPO_ROOT / "paper" / "external" / "TrAISformer"
sys.path.insert(0, str(REPO_ROOT))


def make_init_history(
    val_traj: np.ndarray,   # (N, T, 2) [lon, lat]
    init_seqlen: int,
    k_anchor: int,
    lat_min: float, lat_max: float,
    lon_min: float, lon_max: float,
) -> np.ndarray:
    """Build TrAISformer's input tensor: (N, init_seqlen, 4) [lat, lon, sog, cog]
    normalized to [0, 1] per their convention.

    For k_anchor == 2: zero-velocity start replication (degeneration test).
    For k_anchor == 18 and init_seqlen == 18: use first 18 GT steps.
    Other combinations are not supported yet.
    """
    N, T, _ = val_traj.shape
    out = np.zeros((N, init_seqlen, 4), dtype=np.float32)

    if k_anchor == 2:
        # Replicate start position; SOG=0; COG=0.
        lon0 = val_traj[:, 0, 0]
        lat0 = val_traj[:, 0, 1]
        lat_norm = (lat0 - lat_min) / (lat_max - lat_min)
        lon_norm = (lon0 - lon_min) / (lon_max - lon_min)
        out[:, :, 0] = lat_norm[:, None]
        out[:, :, 1] = lon_norm[:, None]
        # SOG / COG stay at 0 → "stopped" history
    elif k_anchor == 18 and init_seqlen == 18:
        for n in range(N):
            for t in range(init_seqlen):
                lon = val_traj[n, t, 0]
                lat = val_traj[n, t, 1]
                out[n, t, 0] = (lat - lat_min) / (lat_max - lat_min)
                out[n, t, 1] = (lon - lon_min) / (lon_max - lon_min)
            # SOG / COG from finite differences of (lat, lon) over GT spacing
            # SOG in TrAISformer units is normalised to [0, 1] over [0, 30] knots
            # COG is normalised over [0, 360) degrees.
            # We approximate: SOG ∝ |Δposition|, COG = atan2(Δlat, Δlon).
            d = np.diff(val_traj[n, :init_seqlen + 1], axis=0)   # (init_seqlen, 2)
            sog_proxy = np.linalg.norm(d, axis=1)
            sog_norm = np.clip(sog_proxy / sog_proxy.max() if sog_proxy.max() > 0 else 0.0,
                               0, 1)
            cog_rad = np.arctan2(d[:, 1], d[:, 0])
            cog_norm = ((cog_rad + 2 * np.pi) % (2 * np.pi)) / (2 * np.pi)
            out[n, :, 2] = sog_norm
            out[n, :, 3] = cog_norm
    else:
        raise NotImplementedError(
            f"k_anchor={k_anchor} with init_seqlen={init_seqlen} not "
            "implemented yet. Supported: (k=2, any init) and (k=18, init=18). "
            "For general K, need to modify the AR sampling loop to "
            "re-condition on intermediate anchors.")
    return out


def load_traisformer(ckpt_path: str, device):
    """Defer TrAISformer imports until called so the file imports without CUDA."""
    sys.path.insert(0, str(TRAIS_DIR))
    cwd_before = os.getcwd()
    os.chdir(str(TRAIS_DIR))
    try:
        from config_trAISformer import Config
        import models
        # The Config singleton is mutable; the ckpt's saved config should
        # already match. If not, the caller must monkey-patch Config before
        # this call (see train_traisformer.py for how).
        cf = Config()
        model = models.TrAISformer(cf, partition_model=None).to(device)
        import torch
        state = torch.load(ckpt_path, map_location=device)
        # Their script saves the raw state_dict (no wrapping).
        model.load_state_dict(state)
        model.eval()
        return model, cf
    finally:
        os.chdir(cwd_before)


def predict_trajectories(
    model, cf, init_history: np.ndarray, max_seqlen: int, device,
    *, n_samples: int = 1, temperature: float = 1.0,
) -> np.ndarray:
    """Roll out `max_seqlen - init_seqlen` future steps with their sampler.

    Returns predictions in (N, max_seqlen, 2) [lon, lat] degrees.
    """
    import torch
    sys.path.insert(0, str(TRAIS_DIR))
    cwd_before = os.getcwd()
    os.chdir(str(TRAIS_DIR))
    try:
        import trainers
        init_t = torch.tensor(init_history, dtype=torch.float32, device=device)
        N, init_seqlen, _ = init_t.shape
        future_len = max_seqlen - init_seqlen

        # Their sampler returns the FULL sequence (history + predicted).
        with torch.no_grad():
            preds_norm = trainers.sample(
                model, init_t, future_len, temperature=temperature,
                sample=(n_samples > 1),
                sample_mode=cf.sample_mode, r_vicinity=cf.r_vicinity,
                top_k=cf.top_k,
            )   # (N, max_seqlen, 4) normalized
    finally:
        os.chdir(cwd_before)

    pn = preds_norm.cpu().numpy()
    lat = pn[..., 0] * (cf.lat_max - cf.lat_min) + cf.lat_min
    lon = pn[..., 1] * (cf.lon_max - cf.lon_min) + cf.lon_min
    return np.stack([lon, lat], axis=-1).astype(np.float32)    # (N, T, 2) [lon, lat]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_npz",
                    default="data/processed/trajgen_128_clean.npz")
    ap.add_argument("--ckpt", required=True,
                    help="Path to TrAISformer checkpoint (model.pt).")
    ap.add_argument("--val_frac",   type=float, default=0.15)
    ap.add_argument("--split_seed", type=int,   default=42)
    ap.add_argument("--n_eval",     type=int,   default=500)
    ap.add_argument("--sample_seed", type=int,  default=42)
    ap.add_argument("--k_anchor",   type=int,   default=2,
                    help="K-anchor protocol setting (see "
                         "paper/notes/k_anchor_protocol.md). Supported: 2, 18.")
    ap.add_argument("--out_csv",
                    default="results/paper/traisformer_eval.csv")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    import torch
    device = (torch.device(args.device) if args.device
              else (torch.device("cuda") if torch.cuda.is_available()
                    else torch.device("cpu")))
    print(f"Device: {device}")

    if not Path(args.ckpt).exists():
        print(f"ERROR: checkpoint not found at {args.ckpt}")
        print(f"       Train TrAISformer first via "
              f"scripts/paper/train_traisformer.py")
        sys.exit(2)

    # Load TrAISformer + report what region the model was trained on
    model, cf = load_traisformer(args.ckpt, device)
    print(f"TrAISformer ROI: lat ({cf.lat_min}, {cf.lat_max})  "
          f"lon ({cf.lon_min}, {cf.lon_max})")

    # Load val split with the same protocol the rest of the paper uses
    from src.data_gen import train_val_split_gen
    from src.metrics_gen import (evaluate_generation,
                                  great_circle_trajectory)
    from src.land_mask import LandMask

    data = np.load(args.data_npz, allow_pickle=True)
    trajectories = data["trajectories"].astype(np.float32)
    vessel_types = data["vessel_types"].astype(np.int32)
    track_ids    = data["track_ids"]
    _, _, _, val_traj, _, _ = train_val_split_gen(
        trajectories, vessel_types, track_ids,
        args.val_frac, args.split_seed)
    T = val_traj.shape[1]

    rng = np.random.default_rng(args.sample_seed)
    idx = rng.choice(len(val_traj), size=min(args.n_eval, len(val_traj)),
                     replace=False)
    val_traj = val_traj[idx]

    # Sanity check: TrAISformer was trained on its ROI; our data may extend
    # outside it. Warn and proceed; results will be garbage outside ROI.
    in_roi = np.mean(
        (val_traj[..., 1] >= cf.lat_min) & (val_traj[..., 1] <= cf.lat_max) &
        (val_traj[..., 0] >= cf.lon_min) & (val_traj[..., 0] <= cf.lon_max)
    )
    if in_roi < 0.95:
        print(f"WARNING: only {in_roi*100:.1f}% of val positions fall inside "
              f"the TrAISformer ROI. Predictions outside the ROI will be "
              f"unreliable. Use a US-retrained checkpoint for honest results.")

    # Build history + roll out
    init_history = make_init_history(
        val_traj, init_seqlen=cf.init_seqlen, k_anchor=args.k_anchor,
        lat_min=cf.lat_min, lat_max=cf.lat_max,
        lon_min=cf.lon_min, lon_max=cf.lon_max,
    )
    print(f"Sampling {len(val_traj)} trajectories at K={args.k_anchor}...")
    pred = predict_trajectories(
        model, cf, init_history, max_seqlen=T, device=device)

    # Score (full T for now; K-anchor masking is done in eval_shared_protocol.py)
    m = evaluate_generation(pred, val_traj, compute_frechet=False)
    print(f"\nADE  = {m['ade_m']:>9.1f} m")
    print(f"FDE  = {m['fde_m']:>9.1f} m")
    print(f"normalized ADE = {m['normalized_ade']:.4f}")

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "k_anchor", "n", "ade_m", "fde_m",
                    "normalized_ade", "ckpt"])
        w.writerow(["traisformer", args.k_anchor, len(pred),
                    f"{m['ade_m']:.1f}", f"{m['fde_m']:.1f}",
                    f"{m['normalized_ade']:.6f}", args.ckpt])
    print(f"Wrote {args.out_csv}")


if __name__ == "__main__":
    main()
