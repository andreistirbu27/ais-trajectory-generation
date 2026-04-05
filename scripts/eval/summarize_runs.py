#!/usr/bin/env python3
"""
summarize_runs.py — Collect results from all runs into results/summary.csv.

Reads runs/*/best.pt and runs/*/baseline.json and writes one row per run.

Usage:
    python3 scripts/eval/summarize_runs.py
    python3 scripts/eval/summarize_runs.py --runs_dir runs --out results/summary.csv
"""

import argparse
import json
import os
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--runs_dir", default="runs",        help="Directory containing run folders")
    p.add_argument("--out",      default="results/summary.csv", help="Output CSV path")
    args = p.parse_args()

    runs_dir = Path(args.runs_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    run_dirs = sorted([d for d in runs_dir.iterdir()
                       if d.is_dir() and (d / "best.pt").exists()])

    if not run_dirs:
        print(f"No runs with best.pt found in {runs_dir}")
        return

    rows = []
    for run_dir in tqdm(run_dirs, desc="Reading runs", unit="run"):
        ckpt = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
        a = ckpt.get("args", {})
        metrics = ckpt.get("metrics", {})

        # Baseline from baseline.json
        bl = {}
        bl_path = run_dir / "baseline.json"
        if bl_path.exists():
            with open(bl_path) as f:
                bl = json.load(f)

        rows.append({
            "run":          run_dir.name,
            "data":         Path(a.get("csv", "")).name,
            "seq_len":      a.get("seq_len"),
            "stride":       a.get("stride"),
            "epochs":       ckpt.get("epoch"),
            "num_layers":   a.get("num_layers"),
            "lambda_smooth": a.get("lambda_smooth"),
            "cv_ade_m":     round(bl.get("ade_m"), 1) if bl.get("ade_m") else None,
            "model_ade_m":  round(metrics.get("ade_m"), 1) if metrics.get("ade_m") else None,
            "cv_fde_m":     round(bl.get("fde_m"), 1) if bl.get("fde_m") else None,
            "model_fde_m":  round(metrics.get("fde_m"), 1) if metrics.get("fde_m") else None,
            "val_mse":      round(metrics.get("mse"), 6) if metrics.get("mse") else None,
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)

    print(f"\nSaved {len(df)} runs → {out_path}\n")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
