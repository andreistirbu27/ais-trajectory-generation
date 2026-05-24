#!/usr/bin/env python3
"""
plot_dma_k_anchor.py — Render the K-anchor curve (ADE vs K) on DMA from the
per-K summary CSVs produced by `eval_shared_protocol_dma.py`.

Reads:  results/paper/dma_k_anchor_K{2,4,8,16,64}_summary.csv
Writes: paper/figures/dma_k_anchor.png
        paper/figures/dma_k_anchor.pdf

Each method's curve is mean ADE across the 5 seeds with the std as a shaded
band. Methods are coloured to make the three design-space regions obvious
at a glance:
  - blue   = v10_dma (endpoint-conditioned, flat-low)
  - orange = retr_top1 (endpoint-retrieval, flat-low)
  - green  = great_circle (pure interpolator, monotonic-down)
  - red    = traisformer (next-step predictor, flat-moderate)

Usage:
    paper/.venv/bin/python3 scripts/paper/plot_dma_k_anchor.py
    paper/.venv/bin/python3 scripts/paper/plot_dma_k_anchor.py \\
        --csv_glob 'results/paper/dma_k_anchor_K*_summary.csv' \\
        --out_png paper/figures/dma_k_anchor.png
"""
from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]

METHOD_STYLE = {
    "v10_dma":      {"color": "#1f77b4", "marker": "o", "label": "v10 (ours, DMA)"},
    "retr_top1":    {"color": "#ff7f0e", "marker": "s", "label": "Retrieval-top-1"},
    "great_circle": {"color": "#2ca02c", "marker": "^", "label": "Piecewise great-circle"},
    "traisformer":  {"color": "#d62728", "marker": "D", "label": "TrAISformer"},
}


def read_summary_csv(path: Path) -> dict[str, tuple[float, float]]:
    """{method: (ade_mean_m, ade_std_m)}"""
    out = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            out[row["method"]] = (float(row["ade_m_mean"]),
                                  float(row["ade_m_std"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_glob",
                    default="results/paper/dma_k_anchor_K*_summary.csv")
    ap.add_argument("--out_png",
                    default="paper/figures/dma_k_anchor.png")
    ap.add_argument("--out_pdf",
                    default="paper/figures/dma_k_anchor.pdf")
    args = ap.parse_args()

    # Sort the input files by K.
    files = sorted(Path(REPO_ROOT).glob(args.csv_glob))
    if not files:
        raise SystemExit(f"No CSVs matched {args.csv_glob}")
    pat = re.compile(r"K(\d+)(?:_b\d+)?_summary")
    files_by_k = sorted(((int(pat.search(p.name).group(1)), p) for p in files),
                        key=lambda x: x[0])

    # ade_by_method[m] = list of (K, mean, std) across files
    ade_by_method: dict[str, list[tuple[int, float, float]]] = defaultdict(list)
    for k, path in files_by_k:
        summary = read_summary_csv(path)
        for method, (mean, std) in summary.items():
            ade_by_method[method].append((k, mean, std))

    fig, ax = plt.subplots(figsize=(6.0, 4.0), dpi=150)
    for method in ["great_circle", "traisformer", "retr_top1", "v10_dma"]:
        # Plot in this order so v10 sits on top (it's the smallest line).
        if method not in ade_by_method:
            continue
        rows = sorted(ade_by_method[method], key=lambda x: x[0])
        Ks      = np.array([r[0] for r in rows], dtype=np.int32)
        means   = np.array([r[1] for r in rows], dtype=np.float64) / 1000.0  # km
        stds    = np.array([r[2] for r in rows], dtype=np.float64) / 1000.0
        sty     = METHOD_STYLE.get(method, {"color": "k", "marker": "x", "label": method})
        ax.plot(Ks, means,
                color=sty["color"], marker=sty["marker"],
                label=sty["label"], lw=1.6, markersize=6)
        ax.fill_between(Ks, means - stds, means + stds,
                        color=sty["color"], alpha=0.15, linewidth=0)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks([2, 4, 8, 16, 64])
    ax.set_xticklabels(["2", "4", "8", "16", "64"])
    ax.set_xlabel("K (number of GT anchor points)")
    ax.set_ylabel("ADE on non-anchor positions (km)")
    ax.set_title("K-anchor benchmark on TrAISformer DMA test set")
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()

    out_png = REPO_ROOT / args.out_png
    out_pdf = REPO_ROOT / args.out_pdf
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"Wrote {out_png}")
    print(f"Wrote {out_pdf}")


if __name__ == "__main__":
    main()
