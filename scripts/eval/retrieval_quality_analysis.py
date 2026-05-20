"""Scatter of top-1 KNN distance vs per-route ADE; tests whether the retrieval
mechanism is doing useful work (the closer the historical neighbour, the
better the model should be able to predict).

Reads results/per_route_lambdaland05.csv (per_route_eval.py output).

Outputs:
    figures/retrieval_quality_scatter.png
    report/figures/retrieval_quality_scatter.png
    results/retrieval_quality_summary.txt
"""

from __future__ import annotations

import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_route_csv", default="results/per_route_lambdaland05.csv")
    ap.add_argument("--out_png",       default="figures/retrieval_quality_scatter.png")
    ap.add_argument("--out_png_report", default="report/figures/retrieval_quality_scatter.png")
    ap.add_argument("--out_summary",   default="results/retrieval_quality_summary.txt")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.per_route_csv)))
    # Pool val + test for the scatter (different subsample sizes but same
    # underlying corpus + model).
    knn = np.array([float(r["knn_dist_top1_km"]) for r in rows])
    ade = np.array([float(r["ade_router_m"])    for r in rows])
    arc = np.array([float(r["gt_arc_km"])       for r in rows])
    split = np.array([r["split"] for r in rows])

    # Length buckets follow Section 6.3
    bucket = np.where(arc < 50.0, "short",
             np.where(arc < 500.0, "medium", "long"))

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    summary_lines = []
    for ax, b in zip(axes, ("short", "medium", "long")):
        m = bucket == b
        x = knn[m]
        y = ade[m]
        n = m.sum()
        rho = spearman(x, y) if n > 5 else float("nan")
        r = float(np.corrcoef(x, y)[0, 1]) if n > 5 else float("nan")
        ax.scatter(x, y, s=8, alpha=0.5, color="#2a6dbf", edgecolor="none")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("KNN top-1 distance (km, log)")
        if b == "short":
            ax.set_ylabel("Per-route ADE (m, log)")
        title = (f"{b} ($<$50~km)" if b == "short" else
                 f"{b} (50--500~km)" if b == "medium" else
                 f"{b} ($>$500~km)")
        ax.set_title(f"{title}, n={n}, Spearman ρ={rho:+.2f}")
        ax.grid(alpha=0.2, which="both")
        summary_lines.append(
            f"bucket={b}\tn={n}\tspearman={rho:+.3f}\tpearson={r:+.3f}\t"
            f"knn_median_km={np.median(x):.1f}\tade_median_m={np.median(y):.1f}"
        )
    fig.suptitle("KNN top-1 query distance vs per-route ADE "
                 "(v10 + router, val ∪ test, n="
                 f"{len(rows)} routes)", fontsize=11)
    fig.tight_layout()

    os.makedirs(os.path.dirname(args.out_png) or ".", exist_ok=True)
    fig.savefig(args.out_png, dpi=160)
    if args.out_png_report:
        os.makedirs(os.path.dirname(args.out_png_report) or ".", exist_ok=True)
        fig.savefig(args.out_png_report, dpi=160)
    plt.close(fig)

    # Overall stats (no bucketing)
    rho_all = spearman(knn, ade)
    pearson_all = float(np.corrcoef(knn, ade)[0, 1])
    summary_lines.insert(0,
        f"OVERALL\tn={len(knn)}\tspearman={rho_all:+.3f}\tpearson={pearson_all:+.3f}")

    os.makedirs(os.path.dirname(args.out_summary) or ".", exist_ok=True)
    with open(args.out_summary, "w") as f:
        f.write("\n".join(summary_lines) + "\n")

    print("\n".join(summary_lines))
    print(f"\nWrote {args.out_png}, {args.out_summary}")


if __name__ == "__main__":
    main()
