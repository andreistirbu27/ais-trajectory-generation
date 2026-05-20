"""Paired bootstrap 95% CI for the production v10+router vs each baseline,
on per-route ADE (route is the bootstrap unit, paired across methods).

Reads results/per_route_lambdaland05.csv (per_route_eval.py output) and
optionally results/a_star_baseline.csv (a_star_baseline.py output).

Outputs:
    results/paired_bootstrap_ci.csv
    figures/paired_bootstrap_forest.png
    report/figures/paired_bootstrap_forest.png
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


def paired_ci(diff: np.ndarray, n_boot: int = 10_000, alpha: float = 0.05,
              rng: np.random.Generator | None = None) -> tuple[float, float, float]:
    """Return (mean_diff, lower, upper) for paired bootstrap of diff."""
    rng = rng or np.random.default_rng(42)
    n = len(diff)
    means = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        means[b] = float(diff[idx].mean())
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(diff.mean()), float(lo), float(hi)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_route_csv", default="results/per_route_lambdaland05.csv")
    ap.add_argument("--a_star_csv",    default="results/a_star_baseline.csv")
    ap.add_argument("--n_boot",  type=int, default=10_000)
    ap.add_argument("--out_csv", default="results/paired_bootstrap_ci.csv")
    ap.add_argument("--out_png", default="figures/paired_bootstrap_forest.png")
    ap.add_argument("--out_png_report", default="report/figures/paired_bootstrap_forest.png")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.per_route_csv)))

    # Per-split arrays of (route_idx, ade per method)
    by_split: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    keys = {"router": "ade_router_m", "retr_top1": "ade_retr_top1_m",
            "gc": "ade_gc_m"}
    for r in rows:
        sp = r["split"]
        for k, col in keys.items():
            by_split[sp][k].append(float(r[col]))

    a_star_by_split: dict[str, dict[int, float]] = defaultdict(dict)
    if os.path.exists(args.a_star_csv):
        for r in csv.DictReader(open(args.a_star_csv)):
            a_star_by_split[r["split"]][int(r["route_idx"])] = float(r["ade_m"])

    out_rows = []
    rng = np.random.default_rng(42)
    for sp in ("val", "test"):
        if sp not in by_split:
            continue
        router = np.array(by_split[sp]["router"])
        retr   = np.array(by_split[sp]["retr_top1"])
        gc     = np.array(by_split[sp]["gc"])
        comparisons = [
            ("router - gc",        router - gc),
            ("router - retr_top1", router - retr),
        ]
        # A* baseline (route_idx alignment for val differs because val is
        # a 500-route subsample for A* and 2000 for per_route_eval; we only
        # compare on the intersection — for test A* used the full partition).
        if a_star_by_split.get(sp):
            astar_map = a_star_by_split[sp]
            n_routes = len(router)
            astar_arr = np.array([astar_map.get(i, np.nan) for i in range(n_routes)])
            ok = ~np.isnan(astar_arr)
            if ok.sum() > 20:
                comparisons.append(
                    ("router - a_star", router[ok] - astar_arr[ok]))
        for name, diff in comparisons:
            m, lo, hi = paired_ci(diff, n_boot=args.n_boot, rng=rng)
            row = {"split": sp, "comparison": name, "n_routes": int(len(diff)),
                   "mean_diff_m": m, "ci95_lower_m": lo, "ci95_upper_m": hi,
                   "ci_excludes_zero": "yes" if (lo > 0 or hi < 0) else "no"}
            out_rows.append(row)
            print(f"  {sp:<5s}  {name:<24s}  n={len(diff):>5}  "
                  f"Δ={m:+8.1f}  95% CI [{lo:+8.1f}, {hi:+8.1f}]  "
                  f"{'(excludes 0)' if lo > 0 or hi < 0 else '(includes 0)'}")

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)

    # Forest plot
    labels = [f"{r['split']}: {r['comparison']}" for r in out_rows]
    means = [r["mean_diff_m"]     for r in out_rows]
    los   = [r["ci95_lower_m"]    for r in out_rows]
    his   = [r["ci95_upper_m"]    for r in out_rows]
    yvals = np.arange(len(out_rows))
    fig, ax = plt.subplots(figsize=(7.8, 0.35 * len(out_rows) + 1.7))
    for yi, (m, lo, hi) in enumerate(zip(means, los, his)):
        col = "#1f77b4" if (lo > 0 or hi < 0) else "#999999"
        ax.errorbar([m], [yi], xerr=[[m - lo], [hi - m]], fmt="o",
                    color="black", ecolor=col, capsize=4, lw=1.3,
                    markersize=5)
    ax.axvline(0, color="red", ls=":", lw=1.0)
    ax.set_yticks(yvals)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Paired Δ ADE in metres "
                  "(production v10+router minus baseline, 95% bootstrap CI)")
    ax.set_title(f"Paired-bootstrap CI: production minus baseline "
                 f"({args.n_boot:,} resamples)")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()

    os.makedirs(os.path.dirname(args.out_png) or ".", exist_ok=True)
    fig.savefig(args.out_png, dpi=160)
    if args.out_png_report:
        os.makedirs(os.path.dirname(args.out_png_report) or ".", exist_ok=True)
        fig.savefig(args.out_png_report, dpi=160)
    plt.close(fig)
    print(f"\nWrote {args.out_csv}, {args.out_png}")


if __name__ == "__main__":
    main()
