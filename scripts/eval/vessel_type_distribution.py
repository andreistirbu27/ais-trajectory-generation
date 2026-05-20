"""Bar chart of AIS vessel-type composition in the cleaned 128-pt corpus.

Outputs:
    figures/vessel_type_distribution.png
    report/figures/vessel_type_distribution.png
    results/vessel_type_distribution.csv
"""

from __future__ import annotations

import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np


# Coarse AIS vessel-type classes (codes 60-89), from ITU-R M.1371.
CLASS_OF = {
    60: "passenger", 61: "passenger", 62: "passenger", 63: "passenger", 64: "passenger",
    65: "passenger", 66: "passenger", 67: "passenger", 68: "passenger", 69: "passenger",
    70: "cargo",     71: "cargo",     72: "cargo",     73: "cargo",     74: "cargo",
    75: "cargo",     76: "cargo",     77: "cargo",     78: "cargo",     79: "cargo",
    80: "tanker",    81: "tanker",    82: "tanker",    83: "tanker",    84: "tanker",
    85: "tanker",    86: "tanker",    87: "tanker",    88: "tanker",    89: "tanker",
}
COLOUR_OF = {"passenger": "#1f77b4", "cargo": "#2ca02c", "tanker": "#d62728"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_npz", default="data/processed/trajgen_128_clean.npz")
    ap.add_argument("--out_csv", default="results/vessel_type_distribution.csv")
    ap.add_argument("--out_png", default="figures/vessel_type_distribution.png")
    ap.add_argument("--out_png_report", default="report/figures/vessel_type_distribution.png")
    args = ap.parse_args()

    data = np.load(args.data_npz, allow_pickle=True)
    vts = data["vessel_types"].astype(np.int32)
    codes, counts = np.unique(vts, return_counts=True)
    order = np.argsort(-counts)
    codes, counts = codes[order], counts[order]

    n_total = counts.sum()
    rows = []
    for c, n in zip(codes, counts):
        klass = CLASS_OF.get(int(c), "other")
        rows.append({"vessel_type": int(c), "class": klass,
                     "count": int(n), "fraction": float(n) / n_total})

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    class_totals: dict[str, int] = {}
    for r in rows:
        class_totals[r["class"]] = class_totals.get(r["class"], 0) + r["count"]
    print(f"Corpus: {n_total:,} routes, {len(codes)} distinct vessel-type codes")
    for klass in ("passenger", "cargo", "tanker", "other"):
        if klass in class_totals:
            n = class_totals[klass]
            print(f"  {klass:<10s}  {n:>6,} ({100*n/n_total:5.2f}%)")

    fig, ax = plt.subplots(figsize=(8.5, 4.0))
    sort_idx = np.argsort(codes)
    codes_sorted = codes[sort_idx]
    counts_sorted = counts[sort_idx]
    colours = [COLOUR_OF.get(CLASS_OF.get(int(c), "other"), "#888888")
               for c in codes_sorted]
    ax.bar([str(c) for c in codes_sorted], counts_sorted, color=colours,
           edgecolor="white", linewidth=0.5)
    ax.set_xlabel("AIS vessel-type code")
    ax.set_ylabel("Routes in cleaned corpus")
    ax.set_title(f"Vessel-type composition of the cleaned 128-pt corpus "
                 f"(n = {n_total:,}, {len(codes)} distinct codes)")
    ax.tick_params(axis="x", labelrotation=0, labelsize=7)
    ax.grid(axis="y", alpha=0.3)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLOUR_OF[k])
               for k in ("passenger", "cargo", "tanker")]
    ax.legend(handles, ["passenger (60-69)", "cargo (70-79)", "tanker (80-89)"],
              loc="upper right", frameon=False)
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
