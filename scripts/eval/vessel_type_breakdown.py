"""Per-vessel-type ADE for the production v10 + router checkpoint.

Reads results/per_route_lambdaland05.csv (produced by per_route_eval.py) and:
  - aggregates ADE by vessel-type code (and by coarse class: passenger / cargo / tanker)
  - writes a per-class summary table for the report
  - plots a bar chart of per-type ADE on val + test

Outputs:
    figures/ade_by_vessel_type.png
    report/figures/ade_by_vessel_type.png
    results/vessel_type_ade.csv
    report/tables/vessel_type_ade.tex
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


CLASS_OF = {c: ("passenger" if 60 <= c <= 69 else
                "cargo"     if 70 <= c <= 79 else
                "tanker"    if 80 <= c <= 89 else "other") for c in range(60, 100)}
COLOUR_OF = {"passenger": "#1f77b4", "cargo": "#2ca02c",
             "tanker": "#d62728", "other": "#999999"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_route_csv", default="results/per_route_lambdaland05.csv")
    ap.add_argument("--out_csv",  default="results/vessel_type_ade.csv")
    ap.add_argument("--out_tex",  default="report/tables/vessel_type_ade.tex")
    ap.add_argument("--out_png",  default="figures/ade_by_vessel_type.png")
    ap.add_argument("--out_png_report", default="report/figures/ade_by_vessel_type.png")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.per_route_csv)))
    per_type: dict[tuple[str, int], list[float]] = defaultdict(list)
    per_class: dict[tuple[str, str], list[float]] = defaultdict(list)

    for r in rows:
        ade = float(r["ade_router_m"])
        vt  = int(r["vessel_type"])
        sp  = r["split"]
        klass = CLASS_OF.get(vt, "other")
        per_type[(sp, vt)].append(ade)
        per_class[(sp, klass)].append(ade)

    # Per-type rows
    out_rows = []
    for (sp, vt), ades in sorted(per_type.items()):
        a = np.array(ades)
        out_rows.append({
            "split": sp, "vessel_type": vt, "class": CLASS_OF.get(vt, "other"),
            "n": len(a), "mean_ade_m": float(a.mean()),
            "median_ade_m": float(np.median(a)),
            "std_ade_m": float(a.std(ddof=0)),
        })
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)

    # Per-class table fragment (LaTeX)
    os.makedirs(os.path.dirname(args.out_tex) or ".", exist_ok=True)
    with open(args.out_tex, "w") as f:
        f.write("\\begin{tabular}{lrrrrrr}\n\\toprule\n")
        f.write("\\textbf{Class} & "
                "$n_{\\mathrm{val}}$ & val \\ade{} (m) & val median (m) & "
                "$n_{\\mathrm{test}}$ & test \\ade{} (m) & test median (m) \\\\\n")
        f.write("\\midrule\n")
        for klass in ("passenger", "cargo", "tanker", "other"):
            vk = (("val", klass) in per_class) or (("test", klass) in per_class)
            if not vk:
                continue
            val = np.array(per_class.get(("val",  klass), []))
            tst = np.array(per_class.get(("test", klass), []))
            f.write(f"{klass} & "
                    f"{len(val)} & ${val.mean():.0f} \\pm {val.std(ddof=0):.0f}$ & "
                    f"${np.median(val):.0f}$ & "
                    f"{len(tst)} & ${tst.mean():.0f} \\pm {tst.std(ddof=0):.0f}$ & "
                    f"${np.median(tst):.0f}$ \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")

    # Bar chart: per-type ADE on val + test
    all_types = sorted({vt for (_, vt) in per_type.keys()})
    val_means = [float(np.mean(per_type.get(("val",  vt), [np.nan]))) for vt in all_types]
    tst_means = [float(np.mean(per_type.get(("test", vt), [np.nan]))) for vt in all_types]
    val_ns    = [len(per_type.get(("val",  vt), [])) for vt in all_types]
    tst_ns    = [len(per_type.get(("test", vt), [])) for vt in all_types]

    fig, ax = plt.subplots(figsize=(10.5, 4.4))
    x = np.arange(len(all_types))
    width = 0.38
    colours = [COLOUR_OF[CLASS_OF.get(vt, "other")] for vt in all_types]
    ax.bar(x - width / 2, val_means, width, color=colours, edgecolor="black",
           linewidth=0.4, label="val (n=2000 subsample)")
    ax.bar(x + width / 2, tst_means, width, color=colours, edgecolor="black",
           linewidth=0.4, alpha=0.55, label="test (n=2326)")
    for i, (n_v, n_t) in enumerate(zip(val_ns, tst_ns)):
        ax.text(i, max(val_means[i] or 0, tst_means[i] or 0) + 200,
                f"{n_v}/{n_t}", ha="center", fontsize=6.5, color="#444")
    ax.set_xticks(x, [str(t) for t in all_types], fontsize=7)
    ax.set_xlabel("AIS vessel-type code (sample sizes shown above bars: val / test)")
    ax.set_ylabel("ADE (m, v10 + router)")
    ax.set_title("Per-vessel-type ADE on val (2000-sample) and test "
                 "(2326, full held-out)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLOUR_OF[k]) for k in
               ("passenger", "cargo", "tanker")]
    legend1 = ax.legend(handles, ["passenger", "cargo", "tanker"], loc="upper left",
                         frameon=False, fontsize=8)
    ax.add_artist(legend1)
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    os.makedirs(os.path.dirname(args.out_png) or ".", exist_ok=True)
    fig.savefig(args.out_png, dpi=160)
    if args.out_png_report:
        os.makedirs(os.path.dirname(args.out_png_report) or ".", exist_ok=True)
        fig.savefig(args.out_png_report, dpi=160)
    plt.close(fig)

    print("Per-class ADE (mean ± std):")
    for klass in ("passenger", "cargo", "tanker"):
        for sp in ("val", "test"):
            xs = np.array(per_class.get((sp, klass), []))
            if len(xs):
                print(f"  {klass:<10s} {sp:<5s} n={len(xs):>4}  "
                      f"mean={xs.mean():7.1f}  median={np.median(xs):7.1f}  "
                      f"std={xs.std(ddof=0):6.1f}")
    print(f"\nWrote {args.out_csv}, {args.out_tex}, {args.out_png}")


if __name__ == "__main__":
    main()
