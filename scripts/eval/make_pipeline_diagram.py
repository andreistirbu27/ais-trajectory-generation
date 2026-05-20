#!/usr/bin/env python3
"""
make_pipeline_diagram.py -- end-to-end data + training + inference pipeline
diagram for the v10 + WaterRouter production system.

Writes:  figures/data_pipeline.png + report/figures/data_pipeline.png

Run:     python3 scripts/eval/make_pipeline_diagram.py
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def box(ax, xy, w, h, text, fc="#eaf2fb", ec="#3b6fa7", fs=8.5, lw=1.2,
        style="round,pad=0.02,rounding_size=0.05"):
    x, y = xy
    p = FancyBboxPatch((x, y), w, h, boxstyle=style,
                        facecolor=fc, edgecolor=ec, linewidth=lw)
    ax.add_patch(p)
    ax.text(x + w / 2.0, y + h / 2.0, text,
            ha="center", va="center", fontsize=fs)


def arrow(ax, xy_from, xy_to, color="#222", lw=1.2, rad=0.0):
    a = FancyArrowPatch(xy_from, xy_to,
                         arrowstyle="-|>", mutation_scale=11,
                         color=color, linewidth=lw,
                         connectionstyle=f"arc3,rad={rad}")
    ax.add_patch(a)


def main():
    fig, ax = plt.subplots(figsize=(11.5, 10.5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 14)
    ax.set_aspect("equal")
    ax.axis("off")

    # ── Stage colours
    raw_fc, raw_ec   = "#fdebd0", "#cf6f00"     # raw / input
    proc_fc, proc_ec = "#eaf2fb", "#3b6fa7"     # processing
    art_fc, art_ec   = "#e8f6ef", "#1f7a4c"     # artefacts (cached files)
    side_fc, side_ec = "#f5e6ff", "#6c2dbf"     # side product
    train_fc, train_ec = "#fff5e6", "#a06000"   # training
    out_fc, out_ec   = "#fffae6", "#b58900"     # production output

    # ── Top: raw data ingestion
    ax.text(7.0, 13.55, "End-to-end production pipeline",
            ha="center", fontsize=12, fontweight="bold")

    box(ax, (5.0, 12.5), 4.0, 0.65,
        "NOAA AIS zip archives (2024)\n"
        "approximately 99 M position reports",
        fc=raw_fc, ec=raw_ec, fs=9)

    # ── Filtering pipeline
    box(ax, (5.0, 11.4), 4.0, 0.65,
        "scripts/data/prepare_data.py\n"
        "20-step quality filter, vessel types 60-89",
        fc=proc_fc, ec=proc_ec)
    arrow(ax, (7.0, 12.5), (7.0, 12.05))

    box(ax, (5.0, 10.3), 4.0, 0.65,
        "data/processed/AIS_2024_full.csv\n"
        "598 K open-water tracks",
        fc=art_fc, ec=art_ec)
    arrow(ax, (7.0, 11.4), (7.0, 10.95))

    # ── Resampling to 128 points
    box(ax, (5.0, 9.2), 4.0, 0.65,
        "scripts/data/prepare_trajgen.py\n"
        "resample to 128 equal-arc waypoints",
        fc=proc_fc, ec=proc_ec)
    arrow(ax, (7.0, 10.3), (7.0, 9.85))

    box(ax, (5.0, 8.1), 4.0, 0.65,
        "trajgen_128.npz  (207 K tracks)\n"
        "shape (N, 128, 2)",
        fc=art_fc, ec=art_ec)
    arrow(ax, (7.0, 9.2), (7.0, 8.75))

    # ── E1: land filter
    box(ax, (5.0, 7.0), 4.0, 0.65,
        "filter_trajgen_npz.py  (E1)\n"
        "drop coastal / inland GT via GSHHS SDF",
        fc=proc_fc, ec=proc_ec)
    arrow(ax, (7.0, 8.1), (7.0, 7.65))

    box(ax, (5.0, 5.9), 4.0, 0.65,
        "trajgen_128_clean.npz  (61 K tracks)\n"
        "training corpus for v10 + WaterRouter",
        fc=art_fc, ec=art_ec, lw=1.8)
    arrow(ax, (7.0, 7.0), (7.0, 6.55))

    # ── Left side: GSHHS to SDF + water graph
    box(ax, (0.3, 11.4), 3.7, 0.65,
        "GSHHS coastline shapefile\n"
        "(external, https://soest.hawaii.edu)",
        fc=raw_fc, ec=raw_ec)

    box(ax, (0.3, 10.3), 3.7, 0.65,
        "build_land_sdf.py\n"
        "rasterise to signed distance (0.05 deg)",
        fc=proc_fc, ec=proc_ec)
    arrow(ax, (2.15, 11.4), (2.15, 10.95))

    box(ax, (0.3, 9.2), 3.7, 0.65,
        "land_sdf_050deg.npz  (LandMask)\n"
        "(900 x 1300, +land/-water in km)",
        fc=art_fc, ec=art_ec)
    arrow(ax, (2.15, 10.3), (2.15, 9.85))

    box(ax, (0.3, 8.1), 3.7, 0.65,
        "build_water_graph.py\n"
        "water-cell adjacency at 0.005 deg",
        fc=proc_fc, ec=proc_ec)
    arrow(ax, (2.15, 9.2), (2.15, 8.75))

    box(ax, (0.3, 7.0), 3.7, 0.65,
        "water_graph_005deg.npz\n"
        "A* graph for WaterRouter",
        fc=art_fc, ec=art_ec)
    arrow(ax, (2.15, 8.1), (2.15, 7.65))

    # arrow from SDF to E1 filter
    arrow(ax, (4.0, 9.5), (5.0, 7.4), color="#444", lw=1.0, rad=-0.1)
    ax.text(4.4, 8.3, "SDF used\nfor E1 + L_land",
            fontsize=7, ha="center", color="#555")

    # ── Right side: KNN cache
    box(ax, (10.0, 8.1), 3.7, 0.65,
        "build_and_cache_knn(...)\n"
        "5-D KNN (start, end, vtype)",
        fc=proc_fc, ec=proc_ec)
    arrow(ax, (9.0, 8.45), (10.0, 8.45), rad=0.05)

    box(ax, (10.0, 7.0), 3.7, 0.65,
        "trajgen_128_clean_knn_k5_test005.npz\n"
        "K = 5 retrieved routes per query",
        fc=art_fc, ec=art_ec)
    arrow(ax, (11.85, 8.1), (11.85, 7.65))

    # ── Training stage
    box(ax, (3.5, 4.55), 7.0, 0.95,
        "scripts/train/train_gen_v10.py\n"
        "TrajectoryGeneratorV10  (encoder + decoder + retrieval bank)\n"
        r"loss = $\mathcal{L}_\Delta + \lambda_{\mathrm{end}}\mathcal{L}_{\mathrm{end}}"
        r" + \lambda_{\mathrm{smooth}}\mathcal{L}_{\mathrm{smooth}}"
        r" + \lambda_{\mathrm{land}}\mathcal{L}_{\mathrm{land}}$",
        fc=train_fc, ec=train_ec, fs=8.8, lw=1.6)
    arrow(ax, (7.0, 5.9), (7.0, 5.5))
    arrow(ax, (11.85, 7.0), (10.5, 5.5), color=train_ec, lw=1.2)
    arrow(ax, (2.15, 7.0), (3.5, 5.5), color=train_ec, lw=1.2)

    box(ax, (4.5, 3.55), 5.0, 0.65,
        "runs/trajgen_v10_clean_train_lambdaland05/best.pt  (epoch 52)\n"
        "5.84 M parameters,  $\\lambda_{\\mathrm{land}} = 0.05$,  test_frac = 0.05",
        fc=art_fc, ec=art_ec)
    arrow(ax, (7.0, 4.55), (7.0, 4.2))

    # ── Inference / production
    box(ax, (3.0, 2.0), 8.0, 1.15,
        "scripts/eval/eval_gen_v10_router.py  (inference + post-processing)\n"
        "1. generate() with per-step retrieval + hard land projection\n"
        "2. linear endpoint correction (drives FDE -> 0)\n"
        "3. WaterRouter snap-only on 0.005 deg water graph",
        fc=proc_fc, ec=proc_ec, lw=1.6, fs=8.8)
    arrow(ax, (7.0, 3.55), (7.0, 3.15))

    box(ax, (3.5, 0.4), 7.0, 1.05,
        "Production output\n"
        r"$\mathrm{ADE} = 6051 \pm 225$ m"
        " (5 seeds, n=500 each, clean dataset)\n"
        r"land\%@10 km $= 0.0$, cross\%@10 km $= 0.0$",
        fc=out_fc, ec=out_ec, lw=1.8, fs=10)
    arrow(ax, (7.0, 2.0), (7.0, 1.45))

    # ── Legend
    legend_patches = [
        mpatches.Patch(facecolor=raw_fc,   edgecolor=raw_ec,
                        label="raw / external input"),
        mpatches.Patch(facecolor=proc_fc,  edgecolor=proc_ec,
                        label="processing script"),
        mpatches.Patch(facecolor=art_fc,   edgecolor=art_ec,
                        label="cached artefact (NPZ / PT)"),
        mpatches.Patch(facecolor=train_fc, edgecolor=train_ec,
                        label="training"),
        mpatches.Patch(facecolor=out_fc,   edgecolor=out_ec,
                        label="production output"),
    ]
    ax.legend(handles=legend_patches, loc="lower center",
              ncol=3, fontsize=8, frameon=False,
              bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    for out in ["figures/data_pipeline.png",
                "report/figures/data_pipeline.png"]:
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        plt.savefig(out, dpi=170, bbox_inches="tight")
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
