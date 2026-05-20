#!/usr/bin/env python3
"""
make_architecture_diagram.py -- block-diagram figure of the v10 production
model (encoder + per-step retrieval bank + decoder + land-aware loss +
WaterRouter post-processor).

Writes:  figures/architecture_v10.png + report/figures/architecture_v10.png

Run:     python3 scripts/eval/make_architecture_diagram.py
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def box(ax, xy, w, h, text, fc="#eaf2fb", ec="#3b6fa7", fs=9, lw=1.4):
    x, y = xy
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.05",
        facecolor=fc, edgecolor=ec, linewidth=lw,
    )
    ax.add_patch(p)
    ax.text(x + w / 2.0, y + h / 2.0, text,
            ha="center", va="center", fontsize=fs)


def arrow(ax, xy_from, xy_to, color="#222", lw=1.3, style="-|>",
          rad=0.0):
    a = FancyArrowPatch(
        xy_from, xy_to,
        arrowstyle=style, mutation_scale=14,
        color=color, linewidth=lw,
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(a)


def main():
    fig, ax = plt.subplots(figsize=(11.5, 8.4))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.set_aspect("equal")
    ax.axis("off")

    # =================================================================
    # ENCODER (left/top)
    # =================================================================
    ax.text(3.5, 9.55, "Encoder memory (163 tokens)",
            ha="center", fontsize=11, fontweight="bold")

    # 3 conditioning tokens
    box(ax, (0.4, 8.55), 1.6, 0.65, "start (lon,lat)",
        fc="#fdebd0", ec="#cf6f00", fs=8.5)
    box(ax, (2.2, 8.55), 1.6, 0.65, "end (lon,lat)",
        fc="#fdebd0", ec="#cf6f00", fs=8.5)
    box(ax, (4.0, 8.55), 1.6, 0.65, "vessel type",
        fc="#fdebd0", ec="#cf6f00", fs=8.5)

    # Retrieval bank (K=5 routes x t_retr=32 tokens)
    box(ax, (0.4, 7.30), 5.2, 1.0,
        "K = 5 retrieved historical routes\n"
        r"$\rightarrow$ each subsampled to $t_{\mathrm{retr}}=32$ tokens"
        "\nrich shape information preserved",
        fc="#e8f6ef", ec="#1f7a4c", fs=8.5)

    # Encoder block (visual grouping)
    enc = FancyBboxPatch((0.2, 7.1), 5.6, 2.25,
                          boxstyle="round,pad=0.02,rounding_size=0.05",
                          facecolor="none", edgecolor="#3b6fa7",
                          linewidth=1.0, linestyle="--")
    ax.add_patch(enc)

    # Sub-caption under encoder
    ax.text(3.0, 6.9,
            "3 base + 5 x 32 = 163 memory tokens, refined by 2 Transformer encoder layers",
            ha="center", fontsize=8.5, style="italic", color="#444")

    # =================================================================
    # DECODER (right)
    # =================================================================
    ax.text(10.5, 9.55, "Autoregressive decoder",
            ha="center", fontsize=11, fontweight="bold")

    # decoder stack
    box(ax, (7.4, 7.30), 6.0, 2.0,
        "4 x Transformer decoder layer\n"
        "(d_model = 256, 8 heads, FFN = 1024)\n\n"
        r"windowed causal mask ($k_{\mathrm{past}} = 32$)"
        "\n"
        "input per step: [lon, lat, progress, sin/cos(bearing-to-end)]",
        fc="#eaf2fb", ec="#3b6fa7", fs=9)

    # cross-attention arrow encoder -> decoder
    arrow(ax, (5.8, 8.2), (7.4, 8.2), color="#3b6fa7", lw=1.8)
    ax.text(6.6, 8.4, "cross-attn", ha="center", fontsize=8,
            color="#3b6fa7")

    # =================================================================
    # PER-STEP OUTPUT
    # =================================================================
    box(ax, (5.6, 5.7), 2.8, 0.7,
        r"$\Delta_t = (\delta\mathrm{lon}_t, \delta\mathrm{lat}_t)$"
        "\nper-step displacement",
        fc="#fff5e6", ec="#a06000", fs=9)
    arrow(ax, (10.4, 7.3), (7.4, 6.4), color="#3b6fa7", lw=1.4)

    # Position recovery
    box(ax, (9.0, 5.7), 4.0, 0.7,
        r"$p_{t+1} = p_t + \Delta_t$  (cumulative recovery)",
        fc="#fff5e6", ec="#a06000", fs=9)

    # =================================================================
    # TRAINING LOSS BOX
    # =================================================================
    box(ax, (0.4, 5.1), 5.4, 1.3,
        "Training loss\n\n"
        r"$\mathcal{L} = \mathcal{L}_{\Delta}(\mathrm{Huber}) + $"
        r"$\lambda_{\mathrm{end}} \mathcal{L}_{\mathrm{end}} + $"
        r"$\lambda_{\mathrm{smooth}} \mathcal{L}_{\mathrm{smooth}} + $"
        r"$\lambda_{\mathrm{land}} \mathcal{L}_{\mathrm{land}}$"
        "\n\n"
        r"$\mathcal{L}_{\mathrm{land}} = $ mean ReLU($\mathrm{sdf}_{\mathrm{km}} - 10$)$^2$",
        fc="#f5e6ff", ec="#6c2dbf", fs=9)
    arrow(ax, (5.6, 5.95), (5.8, 5.95), color="#6c2dbf", lw=1.3)

    # =================================================================
    # INFERENCE-TIME POST-PROCESSING
    # =================================================================
    ax.text(7.0, 4.55, "Inference-time post-processing",
            ha="center", fontsize=11, fontweight="bold")

    box(ax, (0.4, 3.0), 4.2, 1.2,
        "1.  Hard land projection\n"
        "    (per step, BFS to nearest\n"
        "     water cell on 0.05 deg SDF)",
        fc="#e8f6ef", ec="#1f7a4c", fs=9)
    box(ax, (4.9, 3.0), 4.2, 1.2,
        "2.  Linear endpoint correction\n"
        "    distribute residual end-error\n"
        r"    proportionally ($\alpha = t / (T-1)$)",
        fc="#e8f6ef", ec="#1f7a4c", fs=9)
    box(ax, (9.4, 3.0), 4.2, 1.2,
        "3.  WaterRouter snap-only\n"
        "    A*-graph water cells on 0.005 deg\n"
        "    (~22 ms / trajectory)",
        fc="#e8f6ef", ec="#1f7a4c", fs=9)

    arrow(ax, (11.0, 5.7), (11.5, 4.2), color="#1f7a4c", lw=1.3)
    arrow(ax, (4.6, 3.6), (4.9, 3.6), color="#1f7a4c", lw=1.0)
    arrow(ax, (9.1, 3.6), (9.4, 3.6), color="#1f7a4c", lw=1.0)

    # =================================================================
    # FINAL OUTPUT
    # =================================================================
    box(ax, (4.0, 1.2), 6.0, 1.2,
        "Final 128-waypoint trajectory\n"
        r"$\mathrm{ADE} = 6051 \pm 225$ m (5 seeds)"
        "\n"
        r"land\%@10 km $= 0.0$, cross\%@10 km $= 0.0$",
        fc="#fffae6", ec="#b58900", fs=10, lw=1.8)
    arrow(ax, (11.5, 3.0), (10.0, 2.4), color="#b58900", lw=1.6)

    # =================================================================
    # Legend
    # =================================================================
    legend_patches = [
        mpatches.Patch(facecolor="#fdebd0", edgecolor="#cf6f00",
                        label="conditioning tokens"),
        mpatches.Patch(facecolor="#e8f6ef", edgecolor="#1f7a4c",
                        label="retrieval / water-aware"),
        mpatches.Patch(facecolor="#eaf2fb", edgecolor="#3b6fa7",
                        label="transformer blocks"),
        mpatches.Patch(facecolor="#fff5e6", edgecolor="#a06000",
                        label="per-step output"),
        mpatches.Patch(facecolor="#f5e6ff", edgecolor="#6c2dbf",
                        label="training loss"),
        mpatches.Patch(facecolor="#fffae6", edgecolor="#b58900",
                        label="production output"),
    ]
    ax.legend(handles=legend_patches, loc="lower center",
              ncol=3, fontsize=8, frameon=False,
              bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    for out in ["figures/architecture_v10.png",
                "report/figures/architecture_v10.png"]:
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        plt.savefig(out, dpi=170, bbox_inches="tight")
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
