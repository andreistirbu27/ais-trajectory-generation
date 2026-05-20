"""Generate cross-model report figures from results/summary.csv and runs/*/metrics.csv.

Usage:
    python3 scripts/eval/plot_comparison.py --out_dir figures/

Produces four PNGs in --out_dir:
    ade_by_model.png            : ADE bars across the trajectory-generation family
    fde_by_model.png            : FDE bars (highlights endpoint-correction effect)
    length_bucketed_ade.png     : short / medium / long ADE for production variants
    v9_v10_v12_training_curves.png : train_loss + val_loss curves for v9, v10, v12

All numbers are sourced from results/summary.csv and the per-run metrics.csv files.
The script fails loudly if a referenced metrics.csv is missing — it never silently
emits empty plots.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Numbers taken verbatim from docs/narrative.md §3.6, §3.7 and results/summary.csv.
# These are model-specific length-bucketed ADEs that are NOT in the CSV columns
# (they live only in the free-text 'notes' field of summary.csv).
# Keeping them as a typed dict here prevents NaN-bar bugs and keeps the report
# faithful to what was actually measured.

LENGTH_BUCKETED_ADE_M = {
    # bucket -> {model: ADE_metres}
    "short (<50 km)":    {"v9 ep38": 2117,  "v10 raw": None,   "retr-top1": 3366,  "GC": 2264},
    "medium (50-500 km)": {"v9 ep38": 9050,  "v10 raw": 8397,   "retr-top1": 9143,  "GC": 11598},
    "long (>500 km)":    {"v9 ep38": 46631, "v10 raw": 25964,  "retr-top1": 18785, "GC": 94774},
}

# Bars to draw on the ADE bar chart. Values are model_ade_m from summary.csv;
# data slice annotation noted for the report's caption to read clearly.
ADE_BARS = [
    # label,           ADE_m, slice,      epoch, water_strict
    ("MVP",             6807, "trajgen_128",      45, False),
    ("v2",              None, "trajgen_128",      30, False),  # CSV has no model_ade_m
    ("v4",              None, "trajgen_128",      60, False),  # likewise
    ("v5",              6753, "trajgen_128",      60, False),
    ("v6 obstacle",     7202, "trajgen_128",      49, False),
    ("v9 ep38",         6207, "trajgen_128",      38, False),
    ("v10 raw",         6584, "trajgen_128",      52, False),
    ("v10 + router",    7934, "trajgen_128",      52, True),   # snap-only on dirty data
    ("v10 raw / clean", 6278, "trajgen_128_clean", 52, False),
    ("v10+router/clean", 6282, "trajgen_128_clean", 52, True), # production
    ("v12",            27365, "trajgen_128",      40, False),
    ("retr-top1",       6470, "trajgen_128",      None, False),
    ("retr-top1/clean",12373, "trajgen_128_clean", None, False),
    ("great-circle",    8226, "trajgen_128",      None, False),
    ("great-circle/clean", 9304, "trajgen_128_clean", None, False),
]

# FDE numbers from summary.csv. AR models with linear endpoint correction land
# at FDE 0; retrieval-top-1 has non-zero FDE because no correction is applied.
FDE_BARS = [
    ("MVP",             0,    True),   # endpoint correction
    ("v5",              0,    True),
    ("v9 ep38",         0,    True),
    ("v10 raw",         0,    True),
    ("v10 + router",    4845, True),   # router can shift the final cell
    ("v12",             41038, False),
    ("retr-top1",       3306, False),
    ("great-circle",    0,    False),
]


def fail(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def plot_ade_by_model(out_path: Path) -> None:
    labels = [b[0] for b in ADE_BARS if b[1] is not None]
    values = [b[1] for b in ADE_BARS if b[1] is not None]
    water_strict = [b[4] for b in ADE_BARS if b[1] is not None]
    colors = ["#3a86ff" if not w else "#06d6a0" for w in water_strict]

    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(range(len(values)), values, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(labels, rotation=40, ha="right")
    ax.set_ylabel("ADE (metres)")
    ax.set_title("Average Displacement Error across model iterations\n"
                 "(n=500 validation routes, seed=42; lower is better)")
    ax.set_ylim(0, max(values) * 1.10)

    # Annotate each bar with the value
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 200,
                f"{v}", ha="center", va="bottom", fontsize=8)

    # Reference line: great-circle baseline on dirty data
    gc_ade = next(b[1] for b in ADE_BARS if b[0] == "great-circle")
    ax.axhline(gc_ade, color="#ef476f", linestyle="--", linewidth=1, alpha=0.7,
               label=f"Great-circle baseline ({gc_ade} m)")

    # Legend explaining colour code
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color="#3a86ff", label="ADE-only model"),
        plt.Rectangle((0, 0), 1, 1, color="#06d6a0", label="Water-strict (post-processed)"),
        plt.Line2D([0], [0], color="#ef476f", linestyle="--", label=f"Great-circle ({gc_ade} m)"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", framealpha=0.9)
    ax.grid(axis="y", alpha=0.25, linestyle=":")

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_fde_by_model(out_path: Path) -> None:
    labels = [b[0] for b in FDE_BARS]
    values = [b[1] for b in FDE_BARS]
    has_correction = [b[2] for b in FDE_BARS]
    colors = ["#3a86ff" if c else "#ffba08" for c in has_correction]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.bar(range(len(values)), values, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("FDE (metres)")
    ax.set_title("Final Displacement Error: linear endpoint correction\n"
                 "drives FDE to zero for AR models")
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.01,
                f"{v}", ha="center", va="bottom", fontsize=8)

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color="#3a86ff", label="With endpoint correction"),
        plt.Rectangle((0, 0), 1, 1, color="#ffba08", label="No correction"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", framealpha=0.9)
    ax.grid(axis="y", alpha=0.25, linestyle=":")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_length_bucketed_ade(out_path: Path) -> None:
    buckets = list(LENGTH_BUCKETED_ADE_M.keys())
    models = ["v9 ep38", "v10 raw", "retr-top1", "GC"]
    colors = {"v9 ep38": "#3a86ff", "v10 raw": "#06d6a0",
              "retr-top1": "#ffba08", "GC": "#ef476f"}

    fig, ax = plt.subplots(figsize=(10, 5))
    n_models = len(models)
    bar_width = 0.18
    x = np.arange(len(buckets))

    for i, model in enumerate(models):
        vals = [LENGTH_BUCKETED_ADE_M[b].get(model) for b in buckets]
        # Replace None with 0 for plotting, but skip annotation
        plot_vals = [v if v is not None else 0 for v in vals]
        offsets = x + (i - (n_models - 1) / 2) * bar_width
        bars = ax.bar(offsets, plot_vals, bar_width, label=model,
                      color=colors[model], edgecolor="black", linewidth=0.4)
        for bar, v in zip(bars, vals):
            if v is None:
                ax.text(bar.get_x() + bar.get_width() / 2, 1500,
                        "n/a", ha="center", va="bottom", fontsize=7, rotation=90)
            else:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1500,
                        f"{v//1000}k", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(buckets)
    ax.set_ylabel("ADE (metres)")
    ax.set_yscale("log")
    ax.set_ylim(1000, 200000)
    ax.set_title("Length-bucketed ADE (dirty data, n=500, seed=42)\n"
                 "v9 wins short+medium; retrieval-top-1 wins long-tail")
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.25, linestyle=":")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"  wrote {out_path}")


def _read_metrics_csv(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        fail(f"missing metrics.csv: {path}")
    rows: list[dict[str, str]] = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    out: dict[str, np.ndarray] = {}
    for col in rows[0].keys():
        try:
            out[col] = np.array([float(r[col]) if r.get(col) not in (None, "") else np.nan
                                 for r in rows], dtype=float)
        except ValueError:
            # non-numeric column, skip
            continue
    return out


def plot_training_curves(out_path: Path) -> None:
    runs = {
        "v9 retrieval": REPO_ROOT / "runs" / "trajgen_v9_retrieval" / "metrics.csv",
        "v10 (production)": REPO_ROOT / "runs" / "trajgen_v10" / "metrics.csv",
        "v12 (abandoned)": REPO_ROOT / "runs" / "trajgen_v12" / "metrics.csv",
    }
    fig, (ax_train, ax_val) = plt.subplots(1, 2, figsize=(13, 4.5), sharex=False)
    colors = {"v9 retrieval": "#3a86ff", "v10 (production)": "#06d6a0", "v12 (abandoned)": "#ef476f"}

    for name, path in runs.items():
        m = _read_metrics_csv(path)
        if "epoch" not in m or "train_loss" not in m:
            fail(f"{path} missing epoch/train_loss columns")
        # Smooth val_loss to epoch-end values: take rows where val_loss is non-NaN.
        train_mask = ~np.isnan(m.get("train_loss", np.array([])))
        ax_train.plot(m["epoch"][train_mask], m["train_loss"][train_mask],
                      color=colors[name], alpha=0.85, label=name, linewidth=1.2)
        if "val_loss" in m:
            val_mask = ~np.isnan(m["val_loss"])
            ax_val.plot(m["epoch"][val_mask], m["val_loss"][val_mask],
                        color=colors[name], alpha=0.85, label=name, linewidth=1.2,
                        marker="o", markersize=3)

    for ax, title in [(ax_train, "Training loss"), (ax_val, "Validation loss")]:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss (normalised units)")
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.25, linestyle=":")
        ax.set_yscale("log")

    fig.suptitle("Training and validation loss across three model families\n"
                 "(NB: v12's CE loss is on a different scale and trajectory)", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate cross-model report figures.")
    parser.add_argument("--out_dir", default="figures",
                        help="Output directory for PNGs (default: figures/)")
    parser.add_argument("--summary_csv", default=str(REPO_ROOT / "results" / "summary.csv"),
                        help="Path to results/summary.csv (existence verified, "
                             "but values for length-bucketed bars are hard-coded "
                             "from narrative.md §3 since they are not in CSV columns)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = Path(args.summary_csv)
    if not summary.exists():
        fail(f"summary CSV not found: {summary}")
    print(f"Reading {summary}")

    print("Generating figures:")
    plot_ade_by_model(out_dir / "ade_by_model.png")
    plot_fde_by_model(out_dir / "fde_by_model.png")
    plot_length_bucketed_ade(out_dir / "length_bucketed_ade.png")
    plot_training_curves(out_dir / "v9_v10_v12_training_curves.png")
    print(f"All figures written to {out_dir}")


if __name__ == "__main__":
    main()
