"""plot_comparison.py — produce the cross-model report figures.

All numeric values are read from CSV files, never hardcoded:

  results/headline_clean_5seed.csv          one row per (variant × seed)
  results/headline_clean_5seed_summary.csv  one row per variant, aggregated
  runs/trajgen_v9_retrieval/metrics.csv     v9 train/val loss curves
  runs/trajgen_v10/metrics.csv              v10 train/val loss curves
  runs/trajgen_v12/metrics.csv              v12 train/val loss curves

Generated PNGs (in --out_dir, default ``figures/``):

    ade_by_model.png            mean ± std ADE per variant on the cleaned
                                corpus, 5 subsample seeds (error bars)
    length_bucketed_ade.png     short / medium / long ADE per variant,
                                5-seed mean ± std error bars
    v9_v10_v12_training_curves.png   training and validation loss curves

The script fails loudly if any required CSV is missing — it never silently
falls back to hardcoded constants. Re-running it after a new evaluation
sweep (``scripts/eval/eval_all_clean.py``) refreshes every figure.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]


# ─── Variant ordering / display ─────────────────────────────────────────
# Order in which variants appear left-to-right in bar charts. Variants not
# present in the CSV are silently skipped.
VARIANT_ORDER: List[str] = [
    "v4", "v5",
    "v9",
    "v10_raw", "v10_hardproj", "v10_router",
    "retr_top1",
    "great_circle",
]

VARIANT_LABELS: Dict[str, str] = {
    "v4":           "v4 (encoder + bearing)",
    "v5":           "v5 (+ sched. sampling)",
    "v9":           "v9 (mean-pool retrieval)",
    "v10_raw":      "v10 raw",
    "v10_hardproj": "v10 + hard projection",
    "v10_router":   "v10 + router (production)",
    "retr_top1":    "Retrieval top-1",
    "great_circle": "Great-circle",
}

# colour palette: water-strict in green, retrieval in orange, GC in red
VARIANT_COLOR: Dict[str, str] = {
    "v4":           "#3a86ff",
    "v5":           "#3a86ff",
    "v9":           "#3a86ff",
    "v10_raw":      "#3a86ff",
    "v10_hardproj": "#06d6a0",
    "v10_router":   "#06d6a0",
    "retr_top1":    "#ffba08",
    "great_circle": "#ef476f",
}


# ─── CSV utilities ──────────────────────────────────────────────────────

def fail(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        fail(f"missing CSV: {path}")
    with path.open() as f:
        return list(csv.DictReader(f))


def _read_summary(path: Path) -> Dict[str, Dict[str, float]]:
    rows = _read_csv(path)
    out: Dict[str, Dict[str, float]] = {}
    for r in rows:
        out[r["variant"]] = {k: float(v) for k, v in r.items()
                             if k not in ("variant", "dataset")}
    return out


def _per_seed_rows_by_variant(path: Path) -> Dict[str, List[Dict[str, float]]]:
    rows = _read_csv(path)
    out: Dict[str, List[Dict[str, float]]] = {}
    for r in rows:
        v = r["variant"]
        clean = {}
        for k, val in r.items():
            if k in ("variant", "dataset"):
                continue
            try:
                clean[k] = float(val)
            except (TypeError, ValueError):
                clean[k] = val  # type: ignore[assignment]
        out.setdefault(v, []).append(clean)
    return out


# ─── ADE bar chart with 5-seed error bars ───────────────────────────────

def plot_ade_by_model(summary: Dict[str, Dict[str, float]],
                      out_path: Path,
                      n_seeds: int = 5) -> None:
    ordered = [v for v in VARIANT_ORDER if v in summary]
    if not ordered:
        fail("no known variants found in summary CSV")

    labels = [VARIANT_LABELS[v] for v in ordered]
    means  = [summary[v]["ade_m_mean"] for v in ordered]
    stds   = [summary[v]["ade_m_std"]  for v in ordered]
    colors = [VARIANT_COLOR[v] for v in ordered]

    fig, ax = plt.subplots(figsize=(11, 5.2))
    x = np.arange(len(ordered))
    bars = ax.bar(x, means, yerr=stds, capsize=6,
                  color=colors, edgecolor="black", linewidth=0.6,
                  error_kw={"linewidth": 1.4})

    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + std + 100,
                f"{mean:.0f}±{std:.0f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=10)
    ax.set_ylabel("ADE (metres) — lower is better")
    ax.set_title(f"Average Displacement Error on the cleaned corpus\n"
                 f"(mean ± std over {n_seeds} disjoint 500-route subsamples,"
                 f" same trained checkpoint per variant)")
    ax.set_ylim(0, max(means + stds) * 1.20)
    ax.grid(axis="y", alpha=0.25, linestyle=":")

    # Colour legend
    handles = [
        plt.Rectangle((0, 0), 1, 1, color="#3a86ff", label="parametric model"),
        plt.Rectangle((0, 0), 1, 1, color="#06d6a0", label="water-strict (post-processed)"),
        plt.Rectangle((0, 0), 1, 1, color="#ffba08", label="non-parametric retrieval"),
        plt.Rectangle((0, 0), 1, 1, color="#ef476f", label="geometric baseline"),
    ]
    ax.legend(handles=handles, loc="upper left", framealpha=0.9, fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"  wrote {out_path}")


# ─── Length-bucketed bar chart ─────────────────────────────────────────

def plot_length_bucketed_ade(summary: Dict[str, Dict[str, float]],
                             out_path: Path,
                             n_seeds: int = 5,
                             include: List[str] = None) -> None:
    if include is None:
        include = ["v5", "v9", "v10_router", "retr_top1", "great_circle"]
    include = [v for v in include if v in summary]
    if not include:
        fail("no variants matched for length-bucketed plot")

    buckets = ["short (<50 km)", "medium (50-500 km)", "long (>500 km)"]
    bucket_keys = ["ade_short_m", "ade_medium_m", "ade_long_m"]

    fig, ax = plt.subplots(figsize=(11, 5.2))
    n_models = len(include)
    bar_width = 0.85 / n_models
    x = np.arange(len(buckets))

    for i, variant in enumerate(include):
        s = summary[variant]
        means = [s[f"{k}_mean"] for k in bucket_keys]
        stds  = [s[f"{k}_std"]  for k in bucket_keys]
        offset = x + (i - (n_models - 1) / 2) * bar_width
        bars = ax.bar(offset, means, bar_width, yerr=stds, capsize=4,
                      color=VARIANT_COLOR[variant],
                      label=VARIANT_LABELS[variant],
                      edgecolor="black", linewidth=0.4,
                      error_kw={"linewidth": 1.0})
        for bar, mean in zip(bars, means):
            if not np.isfinite(mean):
                continue
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(1.0, mean * 0.05),
                    f"{mean/1000:.0f}k", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(buckets)
    ax.set_ylabel("ADE (metres)")
    ax.set_yscale("log")
    ax.set_ylim(800, 1.2e5)
    ax.set_title(f"Length-bucketed ADE on the cleaned corpus\n"
                 f"(mean over {n_seeds} subsample seeds; error bars = std)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="y", alpha=0.25, linestyle=":")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"  wrote {out_path}")


# ─── Training curves (unchanged: reads runs/*/metrics.csv) ──────────────

def _read_metrics_csv(path: Path) -> Dict[str, np.ndarray]:
    if not path.exists():
        fail(f"missing metrics.csv: {path}")
    rows = list(csv.DictReader(path.open()))
    if not rows:
        fail(f"{path} is empty")
    out: Dict[str, np.ndarray] = {}
    for col in rows[0]:
        try:
            out[col] = np.array(
                [float(r[col]) if r.get(col) not in (None, "") else np.nan
                 for r in rows], dtype=float)
        except ValueError:
            continue
    return out


def plot_training_curves(out_path: Path) -> None:
    runs = {
        "v9 retrieval":     REPO_ROOT / "runs" / "trajgen_v9_retrieval" / "metrics.csv",
        "v10 (production)": REPO_ROOT / "runs" / "trajgen_v10"          / "metrics.csv",
        "v12 (abandoned)":  REPO_ROOT / "runs" / "trajgen_v12"          / "metrics.csv",
    }
    colors = {"v9 retrieval": "#3a86ff",
              "v10 (production)": "#06d6a0",
              "v12 (abandoned)": "#ef476f"}

    fig, (ax_train, ax_val) = plt.subplots(1, 2, figsize=(13, 4.5), sharex=False)
    for name, path in runs.items():
        m = _read_metrics_csv(path)
        if "epoch" not in m or "train_loss" not in m:
            fail(f"{path} missing epoch/train_loss columns")
        train_mask = ~np.isnan(m["train_loss"])
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

    fig.suptitle("Training and validation loss for v9, v10 and v12\n"
                 "(v12 cross-entropy on a discrete pointer is on a different scale)",
                 y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


# ─── Main ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate cross-model report figures (CSV-driven).")
    parser.add_argument("--summary_csv",
        default=str(REPO_ROOT / "results" / "headline_clean_5seed_summary.csv"),
        help="Per-variant aggregated CSV (mean / std over seeds)")
    parser.add_argument("--per_seed_csv",
        default=str(REPO_ROOT / "results" / "headline_clean_5seed.csv"),
        help="One row per (variant, seed) — used for sanity checks")
    parser.add_argument("--out_dir", default="figures",
        help="Output directory for PNGs (default: figures/)")
    parser.add_argument("--n_seeds", type=int, default=5,
        help="Number of seeds used (for captions); must match the CSV.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = _read_summary(Path(args.summary_csv))
    print(f"Read {len(summary)} variants from {args.summary_csv}")

    # Sanity check: per_seed CSV exists and matches.
    per_seed = _per_seed_rows_by_variant(Path(args.per_seed_csv))
    for v, rows in per_seed.items():
        if v in summary and len(rows) != args.n_seeds:
            print(f"  [WARN] {v}: {len(rows)} seed rows (expected {args.n_seeds})")

    print("\nGenerating figures:")
    plot_ade_by_model(
        summary, out_dir / "ade_by_model.png", n_seeds=args.n_seeds)
    plot_length_bucketed_ade(
        summary, out_dir / "length_bucketed_ade.png", n_seeds=args.n_seeds)
    plot_training_curves(out_dir / "v9_v10_v12_training_curves.png")

    print(f"\nAll figures written to {out_dir}/")


if __name__ == "__main__":
    main()
