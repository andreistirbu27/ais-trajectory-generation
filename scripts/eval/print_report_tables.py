"""Emit LaTeX table fragments for the rewritten report from the unified CSV.

Reads ``results/headline_clean_5seed_summary.csv`` and
``results/headline_clean_5seed.csv`` and writes self-contained
``\\input``-able tables under ``report/tables/``:

    headline.tex          — Table 1 (production + retr-top1 + GC)
    full_ladder.tex       — Table 6 (every variant in the unified run)
    landstats.tex         — Table 4 (land/cross at thresholds 10/25 km)
    length_bucketed.tex   — bucketed ADE per variant

The script is a pure CSV → LaTeX renderer with no hard-coded numbers.
Every entry traces back to a row in the CSV files."""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]


DISPLAY = {
    "v4":           "v4 (encoder + bearing)",
    "v5":           "v5 (+ scheduled sampling)",
    "v9":           "v9 (mean-pool retrieval)",
    "v10_raw":      "v10 raw",
    "v10_hardproj": "v10 + hard projection",
    "v10_router":   r"\textbf{v10 + router (production)}",
    "retr_top1":    "Retrieval top-1 (zero training)",
    "great_circle": "Great-circle baseline",
}


def _read_summary(path: Path) -> Dict[str, Dict[str, float]]:
    with path.open() as f:
        rows = list(csv.DictReader(f))
    out: Dict[str, Dict[str, float]] = {}
    for r in rows:
        d = {"_dataset": r["dataset"]}
        for k, v in r.items():
            if k in ("variant", "dataset", "n_seeds", "n_per_seed"):
                d[k] = v
            else:
                try:
                    d[k] = float(v)
                except (TypeError, ValueError):
                    d[k] = v
        out[r["variant"]] = d
    return out


def _msd(d: Dict[str, float], key: str, fmt: str = "{:.0f} $\\pm$ {:.0f}") -> str:
    m = d.get(f"{key}_mean")
    s = d.get(f"{key}_std")
    if m is None or not isinstance(m, float):
        return "--"
    if s is None or not isinstance(s, float):
        return fmt.split(" $\\pm$ ")[0].format(m)
    return fmt.format(m, s)


def write_headline(summary, out_path: Path) -> None:
    """Table 1 — production headline (3 variants, 5-seed mean ± std)."""
    rows = []
    order = ["v10_router", "retr_top1", "great_circle"]
    for v in order:
        if v not in summary:
            continue
        d = summary[v]
        rows.append(
            f"{DISPLAY[v]} & {_msd(d, 'ade_m')} & "
            f"{_msd(d, 'fde_m')} & {_msd(d, 'cross_pct_10', '{:.2f} $\\pm$ {:.2f}')} \\\\")
    body = "\n".join(rows)
    out_path.write_text(
        r"""\begin{tabular}{lrrr}
\toprule
\textbf{Method} & \ade~(m) & \fde~(m) & cross\%@10\,km \\
\midrule
""" + body + r"""
\bottomrule
\end{tabular}
""")
    print(f"wrote {out_path}")


def write_full_ladder(summary, out_path: Path) -> None:
    """Table 6 — every variant in the unified driver, with n_seeds."""
    order = ["v4", "v5", "v9",
             "v10_raw", "v10_hardproj", "v10_router",
             "retr_top1", "great_circle"]
    rows = []
    for v in order:
        if v not in summary:
            continue
        d = summary[v]
        n_seeds = d.get("n_seeds", "")
        rows.append(
            f"{DISPLAY[v]} & {_msd(d, 'ade_m')} & "
            f"{_msd(d, 'fde_m')} & "
            f"{_msd(d, 'land_pct_10', '{:.2f} $\\pm$ {:.2f}')} & "
            f"{_msd(d, 'cross_pct_10', '{:.2f} $\\pm$ {:.2f}')} & "
            f"{n_seeds} \\\\")
    body = "\n".join(rows)
    out_path.write_text(
        r"""\begin{tabular}{lrrrrr}
\toprule
\textbf{Variant} & \ade~(m) & \fde~(m) & land\%@10\,km & cross\%@10\,km & $n_{seeds}$ \\
\midrule
""" + body + r"""
\bottomrule
\end{tabular}
""")
    print(f"wrote {out_path}")


def write_landstats(summary, out_path: Path) -> None:
    """Table 4 — land-validity diagnostics at 10 and 25 km."""
    order = ["v10_router", "v10_hardproj", "v10_raw", "v9",
             "retr_top1", "great_circle"]
    rows = []
    for v in order:
        if v not in summary:
            continue
        d = summary[v]
        rows.append(
            f"{DISPLAY[v]} & "
            f"{_msd(d, 'land_pct_10', '{:.2f} $\\pm$ {:.2f}')} & "
            f"{_msd(d, 'cross_pct_10', '{:.2f} $\\pm$ {:.2f}')} & "
            f"{_msd(d, 'land_pct_25', '{:.2f} $\\pm$ {:.2f}')} & "
            f"{_msd(d, 'cross_pct_25', '{:.2f} $\\pm$ {:.2f}')} \\\\")
    body = "\n".join(rows)
    out_path.write_text(
        r"""\begin{tabular}{lrrrr}
\toprule
 & \multicolumn{2}{c}{$\theta = 10$~km} & \multicolumn{2}{c}{$\theta = 25$~km} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}
\textbf{Method} & land\,\% & cross\,\% & land\,\% & cross\,\% \\
\midrule
""" + body + r"""
\bottomrule
\end{tabular}
""")
    print(f"wrote {out_path}")


def write_length_bucketed(summary, out_path: Path) -> None:
    """Length-bucketed ADE table (short / medium / long, with std)."""
    order = ["v5", "v9", "v10_raw", "v10_router",
             "retr_top1", "great_circle"]
    rows = []
    for v in order:
        if v not in summary:
            continue
        d = summary[v]
        rows.append(
            f"{DISPLAY[v]} & "
            f"{_msd(d, 'ade_short_m')} & "
            f"{_msd(d, 'ade_medium_m')} & "
            f"{_msd(d, 'ade_long_m')} \\\\")
    body = "\n".join(rows)
    out_path.write_text(
        r"""\begin{tabular}{lrrr}
\toprule
\textbf{Method} & \ade{} short (m) & \ade{} medium (m) & \ade{} long (m) \\
                & ($<$50~km)       & (50--500~km)      & ($>$500~km) \\
\midrule
""" + body + r"""
\bottomrule
\end{tabular}
""")
    print(f"wrote {out_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--summary_csv",
        default=str(REPO_ROOT / "results" / "headline_clean_5seed_summary.csv"))
    p.add_argument("--out_dir", default=str(REPO_ROOT / "report" / "tables"))
    args = p.parse_args()

    summary = _read_summary(Path(args.summary_csv))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    write_headline(summary,        out_dir / "headline.tex")
    write_full_ladder(summary,     out_dir / "full_ladder.tex")
    write_landstats(summary,       out_dir / "landstats.tex")
    write_length_bucketed(summary, out_dir / "length_bucketed.tex")
    print(f"\nAll tables written to {out_dir}/")


if __name__ == "__main__":
    main()
