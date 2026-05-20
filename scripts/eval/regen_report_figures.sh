#!/usr/bin/env bash
# regen_report_figures.sh — one-shot regenerator for every figure
# and every LaTeX-table fragment referenced by report/report_v2.tex.
# Assumes that scripts/eval/eval_all_clean.py has already produced
# results/headline_clean_5seed.csv and *_summary.csv.

set -e

PY=/usr/local/Anaconda3-2025.06/bin/python
cd "$(dirname "$0")"/../..

echo "── LaTeX table fragments ─────────────────────────────"
$PY scripts/eval/print_report_tables.py
echo ""

echo "── Figures (CSV-driven) ──────────────────────────────"
$PY scripts/eval/plot_comparison.py --out_dir figures
echo ""

echo "── Production seed-variance figure ───────────────────"
$PY scripts/eval/multi_seed_variance.py \
    --out_csv results/seed_variance.csv \
    --out_png figures/seed_variance.png \
    > /tmp/seed_variance.log 2>&1
echo "    figures/seed_variance.png updated (log: /tmp/seed_variance.log)"
echo ""

echo "── Qualitative grid ──────────────────────────────────"
$PY scripts/eval/regen_viz_v10_clean.py \
    --out figures/viz_v10_clean.png \
    --mirror_to report/figures/viz_v10_clean.png \
    > /tmp/viz_v10_clean.log 2>&1
echo "    figures/viz_v10_clean.png updated (log: /tmp/viz_v10_clean.log)"
echo ""

echo "── Motivation / data figures (unchanged inputs, refresh) ─"
$PY scripts/eval/regen_report_data_figs.py \
    --dirty data/processed/trajgen_128.npz \
    --clean data/processed/trajgen_128_clean.npz \
    --out_dir report/figures \
    > /tmp/regen_data_figs.log 2>&1
echo "    report/figures/data_*.png refreshed"
echo ""

echo "── Mirror figures/ → report/figures/ ─────────────────"
for f in ade_by_model length_bucketed_ade v9_v10_v12_training_curves \
         seed_variance viz_v10_clean; do
    cp -v "figures/${f}.png" "report/figures/${f}.png"
done

echo ""
echo "── PDF build ─────────────────────────────────────────"
cd report
pdflatex -interaction=nonstopmode report_v2.tex > /tmp/pdflatex_pass1.log 2>&1
pdflatex -interaction=nonstopmode report_v2.tex > /tmp/pdflatex_pass2.log 2>&1
echo ""
echo "── Build warnings / errors ──────────────────────────"
grep -E "Warning|Error|undefined|Missing" /tmp/pdflatex_pass2.log | head -30 || true
echo ""
ls -la report_v2.pdf
