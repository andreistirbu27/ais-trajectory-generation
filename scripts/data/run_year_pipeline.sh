#!/usr/bin/env bash
# Full-year AIS pipeline: download + filter + merge, quarter by quarter.
# Processes 3 months at a time to stay within ~16 GB RAM.
# Each quarter has 1-day overlap with the previous to preserve cross-boundary vessel tracks.
# Batch IDs are globally unique across quarters (no segment ID collisions in the final merge).
#
# Usage:
#   bash scripts/data/run_year_pipeline.sh
#
# To resume after an interruption: just run again — already-processed batches are skipped.
# To add --delete_raw (free disk space after each batch): add it to each run_pipeline call.

set -e  # stop immediately if any command fails

echo "============================================================"
echo "  AIS Full-Year Pipeline — 2024"
echo "  Quarters: Q1 (Jan–Mar), Q2 (Apr–Jun), Q3 (Jul–Sep), Q4 (Oct–Dec)"
echo "  Each quarter overlaps 1 day with the previous"
echo "============================================================"

# Q1: Jan 1 – Mar 31 (batches 0–12)
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Q1: 2024-01-01 → 2024-03-31  (batch IDs b000–b012)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python3 scripts/data/run_pipeline.py \
    --start 2024-01-01 --end 2024-03-31 \
    --proc_dir data/processed/batches/q1 \
    --out data/processed/AIS_2024_Q1.csv \
    --batch_start_idx 0 \
    --delete_raw

# Q2: Mar 31 – Jun 30 (1-day overlap with Q1; batches 13–25)
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Q2: 2024-03-31 → 2024-06-30  (batch IDs b020–b034)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python3 scripts/data/run_pipeline.py \
    --start 2024-03-31 --end 2024-06-30 \
    --proc_dir data/processed/batches/q2 \
    --out data/processed/AIS_2024_Q2.csv \
    --batch_start_idx 20 \
    --delete_raw

# Q3: Jun 30 – Sep 30 (1-day overlap with Q2; batches 26–38)
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Q3: 2024-06-30 → 2024-09-30  (batch IDs b040–b055)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python3 scripts/data/run_pipeline.py \
    --start 2024-06-30 --end 2024-09-30 \
    --proc_dir data/processed/batches/q3 \
    --out data/processed/AIS_2024_Q3.csv \
    --batch_start_idx 40 \
    --delete_raw

# Q4: Sep 30 – Dec 31 (1-day overlap with Q3; batches 39–51)
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Q4: 2024-09-30 → 2024-12-31  (batch IDs b060–b075)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python3 scripts/data/run_pipeline.py \
    --start 2024-09-30 --end 2024-12-31 \
    --proc_dir data/processed/batches/q4 \
    --out data/processed/AIS_2024_Q4.csv \
    --batch_start_idx 60 \
    --delete_raw

# Final merge
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Final merge: Q1 + Q2 + Q3 + Q4 → AIS_2024_full.csv"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python3 scripts/data/merge_quarters.py \
    --inputs data/processed/AIS_2024_Q1.csv \
             data/processed/AIS_2024_Q2.csv \
             data/processed/AIS_2024_Q3.csv \
             data/processed/AIS_2024_Q4.csv \
    --out data/processed/AIS_2024_full.csv

echo ""
echo "============================================================"
echo "  Pipeline complete → data/processed/AIS_2024_full.csv"
echo "============================================================"
