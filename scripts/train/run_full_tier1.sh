#!/usr/bin/env bash
# Single-command wrapper for the entire Tier-1 pipeline.
#
# Sequence:
#   1. Tier 1a: 4 v10 clean-corpus retrains in parallel (~90 min)
#   2. Tier 1b: 6 ladder + HP-sweep runs in parallel  (~90 min)
#   3. Eval:    unified evaluator over all 10 checkpoints
#               -> results/tier1_eval.csv             (~5-10 min)
#
# Total: ~3-3.5 h. Logs go to logs/tier1{,b}/<name>.log and the wrapper's
# own progress goes to stdout. Each stage skips work already done
# (resuming a half-completed pipeline is safe).
#
# Usage:
#   bash scripts/train/run_full_tier1.sh
#
# If you want to launch it and detach:
#   nohup bash scripts/train/run_full_tier1.sh >logs/tier1_full.log 2>&1 &
#   tail -f logs/tier1_full.log

set -euo pipefail

cd "$(dirname "$0")/../.."   # repo root

echo "=== Tier 1 full pipeline ==="
echo "  start: $(date -Is)"
echo

echo "--- Stage 1/3: Tier 1a (4 v10 variants) ---"
bash scripts/train/run_all_tier1_parallel.sh
echo

echo "--- Stage 2/3: Tier 1b (v4/v5/v9 + v10 HP sweep) ---"
bash scripts/train/run_all_tier1b_parallel.sh
echo

echo "--- Stage 3/3: unified eval ---"
PY=${PY:-/usr/local/Anaconda3-2025.06/bin/python3}
"$PY" scripts/eval/eval_all_tier1.py
echo

echo "=== Tier 1 full pipeline complete ==="
echo "  end:   $(date -Is)"
echo "  results: results/tier1_eval.csv"
