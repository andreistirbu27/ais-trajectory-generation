#!/usr/bin/env bash
# Tier-1 training runs for the report_v2 revision (see
# .claude/plans/i-looked-at-report-v2-dapper-garden.md).
#
# Four v10 training runs on the CLEANED corpus (trajgen_128_clean.npz):
#   1. seed42        — replaces the leakage-affected production checkpoint
#   2. seed0, seed1  — training-seed variance (3 seeds total with seed42)
#   3. no_lland      — L_land = 0 ablation (does the land-aware loss matter?)
#
# Wall-clock: ~1.5-3 h per run on the RTX 3090 (24 GB) with bf16 AMP.
# Clean corpus is ~52k tracks (vs ~175k dirty), 816 steps/epoch x 60 epochs
# = ~49k steps total. v10 (5.84M params) fits in <2 GB activations, so all
# four runs can sit in the same 24 GB card -- launch each in a separate
# terminal to run them in parallel and finish all four in ~1.5-3 h.
# Sanity-check with seed42 first if unsure.
#
# Usage:
#   bash scripts/train/run_all_tier1.sh            # serial, all four
#   bash scripts/train/run_all_tier1.sh seed42     # just one
#   bash scripts/train/run_all_tier1.sh no_lland   # just the ablation
#
# Logs go to logs/tier1/<run_name>.log. Checkpoints to runs/<run_name>/best.pt.

set -euo pipefail

PY=${PY:-/usr/local/Anaconda3-2025.06/bin/python3}
TRAIN=scripts/train/train_gen_v10.py
LOG_DIR=logs/tier1
mkdir -p "$LOG_DIR"

declare -A CONFIGS=(
  [seed42]=configs/trajgen/trajgen_v10_clean_train_seed42.yaml
  [seed0]=configs/trajgen/trajgen_v10_clean_train_seed0.yaml
  [seed1]=configs/trajgen/trajgen_v10_clean_train_seed1.yaml
  [no_lland]=configs/trajgen/trajgen_v10_clean_train_no_lland.yaml
)

run_one () {
  local name=$1
  local cfg=${CONFIGS[$name]:-}
  if [[ -z "$cfg" ]]; then
    echo "Unknown run: $name (choose: ${!CONFIGS[*]})" >&2
    exit 2
  fi
  local out_dir
  out_dir=$(grep -E '^out_dir:' "$cfg" | awk '{print $2}')
  if [[ -f "$out_dir/best.pt" && "${FORCE:-0}" != "1" ]]; then
    echo "[tier1] $(date -Is) skipping $name (already trained: $out_dir/best.pt). Set FORCE=1 to retrain."
    return 0
  fi
  local log="$LOG_DIR/${name}.log"
  echo "[tier1] $(date -Is) starting $name (cfg=$cfg, log=$log)"
  # stderr (tqdm + warnings) -> $log.err so it doesn't spam the main log
  # with carriage-return progress bars; stdout (epoch summaries) -> tee
  "$PY" "$TRAIN" --config "$cfg" 2>"$log.err" | tee "$log"
  echo "[tier1] $(date -Is) finished $name"
}

if [[ $# -eq 0 ]]; then
  for name in seed42 seed0 seed1 no_lland; do
    run_one "$name"
  done
else
  for name in "$@"; do
    run_one "$name"
  done
fi
