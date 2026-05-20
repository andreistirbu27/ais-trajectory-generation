#!/usr/bin/env bash
# Launch all four Tier-1 training runs in parallel on a single GPU.
#
# Each run uses ~2-3 GB VRAM (v10 is 5.84M params with bf16 AMP), so four
# fit comfortably on a 24 GB card. Expected total wall-clock ~20-25 min
# (each run is ~18 min standalone; parallel overhead from GPU contention
# is modest at this batch/model size).
#
# Each run writes its full stdout to logs/tier1/<name>.log and stderr
# (tqdm + warnings) to logs/tier1/<name>.log.err so the main log stays
# readable.
#
# Usage:
#   bash scripts/train/run_all_tier1_parallel.sh
#   tail -f logs/tier1/seed42.log    # watch one run live
#   wait                              # if invoked interactively, this script
#                                     # already waits on all 4 before exiting.

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

launch () {
  local name=$1
  local cfg=${CONFIGS[$name]}
  local out_dir
  out_dir=$(grep -E '^out_dir:' "$cfg" | awk '{print $2}')
  if [[ -f "$out_dir/best.pt" && "${FORCE:-0}" != "1" ]]; then
    echo "[tier1] skipping $name (already trained: $out_dir/best.pt). Set FORCE=1 to retrain."
    return 0
  fi
  local log="$LOG_DIR/${name}.log"
  echo "[tier1] $(date -Is) launching $name in background (log=$log)"
  "$PY" "$TRAIN" --config "$cfg" >"$log" 2>"$log.err" &
}

echo "[tier1] $(date -Is) starting 4 parallel runs on $(nvidia-smi --query-gpu=name --format=csv,noheader)"
launch seed42
launch seed0
launch seed1
launch no_lland

echo "[tier1] PIDs: $(jobs -p | tr '\n' ' ')"
echo "[tier1] waiting for all 4 runs to finish. Tail any log with:"
echo "        tail -f logs/tier1/seed42.log"
echo "        tail -f logs/tier1/seed0.log"
echo "        tail -f logs/tier1/seed1.log"
echo "        tail -f logs/tier1/no_lland.log"
echo

wait
echo "[tier1] $(date -Is) all runs finished."
for name in seed42 seed0 seed1 no_lland; do
  cfg=${CONFIGS[$name]}
  out_dir=$(grep -E '^out_dir:' "$cfg" | awk '{print $2}')
  if [[ -f "$out_dir/best.pt" ]]; then
    echo "  [ok]   $name -> $out_dir/best.pt"
  else
    echo "  [FAIL] $name -> no best.pt produced; check $LOG_DIR/${name}.log.err"
  fi
done
