#!/usr/bin/env bash
# Tier-1b: architecture-controlled ladder retraining + v10 HP micro-sweep.
#
# Three architecture-controlled retrainings on the cleaned corpus:
#   v4_clean    — bigger model + bearing features, no scheduled sampling
#   v5_clean    — v4 + scheduled sampling
#   v9_clean    — mean-pool retrieval encoder
# All use the same 3-way split as v10 (val_frac=0.15, test_frac=0.05, seed=42)
# so any ladder comparison is on matched data.
#
# Three v10 HP micro-sweep runs on the same data:
#   lambdaland05  — lambda_land = 0.05  (5x production)
#   lambdaland1   — lambda_land = 0.1   (10x production)
#   lambdaend25   — lambda_end  = 25    (2.5x production)
#
# Each run uses ~2-3 GB VRAM (5.84M params, bf16 AMP); six fit on a 24 GB
# card. The GPU is compute-bound at this batch/model size, so wall-clock
# scales roughly with batch count rather than count of parallel jobs.
# Expected total: ~30-45 min after current Tier 1a (4 v10 runs) finishes.
#
# Pre-condition: Tier 1a must have built the KNN cache at
#   data/processed/trajgen_128_clean_knn_k5_test005.npz
# Tier 1a does this on first call. Don't run Tier 1b before Tier 1a or
# during it (the four Tier 1a runs already saturate the GPU).
#
# Usage:
#   bash scripts/train/run_all_tier1b_parallel.sh
#   FORCE=1 bash scripts/train/run_all_tier1b_parallel.sh    # retrain even if best.pt exists

set -euo pipefail

PY=${PY:-/usr/local/Anaconda3-2025.06/bin/python3}
LOG_DIR=logs/tier1b
mkdir -p "$LOG_DIR"

declare -A CONFIGS=(
  [v4_clean]=configs/trajgen/trajgen_v4_clean.yaml
  [v5_clean]=configs/trajgen/trajgen_v5_clean.yaml
  [v9_clean]=configs/retrieval/trajgen_v9_retrieval_clean.yaml
  [lambdaland05]=configs/trajgen/trajgen_v10_clean_train_lambdaland05.yaml
  [lambdaland1]=configs/trajgen/trajgen_v10_clean_train_lambdaland1.yaml
  [lambdaend25]=configs/trajgen/trajgen_v10_clean_train_lambdaend25.yaml
)

# Map each run to its training script (different model families).
declare -A SCRIPTS=(
  [v4_clean]=scripts/train/train_gen.py
  [v5_clean]=scripts/train/train_gen.py
  [v9_clean]=scripts/train/train_gen_retrieval.py
  [lambdaland05]=scripts/train/train_gen_v10.py
  [lambdaland1]=scripts/train/train_gen_v10.py
  [lambdaend25]=scripts/train/train_gen_v10.py
)

launch () {
  local name=$1
  local cfg=${CONFIGS[$name]}
  local train=${SCRIPTS[$name]}
  local out_dir
  out_dir=$(grep -E '^out_dir:' "$cfg" | awk '{print $2}')
  if [[ -f "$out_dir/best.pt" && "${FORCE:-0}" != "1" ]]; then
    echo "[tier1b] skipping $name (already trained: $out_dir/best.pt). Set FORCE=1 to retrain."
    return 0
  fi
  local log="$LOG_DIR/${name}.log"
  echo "[tier1b] $(date -Is) launching $name (script=$train, log=$log)"
  "$PY" "$train" --config "$cfg" >"$log" 2>"$log.err" &
}

echo "[tier1b] $(date -Is) starting parallel runs on $(nvidia-smi --query-gpu=name --format=csv,noheader)"
for name in v4_clean v5_clean v9_clean lambdaland05 lambdaland1 lambdaend25; do
  launch "$name"
done

echo "[tier1b] PIDs: $(jobs -p | tr '\n' ' ')"
echo "[tier1b] Tail any log with: tail -f logs/tier1b/<name>.log"
echo
wait
echo "[tier1b] $(date -Is) all runs finished."
for name in v4_clean v5_clean v9_clean lambdaland05 lambdaland1 lambdaend25; do
  cfg=${CONFIGS[$name]}
  out_dir=$(grep -E '^out_dir:' "$cfg" | awk '{print $2}')
  if [[ -f "$out_dir/best.pt" ]]; then
    echo "  [ok]   $name -> $out_dir/best.pt"
  else
    echo "  [FAIL] $name -> no best.pt produced; check $LOG_DIR/${name}.log.err"
  fi
done
