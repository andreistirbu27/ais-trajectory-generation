#!/usr/bin/env bash
# retrain_modern_ladder.sh — train v4, v5, v9, v10 under N training seeds,
# holding the train/val split fixed at seed=42 so that all runs are
# comparable to the existing seed_variance evaluation protocol.
#
# Optimisations enabled by default:
#   --amp bf16          mixed-precision autocast on Ampere+ GPUs
#   --batch_size 256    4× the original 64 (fits in 24 GB)
#   --num_workers 4     overlap dataloading with GPU compute
#   --pin_memory        non-blocking H2D transfers
#   TF32 matmul         enabled via torch.set_float32_matmul_precision('high')
#   cudnn.benchmark     on (autotuner for fixed shapes)
#
# Each run writes to runs/trajgen_<variant>_seed<N>/ with best.pt + metrics.csv.
# Existing runs are skipped — re-running this script after an interruption
# resumes where it left off (per-variant granularity).
#
# Usage:
#   bash scripts/train/retrain_modern_ladder.sh                    # default: 3 seeds {1,2,3}, all 4 variants
#   bash scripts/train/retrain_modern_ladder.sh --seeds "1 2 3 4 5"
#   bash scripts/train/retrain_modern_ladder.sh --variants "v10"   # just v10
#   bash scripts/train/retrain_modern_ladder.sh --batch_size 128   # smaller batch if OOM
#   bash scripts/train/retrain_modern_ladder.sh --epochs 30        # shorter training
#
# Pre-flight check (~5 minutes per run) is recommended before launching the full sweep:
#   bash scripts/train/retrain_modern_ladder.sh --seeds "1" --epochs 1 --variants "v4"
#
# Estimated time per run (60 epochs, batch=256, bf16, single 3090, shared):
#   v4  ≈ 1.5 h   v5  ≈ 1.5 h   v9  ≈ 1.8 h   v10 ≈ 2.0 h
# With a dedicated 3090 those numbers roughly halve.

set -euo pipefail

# ─── Defaults ────────────────────────────────────────────────────────────
SEEDS="1 2 3"
VARIANTS="v4 v5 v9 v10"
BATCH_SIZE=256
NUM_WORKERS=4
EPOCHS=60
AMP=bf16
DRY_RUN=0
PY=${PY:-python3}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --seeds)        SEEDS="$2";        shift 2 ;;
        --variants)     VARIANTS="$2";     shift 2 ;;
        --batch_size)   BATCH_SIZE="$2";   shift 2 ;;
        --num_workers)  NUM_WORKERS="$2";  shift 2 ;;
        --epochs)       EPOCHS="$2";       shift 2 ;;
        --amp)          AMP="$2";          shift 2 ;;
        --python)       PY="$2";           shift 2 ;;
        --dry_run)      DRY_RUN=1;         shift   ;;
        -h|--help)
            sed -n '1,/^# Estimated time/p' "$0" | sed 's/^# \?//'
            exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

# ─── Constants ───────────────────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "$0")"/../.. && pwd)"
cd "$REPO_ROOT"

declare -A CONFIG=(
    [v4]="configs/trajgen/trajgen_v4.yaml"
    [v5]="configs/trajgen/trajgen_v5.yaml"
    [v9]="configs/retrieval/trajgen_v9_retrieval.yaml"
    [v10]="configs/trajgen/trajgen_v10.yaml"
)
declare -A SCRIPT=(
    [v4]="scripts/train/train_gen.py"
    [v5]="scripts/train/train_gen.py"
    [v9]="scripts/train/train_gen_retrieval.py"
    [v10]="scripts/train/train_gen_v10.py"
)
declare -A OUT_PREFIX=(
    [v4]="runs/trajgen_v4_seed"
    [v5]="runs/trajgen_v5_seed"
    [v9]="runs/trajgen_v9_retrieval_seed"
    [v10]="runs/trajgen_v10_seed"
)

# ─── Banner ──────────────────────────────────────────────────────────────
echo "=================================================================="
echo "  Modern-ladder multi-seed retraining"
echo "=================================================================="
echo "  Variants     : $VARIANTS"
echo "  Train seeds  : $SEEDS"
echo "  Split seed   : 42  (fixed — same train/val split as production)"
echo "  Batch size   : $BATCH_SIZE"
echo "  Num workers  : $NUM_WORKERS"
echo "  Epochs       : $EPOCHS"
echo "  AMP          : $AMP"
echo "  Python       : $PY"
echo "  Repo root    : $REPO_ROOT"
echo ""
$PY -c "import torch; print(f'  CUDA available: {torch.cuda.is_available()}, device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"cpu\"}')" 2>/dev/null || true
echo ""

N_TOTAL=0
for v in $VARIANTS; do for s in $SEEDS; do N_TOTAL=$((N_TOTAL+1)); done; done
echo "  Total runs to plan: $N_TOTAL"
echo ""

# ─── Run loop ────────────────────────────────────────────────────────────
RUN_IDX=0
for variant in $VARIANTS; do
    cfg="${CONFIG[$variant]:-}"
    script="${SCRIPT[$variant]:-}"
    prefix="${OUT_PREFIX[$variant]:-}"
    if [[ -z "$cfg" || -z "$script" || -z "$prefix" ]]; then
        echo "[ERROR] unknown variant: $variant" >&2
        exit 1
    fi

    for seed in $SEEDS; do
        RUN_IDX=$((RUN_IDX+1))
        out_dir="${prefix}${seed}"
        echo "──────────────────────────────────────────────────────────────────"
        echo "[$RUN_IDX/$N_TOTAL] variant=$variant  train_seed=$seed  -> $out_dir"
        echo "──────────────────────────────────────────────────────────────────"

        # Skip if already trained (presence of best.pt is the marker).
        if [[ -f "$out_dir/best.pt" ]]; then
            echo "  SKIP: $out_dir/best.pt already exists. Delete the dir to retrain."
            continue
        fi

        cmd=(
            "$PY" "$script"
            --config       "$cfg"
            --out_dir      "$out_dir"
            --train_seed   "$seed"
            --batch_size   "$BATCH_SIZE"
            --num_workers  "$NUM_WORKERS"
            --epochs       "$EPOCHS"
            --amp          "$AMP"
        )

        echo "  command: ${cmd[*]}"
        if [[ "$DRY_RUN" -eq 1 ]]; then
            echo "  (dry run — not executing)"
            continue
        fi

        log="${out_dir}.log"
        mkdir -p "$(dirname "$log")"
        echo "  log    : $log"
        echo "  start  : $(date +'%Y-%m-%d %H:%M:%S')"

        # Run; tee both to per-run log and to a master log
        set +e
        "${cmd[@]}" 2>&1 | tee "$log" | grep --line-buffered -E "^Epoch|epoch +[0-9]+ \| step|TRAJECTORY|Device|AMP|Params|train batches|^Saved|^==="
        EXIT_CODE=${PIPESTATUS[0]}
        set -e

        echo "  end    : $(date +'%Y-%m-%d %H:%M:%S')"
        if [[ $EXIT_CODE -ne 0 ]]; then
            echo "  FAILED with exit code $EXIT_CODE — aborting sweep" >&2
            echo "  (full log: $log)" >&2
            exit $EXIT_CODE
        fi
    done
done

echo ""
echo "=================================================================="
echo "  All $RUN_IDX runs complete."
echo "  Next step: scripts/train/eval_seed_variance.sh (not yet created)"
echo "  or run scripts/eval/eval_all_clean.py per checkpoint and aggregate."
echo "=================================================================="
