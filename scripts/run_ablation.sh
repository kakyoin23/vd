#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/run_ablation.sh
#   bash scripts/run_ablation.sh --quick
#   bash scripts/run_ablation.sh --seeds 42,43,44
#   bash scripts/run_ablation.sh --keep-raw-log

QUICK=0
SEEDS="42,43,44,45,46"
CUDA_ID="0"
OUT_ROOT="ablation_runs"
KEEP_RAW_LOG=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --quick)
      QUICK=1
      shift
      ;;
    --seeds)
      SEEDS="$2"
      shift 2
      ;;
    --cuda_id)
      CUDA_ID="$2"
      shift 2
      ;;
    --out_root)
      OUT_ROOT="$2"
      shift 2
      ;;
    --keep-raw-log)
      KEEP_RAW_LOG=1
      shift
      ;;
    *)
      echo "Unknown arg: $1"
      exit 1
      ;;
  esac
done

IFS=',' read -r -a SEED_ARR <<< "$SEEDS"
mkdir -p "$OUT_ROOT"

COMMON_ARGS=(
  --do_train --do_test
  --num_train_epochs 50
  --batch_size 32
  --learning_rate 5e-4
  --gradient_accumulation_steps 2
  --label_smoothing 0.05
  --class_weight_gamma 1.0
  --auto_threshold
  --calibrate_temp
  --graph_pooling attn
  --cuda_id "$CUDA_ID"
)

if [[ "$QUICK" -eq 1 ]]; then
  echo "[INFO] QUICK mode enabled: fewer epochs + 2 seeds"
  COMMON_ARGS=(
    --do_train --do_test
    --num_train_epochs 8
    --batch_size 16
    --learning_rate 5e-4
    --gradient_accumulation_steps 1
    --label_smoothing 0.05
    --class_weight_gamma 1.0
    --auto_threshold
    --graph_pooling mean
    --cuda_id "$CUDA_ID"
  )
  SEED_ARR=(42 43)
fi

EXPERIMENTS=(
  "BASE_RGCN|--gnn_model RGCN"
  "MODEL_GCN|--gnn_model GCN"
  "MODEL_GAT|--gnn_model GAT"
  "MODEL_GRAPHCONV|--gnn_model GraphConv"
  "TRAIN_FOCAL|--gnn_model RGCN --use_focal --focal_gamma 2.0 --label_smoothing 0.0"
  "TRAIN_FIXED_THR|--gnn_model RGCN"
)

extract_metrics() {
  local raw_log="$1"
  awk '
    /\*\*\*\*\* Test results \*\*\*\*\*/ {in_block=1; next}
    in_block && /^  / {print; next}
    in_block && !/^  / {in_block=0}
  ' "$raw_log"
}

echo "[INFO] Seeds: ${SEED_ARR[*]}"
echo "[INFO] Experiments: ${#EXPERIMENTS[@]}"

for seed in "${SEED_ARR[@]}"; do
  for item in "${EXPERIMENTS[@]}"; do
    name="${item%%|*}"
    extra="${item#*|}"

    run_dir="$OUT_ROOT/${name}/seed_${seed}"
    mkdir -p "$run_dir"

    this_args=("${COMMON_ARGS[@]}")
    if [[ "$name" == "TRAIN_FIXED_THR" ]]; then
      filtered=()
      for a in "${this_args[@]}"; do
        if [[ "$a" == "--auto_threshold" ]]; then
          continue
        fi
        filtered+=("$a")
      done
      this_args=("${filtered[@]}" --decision_threshold 0.5)
    fi

    result_log="$run_dir/result.log"
    raw_log="$run_dir/raw.log"

    cmd=(python main.py "${this_args[@]}" --seed "$seed")
    # shellcheck disable=SC2206
    extra_arr=($extra)
    cmd+=("${extra_arr[@]}")

    start_ts=$(date '+%F %T')

    echo "[RUN] $name seed=$seed"

    if "${cmd[@]}" > "$raw_log" 2>&1; then
      end_ts=$(date '+%F %T')
      {
        echo "status: SUCCESS"
        echo "experiment: $name"
        echo "seed: $seed"
        echo "start: $start_ts"
        echo "end: $end_ts"
        echo "command: ${cmd[*]}"
        echo "metrics:"
        extract_metrics "$raw_log" || true
      } > "$result_log"
      echo "[DONE] $name seed=$seed -> $result_log"
      if [[ "$KEEP_RAW_LOG" -ne 1 ]]; then
        rm -f "$raw_log"
      fi
    else
      end_ts=$(date '+%F %T')
      {
        echo "status: FAILED"
        echo "experiment: $name"
        echo "seed: $seed"
        echo "start: $start_ts"
        echo "end: $end_ts"
        echo "command: ${cmd[*]}"
        echo "error_tail:"
        tail -n 80 "$raw_log" || true
      } > "$result_log"
      echo "[FAIL] $name seed=$seed -> $result_log"
      if [[ "$KEEP_RAW_LOG" -ne 1 ]]; then
        rm -f "$raw_log"
      fi
      continue
    fi
  done
done

echo "[INFO] All scheduled runs finished. Results under: $OUT_ROOT/*/seed_*/result.log"
