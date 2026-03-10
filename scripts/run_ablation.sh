#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/run_ablation.sh
#   bash scripts/run_ablation.sh --quick
#   bash scripts/run_ablation.sh --seeds 42,43,44

QUICK=0
SEEDS="42,43,44,45,46"
CUDA_ID="0"
OUT_ROOT="ablation_runs"

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
    *)
      echo "Unknown arg: $1"
      exit 1
      ;;
  esac
done

IFS=',' read -r -a SEED_ARR <<< "$SEEDS"
mkdir -p "$OUT_ROOT"

# Common training config (edit here if needed)
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

# Experiment matrix
# name|extra args
EXPERIMENTS=(
  "BASE_RGCN|--gnn_model RGCN"
  "MODEL_GCN|--gnn_model GCN"
  "MODEL_GAT|--gnn_model GAT"
  "MODEL_GRAPHCONV|--gnn_model GraphConv"
  "TRAIN_FOCAL|--gnn_model RGCN --use_focal --focal_gamma 2.0 --label_smoothing 0.0"
  "TRAIN_FIXED_THR|--gnn_model RGCN"
)

echo "[INFO] Seeds: ${SEED_ARR[*]}"
echo "[INFO] Experiments: ${#EXPERIMENTS[@]}"

for seed in "${SEED_ARR[@]}"; do
  for item in "${EXPERIMENTS[@]}"; do
    name="${item%%|*}"
    extra="${item#*|}"

    run_dir="$OUT_ROOT/${name}/seed_${seed}"
    mkdir -p "$run_dir"

    # TRAIN_FIXED_THR ablation: disable auto threshold for this branch
    this_args=("${COMMON_ARGS[@]}")
    if [[ "$name" == "TRAIN_FIXED_THR" ]]; then
      filtered=()
      skip_next=0
      for a in "${this_args[@]}"; do
        if [[ "$a" == "--auto_threshold" ]]; then
          continue
        fi
        filtered+=("$a")
      done
      this_args=("${filtered[@]}" --decision_threshold 0.5)
    fi

    log_file="$run_dir/run.log"
    cmd=(python main.py "${this_args[@]}" --seed "$seed")

    # shellcheck disable=SC2206
    extra_arr=($extra)
    cmd+=("${extra_arr[@]}")

    echo "[RUN] $name seed=$seed"
    echo "[CMD] ${cmd[*]}" | tee "$log_file"
    {
      "${cmd[@]}"
    } >> "$log_file" 2>&1 || {
      echo "[FAIL] $name seed=$seed (see $log_file)"
      continue
    }

    echo "[DONE] $name seed=$seed"
  done
done

echo "[INFO] All scheduled runs finished. Logs under: $OUT_ROOT"
