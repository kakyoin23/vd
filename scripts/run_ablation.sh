#!/usr/bin/env bash
set -euo pipefail

# 详细消融实验脚本
# - 可控参数网格：模型/损失/类别权重/平滑/阈值/温度校准
# - 每次运行写入简洁 result.log（仅最终结果）
# - 结束后自动汇总到 summary.csv
#
# 示例：
#   bash scripts/run_ablation.sh --quick
#   bash scripts/run_ablation.sh --seeds 42,43,44 --gnn-models RGCN,GAT,GraphConv
#   bash scripts/run_ablation.sh --class-weight-gammas 0.8,1.0,1.2 --label-smoothing-values 0.0,0.05
#
# 注意：默认参数组合较多，请先用 --quick 验证流程。

QUICK=0
KEEP_RAW_LOG=0
CUDA_ID="0"
OUT_ROOT="ablation_runs"
SEEDS="42,43,44,45,46"

# 可控消融参数（逗号分隔）
GNN_MODELS="RGCN,GCN,GAT,GraphConv"
LOSS_MODES="ce,focal"                 # ce|focal
CLASS_WEIGHT_GAMMAS="0.8,1.0,1.2"
LABEL_SMOOTHING_VALUES="0.0,0.05"
AUTO_THRESHOLD_MODES="on,off"         # on|off
CALIBRATE_TEMP_MODES="on,off"         # on|off

NUM_TRAIN_EPOCHS="50"
BATCH_SIZE="32"
LEARNING_RATE="5e-4"
GRAD_ACC_STEPS="2"
GRAPH_POOLING="attn"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --quick) QUICK=1; shift ;;
    --keep-raw-log) KEEP_RAW_LOG=1; shift ;;
    --cuda_id) CUDA_ID="$2"; shift 2 ;;
    --out_root) OUT_ROOT="$2"; shift 2 ;;
    --seeds) SEEDS="$2"; shift 2 ;;
    --gnn-models) GNN_MODELS="$2"; shift 2 ;;
    --loss-modes) LOSS_MODES="$2"; shift 2 ;;
    --class-weight-gammas) CLASS_WEIGHT_GAMMAS="$2"; shift 2 ;;
    --label-smoothing-values) LABEL_SMOOTHING_VALUES="$2"; shift 2 ;;
    --auto-threshold-modes) AUTO_THRESHOLD_MODES="$2"; shift 2 ;;
    --calibrate-temp-modes) CALIBRATE_TEMP_MODES="$2"; shift 2 ;;
    --num-train-epochs) NUM_TRAIN_EPOCHS="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --learning-rate) LEARNING_RATE="$2"; shift 2 ;;
    --grad-acc-steps) GRAD_ACC_STEPS="$2"; shift 2 ;;
    --graph-pooling) GRAPH_POOLING="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

if [[ "$QUICK" -eq 1 ]]; then
  echo "[INFO] QUICK mode enabled"
  SEEDS="42,43"
  GNN_MODELS="RGCN,GAT"
  LOSS_MODES="ce,focal"
  CLASS_WEIGHT_GAMMAS="1.0"
  LABEL_SMOOTHING_VALUES="0.05"
  AUTO_THRESHOLD_MODES="on"
  CALIBRATE_TEMP_MODES="off"
  NUM_TRAIN_EPOCHS="8"
  BATCH_SIZE="16"
  GRAD_ACC_STEPS="1"
  GRAPH_POOLING="mean"
fi

IFS=',' read -r -a SEED_ARR <<< "$SEEDS"
IFS=',' read -r -a GNN_ARR <<< "$GNN_MODELS"
IFS=',' read -r -a LOSS_ARR <<< "$LOSS_MODES"
IFS=',' read -r -a CWG_ARR <<< "$CLASS_WEIGHT_GAMMAS"
IFS=',' read -r -a LS_ARR <<< "$LABEL_SMOOTHING_VALUES"
IFS=',' read -r -a AUTOTHR_ARR <<< "$AUTO_THRESHOLD_MODES"
IFS=',' read -r -a CALTEMP_ARR <<< "$CALIBRATE_TEMP_MODES"

mkdir -p "$OUT_ROOT"
SUMMARY_CSV="$OUT_ROOT/summary.csv"

# 写 CSV 头
cat > "$SUMMARY_CSV" <<CSV
status,experiment_id,seed,gnn_model,loss_mode,class_weight_gamma,label_smoothing,auto_threshold,calibrate_temp,eval_acc,binary_precision,binary_recall,binary_f1,threshold,result_log
CSV

extract_metrics_to_kv() {
  local raw_log="$1"
  awk '
    /\*\*\*\*\* Test results \*\*\*\*\*/ {in_block=1; next}
    in_block && /^  / {
      gsub(/^  /, "", $0)
      split($0, a, " = ")
      if (length(a[1])>0 && length(a[2])>0) {
        printf "%s=%s\n", a[1], a[2]
      }
      next
    }
    in_block && !/^  / {in_block=0}
  ' "$raw_log"
}

get_metric_value() {
  local kv_file="$1"
  local key="$2"
  local val
  val=$(awk -F'=' -v k="$key" '$1==k{print $2}' "$kv_file" | tail -n1)
  if [[ -z "$val" ]]; then
    echo ""
  else
    echo "$val"
  fi
}

run_count=0

for seed in "${SEED_ARR[@]}"; do
  for gnn in "${GNN_ARR[@]}"; do
    for loss_mode in "${LOSS_ARR[@]}"; do
      for cwg in "${CWG_ARR[@]}"; do
        for ls in "${LS_ARR[@]}"; do
          for auto_thr in "${AUTOTHR_ARR[@]}"; do
            for cal_temp in "${CALTEMP_ARR[@]}"; do
              run_count=$((run_count + 1))

              exp_id="M${gnn}_L${loss_mode}_CW${cwg}_LS${ls}_AT${auto_thr}_CT${cal_temp}"
              run_dir="$OUT_ROOT/$exp_id/seed_${seed}"
              mkdir -p "$run_dir"

              result_log="$run_dir/result.log"
              raw_log="$run_dir/raw.log"
              kv_log="$run_dir/metrics.kv"

              cmd=(
                python main.py
                --do_train --do_test
                --seed "$seed"
                --gnn_model "$gnn"
                --num_train_epochs "$NUM_TRAIN_EPOCHS"
                --batch_size "$BATCH_SIZE"
                --learning_rate "$LEARNING_RATE"
                --gradient_accumulation_steps "$GRAD_ACC_STEPS"
                --class_weight_gamma "$cwg"
                --label_smoothing "$ls"
                --graph_pooling "$GRAPH_POOLING"
                --cuda_id "$CUDA_ID"
              )

              if [[ "$loss_mode" == "focal" ]]; then
                cmd+=(--use_focal --focal_gamma 2.0)
              fi

              if [[ "$auto_thr" == "on" ]]; then
                cmd+=(--auto_threshold)
              else
                cmd+=(--decision_threshold 0.5)
              fi

              if [[ "$cal_temp" == "on" ]]; then
                cmd+=(--calibrate_temp)
              fi

              start_ts=$(date '+%F %T')
              echo "[RUN #$run_count] $exp_id seed=$seed"

              if "${cmd[@]}" > "$raw_log" 2>&1; then
                end_ts=$(date '+%F %T')
                extract_metrics_to_kv "$raw_log" > "$kv_log" || true

                eval_acc=$(get_metric_value "$kv_log" "eval_acc")
                precision=$(get_metric_value "$kv_log" "binary_precision")
                recall=$(get_metric_value "$kv_log" "binary_recall")
                f1=$(get_metric_value "$kv_log" "binary_f1")
                threshold=$(get_metric_value "$kv_log" "threshold")

                {
                  echo "status: SUCCESS"
                  echo "experiment: $exp_id"
                  echo "seed: $seed"
                  echo "start: $start_ts"
                  echo "end: $end_ts"
                  echo "command: ${cmd[*]}"
                  echo "metrics:"
                  [[ -n "$eval_acc" ]] && echo "  eval_acc = $eval_acc"
                  [[ -n "$precision" ]] && echo "  binary_precision = $precision"
                  [[ -n "$recall" ]] && echo "  binary_recall = $recall"
                  [[ -n "$f1" ]] && echo "  binary_f1 = $f1"
                  [[ -n "$threshold" ]] && echo "  threshold = $threshold"
                } > "$result_log"

                echo "SUCCESS,$exp_id,$seed,$gnn,$loss_mode,$cwg,$ls,$auto_thr,$cal_temp,$eval_acc,$precision,$recall,$f1,$threshold,$result_log" >> "$SUMMARY_CSV"
                echo "[DONE] $exp_id seed=$seed f1=$f1"
              else
                end_ts=$(date '+%F %T')
                {
                  echo "status: FAILED"
                  echo "experiment: $exp_id"
                  echo "seed: $seed"
                  echo "start: $start_ts"
                  echo "end: $end_ts"
                  echo "command: ${cmd[*]}"
                  echo "error_tail:"
                  tail -n 80 "$raw_log" || true
                } > "$result_log"

                echo "FAILED,$exp_id,$seed,$gnn,$loss_mode,$cwg,$ls,$auto_thr,$cal_temp,,,,,,$result_log" >> "$SUMMARY_CSV"
                echo "[FAIL] $exp_id seed=$seed"
              fi

              rm -f "$kv_log"
              if [[ "$KEEP_RAW_LOG" -ne 1 ]]; then
                rm -f "$raw_log"
              fi
            done
          done
        done
      done
    done
  done
done

echo "[INFO] All runs finished."
echo "[INFO] Summary CSV: $SUMMARY_CSV"
echo "[INFO] Per-run result logs: $OUT_ROOT/*/seed_*/result.log"
