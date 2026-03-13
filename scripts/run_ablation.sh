#!/usr/bin/env bash
set -euo pipefail

# 轻量且聚焦的消融脚本：
# 1) 切片掩码作用（mask_mode）
# 2) 异构边识别作用（edge_mode: heter=RGCN, homo=GCN）
#
# 输出：
# - 每次实验 result.log（只保留最终结果）
# - 全局 summary.csv

QUICK=0
KEEP_RAW_LOG=0
CUDA_ID="0"
OUT_ROOT="ablation_runs"
SEEDS="42,43,44"

# 核心控制点（默认仅围绕这两点）
MASK_MODES="aligned,all_ones,random"
EDGE_MODES="heter,homo"   # heter->RGCN(使用边类型), homo->GCN(忽略边类型)

# 其余参数默认固定，避免冗余
LOSS_MODES="ce"            # ce|focal
CLASS_WEIGHT_GAMMAS="1.0"
LABEL_SMOOTHING_VALUES="0.05"
AUTO_THRESHOLD_MODES="on"
CALIBRATE_TEMP_MODES="off"

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
    --mask-modes) MASK_MODES="$2"; shift 2 ;;
    --edge-modes) EDGE_MODES="$2"; shift 2 ;;
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
  MASK_MODES="aligned,all_ones"
  EDGE_MODES="heter,homo"
  LOSS_MODES="ce"
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
IFS=',' read -r -a MASKMODE_ARR <<< "$MASK_MODES"
IFS=',' read -r -a EDGEMODE_ARR <<< "$EDGE_MODES"
IFS=',' read -r -a LOSS_ARR <<< "$LOSS_MODES"
IFS=',' read -r -a CWG_ARR <<< "$CLASS_WEIGHT_GAMMAS"
IFS=',' read -r -a LS_ARR <<< "$LABEL_SMOOTHING_VALUES"
IFS=',' read -r -a AUTOTHR_ARR <<< "$AUTO_THRESHOLD_MODES"
IFS=',' read -r -a CALTEMP_ARR <<< "$CALIBRATE_TEMP_MODES"

mkdir -p "$OUT_ROOT"
SUMMARY_CSV="$OUT_ROOT/summary.csv"

cat > "$SUMMARY_CSV" <<CSV
status,experiment_id,seed,mask_mode,edge_mode,gnn_model,loss_mode,class_weight_gamma,label_smoothing,auto_threshold,calibrate_temp,eval_acc,binary_precision,binary_recall,binary_f1,threshold,result_log
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
  echo "$val"
}

map_edge_mode_to_gnn() {
  local edge_mode="$1"
  case "$edge_mode" in
    heter) echo "RGCN" ;;
    homo) echo "GCN" ;;
    *) echo "" ;;
  esac
}

total_runs=$(( ${#SEED_ARR[@]} * ${#MASKMODE_ARR[@]} * ${#EDGEMODE_ARR[@]} * ${#LOSS_ARR[@]} * ${#CWG_ARR[@]} * ${#LS_ARR[@]} * ${#AUTOTHR_ARR[@]} * ${#CALTEMP_ARR[@]} ))
echo "[INFO] Planned runs: $total_runs"

run_count=0
for seed in "${SEED_ARR[@]}"; do
  for mask_mode in "${MASKMODE_ARR[@]}"; do
    for edge_mode in "${EDGEMODE_ARR[@]}"; do
      gnn=$(map_edge_mode_to_gnn "$edge_mode")
      if [[ -z "$gnn" ]]; then
        echo "[WARN] skip unknown edge_mode=$edge_mode"
        continue
      fi

      for loss_mode in "${LOSS_ARR[@]}"; do
        for cwg in "${CWG_ARR[@]}"; do
          for ls in "${LS_ARR[@]}"; do
            for auto_thr in "${AUTOTHR_ARR[@]}"; do
              for cal_temp in "${CALTEMP_ARR[@]}"; do
                run_count=$((run_count + 1))

                exp_id="MM${mask_mode}_E${edge_mode}_L${loss_mode}_CW${cwg}_LS${ls}_AT${auto_thr}_CT${cal_temp}"
                run_dir="$OUT_ROOT/$exp_id/seed_${seed}"
                mkdir -p "$run_dir"

                result_log="$run_dir/result.log"
                raw_log="$run_dir/raw.log"
                kv_log="$run_dir/metrics.kv"

                cmd=(
                  python main.py
                  --do_train --do_test
                  --seed "$seed"
                  --mask_mode "$mask_mode"
                  --mask_seed "$seed"
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

                echo "[RUN #$run_count/$total_runs] $exp_id seed=$seed gnn=$gnn"
                start_ts=$(date '+%F %T')

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
                    echo "mask_mode: $mask_mode"
                    echo "edge_mode: $edge_mode"
                    echo "gnn_model: $gnn"
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

                  echo "SUCCESS,$exp_id,$seed,$mask_mode,$edge_mode,$gnn,$loss_mode,$cwg,$ls,$auto_thr,$cal_temp,$eval_acc,$precision,$recall,$f1,$threshold,$result_log" >> "$SUMMARY_CSV"
                  echo "[DONE] $exp_id seed=$seed f1=$f1"
                else
                  end_ts=$(date '+%F %T')
                  {
                    echo "status: FAILED"
                    echo "experiment: $exp_id"
                    echo "seed: $seed"
                    echo "mask_mode: $mask_mode"
                    echo "edge_mode: $edge_mode"
                    echo "gnn_model: $gnn"
                    echo "start: $start_ts"
                    echo "end: $end_ts"
                    echo "command: ${cmd[*]}"
                    echo "error_tail:"
                    tail -n 80 "$raw_log" || true
                  } > "$result_log"

                  echo "FAILED,$exp_id,$seed,$mask_mode,$edge_mode,$gnn,$loss_mode,$cwg,$ls,$auto_thr,$cal_temp,,,,,,$result_log" >> "$SUMMARY_CSV"
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
done

echo "[INFO] All runs finished."
echo "[INFO] Summary CSV: $SUMMARY_CSV"
echo "[INFO] Per-run result logs: $OUT_ROOT/*/seed_*/result.log"
