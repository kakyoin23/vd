#!/usr/bin/env bash
set -euo pipefail

# ABCD optimization runner for explainer quality
# A: line post-processing handled in main.py via --exp_line_agg / --exp_top_lines
# B: explainer hyper-parameter tuning
# C: high-confidence-only explanation via --explain_min_prob
# D: slice-mask ablation via --mask_mode

GNN_MODEL=${GNN_MODEL:-GCNConv}
CKPT_DIR=${CKPT_DIR:-saved_models_gcn}
CUDA_ID=${CUDA_ID:-0}
IPT_METHOD=${IPT_METHOD:-gnnexplainer}

MASK_MODES=(aligned all_ones random)
MASK_SEEDS=(42 3407)
KMS=(4 8 12)
THRESHOLDS=(-1 0.2 0.3)
EXPLAIN_MIN_PROBS=(0.0 0.7)
LINE_AGGS=(sum max)
TOP_LINES=(3 5)
GNNE_LRS=(0.03 0.05)
GNNE_EPOCHS=(300 600)
CF_MASK_PRIOR_LAMBDAS=(0.0 0.2)
CF_MASK_PRIOR_MODES=(mean max)

for mask_mode in "${MASK_MODES[@]}"; do
  for mask_seed in "${MASK_SEEDS[@]}"; do
    # random mode uses mask_seed; other modes ignore it but keep a consistent tag
    for km in "${KMS[@]}"; do
      for th in "${THRESHOLDS[@]}"; do
        for minp in "${EXPLAIN_MIN_PROBS[@]}"; do
          for agg in "${LINE_AGGS[@]}"; do
            for topn in "${TOP_LINES[@]}"; do
              for lr in "${GNNE_LRS[@]}"; do
                for ep in "${GNNE_EPOCHS[@]}"; do
                  for cfl in "${CF_MASK_PRIOR_LAMBDAS[@]}"; do
                    for cfm in "${CF_MASK_PRIOR_MODES[@]}"; do
                      tag="abcd_${mask_mode}_s${mask_seed}_km${km}_th${th}_p${minp}_${agg}_top${topn}_lr${lr}_ep${ep}_cfl${cfl}_${cfm}"
                      echo "[RUN] ${tag}"
                      python main.py \
                        --do_test --do_explain \
                        --gnn_model "${GNN_MODEL}" \
                        --ipt_method "${IPT_METHOD}" \
                        --KM "${km}" \
                        --exp_edge_thresh "${th}" \
                        --explain_min_prob "${minp}" \
                        --exp_line_agg "${agg}" \
                        --exp_top_lines "${topn}" \
                        --gnnexplainer_lr "${lr}" \
                        --gnnexplainer_epochs "${ep}" \
                        --cfexp_mask_prior_lambda "${cfl}" \
                        --cfexp_mask_prior_mode "${cfm}" \
                        --cfexp_init_with_mask \
                        --mask_mode "${mask_mode}" \
                        --mask_seed "${mask_seed}" \
                        --model_checkpoint_dir "${CKPT_DIR}" \
                        --cuda_id "${CUDA_ID}" \
                        --overwrite_explain \
                        --explain_cache_tag "${tag}"
                    done
                  done
                done
              done
            done
          done
        done
      done
    done
  done
done
