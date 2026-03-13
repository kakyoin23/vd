# 完整消融实验指南

本指南配合以下文件使用：
- `ablation_plan_template.csv`：实验清单模板（可跟踪状态）
- `scripts/run_ablation.sh`：自动化运行脚本（支持参数网格）

## 1. 推荐分组

1) 数据层消融
- aligned mask（默认）
- no-mask（全 1）
- random-mask（随机）

2) 异构边识别消融（edge mode）
- `heter`: 使用 `RGCN`（利用边类型）
- `homo`: 使用 `GCN`（不利用边类型）

3) 训练策略（默认固定，避免冗余）
- 默认固定 CE + gamma=1.0 + label_smoothing=0.05 + auto_threshold=on + calibrate_temp=off
- 如需扩展可手动开启 loss/gamma/smoothing/threshold/calibration 网格

4) 稳健性
- 多随机种子（建议 5 个）
- 报告均值±标准差

## 2. 快速运行（推荐先试）

```bash
bash scripts/run_ablation.sh --quick --cuda_id 0
```

该模式使用较小实验组合验证流程是否正常。

## 3. 完整运行

```bash
bash scripts/run_ablation.sh \
  --seeds 42,43,44,45,46 \
  --mask-modes aligned,all_ones,random \
  --edge-modes heter,homo \
  --cuda_id 0 \
  --out_root ablation_runs
```

## 4. 结果日志与汇总

输出包含两类：

1) 每次实验的结果日志（仅最终结果）：
- `ablation_runs/<experiment_id>/seed_<seed>/result.log`

2) 全局汇总 CSV：
- `ablation_runs/summary.csv`

CSV 字段包括：
- `status, experiment_id, seed, mask_mode, edge_mode, gnn_model, loss_mode, class_weight_gamma, label_smoothing, auto_threshold, calibrate_temp`
- `eval_acc, binary_precision, binary_recall, binary_f1, threshold`
- `result_log`

> 默认会删除完整过程日志（`raw.log`），保持目录简洁。
> 如果你需要排错可加 `--keep-raw-log`。

## 5. 常用控制参数

```bash
--seeds 42,43,44
--mask-modes aligned,all_ones,random
--edge-modes heter,homo
--loss-modes ce,focal              # 可选扩展
--class-weight-gammas 1.0,1.2      # 可选扩展
--label-smoothing-values 0.0,0.05  # 可选扩展
--auto-threshold-modes on,off      # 可选扩展
--calibrate-temp-modes on,off      # 可选扩展
--num-train-epochs 50
--batch-size 32
--learning-rate 5e-4
--grad-acc-steps 2
--graph-pooling attn
```

## 6. 数据层消融实现建议

当前脚本默认不改数据处理逻辑。建议这样做：

- no-mask：在 `graph_dataset.py` 中将 `x_mask` 强制为全 1 后重新处理数据。
- random-mask：在 `graph_dataset.py` 中将 `x_mask` 置为 Bernoulli 随机向量后重新处理数据。

注意：每种数据设定需要独立缓存目录，避免互相覆盖。

## 7. 复现清单

每次实验至少记录：
- git commit hash
- 完整命令行
- seed
- 数据设定（aligned/all_ones/random）
- 边模式（heter/homo）
- 最终测试指标（F1/Precision/Recall/Acc/Threshold）
