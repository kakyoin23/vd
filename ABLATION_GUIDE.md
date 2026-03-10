# 完整消融实验指南

本指南配合以下文件使用：
- `ablation_plan_template.csv`：实验清单模板（可跟踪状态）
- `scripts/run_ablation.sh`：自动化运行脚本

## 1. 推荐分组

1) 数据层消融
- aligned mask（默认）
- no-mask（全 1）
- random-mask（随机）

2) 模型层消融
- `RGCN`（baseline）
- `GCN`
- `GAT`
- `GraphConv`

3) 训练策略消融
- CE vs Focal
- 自动阈值 vs 固定阈值 0.5
- 温度校准 on/off

4) 稳健性
- 多随机种子（建议 5 个）
- 报告均值±标准差

## 2. 快速运行

```bash
bash scripts/run_ablation.sh --quick --cuda_id 0
```

该模式用于确认流程能跑通，不用于最终论文结论。

## 3. 完整运行

```bash
bash scripts/run_ablation.sh --seeds 42,43,44,45,46 --cuda_id 0 --out_root ablation_runs
```

输出日志目录：
- `ablation_runs/<实验名>/seed_<seed>/run.log`

## 4. 数据层消融实现建议

当前脚本默认不改数据处理逻辑。建议这样做：

- no-mask：在 `graph_dataset.py` 中将 `x_mask` 强制为全 1 后重新处理数据。
- random-mask：在 `graph_dataset.py` 中将 `x_mask` 置为 Bernoulli 随机向量后重新处理数据。

注意：每种数据设定需要独立缓存目录，避免互相覆盖。

## 5. 结果汇总模板

建议从 `run.log` 中提取测试指标并形成表格：

- 主指标：`binary_f1`
- 次指标：`binary_precision`、`binary_recall`、`eval_acc`、`threshold`

每个实验报告：
- mean ± std（跨 seed）
- 相对 baseline 的 ΔF1

## 6. 复现清单

每次实验至少记录：
- git commit hash
- 完整命令行
- seed
- 数据设定（aligned/no-mask/random-mask）
- 最终测试指标
