# 优化改进报告（已落实）

本报告总结当前分支中已落地的解释器与评估优化点，便于实验复现与后续对照。

## 已落实项

1. **行号语义修复**
   - `line_index` 以源码行号为主，优先使用 `lineNumber`，并补充 `line_number` 别名。
   - 缺失或异常时回退到节点顺序索引，减少解释评估坐标系偏差。

2. **训练/测试稳定性增强**
   - 当训练未触发 best checkpoint 时，训练结束自动保存 final 模型到 `checkpoint-best-f1/model.bin`，避免测试阶段缺失模型。

3. **解释后处理（A 方案）**
   - 支持行级聚合（`sum/max`）、行去重、非法行号过滤与 Top-N 截断。

4. **解释评估稳健性增强**
   - 对无标签 key、空 GT 样本进行跳过；
   - 输出 `evaluated/skipped` 统计；
   - 兼容 legacy `line_index` 并给出告警；
   - 空样本场景下安全计算指标。

5. **解释样本筛选（C 方案）**
   - 引入 `--explain_min_prob`，按置信度过滤低质量解释样本。

6. **CFExplainer 与切片掩码融合（核心）**
   - 新增 `mask_prior_lambda`、`init_with_mask`、`mask_prior_mode`；
   - 基于 `x[:,768]` 构建边先验；
   - 先验用于初始化并加入正则损失（prior loss）。

7. **ABCD 自动化脚本（B/D）**
   - `scripts/run_abcd_optimization.sh` 已可批量扫描掩码策略、解释器参数与先验超参数组合。

8. **解释缓存隔离**
   - 新增 `explain_cache_tag` 到缓存命名，避免不同实验配置互相覆盖。

## 建议的最小对照实验

- Baseline：`cfexp_mask_prior_lambda=0.0`
- Prior：`cfexp_mask_prior_lambda=0.2`, `cfexp_mask_prior_mode=mean`, `cfexp_init_with_mask`
- Prior+：`cfexp_mask_prior_lambda=0.4`
- Prior+Sparse：`cfexp_mask_prior_lambda=0.2` + `exp_edge_thresh=0.2`

推荐同时对比行级 Precision/F1 与 PN，并补充 Top-K 命中指标用于定位能力分析。
