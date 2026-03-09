import pandas as pd
import os
from pathlib import Path

# === 配置区 (已修正) ===
# 你的实际 CSV 路径
CSV_PATH = "/root/autodl-tmp/counterfactual-vulnerability-detection-main/cfexplainer/storage/external/MSR_data_cleaned.csv"

# 切片结果存放的目录 (你的 Target Dir)
SLICE_DIR = "/root/autodl-tmp/counterfactual-vulnerability-detection-main/cfexplainer/storage/processed/bigvul/sliced_before"

# 你的白名单文件 (用来对比期望与实际)
WHITELIST_FILE = "target_ids.txt"

def main():
    print(f"正在加载 CSV 数据: {CSV_PATH} ...")
    if not os.path.exists(CSV_PATH):
        print(f"❌ 错误: 找不到文件 {CSV_PATH}")
        return

    # BigVul 数据集可能有一些混合类型警告，low_memory=False 可以避免
    df = pd.read_csv(CSV_PATH, low_memory=False)
    
    # 打印一下列名，防止列名不匹配
    # print(f"CSV 列名: {df.columns.tolist()}")

    # 统一 ID 列名处理 (通常是 Unnamed: 0 或者 id)
    id_col = 'id'
    if 'id' not in df.columns:
        if 'Unnamed: 0' in df.columns:
            df.rename(columns={'Unnamed: 0': 'id'}, inplace=True)
        else:
            print("❌ 错误: CSV 中找不到 'id' 或 'Unnamed: 0' 列，请检查列名。")
            return

    # 统一 ID 类型为字符串
    df['id'] = df['id'].astype(str)
    
    # 创建映射字典: ID -> Vul Label
    # 假设标签列叫 'vul' (如果报错 KeyError: 'vul'，请改为 'target' 或实际列名)
    label_col = 'vul' 
    if label_col not in df.columns:
        if 'target' in df.columns: label_col = 'target'
        else:
            print(f"❌ 错误: 找不到标签列 ({label_col} 或 target)")
            return

    df_map = df.set_index('id')[label_col].to_dict()

    # 1. 读取白名单 (期望切片的数量)
    wanted_ids = set()
    if os.path.exists(WHITELIST_FILE):
        with open(WHITELIST_FILE, 'r') as f:
            wanted_ids = set(line.strip() for line in f if line.strip())
        print(f"原始计划切片数 (白名单): {len(wanted_ids)}")
    else:
        print("⚠️ 警告: 未找到白名单文件，统计结果可能不准确（分母未知）。")

    # 2. 读取实际切片目录 (实际成功的数量)
    print(f"正在扫描目录: {SLICE_DIR} ...")
    if not os.path.exists(SLICE_DIR):
        print(f"❌ 错误: 切片目录不存在 {SLICE_DIR}")
        return

    actual_files = list(Path(SLICE_DIR).glob("*.c"))
    actual_ids = set(f.stem for f in actual_files) # 获取文件名作为 ID
    
    print(f"实际生成切片数: {len(actual_ids)}")

    # 3. 统计逻辑
    # 实际成功的样本里，有多少是在我们白名单计划内的
    valid_actual_ids = actual_ids.intersection(wanted_ids) if wanted_ids else actual_ids
    
    pos_count = 0
    neg_count = 0
    unknown_count = 0
    
    for uid in valid_actual_ids:
        if uid in df_map:
            label = df_map[uid]
            if label == 1:
                pos_count += 1
            else:
                neg_count += 1
        else:
            unknown_count += 1

    # 4. 计算原来的正样本总数 (在白名单里的)
    wanted_pos = 0
    if wanted_ids:
        for uid in wanted_ids:
            if uid in df_map and df_map[uid] == 1:
                wanted_pos += 1
    
    # === 输出报告 ===
    print(f"\n{'='*40}")
    print(f"📊 切片结果统计报告")
    print(f"{'='*40}")
    
    if wanted_ids:
        success_rate = len(valid_actual_ids) / len(wanted_ids) * 100
        print(f"总体成功率: {len(valid_actual_ids)} / {len(wanted_ids)} = {success_rate:.2f}%")
    
    if wanted_pos > 0:
        pos_rate = pos_count / wanted_pos * 100
        print(f"\n[正样本 - Vulnerable]")
        print(f"  - 计划: {wanted_pos}")
        print(f"  - 实际: {pos_count}")
        print(f"  - 存活率: {pos_rate:.2f}%")
        
        if pos_rate < 30:
            print("  ⚠️ 警告: 正样本丢失过多，可能需要检查切片逻辑 (Joern解析失败或无数据流)。")
    else:
        print(f"\n[正样本]: 实际 {pos_count} (白名单中未统计到正样本)")

    print(f"\n[负样本 - Non-Vulnerable]")
    print(f"  - 实际: {neg_count}")
    
    print(f"\n[最终数据集比例]")
    if pos_count > 0:
        ratio = neg_count / pos_count
        print(f"  - 正负比例: 1 : {ratio:.2f}")
    else:
        print(f"  - 正负比例: 无法计算 (无正样本)")

    if unknown_count > 0:
        print(f"\n⚠️ 有 {unknown_count} 个切片文件名在 CSV ID 中找不到对应记录。")

if __name__ == "__main__":
    main()