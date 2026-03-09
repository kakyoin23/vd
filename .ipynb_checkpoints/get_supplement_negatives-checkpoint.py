import pandas as pd
import os
from pathlib import Path

# === 配置 ===
CSV_PATH = "/root/autodl-tmp/counterfactual-vulnerability-detection-main/cfexplainer/storage/external/MSR_data_cleaned.csv"
SLICE_DIR = "/root/autodl-tmp/counterfactual-vulnerability-detection-main/cfexplainer/storage/processed/bigvul/sliced_before"
OUTPUT_FILE = "target_ids_supplement.txt"

# 目标参数
CURRENT_POS_COUNT = 2887  # 现有的正样本数
TARGET_RATIO = 5.0        # 1:2
NEEDED_NEG_TOTAL = int(CURRENT_POS_COUNT * TARGET_RATIO)

def main():
    print("1. 读取 CSV...")
    df = pd.read_csv(CSV_PATH, low_memory=False)
    
    # 统一 ID 格式
    if 'id' not in df.columns:
        if 'Unnamed: 0' in df.columns: df.rename(columns={'Unnamed: 0': 'id'}, inplace=True)
    df['id'] = df['id'].astype(str)
    
    # 2. 获取所有潜在的负样本 ID
    if 'vul' in df.columns: label_col = 'vul'
    elif 'target' in df.columns: label_col = 'target'
    else: return
    
    all_neg_ids = set(df[df[label_col] == 0]['id'].tolist())
    print(f"CSV 中负样本总池子: {len(all_neg_ids)}")

    # 3. 扫描已有的切片
    print("2. 扫描现有切片...")
    existing_files = list(Path(SLICE_DIR).glob("*.c"))
    existing_ids = set(f.stem for f in existing_files)
    
    # 找出已有的负样本
    existing_negs = all_neg_ids.intersection(existing_ids)
    current_neg_count = len(existing_negs)
    print(f"现有负样本切片: {current_neg_count}")
    
    # 4. 计算还需要多少
    shortfall = NEEDED_NEG_TOTAL - current_neg_count
    if shortfall <= 0:
        print("🎉 现有负样本已足够！无需补充。")
        return
    
    print(f"目标负样本数: {NEEDED_NEG_TOTAL} (正样本 {CURRENT_POS_COUNT} * {TARGET_RATIO})")
    print(f"需要补充: {shortfall} 个")
    
    # 为了保险，多取 10% 防止切片失败
    sample_size = int(shortfall * 1.1)
    
    # 5. 筛选出“未处理”的负样本
    # 从总负样本池中 减去 已经存在的
    candidates = list(all_neg_ids - existing_ids)
    
    if len(candidates) < sample_size:
        print("⚠️ 剩余负样本不足，全部提取。")
        selected_ids = candidates
    else:
        import random
        random.seed(42)
        selected_ids = random.sample(candidates, sample_size)
        
    print(f"本次计划提取 ID 数: {len(selected_ids)}")
    
    # 6. 写入新的白名单
    with open(OUTPUT_FILE, "w") as f:
        for uid in selected_ids:
            f.write(f"{uid}\n")
            
    print(f"✅ 补充名单已生成: {OUTPUT_FILE}")
    print("请修改 slice_process_v4_fallback.py 中的 TARGET_IDS_FILE 为这个新文件名，然后运行。")

if __name__ == "__main__":
    main()