import pandas as pd
from data_pre import bigvul

def get_needed_ids(negative_ratio=5.0):
    """
    获取训练、验证、测试集所需的所有样本 ID。
    negative_ratio: 负样本与正样本的比例。
                    如果你代码里是 1:1，这里填 1.0。
                    如果你想要 1:2 (二倍负样本)，这里填 2.0。
    """
    print("正在加载并预处理数据 (这可能需要几分钟)...")
    # 1. 加载清洗后的全量数据 (包含 train/val/test 标签)
    df = bigvul()
    
    # 确保 ID 是字符串类型
    df['id'] = df['id'].astype(str)
    
    all_needed_ids = set()
    
    # 2. 模拟 GraphDataset 对每个分区的采样逻辑
    for partition in ['train', 'val', 'test']:
        part_df = df[df.label == partition].copy()
        
        # 获取正样本 (保留全部)
        vul = part_df[part_df.vul == 1]
        
        # 获取负样本 (下采样)
        # 注意：这里必须和 graph_dataset.py 里的 random_state 保持一致 (通常是 0)
        # 如果负样本数量不够采样，就取全部
        n_sample = int(len(vul) * negative_ratio)
        if n_sample > len(part_df[part_df.vul == 0]):
            n_sample = len(part_df[part_df.vul == 0])
            
        nonvul = part_df[part_df.vul == 0].sample(n_sample, random_state=0)
        
        # 收集 ID
        current_ids = set(vul.id.tolist()) | set(nonvul.id.tolist())
        all_needed_ids.update(current_ids)
        
        print(f"[{partition}] 正样本: {len(vul)}, 负样本: {len(nonvul)}, 总计: {len(current_ids)}")

    print(f"\n=== 最终统计 ===")
    print(f"原始样本总数: {len(df)}")
    print(f"实际需要切片的样本数: {len(all_needed_ids)}")
    print(f"节省计算量: {(1 - len(all_needed_ids)/len(df))*100:.2f}%")
    
    return all_needed_ids

if __name__ == "__main__":
    # 你说想要二倍负样本，就把 ratio 设为 2.0
    # 但请注意：你的 graph_dataset.py 里面目前写的是 sample(len(vul)) 即 1:1
    # 如果你想改变比例，记得同时修改 graph_dataset.py 和这里
    ids = get_needed_ids(negative_ratio=1.0) 
    
    # 保存为临时文件供切片脚本读取
    with open("target_ids.txt", "w") as f:
        for _id in ids:
            f.write(f"{_id}\n")
    print("目标 ID 已保存到 target_ids.txt")