import sys, json, os
import os.path as osp
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union
import pickle as pkl
from pathlib import Path
from glob import glob
import shutil
import ast
import re

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Dataset, Data, Batch
from tqdm.std import trange
from transformers import (BertConfig, BertForMaskedLM, BertTokenizer,
                          GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
                          OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
                          RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer,
                          DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer,
                          T5Config, T5ForConditionalGeneration, T5Tokenizer)

from helpers import utils
from data_pre import bigvul

# ==========================================
# [SR-GVD 配置] 路径设置
# ==========================================

# 1. 切片源代码目录 (用于白名单筛选 + 掩码生成)
try:
    SLICE_SOURCE_DIR = utils.processed_dir() / "bigvul/sliced_before"
except:
    SLICE_SOURCE_DIR = Path("/root/autodl-tmp/counterfactual-vulnerability-detection-main/cfexplainer/storage/processed/bigvul/sliced_before")

# 2. 全量图数据目录 (用于构建图结构 + 语义提取)
try:
    DATA_FULL_DIR = utils.processed_dir() / "bigvul/before"
except:
    DATA_FULL_DIR = Path("/root/autodl-tmp/counterfactual-vulnerability-detection-main/cfexplainer/storage/processed/bigvul/before")


# ==========================================
# [SR-GVD 核心算法] 文本对齐工具函数
# ==========================================
def normalize_code(code_str):
    """
    标准化代码字符串：去除首尾空格，用于忽略缩进差异进行匹配
    """
    if not isinstance(code_str, str): return ""
    # 去除多余空白符，只保留单空格间隔，忽略缩进影响
    return " ".join(code_str.split())

def get_slice_mask_by_alignment(full_nodes, slice_file_path):
    """
    通过贪婪子序列匹配算法，计算哪些原始行号在切片中被保留了。
    这是解决“切片行号漂移”问题的关键算法，无需重跑切片。
    
    Args:
        full_nodes: DataFrame, 包含全量图的节点信息
        slice_file_path: 切片 .c 文件的路径
    
    Returns:
        dict: {original_line_number: 1.0 (保留) or 0.0 (切除)}
    """
    if not slice_file_path.exists():
        return {}

    # 1. 读取切片文件的所有有效代码行 (作为“目标序列”)
    try:
        with open(slice_file_path, 'r', errors='ignore') as f:
            # 过滤掉空行，并标准化
            slice_lines = [normalize_code(line) for line in f if normalize_code(line)]
    except:
        return {}

    if not slice_lines:
        return {}

    # 2. 提取全量代码行 (按原始行号排序)
    # 这一步是为了重建“原始文件”的逻辑顺序
    if 'lineNumber' not in full_nodes.columns or 'code' not in full_nodes.columns:
        return {}
        
    valid_nodes = full_nodes[full_nodes['lineNumber'] > 0].sort_values('lineNumber')
    if valid_nodes.empty:
        return {}

    # 去重：同一行可能有多个AST节点，只取一个代表即可
    code_map = valid_nodes.groupby('lineNumber')['code'].apply(lambda x: x.iloc[0]).to_dict()
    sorted_line_nums = sorted(code_map.keys())
    
    # 3. 双指针贪婪匹配 (Greedy Subsequence Matching)
    mask_map = {}
    slice_idx = 0
    total_slice = len(slice_lines)
    
    for line_num in sorted_line_nums:
        original_code = normalize_code(code_map[line_num])
        
        if not original_code:
            mask_map[line_num] = 0.0
            continue
            
        # 尝试匹配切片中的当前行
        if slice_idx < total_slice:
            target_code = slice_lines[slice_idx]
            
            # === 核心匹配逻辑 ===
            # 如果内容相同，或者是包含关系 (应对部分切片截断情况)
            # 例如: 切片里是 "if(a){", 原文是 "if (a) {"
            if original_code == target_code or (len(target_code) > 4 and target_code in original_code):
                # 匹配成功！这行在切片里
                mask_map[line_num] = 1.0
                slice_idx += 1 # 切片指针下移
            else:
                # 没匹配上，说明这行被切掉了（噪声）
                mask_map[line_num] = 0.0
        else:
            # 切片已经匹配完了，剩下的都是被切掉的
            mask_map[line_num] = 0.0
            
    return mask_map


class VulGraphDataset(Dataset):
    def __init__(self, root: Optional[str] = "storage/processed/vul_graph_dataset", 
                 transform: Optional[Callable] = None, pre_transform: Optional[Callable] = None, pre_filter: Optional[Callable] = None, log: bool = True, 
                 encoder = None, tokenizer = None, partition = None,
                 vulonly = False, sample = -1, splits = "default",
                 debug: bool = False,
                 clear_cache: bool = False,
                 mask_mode: str = "aligned",
                 mask_seed: int = 42
                 ):
        os.makedirs(root, exist_ok=True)
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.partition = partition
        self.vulonly = vulonly
        self.sample = sample
        self.splits = splits
        self.debug = debug
        self.id2filename = {}
        self.mask_mode = mask_mode
        self.mask_seed = mask_seed
        valid_modes = {"aligned", "all_ones", "random"}
        if self.mask_mode not in valid_modes:
            raise ValueError(f"Unsupported mask_mode={self.mask_mode}. choose from {sorted(valid_modes)}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if self.encoder:
            self.encoder.to(self.device)
            self.encoder.eval()
            # 预加载 embedding 权重，备用
            if hasattr(self.encoder, 'roberta'):
                self.word_embeddings = self.encoder.roberta.embeddings.word_embeddings.weight.detach().cpu().numpy()
            elif hasattr(self.encoder, 'bert'):
                self.word_embeddings = self.encoder.bert.embeddings.word_embeddings.weight.detach().cpu().numpy()
            else:
                self.word_embeddings = None
        
        if clear_cache:
            cache_dir = utils.get_dir(utils.cache_dir() / "vul_graph_feat_target")
            if cache_dir.exists():
                print(f"Clearing cache directory: {cache_dir}")
                shutil.rmtree(cache_dir)
                cache_dir.mkdir(parents=True, exist_ok=True)
        
        super().__init__(root, transform, pre_transform, pre_filter, log)

        if not osp.exists(self.processed_paths[0]) and (self.encoder is None or self.tokenizer is None):
            raise ValueError(
                f"Processed dataset not found at {self.processed_paths[0]}, and encoder/tokenizer is missing. "
                "Provide encoder+tokenizer for first-time dataset processing."
            )

        if osp.exists(self.processed_paths[0]):
            self.data_list = torch.load(self.processed_paths[0])
        else:
            self.data_list = []
            
    @property
    def processed_dir(self) -> str:
        suffix = '' if self.mask_mode == 'aligned' else f'_{self.mask_mode}'
        return osp.join(self.root, f'{self.partition}_processed_target{suffix}')
    
    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return 'data.pt'
    
    def process(self):
        # 1. 扫描切片源代码 (建立白名单)
        if not SLICE_SOURCE_DIR.exists():
             print(f"⚠️ Warning: Slice Directory not found: {SLICE_SOURCE_DIR}")
             os.makedirs(SLICE_SOURCE_DIR, exist_ok=True)

        print(f"1. Scanning Slice Source Code (Filter & Mask Source): {SLICE_SOURCE_DIR}")
        slice_files = list(SLICE_SOURCE_DIR.glob("*.c"))
        
        whitelist_ids = set()
        for f in slice_files:
            fid = f.name.replace('.c', '')
            if not fid.isdigit():
                match = re.match(r"(\d+)", fid)
                if match: fid = match.group(1)
            
            if fid and fid.isdigit():
                whitelist_ids.add(fid)
        
        print(f"   -> Found {len(whitelist_ids)} valid sliced samples.")
        
        # 2. 扫描全量图数据
        if not DATA_FULL_DIR.exists():
             raise FileNotFoundError(f"Full Graph Directory not found: {DATA_FULL_DIR}")

        print(f"2. Scanning Full Graph Directory (Structure Source): {DATA_FULL_DIR}")
        full_files = list(DATA_FULL_DIR.glob("**/*.nodes.json"))
        
        self.id2filename = {}
        self.finished = []
        
        for f in full_files:
            fid = f.name.split('.')[0]
            if not fid.isdigit():
                match = re.match(r"(\d+)", fid)
                if match: fid = match.group(1)
            
            # 只有在切片白名单中的样本才会被处理
            if fid in whitelist_ids:
                try:
                    rel_path = f.relative_to(DATA_FULL_DIR)
                except ValueError:
                    rel_path = f.name
                self.id2filename[fid] = rel_path
                self.finished.append(fid)
        
        print(f"   -> [Hybrid Strategy] Matched {len(self.finished)} samples ready for processing.")
        
        if len(self.finished) == 0:
            print("❌ Error: No intersection found! Check IDs.")
            return

        # 3. 读取 BigVul 标签数据
        self.df = bigvul(splits=self.splits)
        self.df['id'] = self.df['id'].astype(str)
        self.df = self.df[self.df.label == self.partition]
        self.df = self.df[self.df.id.isin(self.finished)]

        vul_df = self.df[self.df.vul == 1]
        nonvul_df = self.df[self.df.vul == 0]
        
        print(f"\n[Data Distribution] Vul: {len(vul_df)}, Non-Vul: {len(nonvul_df)}")
        
        self.df = pd.concat([vul_df, nonvul_df])
        self.df = self.df.sample(frac=1, random_state=42).reset_index(drop=True)

        if self.sample > 0: self.df = self.df.sample(self.sample, random_state=0)
        if self.vulonly: self.df = self.df[self.df.vul == 1]

        self.df = self.df.reset_index(drop=True).reset_index().rename(columns={"index": "idx"})
        self.idx2id = pd.Series(self.df.id.values, index=self.df.idx).to_dict()

        data_list = []
        stats = {"vul": 0, "non": 0, "dropped": 0}

        print(f"[Processing] Extracting Features with Slice Masking...")
        for idx in trange(self.df.shape[0]):
            _id = str(self.idx2id[idx])
            row = self.df.iloc[idx]
            true_label = int(row.vul)

            try:
                # 调用特征提取
                n, e, edge_types = self.feature_extraction(_id)

                if n is None or e is None or len(n) == 0: 
                    stats["dropped"] += 1
                    continue
                if 'subseq_feat' not in n.columns: 
                    stats["dropped"] += 1
                    continue

                # --- 核心修改：特征拼接 ---
                valid_feats = []
                feat_dim = self.encoder.config.hidden_size if hasattr(self.encoder.config, 'hidden_size') else 768
                
                # 1. 获取语义特征 (768维)
                for val in n['subseq_feat'].values:
                    if isinstance(val, (np.ndarray, list)):
                        valid_feats.append(val)
                    else:
                        valid_feats.append(np.zeros(feat_dim))
                
                x_base = np.array(valid_feats) # (N, 768)
                
                # 2. 获取切片掩码 (1维)
                # feature_extraction 里已经算好了 slice_mask 列
                if "slice_mask" in n.columns:
                    base_mask = n["slice_mask"].values.reshape(-1, 1) # (N, 1)
                else:
                    base_mask = np.ones((len(n), 1)) # 如果没算出来，默认全1

                if self.mask_mode == "all_ones":
                    x_mask = np.ones((len(n), 1), dtype=np.float32)
                elif self.mask_mode == "random":
                    rng = np.random.RandomState(self.mask_seed + int(_id))
                    x_mask = rng.binomial(1, 0.5, size=(len(n), 1)).astype(np.float32)
                else:
                    x_mask = base_mask.astype(np.float32)

                # 3. 拼接 -> 769维
                x_enhanced = np.concatenate([x_base, x_mask], axis=1)

                if len(e) == 2:
                     edge_index = np.array(e)
                else:
                     edge_index = np.array([[], []])

                code_graph = Data(x=torch.FloatTensor(x_enhanced), edge_index=torch.LongTensor(edge_index))

                # --- 标签与元数据 ---
                code_graph.y = torch.tensor([true_label], dtype=torch.long)
                
                if 'lineNumber' in n.columns:
                    vuln_indices = self.get_vuln_indices(_id)
                    line_nums = n['lineNumber'].astype(int)
                    n["vuln"] = line_nums.map(vuln_indices).fillna(0)
                else:
                    n["vuln"] = 0
                
                code_graph.vuln_label = torch.Tensor(n["vuln"].astype(int).to_numpy())
                
                if true_label == 1: stats["vul"] += 1
                else: stats["non"] += 1

                node_ids = n['id'] if 'id' in n.columns else n.index
                try:
                    code_graph.line_index = torch.Tensor(node_ids.astype(int).to_numpy())
                except:
                    code_graph.line_index = torch.Tensor(n.index.to_numpy())

                code_graph.sample_id = torch.Tensor([int(_id)] * len(n))

                num_edges = code_graph.edge_index.shape[1]
                if edge_types is None or len(edge_types) == 0:
                    final_edge_types = torch.LongTensor([0] * num_edges)
                elif len(edge_types) != num_edges:
                    if len(edge_types) > num_edges: final_edge_types = edge_types[:num_edges]
                    else:
                        padding = torch.zeros(num_edges - len(edge_types), dtype=torch.long)
                        final_edge_types = torch.cat([edge_types, padding])
                else:
                    final_edge_types = edge_types

                code_graph.edge_types = final_edge_types
                data_list.append(code_graph)

            except Exception as ex:
                if self.debug: print(f"Error processing {_id}: {ex}")
                stats["dropped"] += 1
                continue

        print(f"\n=== Dataset Generation Complete ===")
        print(f"Total Graphs: {len(data_list)}")
        print(f"Stats: Vul={stats['vul']}, Non-Vul={stats['non']}, Dropped={stats['dropped']}")
        print(f"NOTE: Feature dimension is now 769 (768 + 1 mask). Please update your model input_dim!")
        print("===================================\n")
        
        if len(data_list) > 0:
            print(f'Saving to {self.processed_paths[0]}...')
            torch.save(data_list, self.processed_paths[0])
        else:
            print("❌ Error: No graphs generated.")
        
    def len(self) -> int: return len(self.data_list)
    def get(self, idx: int) -> Data: return self.data_list[idx]
    
    @staticmethod
    def itempath(_id): return Path(f"{_id}.nodes.json")
    @staticmethod
    def check_validity(_id): return True 
        
    def get_vuln_indices(self, _id):
        df = self.df[self.df.id == str(_id)]
        if df.empty: return {}
        try:
            raw_val = df.removed.values[0]
            val_list = []
            if isinstance(raw_val, (list, np.ndarray)): val_list = raw_val
            elif isinstance(raw_val, str):
                try: val_list = ast.literal_eval(raw_val)
                except:
                    raw_val = raw_val.strip("[]")
                    if raw_val: val_list = [int(float(x.strip())) for x in raw_val.split(",")]
            return {int(i): 1 for i in val_list}
        except: return {}
    
    def feature_extraction(self, _id):
        """
        特征提取：GraphCodeBERT + Slice Masking
        """
        if self.encoder is None or self.tokenizer is None:
            raise ValueError(
                "encoder/tokenizer is required to build graph features. "
                "Please pass both when constructing VulGraphDataset, or prepare processed data in advance."
            )

        cache_name = f"graph_feat_target_{_id}"
        cachefp = utils.get_dir(utils.cache_dir() / "vul_graph_feat_target") / cache_name

        if cachefp.exists():
            try:
                with open(cachefp, "rb") as f:
                    result = pkl.load(f)
                    if len(result) == 3: return result
            except: pass

        if _id not in self.id2filename: return None, None, None
        rel_path = self.id2filename[_id]
        
        node_path = DATA_FULL_DIR / rel_path
        edge_filename = node_path.name.replace(".nodes.json", ".edges.json")
        edge_path = node_path.parent / edge_filename

        if not node_path.exists(): return None, None, None

        try:
            with open(node_path, "r") as f:
                nodes_data = json.load(f)
                nodes = pd.DataFrame(nodes_data)
            
            if nodes.empty: return None, None, None

            if edge_path.exists():
                with open(edge_path, "r") as f:
                    edges_data = json.load(f)
                    edges = pd.DataFrame(edges_data, columns=["innode", "outnode", "etype", "variable"])
            else:
                edges = pd.DataFrame(columns=["innode", "outnode", "etype", "variable"])

        except: return None, None, None

        # -----------------------------------------------------------
        # 1. 语义嵌入 (Embedding)
        # -----------------------------------------------------------
        try:
            if "lineNumber" not in nodes.columns: nodes["lineNumber"] = -1
            nodes["lineNumber"] = pd.to_numeric(nodes["lineNumber"], errors='coerce').fillna(-1).astype(int)
            
            valid_lines = nodes[nodes.lineNumber > 0].copy()
            if valid_lines.empty: return None, None, None

            if "code" not in valid_lines.columns: valid_lines["code"] = ""

            subseq = (valid_lines.sort_values(by="code", key=lambda x: x.str.len(), ascending=False)
                      .groupby("lineNumber").head(1))
            subseq = subseq[["lineNumber", "code"]].copy()
            subseq["code_text"] = subseq.code.fillna("")
            subseq = subseq[subseq.code_text.str.strip() != ""]
            
            line2text = subseq.set_index("lineNumber")["code_text"].to_dict()
            sorted_lines = sorted(subseq.lineNumber.unique())
            
            context_texts = []
            sep = self.tokenizer.sep_token if self.tokenizer else " [SEP] "
            
            for i, line_num in enumerate(sorted_lines):
                curr_text = line2text[line_num]
                prev_text = line2text.get(sorted_lines[i-1], "") if i > 0 else ""
                next_text = line2text.get(sorted_lines[i+1], "") if i < len(sorted_lines) - 1 else ""
                full_text = f"{prev_text} {sep} {curr_text} {sep} {next_text}"
                context_texts.append(full_text)
            
            embeddings = []
            batch_size = 64 
            
            if len(context_texts) > 0:
                for i in range(0, len(context_texts), batch_size):
                    batch_texts = context_texts[i : i + batch_size]
                    encoded_inputs = self.tokenizer(
                        batch_texts, padding=True, truncation=True, max_length=512, return_tensors="pt"
                    )
                    input_ids = encoded_inputs['input_ids'].to(self.device)
                    attention_mask = encoded_inputs['attention_mask'].to(self.device)
                    
                    with torch.no_grad():
                        if hasattr(self.encoder, 'roberta'):
                            outputs = self.encoder.roberta(input_ids=input_ids, attention_mask=attention_mask)
                        elif hasattr(self.encoder, 'bert'):
                            outputs = self.encoder.bert(input_ids=input_ids, attention_mask=attention_mask)
                        else:
                            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
                        
                        last_hidden_state = outputs.last_hidden_state
                        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
                        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
                        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                        batch_embeddings = (sum_embeddings / sum_mask).cpu().numpy()
                        embeddings.extend(list(batch_embeddings))
            
            feat_map = dict(zip(sorted_lines, embeddings))

            nodesline = nodes.copy()
            if 'id' not in nodesline.columns: nodesline["id"] = nodesline.index
            
            nodesline["subseq_feat"] = nodesline.lineNumber.map(feat_map)

            if hasattr(self.encoder.config, 'hidden_size'):
                feat_dim = self.encoder.config.hidden_size
            else:
                feat_dim = 768
            zero_feat = np.zeros(feat_dim)
            
            nodesline["subseq_feat"] = nodesline["subseq_feat"].apply(
                lambda x: x if isinstance(x, (np.ndarray, list)) else zero_feat
            )

            # -----------------------------------------------------------
            # 2. [SR-GVD 新增] 切片掩码生成 (Slice Masking)
            # -----------------------------------------------------------
            try:
                slice_path = SLICE_SOURCE_DIR / f"{_id}.c"
                # 计算对齐：哪些全量图的行在切片中保留了
                mask_mapping = get_slice_mask_by_alignment(nodes, slice_path)
                
                if mask_mapping:
                    # 匹配成功：填入 0 或 1
                    nodesline["slice_mask"] = nodesline["lineNumber"].map(mask_mapping).fillna(0.0).astype(float)
                else:
                    # 匹配失败/文件不存在：默认全 1 (保守策略，认为全重要，避免丢失信息)
                    nodesline["slice_mask"] = 1.0
            except Exception as e:
                if self.debug: print(f"Mask alignment warning {_id}: {e}")
                nodesline["slice_mask"] = 1.0

            # -----------------------------------------------------------
            
            edgesline = edges.copy()
            if 'etype' in edgesline.columns:
                keep_types = ['CFG', 'CDG', 'REACHING_DEF', 'DDG'] 
                edgesline = edgesline[edgesline['etype'].isin(keep_types)]
            
            edgesline["src_id"] = edgesline["outnode"].astype(str)
            edgesline["dst_id"] = edgesline["innode"].astype(str)
            
            valid_node_ids = set(nodesline.id.astype(str).values)
            edgesline = edgesline[edgesline.src_id.isin(valid_node_ids) & edgesline.dst_id.isin(valid_node_ids)]
            
            if 'etype' in edgesline.columns:
                unique_types = sorted(edgesline.etype.unique())
                type_map = {t: i for i, t in enumerate(unique_types)}
                edgesline["etype_idx"] = edgesline.etype.map(type_map).fillna(0).astype(int)
            else:
                edgesline["etype_idx"] = 0

            nodesline = nodesline.reset_index(drop=True)
            node_id_map = {str(original_id): new_idx for new_idx, original_id in enumerate(nodesline.id)}
            
            edgesline["src"] = edgesline.src_id.map(node_id_map)
            edgesline["dst"] = edgesline.dst_id.map(node_id_map)
            
            edgesline = edgesline.dropna(subset=["src", "dst"])

            edge_index = [edgesline.src.tolist(), edgesline.dst.tolist()]
            edge_types = torch.LongTensor(edgesline.etype_idx.tolist())
            
            with open(cachefp, "wb") as f:
                # 缓存也包含了 nodesline，其中现在有 'slice_mask' 列
                pkl.dump([nodesline, edge_index, edge_types], f)
                
            return nodesline, edge_index, edge_types
            
        except Exception as e:
            if self.debug: print(f"[FEAT EXTRACT ERROR] {_id}: {e}")
            return None, None, None

def collate(data_list):
    batch = Batch.from_data_list(data_list)
    return batch

if __name__ == '__main__':
    MODEL_CLASSES = {
        'bert': (BertConfig, BertForMaskedLM, BertTokenizer),
        'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    }
    model_type = "roberta"
    model_name_or_path = "/root/autodl-tmp/counterfactual-vulnerability-detection-main/cfexplainer/storage/processed/graphcodebert-base"
    
    partition = sys.argv[1] if len(sys.argv) > 1 else "train"
    debug_mode = "debug" in sys.argv
    clear_cache = "clear" in sys.argv

    config_class, model_class, tokenizer_class = MODEL_CLASSES[model_type]
    print(f"Loading {model_type} from {model_name_or_path}...")
    
    config = config_class.from_pretrained(model_name_or_path, local_files_only=True)
    tokenizer = tokenizer_class.from_pretrained(model_name_or_path, local_files_only=True)
    language_model = model_class.from_pretrained(model_name_or_path, config=config, local_files_only=True)

    dataset = VulGraphDataset(
        root=str(utils.processed_dir() / "vul_graph_dataset"),
        encoder=language_model,
        tokenizer=tokenizer,
        partition=partition,
        debug=debug_mode,
        clear_cache=clear_cache
    )

    print(f"Dataset ready. Size: {len(dataset)}")
