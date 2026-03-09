import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, ReLU, GELU, Dropout
from torch_geometric.utils import softmax 

# 注意：请确保安装了较新版本的 torch_geometric 以支持 GraphNorm
from torch_geometric.nn import (
    GCNConv, GatedGraphConv, GraphConv, 
    GATv2Conv, RGCNConv, RGATConv, TransformerConv,
    GlobalAttention, global_mean_pool, global_max_pool, global_add_pool,
    GraphNorm
)

# ==========================================\n# 1. 辅助池化类 (保持不变)
# ==========================================
class GNNPool(nn.Module):
    def __init__(self):
        super().__init__()

class GlobalMeanPool(GNNPool):
    def __init__(self):
        super().__init__()
    def forward(self, x, batch):
        return global_mean_pool(x, batch)

class GlobalAddPool(GNNPool):
    def __init__(self):
        super().__init__()
    def forward(self, x, batch):
        return global_add_pool(x, batch)
    
class GlobalMaxPool(GNNPool):
    def __init__(self):
        super().__init__()
    def forward(self, x, batch):
        return global_max_pool(x, batch)

# ==========================================
# 2. 增强后的主模型类 (EnhancedDetector)
# ==========================================
class EnhancedDetector(nn.Module):
    def __init__(self, args):
        super(EnhancedDetector, self).__init__()
        self.args = args
        self.num_classes = args.num_classes
        
        # ------------------------------------------------------------------
        # [修改点 1] 输入层改造：分离语义特征与切片掩码
        # ------------------------------------------------------------------
        # 假设输入特征总长 769 = 768 (BERT/RoBERTa) + 1 (Slice Mask)
        self.bert_dim = 768  
        self.hidden_dim = args.gnn_hidden_size
        
        # A. 语义特征投影: 768 -> hidden_dim
        self.sem_proj = Linear(self.bert_dim, self.hidden_dim)
        
        # B. 切片掩码嵌入: 0/1 -> hidden_dim
        # 这将 Mask 视为一种状态，学习出对应的向量表示，避免被淹没
        self.slice_embedding = nn.Embedding(2, self.hidden_dim)
        
        # C. 融合后的归一化与激活
        self.input_norm = nn.LayerNorm(self.hidden_dim)
        self.input_act = GELU()
        
        # D. 移除旧的直接投影层
        # self.lin = Linear(args.gnn_feature_dim_size, args.gnn_hidden_size)
        
        # ------------------------------------------------------------------
        # GNN 骨干网络定义 (保持不变)
        # ------------------------------------------------------------------
        self.dropout = Dropout(args.dropout_rate)
        self.act = GELU()
        
        # 第一层 GNN
        if args.gnn_model == 'GCN':
            self.conv1 = GCNConv(self.hidden_dim, self.hidden_dim)
        elif args.gnn_model == 'GAT':
             # GATv2通常比GAT更强
            self.conv1 = GATv2Conv(self.hidden_dim, self.hidden_dim // args.num_heads, heads=args.num_heads)
        elif args.gnn_model == 'GraphConv':
            self.conv1 = GraphConv(self.hidden_dim, self.hidden_dim)
        elif args.gnn_model == 'GatedGraph':
            self.conv1 = GatedGraphConv(self.hidden_dim, num_layers=args.num_ggnn_steps)
        elif args.gnn_model == 'Transformer':
            self.conv1 = TransformerConv(self.hidden_dim, self.hidden_dim // args.num_heads, heads=args.num_heads, edge_dim=args.num_relations if args.residual else None)
        elif args.gnn_model == 'RGCN':
            self.conv1 = RGCNConv(self.hidden_dim, self.hidden_dim, num_relations=args.num_relations)
        else:
            raise ValueError(f"Unsupported GNN model: {args.gnn_model}")

        # 残差连接或第二层
        self.conv2 = None
        if args.num_gnn_layers >= 2:
            if args.gnn_model == 'GCN':
                self.conv2 = GCNConv(self.hidden_dim, self.hidden_dim)
            elif args.gnn_model == 'GAT':
                self.conv2 = GATv2Conv(self.hidden_dim, self.hidden_dim // args.num_heads, heads=args.num_heads)
            elif args.gnn_model == 'GraphConv':
                self.conv2 = GraphConv(self.hidden_dim, self.hidden_dim)
            elif args.gnn_model == 'GatedGraph':
                pass # GGNN 自身就是多步的
            elif args.gnn_model == 'Transformer':
                self.conv2 = TransformerConv(self.hidden_dim, self.hidden_dim // args.num_heads, heads=args.num_heads)
            elif args.gnn_model == 'RGCN':
                self.conv2 = RGCNConv(self.hidden_dim, self.hidden_dim, num_relations=args.num_relations)

        # 图池化层
        if args.graph_pooling == "sum":
            self.pool = GlobalAddPool()
        elif args.graph_pooling == "mean":
            self.pool = GlobalMeanPool()
        elif args.graph_pooling == "max":
            self.pool = GlobalMaxPool()
        elif args.graph_pooling == "attn":
             self.pool = GlobalAttention(Linear(self.hidden_dim, 1))
        else:
             self.pool = GlobalMeanPool()

        # 分类头
        self.classifier = nn.Sequential(
            Linear(self.hidden_dim, self.hidden_dim),
            GELU(),
            Dropout(args.dropout_rate),
            Linear(self.hidden_dim, args.num_classes)
        )
        
        print(f"[Model Init] Mode: Slice-Embedding Enhanced | Backbone: {args.gnn_model}")


    def forward(self, x, edge_index, batch=None, edge_attr=None):
        # =========================================================
        # [修改点 2] 前向传播逻辑：特征融合
        # =========================================================
        
        # 1. 拆分输入 [N, 769] -> [N, 768] & [N, 1]
        sem_feat = x[:, :768]       # 语义特征
        slice_mask = x[:, 768]      # 切片掩码 (float)
        
        # 确保 mask 是 long 类型用于 Embedding 查表 (0 或 1)
        # 这里假设 mask 只有 0.0 和 1.0 两个值
        slice_mask_idx = slice_mask.long() 
        
        # 2. 分别映射
        h_sem = self.sem_proj(sem_feat)        # [N, hidden]
        h_mask = self.slice_embedding(slice_mask_idx) # [N, hidden]
        
        # 3. 融合 (Add) + 归一化 + 激活
        # 此时 Mask 的信息已经作为一个强特征叠加到了语义特征上
        h = h_sem + h_mask
        h = self.input_norm(h)
        h = self.input_act(h)
        
        # =========================================================
        # 后续 GNN 处理逻辑 (与原代码保持一致)
        # =========================================================
        
        # GNN Layer 1
        if self.args.gnn_model == 'Transformer':
            # TransformerConv 支持 edge_attr 但具体看实现，这里简化处理
            h_gnn = self.conv1(h, edge_index) 
        elif self.args.gnn_model == 'RGCN':
             # RGCN 需要 edge_type，这里假设 edge_attr 是 type 索引
             # 注意：dataset 中 edge_attr 需要是 long 类型
            if edge_attr is not None:
                h_gnn = self.conv1(h, edge_index, edge_attr)
            else:
                 # 如果没有边类型，RGCN 可能会报错，需确保数据处理正确
                h_gnn = self.conv1(h, edge_index, torch.zeros(edge_index.size(1)).long().to(x.device))
        else:
            h_gnn = self.conv1(h, edge_index)

        h = self.act(h_gnn)
        h = self.dropout(h)

        # GNN Layer 2 (Residual)
        if self.conv2 is not None:
            if self.args.residual:
                identity = h
            
            if self.args.gnn_model == 'RGCN' and edge_attr is not None:
                h_new = self.conv2(h, edge_index, edge_attr)
            elif self.args.gnn_model == 'RGCN':
                h_new = self.conv2(h, edge_index, torch.zeros(edge_index.size(1)).long().to(x.device))
            else:
                h_new = self.conv2(h, edge_index)
            
            h_new = self.act(h_new)
            h_new = self.dropout(h_new)
            
            if self.args.residual:
                h = h + h_new
            else:
                h = h_new

        # Pooling
        if batch is None:
             # 如果没有 batch 向量，默认为全图池化 (batch size = 1)
            batch = torch.zeros(h.size(0), dtype=torch.long, device=h.device)
            
        hg = self.pool(h, batch)
        
        # Classification
        out = self.classifier(hg)
        
        return out


# ==========================================
# 3. 模型自检函数 (Model Diagnosis)
# ==========================================
def model_diagnosis(model, device="cpu"):
    """
    检查模型是否能正常跑通一次 Forward
    """
    print("\n[Diagnosis] Running model self-check...")
    model.to(device)
    model.eval()
    
    # 构造假数据
    # 模拟 [N=10, Feat=769]
    # 注意：第 769 维必须是 0 或 1
    N = 10
    feat_dim = 769
    
    # 随机生成前 768 维
    x_sem = torch.randn(N, 768).to(device)
    # 随机生成 Mask (0 或 1)
    x_mask = torch.randint(0, 2, (N, 1)).float().to(device)
    
    x = torch.cat([x_sem, x_mask], dim=1) # [N, 769]
    
    edge_index = torch.randint(0, N, (2, 20)).to(device)
    batch = torch.zeros(N, dtype=torch.long).to(device)
    
    # 模拟 RGCN 可能需要的边类型
    edge_attr = torch.randint(0, 3, (20,)).long().to(device)

    with torch.no_grad():
        try:
            output = model(x, edge_index, batch, edge_attr)
            print(f"Model output shape: {output.shape}")
            if output.shape[1] == model.num_classes:
                print("=== Diagnosis Passed: Output shape matches num_classes ===\n")
                return True
            else:
                 print(f"=== Diagnosis Warning: Expected {model.num_classes} classes, got {output.shape[1]} ===\n")
                 return False
            
        except Exception as e:
            print(f"\n[!!!] Model diagnosis failed: {e}")
            import traceback
            traceback.print_exc()
            return False

# ==========================================
# 4. 参数配置类 (测试用)
# ==========================================
class EnhancedArgs:
    def __init__(self):
        self.gnn_feature_dim_size = 769 # 必须匹配你的数据处理
        self.gnn_hidden_size = 256 
        self.dropout_rate = 0.3
        self.num_gnn_layers = 2
        self.residual = True
        self.num_classes = 2
        self.gnn_model = "RGCN" 
        self.num_relations = 5  
        self.num_heads = 4
        self.graph_pooling = "attn"
        self.num_ggnn_steps = 3

if __name__ == "__main__":
    args = EnhancedArgs()
    try:
        print(f"初始化模型: {args.gnn_model} ...")
        model = EnhancedDetector(args)
        
        # 运行诊断
        model_diagnosis(model)
        
    except Exception as e:
        print(f"Main block error: {e}")