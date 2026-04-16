from math import sqrt
import torch
from torch import Tensor
import torch.nn.functional as F
from torch_geometric.utils.loop import add_remaining_self_loops
from dig.xgraph.models.utils import subgraph
from dig.version import debug
from dig.xgraph.method.utils import symmetric_edge_mask_indirect_graph
from torch_geometric.nn import MessagePassing
from dig.xgraph.method.base_explainer import ExplainerBase
from typing import Union


class CFExplainer(ExplainerBase):
    def __init__(self, model: torch.nn.Module, epochs: int = 100, lr: float = 0.01, 
                 alpha: float = 0.9, explain_graph: bool = False, 
                 indirect_graph_symmetric_weights: bool = False, L1_dist: bool = False):
        super(CFExplainer, self).__init__(model, epochs, lr, explain_graph)
        self.alpha = alpha
        self.L1_dist = L1_dist
        self._symmetric_edge_mask_indirect_graph = indirect_graph_symmetric_weights

        # 🆕 安全地冻结参数
        self._safe_freeze_parameters()
        
        self.model.eval()

    def _safe_freeze_parameters(self):
        """安全地冻结模型参数"""
        print("开始冻结模型参数...")

        # 首先确保模型已初始化
        try:
            # 使用更健壮的方法检查未初始化参数
            uninitialized = []
            for name, param in self.model.named_parameters():
                # 检查参数是否包含数据
                if not hasattr(param, 'data') or param.data.nelement() == 0:
                    uninitialized.append(name)
                # 检查是否是UninitializedParameter类型
                elif hasattr(param, 'is_uninitialized') and param.is_uninitialized():
                    uninitialized.append(name)

            if uninitialized:
                print(f"发现 {len(uninitialized)} 个未初始化的参数")
                print("使用更全面的虚拟数据初始化模型...")
                self._initialize_with_comprehensive_dummy_data()

            # 分步骤冻结参数
            total_params = 0
            frozen_count = 0
            skipped_count = 0

            for name, param in self.model.named_parameters():
                total_params += 1
                try:
                    # 再次检查参数状态
                    if hasattr(param, 'is_uninitialized') and param.is_uninitialized():
                        print(f"跳过未初始化的参数: {name}")
                        skipped_count += 1
                        continue

                    param.requires_grad_(False)
                    frozen_count += 1
                except Exception as e:
                    print(f"冻结参数 {name} 时出错: {e}")
                    skipped_count += 1
                    continue

            print(f"参数冻结统计: 总计 {total_params}, 成功冻结 {frozen_count}, 跳过 {skipped_count}")

        except Exception as e:
            print(f"参数冻结过程中出错: {e}")

    def _initialize_with_comprehensive_dummy_data(self):
        """使用更全面的虚拟数据初始化模型"""
        device = next(self.model.parameters()).device

        with torch.no_grad():
            test_sizes = [2, 5, 10]
            for size in test_sizes:
                try:
                    print(f"尝试使用 {size} 个节点的虚拟数据初始化...")
                    dummy_x = torch.randn(size, 768).to(device)
                    dummy_edge_index = torch.combinations(torch.arange(size), r=2).t().contiguous().to(device)
                    dummy_batch = torch.zeros(size, dtype=torch.long).to(device)

                    if hasattr(self.model, 'args') and getattr(self.model.args, 'use_edge_features', False):
                        dummy_edge_attr = torch.randn(dummy_edge_index.size(1), 64).to(device)
                        output = self.model(dummy_x, dummy_edge_index, dummy_batch, edge_attr=dummy_edge_attr)
                    else:
                        output = self.model(dummy_x, dummy_edge_index, dummy_batch)

                    print(f"使用 {size} 个节点初始化成功，输出形状: {output.shape}")
                    break
                except Exception as e:
                    print(f"使用 {size} 个节点初始化失败: {e}")
                    continue

    def _initialize_with_dummy_data(self):
        """使用虚拟数据初始化模型"""
        device = next(self.model.parameters()).device

        with torch.no_grad():
            num_nodes = 10
            dummy_x = torch.randn(num_nodes, 768).to(device)
            dummy_edge_index = torch.tensor([
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
            ], dtype=torch.long).to(device)
            dummy_batch = torch.zeros(num_nodes, dtype=torch.long).to(device)

            if hasattr(self.model, 'args') and getattr(self.model.args, 'use_edge_features', False):
                dummy_edge_attr = torch.randn(dummy_edge_index.size(1), 64).to(device)
                _ = self.model(dummy_x, dummy_edge_index, dummy_batch, edge_attr=dummy_edge_attr)
            else:
                _ = self.model(dummy_x, dummy_edge_index, dummy_batch)

    @staticmethod
    def _model_forward_kwargs(kwargs):
        """Remove explainer-only kwargs before calling detector forward."""
        model_kwargs = dict(kwargs)
        for key in ["num_classes", "sparsity", "edge_masks", "node_idx", "target_label"]:
            model_kwargs.pop(key, None)
        return model_kwargs

    def __set_masks__(self, x: Tensor, edge_index: Tensor, init: str = "normal"):
        (N, F), E = x.size(), edge_index.size(1)

        self.node_feat_mask = torch.nn.Parameter(
            torch.randn(F, requires_grad=True, device=self.device) * 0.1
        )
        std = torch.nn.init.calculate_gain('relu') * sqrt(2.0 / (2 * N))
        self.edge_mask = torch.nn.Parameter(
            torch.randn(E, requires_grad=True, device=self.device) * std
        )

        loop_mask = edge_index[0] != edge_index[1]
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                module.explain = True
                module._edge_mask = self.edge_mask
                module._loop_mask = loop_mask
                module._apply_sigmoid = True

    def __clear_masks__(self):
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                module.explain = False
                module._edge_mask = None
                module._loop_mask = None
                module._apply_sigmoid = True
        self.node_feat_mask = None
        self.edge_mask = None

    def __loss__(self, raw_preds: Tensor, x_label: Union[Tensor, int]) -> Tensor:
        if isinstance(x_label, torch.Tensor):
            x_label = int(x_label.item())

        margin = 1.0
        if self.explain_graph:
            logits = raw_preds
            if logits.dim() == 2 and logits.size(0) >= 1:
                logits = logits[0]
            pos = logits[x_label]
            mask = torch.ones_like(logits, dtype=torch.bool)
            mask[x_label] = False
            neg = logits[mask].max()
        else:
            node_logits = raw_preds[self.node_idx]
            pos = node_logits[x_label]
            mask = torch.ones_like(node_logits, dtype=torch.bool)
            mask[x_label] = False
            neg = node_logits[mask].max()

        cf_loss = torch.relu(pos - neg + margin)
        m = self.edge_mask.sigmoid()
        if self.L1_dist:
            edge_dist_loss = m.abs().sum()
        else:
            edge_dist_loss = F.binary_cross_entropy(m, torch.zeros_like(m, device=m.device))
        return self.alpha * cf_loss + (1 - self.alpha) * edge_dist_loss

    def gnn_explainer_alg(self,
                          x: Tensor,
                          edge_index: Tensor,
                          ex_label: Tensor,
                          mask_features: bool = False,
                          **kwargs) -> Tensor:
        self.to(x.device)
        self.mask_features = mask_features
        optimizer = torch.optim.Adam([self.node_feat_mask, self.edge_mask], lr=self.lr)

        for epoch in range(1, self.epochs + 1):
            h = x * self.node_feat_mask.view(1, -1).sigmoid() if mask_features else x
            raw_preds = self.model(x=h, edge_index=edge_index, **self._model_forward_kwargs(kwargs))
            loss = self.__loss__(raw_preds, ex_label)

            if epoch % 20 == 0 and debug:
                print(f'Loss:{loss.item():.6f}')

            optimizer.zero_grad()
            loss.backward()
            if self.edge_mask.grad is not None:
                torch.nn.utils.clip_grad_norm_([self.edge_mask], max_norm=5.0)
            if mask_features and self.node_feat_mask.grad is not None:
                torch.nn.utils.clip_grad_norm_([self.node_feat_mask], max_norm=5.0)
            optimizer.step()

        return self.edge_mask.data

    def forward(self, x, edge_index, mask_features=False, target_label=None, **kwargs):
        super().forward(x=x, edge_index=edge_index, **kwargs)
        self.model.eval()

        self_loop_edge_index, _ = add_remaining_self_loops(edge_index, num_nodes=self.num_nodes)

        if not self.explain_graph:
            self.node_idx = node_idx = kwargs.get('node_idx')
            assert node_idx is not None, 'An node explanation needs kwarg node_idx, but got None.'
            if isinstance(node_idx, torch.Tensor) and not node_idx.dim():
                node_idx = node_idx.to(self.device).flatten()
            elif isinstance(node_idx, (int, list, tuple)):
                node_idx = torch.tensor([node_idx], device=self.device, dtype=torch.int64).flatten()
            else:
                raise TypeError(f'node_idx should be in types of int, list, tuple, or torch.Tensor, but got {type(node_idx)}')
            self.subset, _, _, self.hard_edge_mask = subgraph(
                node_idx, self.__num_hops__, self_loop_edge_index, relabel_nodes=True,
                num_nodes=None, flow=self.__flow__())
            self.new_node_idx = torch.where(self.subset == node_idx)[0]

        if kwargs.get('edge_masks'):
            edge_masks = kwargs.pop('edge_masks')
            self.__set_masks__(x, self_loop_edge_index)
        else:
            num_classes = kwargs.get('num_classes')
            if num_classes is None:
                num_classes = getattr(self.model, 'num_classes', 2)
            labels = tuple(i for i in range(num_classes))
            ex_labels = tuple(torch.tensor([label]).to(self.device) for label in labels)

            edge_masks = []
            for ex_label in ex_labels:
                if target_label is None or ex_label.item() == target_label.item():
                    self.__clear_masks__()
                    self.__set_masks__(x, self_loop_edge_index)
                    edge_mask = self.gnn_explainer_alg(x, edge_index, ex_label, **kwargs).sigmoid()
                    if self._symmetric_edge_mask_indirect_graph:
                        edge_mask = symmetric_edge_mask_indirect_graph(self_loop_edge_index, edge_mask)
                    edge_masks.append(edge_mask)

        hard_edge_masks = [self.control_sparsity(mask, sparsity=kwargs.get('sparsity'))
                           for mask in edge_masks]
        with torch.no_grad():
            related_preds = self.eval_related_pred(
                x, edge_index, hard_edge_masks, **self._model_forward_kwargs(kwargs)
            )
        self.__clear_masks__()
        return edge_masks, hard_edge_masks, related_preds, self_loop_edge_index

    def __repr__(self):
        return f'{self.__class__.__name__}()'

def _initialize_with_comprehensive_dummy_data(self):
    """使用更全面的虚拟数据初始化模型"""
    device = next(self.model.parameters()).device
    
    with torch.no_grad():
        # 尝试多种大小的虚拟数据
        test_sizes = [2, 5, 10]
        
        for size in test_sizes:
            try:
                print(f"尝试使用 {size} 个节点的虚拟数据初始化...")
                
                dummy_x = torch.randn(size, 768).to(device)
                # 创建完整的图结构
                dummy_edge_index = torch.combinations(torch.arange(size), r=2).t().contiguous().to(device)
                dummy_batch = torch.zeros(size, dtype=torch.long).to(device)
                
                # 根据模型配置决定是否使用边特征
                if hasattr(self.model, 'args') and getattr(self.model.args, 'use_edge_features', False):
                    dummy_edge_attr = torch.randn(dummy_edge_index.size(1), 64).to(device)
                    output = self.model(dummy_x, dummy_edge_index, dummy_batch, edge_attr=dummy_edge_attr)
                else:
                    output = self.model(dummy_x, dummy_edge_index, dummy_batch)
                
                print(f"使用 {size} 个节点初始化成功，输出形状: {output.shape}")
                break  # 如果成功就跳出循环
                
            except Exception as e:
                print(f"使用 {size} 个节点初始化失败: {e}")
                continue

    def _initialize_with_dummy_data(self):
        """使用虚拟数据初始化模型"""
        device = next(self.model.parameters()).device
        
        with torch.no_grad():
            # 创建合适的虚拟数据
            batch_size = 2
            num_nodes = 10
            
            dummy_x = torch.randn(num_nodes, 768).to(device)
            dummy_edge_index = torch.tensor([
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
            ], dtype=torch.long).to(device)
            dummy_batch = torch.zeros(num_nodes, dtype=torch.long).to(device)
            
            # 根据模型配置决定是否使用边特征
            if hasattr(self.model, 'args') and getattr(self.model.args, 'use_edge_features', False):
                dummy_edge_attr = torch.randn(dummy_edge_index.size(1), 64).to(device)
                _ = self.model(dummy_x, dummy_edge_index, dummy_batch, edge_attr=dummy_edge_attr)
            else:
                _ = self.model(dummy_x, dummy_edge_index, dummy_batch)
    def __set_masks__(self, x: Tensor, edge_index: Tensor, init: str = "normal"):
        (N, F), E = x.size(), edge_index.size(1)

        # 可选的特征掩码（默认不用）
        self.node_feat_mask = torch.nn.Parameter(
            torch.randn(F, requires_grad=True, device=self.device) * 0.1
        )

        # 边掩码
        std = torch.nn.init.calculate_gain('relu') * sqrt(2.0 / (2 * N))
        self.edge_mask = torch.nn.Parameter(
            torch.randn(E, requires_grad=True, device=self.device) * std
        )

        loop_mask = edge_index[0] != edge_index[1]
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                module.explain = True
                module._edge_mask = self.edge_mask
                module._loop_mask = loop_mask
                module._apply_sigmoid = True

    def __clear_masks__(self):
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                module.explain = False
                module._edge_mask = None
                module._loop_mask = None
                module._apply_sigmoid = True
        # 修正变量名（原来是 node_feat_masks）
        self.node_feat_mask = None
        self.edge_mask = None

    def __loss__(self, raw_preds: Tensor, x_label: Union[Tensor, int]) -> Tensor:
        """
        反事实目标：
        - 压低“当前类”logit（pos），抬高“对立类”logit（neg），满足 pos - neg + margin <= 0
        - 同时最小化掩码大小（让边尽量少）
        要求 raw_preds 为 logits（未过 softmax）。
        """
        if isinstance(x_label, torch.Tensor):
            x_label = int(x_label.item())

        margin = 1.0

        if self.explain_graph:
            # 兼容 [C] 或 [1, C] 或 [B, C]（取第 0 个样本）
            logits = raw_preds
            if logits.dim() == 2 and logits.size(0) == 1:
                logits = logits[0]
            elif logits.dim() == 2 and logits.size(0) > 1:
                logits = logits[0]  # 通常我们一次解释一个图
            # pos: 当前类；neg: 其它类最大
            pos = logits[x_label]
            mask = torch.ones_like(logits, dtype=torch.bool)
            mask[x_label] = False
            neg = logits[mask].max()
        else:
            # 节点解释：取该节点的 logits
            node_logits = raw_preds[self.node_idx]
            pos = node_logits[x_label]
            mask = torch.ones_like(node_logits, dtype=torch.bool)
            mask[x_label] = False
            neg = node_logits[mask].max()

        cf_loss = torch.relu(pos - neg + margin)

        # 稀疏正则：把 m → 0（更符合“尽量少改动”）
        m = self.edge_mask.sigmoid()
        if self.L1_dist:
            edge_dist_loss = m.abs().sum()
        else:
            edge_dist_loss = F.binary_cross_entropy(m, torch.zeros_like(m, device=m.device))

        return self.alpha * cf_loss + (1 - self.alpha) * edge_dist_loss

    def gnn_explainer_alg(self,
                          x: Tensor,
                          edge_index: Tensor,
                          ex_label: Tensor,
                          mask_features: bool = False,
                          **kwargs) -> Tensor:

        # 初始化掩码
        self.to(x.device)
        self.mask_features = mask_features

        # 只优化掩码参数
        optimizer = torch.optim.Adam([self.node_feat_mask, self.edge_mask], lr=self.lr)

        for epoch in range(1, self.epochs + 1):

            # 可选特征掩码（默认不用）
            if mask_features:
                h = x * self.node_feat_mask.view(1, -1).sigmoid()
            else:
                h = x

            # raw_preds 必须是 logits（Detector 已修改为返回 logits）
            raw_preds = self.model(x=h, edge_index=edge_index, **kwargs)

            loss = self.__loss__(raw_preds, ex_label)

            if epoch % 20 == 0 and debug:
                print(f'Loss:{loss.item():.6f}')

            optimizer.zero_grad()
            loss.backward()

            # 只裁剪“掩码”的梯度，避免误裁模型参数（模型已冻结）
            if self.edge_mask.grad is not None:
                torch.nn.utils.clip_grad_norm_([self.edge_mask], max_norm=5.0)
            if mask_features and self.node_feat_mask.grad is not None:
                torch.nn.utils.clip_grad_norm_([self.node_feat_mask], max_norm=5.0)

            optimizer.step()

        return self.edge_mask.data

    def forward(self, x, edge_index, mask_features=False, target_label=None, **kwargs):
        r"""
        Run the explainer for a specific graph instance.
        Args:
            x (torch.Tensor): The graph instance's input node features.
            edge_index (torch.Tensor): The graph instance's edge index.
            mask_features (bool, optional): Whether to use feature mask. Not recommended. (Default: False)
            target_label (torch.Tensor, optional): if given then apply optimization only on this label
            **kwargs:
                node_idx (for node classification)
                sparsity (float): The Sparsity we need to control to transform a soft mask to a hard mask. (Default: 0.7)
                num_classes (int): The number of task's classes.
        Returns:
            edge_masks, hard_edge_masks, related_predictions, self_loop_edge_index
        """
        super().forward(x=x, edge_index=edge_index, **kwargs)
        self.model.eval()

        self_loop_edge_index, _ = add_remaining_self_loops(edge_index, num_nodes=self.num_nodes)

        # 节点任务：裁到 k-hop 诱导子图
        if not self.explain_graph:
            self.node_idx = node_idx = kwargs.get('node_idx')
            assert node_idx is not None, 'An node explanation needs kwarg node_idx, but got None.'
            if isinstance(node_idx, torch.Tensor) and not node_idx.dim():
                node_idx = node_idx.to(self.device).flatten()
            elif isinstance(node_idx, (int, list, tuple)):
                node_idx = torch.tensor([node_idx], device=self.device, dtype=torch.int64).flatten()
            else:
                raise TypeError(f'node_idx should be in types of int, list, tuple, or torch.Tensor, but got {type(node_idx)}')
            self.subset, _, _, self.hard_edge_mask = subgraph(
                node_idx, self.__num_hops__, self_loop_edge_index, relabel_nodes=True,
                num_nodes=None, flow=self.__flow__())
            self.new_node_idx = torch.where(self.subset == node_idx)[0]

        if kwargs.get('edge_masks'):
            edge_masks = kwargs.pop('edge_masks')
            self.__set_masks__(x, self_loop_edge_index)
        else:
            # 为所有类别分别学习一次掩码（与原流程兼容）
            labels = tuple(i for i in range(kwargs.get('num_classes')))
            ex_labels = tuple(torch.tensor([label]).to(self.device) for label in labels)

            edge_masks = []
            for ex_label in ex_labels:
                # 若指定 target_label，只对该类优化；否则对所有类优化
                if target_label is None or ex_label.item() == target_label.item():
                    self.__clear_masks__()
                    self.__set_masks__(x, self_loop_edge_index)
                    edge_mask = self.gnn_explainer_alg(x, edge_index, ex_label).sigmoid()

                    if self._symmetric_edge_mask_indirect_graph:
                        edge_mask = symmetric_edge_mask_indirect_graph(self_loop_edge_index, edge_mask)

                    edge_masks.append(edge_mask)

        # 将软掩码硬化（遵循原接口）
        hard_edge_masks = [self.control_sparsity(mask, sparsity=kwargs.get('sparsity'))
                           for mask in edge_masks]

        with torch.no_grad():
            related_preds = self.eval_related_pred(x, edge_index, hard_edge_masks, **kwargs)

        self.__clear_masks__()

        return edge_masks, hard_edge_masks, related_preds, self_loop_edge_index

    def __repr__(self):
        return f'{self.__class__.__name__}()'
