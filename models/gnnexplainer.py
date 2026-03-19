from math import sqrt
import torch
from torch import Tensor
from torch_geometric.utils.loop import add_remaining_self_loops
from dig.version import debug
from dig.xgraph.models.utils import subgraph
from dig.xgraph.method.utils import symmetric_edge_mask_indirect_graph
from torch.nn.functional import cross_entropy
from torch_geometric.nn import MessagePassing, GATConv
from dig.xgraph.method.base_explainer import ExplainerBase
from typing import Union, List, Optional
import torch.nn.functional as F

EPS = 1e-15

def cross_entropy_with_logit(y_pred: torch.Tensor, y_true: torch.Tensor, **kwargs):
    return cross_entropy(y_pred, y_true.long(), **kwargs)

class GATEnhancedGNNExplainer(ExplainerBase):
    r"""GNN-Explainer optimized for GAT models with attention-aware initialization
    and consistency constraints.
    
    Args:
        model (torch.nn.Module): The GNN module to explain.
        epochs (int, optional): The number of epochs to train.
            (default: :obj:`100`)
        lr (float, optional): The learning rate to apply.
            (default: :obj:`0.01`)
        coff_edge_size (float, optional): Coefficient for edge mask size regularization.
            (default: :obj:`0.001`)
        coff_edge_ent (float, optional): Coefficient for edge mask entropy regularization.
            (default: :obj:`0.001`)
        coff_node_feat_size (float, optional): Coefficient for node feature mask size regularization.
            (default: :obj:`1.0`)
        coff_node_feat_ent (float, optional): Coefficient for node feature mask entropy regularization.
            (default: :obj:`0.1`)
        explain_graph (bool, optional): Whether to explain graph classification model
            (default: :obj:`False`)
        indirect_graph_symmetric_weights (bool, optional): If `True`, then the explainer
            will first realize whether this graph input has indirect edges, 
            then makes its edge weights symmetric. (default: :obj:`False`)
        attention_consistency_weight (float, optional): Weight for attention consistency loss.
            (default: :obj:`0.1`)
        use_attention_guidance (bool, optional): Whether to use GAT attention for initialization.
            (default: :obj:`True`)
        early_stopping_patience (int, optional): Patience for early stopping.
            (default: :obj:`50`)
    """
    def __init__(self,
                 model: torch.nn.Module,
                 epochs: int = 100,
                 lr: float = 0.01,
                 coff_edge_size: float = 0.001,
                 coff_edge_ent: float = 0.001,
                 coff_node_feat_size: float = 1.0,
                 coff_node_feat_ent: float = 0.1,
                 explain_graph: bool = False,
                 indirect_graph_symmetric_weights: bool = False,
                 attention_consistency_weight: float = 0.1,
                 use_attention_guidance: bool = True,
                 early_stopping_patience: int = 50):
        super(GATEnhancedGNNExplainer, self).__init__(model, epochs, lr, explain_graph)
        self.coff_node_feat_size = coff_node_feat_size
        self.coff_node_feat_ent = coff_node_feat_ent
        self.coff_edge_size = coff_edge_size
        self.coff_edge_ent = coff_edge_ent
        self._symmetric_edge_mask_indirect_graph: bool = indirect_graph_symmetric_weights
        
        # GAT-specific enhancements
        self.attention_consistency_weight = attention_consistency_weight
        self.use_attention_guidance = use_attention_guidance
        self.early_stopping_patience = early_stopping_patience
        self.gat_attention_weights: Optional[torch.Tensor] = None
        self.attention_hooks: List = []
        
        # Detect if model contains GAT layers
        self._is_gat_model = any(isinstance(module, GATConv) for module in self.model.modules())
        if self._is_gat_model:
            print("Detected GAT model, enabling GAT-optimized GNNExplainer")

    def _register_attention_hooks(self):
        """Register hooks to capture GAT attention weights"""
        def attention_hook(module, input, output):
            if hasattr(module, 'alpha') and module.alpha is not None:
                # Store attention weights for later use
                self.gat_attention_weights = module.alpha.detach()
            elif hasattr(module, 'att') and module.att is not None:
                # Alternative attribute name in some GAT implementations
                self.gat_attention_weights = module.att.detach()
        
        # Clear previous hooks
        self._remove_attention_hooks()
        
        # Register hooks on all GAT layers
        for name, module in self.model.named_modules():
            if isinstance(module, GATConv):
                hook = module.register_forward_hook(attention_hook)
                self.attention_hooks.append(hook)

    def _remove_attention_hooks(self):
        """Remove all registered attention hooks"""
        for hook in self.attention_hooks:
            hook.remove()
        self.attention_hooks.clear()

    def _extract_gat_attention(self, x: Tensor, edge_index: Tensor, **kwargs):
        """Extract attention weights from GAT model"""
        if not self._is_gat_model:
            return None
            
        self._register_attention_hooks()
        
        # Perform forward pass to capture attention weights
        with torch.no_grad():
            _ = self.model(x, edge_index, **self._model_forward_kwargs(kwargs))
        
        self._remove_attention_hooks()
        return self.gat_attention_weights

    @staticmethod
    def _model_forward_kwargs(kwargs):
        model_kwargs = dict(kwargs)
        for key in ["num_classes", "sparsity", "edge_masks", "node_idx"]:
            model_kwargs.pop(key, None)
        return model_kwargs

    def __set_masks__(self, x: Tensor, edge_index: Tensor, init: str = "normal"):
        (N, F), E = x.size(), edge_index.size(1)

        self.node_feat_mask = torch.nn.Parameter(torch.randn(F, requires_grad=True, device=self.device) * 0.1)

        # Enhanced initialization for GAT models
        if init == "attention_guided" and self._is_gat_model and self.gat_attention_weights is not None:
            att_weights = self.gat_attention_weights
            
            # Handle multi-head attention (average across heads)
            if att_weights.dim() > 1 and att_weights.size(-1) > 1:
                att_weights = att_weights.mean(dim=-1)
            
            # Ensure attention weights match edge count
            if att_weights.size(0) == E:
                # Use attention weights as prior with small noise
                std = torch.nn.init.calculate_gain('relu') * sqrt(2.0 / (2 * N)) * 0.1
                initial_mask = att_weights + torch.randn_like(att_weights) * std
                print(f"Using attention-guided initialization for {E} edges")
            else:
                # Fallback to normal initialization if sizes don't match
                std = torch.nn.init.calculate_gain('relu') * sqrt(2.0 / (2 * N))
                initial_mask = torch.randn(E, requires_grad=True, device=self.device) * std
                print(f"Attention size mismatch: {att_weights.size(0)} vs {E}, using normal init")
        else:
            # Standard initialization
            std = torch.nn.init.calculate_gain('relu') * sqrt(2.0 / (2 * N))
            initial_mask = torch.randn(E, requires_grad=True, device=self.device) * std

        self.edge_mask = torch.nn.Parameter(initial_mask)
        
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
        self.gat_attention_weights = None

    def __loss__(self, raw_preds: Tensor, x_label: Union[Tensor, int]):
        """Enhanced loss function with GAT attention consistency"""
        if self.explain_graph:
            loss = cross_entropy_with_logit(raw_preds, x_label)
        else:
            loss = cross_entropy_with_logit(raw_preds[self.node_idx].reshape(1, -1), x_label)

        m = self.edge_mask.sigmoid()
        
        # Base sparsity regularization
        loss = loss + self.coff_edge_size * m.sum()
        ent = -m * torch.log(m + EPS) - (1 - m) * torch.log(1 - m + EPS)
        loss = loss + self.coff_edge_ent * ent.mean()

        # GAT-specific attention consistency loss
        if (self._is_gat_model and 
            self.attention_consistency_weight > 0 and 
            self.gat_attention_weights is not None):
            
            att_weights = self.gat_attention_weights
            
            # Handle multi-head attention
            if att_weights.dim() > 1 and att_weights.size(-1) > 1:
                att_weights = att_weights.mean(dim=-1)
            
            # Ensure sizes match
            if att_weights.size(0) == m.size(0):
                # Normalize both to [0, 1] range for comparison
                att_norm = (att_weights - att_weights.min()) / (att_weights.max() - att_weights.min() + EPS)
                mask_norm = (m - m.min()) / (m.max() - m.min() + EPS)
                
                # Attention consistency loss (MSE between normalized distributions)
                attention_consistency = F.mse_loss(mask_norm, att_norm.detach())
                loss = loss + self.attention_consistency_weight * attention_consistency

        if self.mask_features:
            m_feat = self.node_feat_mask.sigmoid()
            loss = loss + self.coff_node_feat_size * m_feat.sum()
            ent_feat = -m_feat * torch.log(m_feat + EPS) - (1 - m_feat) * torch.log(1 - m_feat + EPS)
            loss = loss + self.coff_node_feat_ent * ent_feat.mean()

        return loss

    def gnn_explainer_alg(self,
                          x: Tensor,
                          edge_index: Tensor,
                          ex_label: Tensor,
                          mask_features: bool = False,
                          **kwargs
                          ) -> Tensor:
        """Enhanced training algorithm with GAT optimizations"""
        # Initialize masks
        self.to(x.device)
        self.mask_features = mask_features

        # Extract GAT attention weights for guidance
        if self._is_gat_model and self.use_attention_guidance:
            self._extract_gat_attention(x, edge_index, **kwargs)
            init_method = "attention_guided"
        else:
            init_method = "normal"

        self.__set_masks__(x, edge_index, init=init_method)

        # Optimizer with parameter groups for different learning rates
        optimizer = torch.optim.Adam([
            {'params': [self.node_feat_mask], 'lr': self.lr * 0.1},  # Lower LR for feature mask
            {'params': [self.edge_mask], 'lr': self.lr}
        ])

        # Learning rate scheduler for better convergence
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)

        # Early stopping setup
        best_mask = None
        best_loss = float('inf')
        patience_counter = 0

        for epoch in range(1, self.epochs + 1):
            # Apply masks
            if mask_features:
                h = x * self.node_feat_mask.view(1, -1).sigmoid()
            else:
                h = x

            # Forward pass
            raw_preds = self.model(
                x=h,
                edge_index=edge_index,
                **self._model_forward_kwargs(kwargs),
            )
            loss = self.__loss__(raw_preds, ex_label)

            if epoch % 20 == 0 and debug:
                current_lr = scheduler.get_last_lr()[0]
                print(f'Epoch {epoch}: Loss:{loss.item():.6f}, LR:{current_lr:.6f}')

            # Early stopping check
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_mask = self.edge_mask.data.clone()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.early_stopping_patience:
                if debug:
                    print(f"Early stopping at epoch {epoch}, best loss: {best_loss:.6f}")
                break

            # Optimization step
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping (adjusted for GAT)
            if self.edge_mask.grad is not None:
                torch.nn.utils.clip_grad_value_([self.edge_mask], clip_value=1.0)
            if mask_features and self.node_feat_mask.grad is not None:
                torch.nn.utils.clip_grad_value_([self.node_feat_mask], clip_value=0.5)
                
            optimizer.step()
            scheduler.step()

        # Return best mask found during training
        return best_mask if best_mask is not None else self.edge_mask.data

    def forward(self, x, edge_index, mask_features=False, target_label=None, **kwargs):
        r"""
        Run the explainer for a specific graph instance with GAT optimizations.
        """
        super().forward(x=x, edge_index=edge_index, **kwargs)
        self.model.eval()
        self.__clear_masks__()

        try:
            # Handle self-loops based on GAT configuration
            add_self_loops_for_explainer = True
            if self._is_gat_model:
                for module in self.model.modules():
                    if isinstance(module, GATConv) and not module.add_self_loops:
                        add_self_loops_for_explainer = False
                        break

            if self.explain_graph:
                # 图分类解释必须与真实 forward 使用同一条 edge_index，
                # 否则 heterogeneous edge_types 与 edge_mask 长度会失配。
                self_loop_edge_index = edge_index
            elif add_self_loops_for_explainer:
                self_loop_edge_index, _ = add_remaining_self_loops(edge_index, num_nodes=self.num_nodes)
            else:
                self_loop_edge_index = edge_index

            # Only operate on a k-hop subgraph around `node_idx` for node classification
            if not self.explain_graph:
                self.node_idx = node_idx = kwargs.get('node_idx')
                assert node_idx is not None, 'An node explanation needs kwarg node_idx, but got None.'
                if isinstance(node_idx, torch.Tensor) and not node_idx.dim():
                    node_idx = node_idx.to(self.device).flatten()
                elif isinstance(node_idx, (int, list, tuple)):
                    node_idx = torch.tensor([node_idx], device=self.device, dtype=torch.int64).flatten()
                else:
                    raise TypeError(f'node_idx should be in types of int, list, tuple, '
                                    f'or torch.Tensor, but got {type(node_idx)}')
                self.subset, _, _, self.hard_edge_mask = subgraph(
                    node_idx, self.__num_hops__, self_loop_edge_index, relabel_nodes=True,
                    num_nodes=None, flow=self.__flow__())
                self.new_node_idx = torch.where(self.subset == node_idx)[0]

            if kwargs.get('edge_masks'):
                edge_masks = kwargs.pop('edge_masks')
                self.__set_masks__(x, self_loop_edge_index)
            else:
                # Calculate masks for all classes
                num_classes = kwargs.get('num_classes')
                if num_classes is None:
                    num_classes = getattr(self.model, 'num_classes', None)

                if target_label is not None:
                    if isinstance(target_label, torch.Tensor):
                        ex_labels = (target_label.to(self.device).view(-1)[0:1],)
                    else:
                        ex_labels = (torch.tensor([int(target_label)], device=self.device),)
                else:
                    if num_classes is None:
                        raise ValueError(
                            "num_classes is required for GNNExplainer when target_label is not provided."
                        )
                    labels = tuple(i for i in range(num_classes))
                    ex_labels = tuple(torch.tensor([label]).to(self.device) for label in labels)

                edge_masks = []
                for ex_label in ex_labels:
                    if target_label is None or ex_label.item() == target_label.item():
                        self.__clear_masks__()
                        self.__set_masks__(x, self_loop_edge_index)
                        edge_mask = self.gnn_explainer_alg(x, edge_index, ex_label, mask_features, **kwargs).sigmoid()
                        
                        if self._symmetric_edge_mask_indirect_graph:
                            edge_mask = symmetric_edge_mask_indirect_graph(self_loop_edge_index, edge_mask)

                        edge_masks.append(edge_mask)

            # Convert soft masks to hard masks based on sparsity
            hard_edge_masks = [self.control_sparsity(mask, sparsity=kwargs.get('sparsity'))
                               for mask in edge_masks]

            # Evaluate explanations
            with torch.no_grad():
                related_preds = self.eval_related_pred(
                    x,
                    edge_index,
                    hard_edge_masks,
                    **self._model_forward_kwargs(kwargs),
                )

            return edge_masks, hard_edge_masks, related_preds, self_loop_edge_index
        finally:
            self.__clear_masks__()

    def __repr__(self):
        return f'{self.__class__.__name__}(gat_optimized={self._is_gat_model})'


# 保持原始XGNNExplainer类的兼容性
class XGNNExplainer(GATEnhancedGNNExplainer):
    """Original XGNNExplainer maintained for backward compatibility"""
    def __init__(self, *args, **kwargs):
        # Filter out GAT-specific parameters for original behavior
        gat_kwargs = {}
        for key in ['attention_consistency_weight', 'use_attention_guidance', 'early_stopping_patience']:
            if key in kwargs:
                gat_kwargs[key] = kwargs.pop(key)
        
        # Disable GAT optimizations for original behavior
        gat_kwargs['attention_consistency_weight'] = 0.0
        gat_kwargs['use_attention_guidance'] = False
        
        super().__init__(*args, **kwargs, **gat_kwargs)
