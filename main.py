import os
import gc
import json
import random
import argparse
import warnings

import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, fbeta_score
import torch
import torch.nn.functional as F
from torch_geometric.nn import global_max_pool
from torch_geometric.loader import DataLoader
from torch_geometric.utils import *
import torch_scatter
from torch_scatter import scatter_softmax
from transformers import AdamW, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from transformers import RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer
from torch.cuda.amp import autocast, GradScaler  # AMP

from models.vul_detector import EnhancedDetector, model_diagnosis
from helpers import utils
from line_extract import get_dep_add_lines_bigvul
from graph_dataset import VulGraphDataset, collate
from models.gnnexplainer import XGNNExplainer, GATEnhancedGNNExplainer
from models.cfexplainer import CFExplainer
# from models.pgexplainer import XPGExplainer, PGExplainer_edges # 暂时注释，除非你用了 PGExplainer
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning)

class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss) # p_t
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def plot_training_curves(train_losses, valid_losses, train_accuracies, valid_accuracies, epochs, save_dir):
    """绘制训练和验证曲线"""
    os.makedirs(save_dir, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, valid_losses, 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(epochs, train_accuracies, 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, valid_accuracies, 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(save_dir, 'training_curves.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Training curves saved to: {plot_path}")
    
    data_path = os.path.join(save_dir, 'training_data.json')
    training_data = {
        'epochs': epochs,
        'train_losses': train_losses,
        'valid_losses': valid_losses,
        'train_accuracies': train_accuracies,
        'valid_accuracies': valid_accuracies
    }
    with open(data_path, 'w') as f:
        json.dump(training_data, f, indent=2)
    
    print(f"Training data saved to: {data_path}")

def set_seed(seed=42, deterministic=True):
    import os as _os, random as _random, numpy as _np, torch as _torch
    _random.seed(seed)
    _os.environ['PYTHONHASHSEED'] = str(seed)
    _np.random.seed(seed)
    _torch.manual_seed(seed)
    _torch.cuda.manual_seed(seed)

    if deterministic:
        try:
            _torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            _torch.use_deterministic_algorithms(True)
        _torch.backends.cudnn.benchmark = False
        _torch.backends.cudnn.deterministic = True
        _os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    else:
        try:
            _torch.use_deterministic_algorithms(False)
        except TypeError:
            pass
        _torch.backends.cudnn.benchmark = True

def calculate_metrics(y_true, y_pred):
    results = {
        'binary_precision': round(precision_score(y_true, y_pred, average='binary', zero_division=0), 4),
        'binary_recall': round(recall_score(y_true, y_pred, average='binary', zero_division=0), 4),
        'binary_f1': round(f1_score(y_true, y_pred, average='binary', zero_division=0), 4),
    }
    return results

def check_dataset_stats(dataloader, name):
    print(f"\n=== {name} Dataset Stats ===")
    all_labels = []
    for batch in dataloader:
        labels = batch.y.long()
        all_labels.extend(labels.cpu().numpy())
    
    all_labels = np.array(all_labels)
    print(f"Total samples: {len(all_labels)}")
    print(f"Positive samples: {np.sum(all_labels)} ({np.mean(all_labels)*100:.2f}%)")
    print(f"Negative samples: {len(all_labels) - np.sum(all_labels)}")
    return all_labels

def train(args, train_dataloader, valid_dataloader, test_dataloader, model):
    train_losses = []
    valid_losses = []
    train_accuracies = []
    valid_accuracies = []
    epochs_list = []

    if args.max_steps <= 0:
        args.max_steps = int(args.num_train_epochs * len(train_dataloader))
    if args.save_steps == -1:
        args.save_steps = len(train_dataloader)
    if args.warmup_steps == -1:
        args.warmup_steps = int(args.max_steps * 0.1)
    if args.logging_steps == -1:
        args.logging_steps = len(train_dataloader)

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_steps
    )

    pos, neg = 0, 0
    for bd in train_dataloader:
        y = bd.y.long()
        pos += int((y == 1).sum())
        neg += int((y == 0).sum())
    
    ratio = neg / max(pos, 1)
    gamma = getattr(args, "class_weight_gamma", 1.0)
    w0, w1 = 1.0, (ratio ** gamma)
    class_weights = torch.tensor([w0, w1], device=args.device, dtype=torch.float32)
    print(f"[info] CE class weights: {class_weights.tolist()} (gamma={gamma}, neg={neg}, pos={pos})")

    scaler = GradScaler(enabled=(args.device.type == "cuda"))

    best_valid_f1 = 0.0
    patience = 10
    patience_counter = 0
    best_epoch = 0
    min_epochs = 6
    
    print("***** Running training *****")
    print("  Num examples = {}".format(len(train_dataloader)))
    print("  Num Epochs = {}".format(args.num_train_epochs))
    
    global_step = args.start_step
    tr_loss, avg_loss, tr_num, train_loss = 0.0, 0.0, 0, 0.0

    model.zero_grad()
    
    for idx in range(args.start_epoch, int(args.num_train_epochs)):
        bar = tqdm(train_dataloader, total=len(train_dataloader))
        tr_num = 0
        train_loss = 0.0
        epoch_train_loss = 0.0
        epoch_train_samples = 0
        
        for step, batch_data in enumerate(bar):
            batch_data.to(args.device)
            x, edge_index, batch = batch_data.x, batch_data.edge_index.long(), batch_data.batch
            
            # --- [兼容] 处理 edge_types (RGCN/RGAT) ---
            edge_type = getattr(batch_data, 'edge_types', None)
            if edge_type is not None:
                edge_type = edge_type.to(args.device).long()
            elif args.gnn_model in ["RGCN", "RGAT"]:
                edge_type = torch.zeros(edge_index.shape[1], dtype=torch.long, device=args.device)

            # --- [兼容] 处理 edge_attr (Transformer/GAT) ---
            edge_attr = getattr(batch_data, 'edge_attr', None)
            if edge_attr is not None and args.use_edge_features:
                edge_attr = edge_attr.to(args.device)

            labels = batch_data.y.long()

            model.train()

            with autocast(enabled=(args.device.type == "cuda")):
                # 统一调用 forward，EnhancedDetector 内部会根据 args.gnn_model 自动分发
                logits = model(x, edge_index, batch, edge_attr=edge_attr, edge_types=edge_type)

                target = labels.long()
                
                if getattr(args, "use_focal", False):
                    criterion = FocalLoss(alpha=class_weights, gamma=args.focal_gamma)
                    loss = criterion(logits, target)
                else:
                    loss = F.cross_entropy(
                        logits, target,
                        weight=class_weights,
                        label_smoothing=getattr(args, "label_smoothing", 0.0)
                    )

                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

            if not torch.isfinite(loss):
                print(f"[warn] non-finite loss at epoch={idx} step={step}.")
                optimizer.zero_grad(set_to_none=True)
                scaler.update()
                continue

            scaler.scale(loss).backward()

            epoch_train_loss += loss.item() * len(labels)
            epoch_train_samples += len(labels)
            tr_loss += loss.item()
            tr_num += 1
            train_loss += loss.item()
            avg_loss = round(train_loss / tr_num, 5)
            bar.set_description("epoch {} loss {}".format(idx, avg_loss))

            if (step + 1) % args.gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

        bar.close()
            
        avg_epoch_train_loss = epoch_train_loss / epoch_train_samples if epoch_train_samples > 0 else 0
        train_losses.append(avg_epoch_train_loss)
        
        valid_loss = calculate_validation_loss(args, valid_dataloader, model, class_weights)
        valid_losses.append(valid_loss)
        
        train_metrics = evaluate(args, train_dataloader, model)
        train_acc = train_metrics['eval_acc']
        
        valid_metrics = evaluate(args, valid_dataloader, model)
        valid_acc = valid_metrics['eval_acc']
        valid_f1 = valid_metrics['binary_f1']
        
        train_accuracies.append(train_acc)
        valid_accuracies.append(valid_acc)
        epochs_list.append(idx)
        
        print(f"Epoch {idx}: Train Loss = {avg_epoch_train_loss:.4f}, Valid Loss = {valid_loss:.4f}")
        print(f"Epoch {idx}: Valid Acc = {valid_acc:.4f}, Valid F1 = {valid_f1:.4f}")

        is_best = valid_f1 > best_valid_f1
        
        if is_best:
            best_valid_f1 = valid_f1
            best_epoch = idx
            
            checkpoint_prefix = 'checkpoint-best-f1'
            output_dir = os.path.join(args.model_checkpoint_dir, '{}'.format(checkpoint_prefix))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            model_to_save = model.module if hasattr(model, 'module') else model
            output_path = os.path.join(output_dir, '{}'.format('model.bin'))
            torch.save(model_to_save.state_dict(), output_path)
            print(f"Saving best model (valid_f1={valid_f1:.4f}) to {output_path}")

        if idx >= min_epochs:
            if is_best:
                patience_counter = 0
            else:
                patience_counter += 1
                print(f"Validation F1 did not improve. Patience: {patience_counter}/{patience}")

            if patience_counter >= patience:
                print(f"Early stopping at epoch {idx}. Best epoch was {best_epoch} with valid_f1={best_valid_f1:.4f}")
                break
        else:
            if not is_best:
                print(f"Warmup epoch {idx}: F1 did not improve ({valid_f1:.4f} <= {best_valid_f1:.4f})")

    plot_training_curves(train_losses, valid_losses, train_accuracies, valid_accuracies, epochs_list, args.model_checkpoint_dir)
    return global_step, tr_loss / global_step if global_step > 0 else 0

def evaluate(args, eval_dataloader, model, threshold=None, return_details=False):
    model.eval()
    all_probs, all_labels, all_preds_list = [], [], []
    thr = args.decision_threshold if threshold is None else threshold

    with torch.no_grad():
        for step, batch_data in enumerate(eval_dataloader):
            batch_data.to(args.device)
            x, edge_index, batch = batch_data.x, batch_data.edge_index.long(), batch_data.batch
            
            edge_type = getattr(batch_data, 'edge_types', None)
            if edge_type is not None:
                edge_type = edge_type.to(args.device).long()
            elif args.gnn_model in ["RGCN", "RGAT"]:
                edge_type = torch.zeros(edge_index.shape[1], dtype=torch.long, device=args.device)

            edge_attr = getattr(batch_data, 'edge_attr', None)
            if edge_attr is not None and args.use_edge_features:
                edge_attr = edge_attr.to(args.device)
            
            labels = batch_data.y.long()

            # 调用 model, EnhancedDetector 会处理 edge_attr 和 edge_types
            logits = model(x, edge_index, batch, edge_attr=edge_attr, edge_types=edge_type)
                
            probs_pos = F.softmax(logits, dim=-1)[:, args.positive_class_id]
            preds = (probs_pos >= thr).long()

            all_probs.append(probs_pos.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_preds_list.append(preds.cpu().numpy())

    all_probs = np.concatenate(all_probs, 0)
    all_labels = np.concatenate(all_labels, 0)
    all_preds = np.concatenate(all_preds_list, 0)

    eval_acc = float(np.mean(all_labels == all_preds))
    p = precision_score(all_labels, all_preds, zero_division=0)
    r = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    result = {
        "eval_acc": round(eval_acc, 4),
        "binary_precision": round(p, 4),
        "binary_recall": round(r, 4),
        "binary_f1": round(f1, 4),
        "threshold": round(thr, 4),
    }

    if return_details:
        result["labels"] = all_labels
        result["preds"] = all_preds
        result["probs"] = all_probs

    return result

def tune_decision_threshold(args, valid_dataloader, model, start=0.20, end=0.80, steps=31):
    grid = np.linspace(start, end, steps)
    best_thr, best_score, best_pack = args.decision_threshold, -1.0, None
    beta = getattr(args, "auto_threshold_beta", 1.0)
    print(f"[tune] searching threshold (F-beta, beta={beta}) in [{start:.2f}, {end:.2f}] ...")
    for t in grid:
        res = evaluate(args, valid_dataloader, model, threshold=float(t), return_details=True)
        score = fbeta_score(res["labels"], res["preds"], beta=beta, zero_division=0)
        if score > best_score:
            best_thr, best_score, best_pack = float(t), score, res
    print(f"[tune] best thr={best_thr:.3f} (F{beta}={best_score:.4f})")
    return best_thr

def calculate_validation_loss(args, valid_dataloader, model, class_weights):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for batch_data in valid_dataloader:
            batch_data.to(args.device)
            x, edge_index, batch = batch_data.x, batch_data.edge_index.long(), batch_data.batch
            
            edge_type = getattr(batch_data, 'edge_types', None)
            if edge_type is not None:
                edge_type = edge_type.to(args.device).long()
            elif args.gnn_model in ["RGCN", "RGAT"]:
                edge_type = torch.zeros(edge_index.shape[1], dtype=torch.long, device=args.device)
            
            edge_attr = getattr(batch_data, 'edge_attr', None)
            if edge_attr is not None and args.use_edge_features:
                edge_attr = edge_attr.to(args.device)
            
            labels = batch_data.y.long()
            
            logits = model(x, edge_index, batch, edge_attr=edge_attr, edge_types=edge_type)
            
            target = labels.long()
            if getattr(args, "use_focal", False):
                criterion = FocalLoss(alpha=class_weights, gamma=args.focal_gamma)
                loss = criterion(logits, target)
            else:
                loss = F.cross_entropy(
                    logits, target,
                    weight=class_weights,
                    label_smoothing=getattr(args, "label_smoothing", 0.0)
                )
            
            total_loss += loss.item() * len(labels)
            total_samples += len(labels)
    
    return total_loss / total_samples

def calibrate_temperature(args, valid_dataloader, model, T_grid=np.linspace(0.5, 3.0, 26)):
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for bd in valid_dataloader:
            bd.to(args.device)
            x, edge_index, batch = bd.x, bd.edge_index.long(), bd.batch
            
            labels = bd.y.long()
            
            edge_type = getattr(bd, 'edge_types', None)
            if edge_type is not None:
                edge_type = edge_type.to(args.device).long()
            elif args.gnn_model in ["RGCN", "RGAT"]:
                edge_type = torch.zeros(edge_index.shape[1], dtype=torch.long, device=args.device)
            
            edge_attr = getattr(bd, 'edge_attr', None)
            if edge_attr is not None and args.use_edge_features:
                edge_attr = edge_attr.to(args.device)
                
            logits = model(x, edge_index, batch, edge_attr=edge_attr, edge_types=edge_type)
            all_logits.append(logits.detach().cpu())
            all_labels.append(labels.detach().cpu())
    logits = torch.cat(all_logits, 0)
    labels = torch.cat(all_labels, 0)

    best_T, best_nll = 1.0, float("inf")
    for T in T_grid:
        nll = F.cross_entropy(logits / T, labels, reduction="mean").item()
        if nll < best_nll:
            best_nll, best_T = nll, float(T)
    print(f"[calib] best temperature T={best_T:.3f}")
    return best_T

def gen_exp_lines(edge_index, edge_weight, index, num_nodes, lines):
    temp = torch.zeros_like(edge_weight).to(edge_index.device)
    temp[index] = edge_weight[index]

    adj_mask = torch.sparse_coo_tensor(edge_index, temp, [num_nodes, num_nodes])
    adj_mask_binary = to_dense_adj(edge_index[:, temp != 0], max_num_nodes=num_nodes).squeeze(0)

    out_degree = torch.sum(adj_mask_binary, dim=1)
    out_degree[out_degree == 0] = 1e-8
    in_degree = torch.sum(adj_mask_binary, dim=0)
    in_degree[in_degree == 0] = 1e-8

    line_importance_init = torch.ones(num_nodes).unsqueeze(-1).to(edge_index.device)
    line_importance_out = torch.spmm(adj_mask, line_importance_init) / out_degree.unsqueeze(-1)
    line_importance_in = torch.spmm(adj_mask.T, line_importance_init) / in_degree.unsqueeze(-1)
    line_importance = line_importance_out + line_importance_in

    ret = sorted(list(zip(line_importance.squeeze(-1).cpu().numpy(), lines)), reverse=True)
    filtered_ret = [int(i[1]) for i in ret if i[0] > 0]
    return filtered_ret

def eval_exp(exp_saved_path, model, correct_lines, args):
    graph_exp_list = torch.load(exp_saved_path, map_location=args.device)
    print("Number of explanations:", len(graph_exp_list))
    if len(graph_exp_list) == 0:
        print("[warn] explanation cache is empty; skip explanation metrics.")
        print("Accuracy:", 0.0)
        print("Precision:", 0.0)
        print("Recall:", 0.0)
        print("F1:", 0.0)
        print("Probability of Necessity:", 0.0)
        return

    accuracy_cnt = 0
    precisions, recalls, F1s, pn = [], [], [], []

    for graph in graph_exp_list:
        graph.to(args.device)
        x, edge_index = graph.x, graph.edge_index.long()
        edge_weight = graph.edge_weight
        pred, batch = graph.pred, graph.batch
        
        edge_type = getattr(graph, 'edge_types', None)
        if edge_type is not None:
             edge_type = edge_type.to(args.device).long()
        elif args.gnn_model in ["RGCN", "RGAT"]:
             edge_type = torch.zeros(edge_index.shape[1], dtype=torch.long, device=args.device)
        
        edge_attr = getattr(graph, 'edge_attr', None)
        if edge_attr is not None and args.use_edge_features:
            edge_attr = edge_attr.to(args.device)

        if hasattr(graph, 'sample_id'):
            sampleid = graph.sample_id.max().int().item()
        elif hasattr(graph, '_SAMPLE'):
            sampleid = graph._SAMPLE.max().int().item()
        else:
            continue

        if int(sampleid) not in correct_lines:
            continue

        exp_label_lines = list(correct_lines[int(sampleid)]["removed"])

        th = getattr(args, "exp_edge_thresh", -1.0)
        if th >= 0:
            mask = edge_weight >= th
            if int(mask.sum()) > 0:
                index = torch.where(mask)[0]
            else:
                k = min(args.KM, edge_weight.numel())
                _, index = torch.topk(edge_weight, k=k)
        else:
            if len(edge_weight) > args.KM:
                _, index = torch.topk(edge_weight, k=args.KM)
            else:
                index = torch.arange(edge_weight.shape[0])

        if hasattr(graph, 'line_index'):
             lines = graph.line_index.cpu().numpy()
        elif hasattr(graph, '_LINE'):
             lines = graph._LINE.cpu().numpy()
        else:
             lines = np.arange(x.shape[0])

        exp_lines = gen_exp_lines(edge_index, edge_weight, index, x.shape[0], lines)

        if any((l in exp_label_lines) for l in exp_lines):
            accuracy_cnt += 1

        hit = sum(1 for l in exp_lines if l in exp_label_lines)
        if hit > 0 and len(exp_lines) > 0 and len(exp_label_lines) > 0:
            precision = hit / len(exp_lines)
            recall = hit / len(exp_label_lines)
            f1 = (2 * precision * recall) / (precision + recall)
        else:
            precision, recall, f1 = 0.0, 0.0, 0.0

        precisions.append(precision)
        recalls.append(recall)
        F1s.append(f1)

        temp = torch.ones_like(edge_weight)
        temp[index] = 0
        cf_index = (temp != 0)

        fac_edge_index = edge_index[:, index]
        
        # 反事实解释也需要适配 edge_attr / edge_types
        # 注意：这里简化处理，假设 edge_attr 不随边索引改变太复杂，暂时传入原始对应切片
        # GNNExplainer 主要是对 Edge Mask，所以特征不变
        if args.gnn_model in ["RGCN", "RGAT"]:
            fac_edge_types = edge_type[index]
            fac_logits = model(x, fac_edge_index, batch, edge_types=fac_edge_types)
        elif args.gnn_model in ["GATv2", "Transformer"] and args.use_edge_features and edge_attr is not None:
            # 这里的 edge_attr 需要对应 index 切片
            fac_edge_attr = edge_attr[index]
            fac_logits = model(x, fac_edge_index, batch, edge_attr=fac_edge_attr)
        else:
            fac_logits = model(x, fac_edge_index, batch)
        
        fac_pred = F.one_hot(torch.argmax(fac_logits, dim=-1), 2)[0][args.positive_class_id]

        cf_edge_index = edge_index[:, cf_index]
        if args.gnn_model in ["RGCN", "RGAT"]:
            cf_edge_types = edge_type[cf_index]
            cf_logits = model(x, cf_edge_index, batch, edge_types=cf_edge_types)
        elif args.gnn_model in ["GATv2", "Transformer"] and args.use_edge_features and edge_attr is not None:
            cf_edge_attr = edge_attr[cf_index]
            cf_logits = model(x, cf_edge_index, batch, edge_attr=cf_edge_attr)
        else:
            cf_logits = model(x, cf_edge_index, batch)

        cf_pred = F.one_hot(torch.argmax(cf_logits, dim=-1), 2)[0][args.positive_class_id]

        pn.append(int(cf_pred != pred))

    N = max(1, len(graph_exp_list))
    print("Accuracy:", round(accuracy_cnt / N, 4))
    print("Precision:", round(float(np.mean(precisions)), 4))
    print("Recall:", round(float(np.mean(recalls)), 4))
    print("F1:", round(float(np.mean(F1s)), 4))
    print("Probability of Necessity:", round(float(sum(pn)) / len(pn) if len(pn) else 0.0, 4))

def gnnexplainer_run(args, model, test_dataset, correct_lines):
    graph_exp_list = []
    visited_sampleids = set()
    print(f"Starting GNNExplainer on {len(test_dataset)} test samples")
    
    if args.gnn_model == "GAT":
        explainer = GATEnhancedGNNExplainer(model=model, explain_graph=True, epochs=args.gnnexplainer_epochs, lr=args.gnnexplainer_lr)
    else:
        explainer = XGNNExplainer(model=model, explain_graph=True, epochs=args.gnnexplainer_epochs, lr=args.gnnexplainer_lr)
    
    explainer.device = args.device

    for graph in test_dataset:
        explainer.__clear_masks__()
        graph.to(args.device)
        x, edge_index, batch = graph.x, graph.edge_index.long(), graph.batch
        
        edge_type = getattr(graph, 'edge_types', None)
        if edge_type is not None:
             edge_type = edge_type.to(args.device).long()
        elif args.gnn_model in ["RGCN", "RGAT"]:
             edge_type = torch.zeros(edge_index.shape[1], dtype=torch.long, device=args.device)
        
        edge_attr = getattr(graph, 'edge_attr', None)
        if edge_attr is not None and args.use_edge_features:
            edge_attr = edge_attr.to(args.device)

        # 保持 edge_index 和属性的同步，不进行 coalesce
        if edge_index.shape[1] == 0: continue
            
        label = graph.y.long()[0]
        
        if hasattr(graph, 'sample_id'):
            sampleid = graph.sample_id.max().int().item()
        elif hasattr(graph, '_SAMPLE'):
            sampleid = graph._SAMPLE.max().int().item()
        else:
            continue
        
        if sampleid not in correct_lines or sampleid in visited_sampleids: continue

        # 检查长度一致性
        if edge_type is not None and edge_type.shape[0] != edge_index.shape[1]:
            continue

        # model 调用需要适配 edge_attr
        prob = model(x, edge_index, batch, edge_attr=edge_attr, edge_types=edge_type)
        exp_prob_label = F.one_hot(torch.argmax(prob, dim=-1), 2)
        
        if label != args.positive_class_id or prob[0][args.positive_class_id] < prob[0][1 - args.positive_class_id]:
            continue
            
        print(f"解释样本: {sampleid}")
        try:
            # GNNExplainer 默认只处理 x, edge_index
            # edge_attr 和 edge_types 是在 model forward 中使用的，Explainer 会优化 mask
            target_label = torch.argmax(prob, dim=-1).detach()
            edge_masks = explainer(
                x,
                edge_index,
                batch=batch,
                edge_attr=edge_attr,
                edge_types=edge_type,
                num_classes=args.num_classes,
                target_label=target_label,
            )
            if isinstance(edge_masks, tuple): edge_mask = edge_masks[0]
            else: edge_mask = edge_masks

            if isinstance(edge_mask, (list, tuple)):
                pred_idx = int(torch.argmax(exp_prob_label, dim=-1).item())
                edge_weight = edge_mask[0] if len(edge_mask) == 1 else edge_mask[pred_idx]
            else:
                edge_weight = edge_mask
            graph.__setitem__("edge_weight", torch.Tensor(edge_weight.detach().cpu()))
            graph.__setitem__("pred", exp_prob_label[0][args.positive_class_id])
            graph_exp_list.append(graph.detach().clone().cpu())
            visited_sampleids.add(sampleid)
        except Exception as e:
            print(f"Error processing {sampleid}: {e}")
            explainer.__clear_masks__()
            continue

    return graph_exp_list

def cfexplainer_run(args, model, test_dataset, correct_lines):
    graph_exp_list = []
    visited_sampleids = set()
    model.eval()
    
    # 简单的 dummy 初始化检测
    try:
        with torch.no_grad():
            sample_graph = next(iter(test_dataset))
            sample_graph.to(args.device)
            x, edge_index, batch = sample_graph.x, sample_graph.edge_index.long(), sample_graph.batch
            et = getattr(sample_graph, 'edge_types', torch.zeros(edge_index.shape[1], dtype=torch.long, device=args.device))
            ea = getattr(sample_graph, 'edge_attr', None)
            if ea is not None and args.use_edge_features: ea = ea.to(args.device)
            _ = model(x, edge_index, batch, edge_attr=ea, edge_types=et)
    except Exception as e:
        print(f"Model init check failed (ignorable): {e}")

    explainer = CFExplainer(model=model, explain_graph=True, epochs=args.cfexp_epochs, lr=args.cfexp_lr, alpha=args.cfexp_alpha, L1_dist=args.cfexp_L1)
    explainer.device = args.device

    for graph in test_dataset:
        graph.to(args.device)
        x, edge_index, batch = graph.x, graph.edge_index.long(), graph.batch
        
        edge_type = getattr(graph, 'edge_types', None)
        if edge_type is not None:
             edge_type = edge_type.to(args.device).long()
        elif args.gnn_model in ["RGCN", "RGAT"]:
             edge_type = torch.zeros(edge_index.shape[1], dtype=torch.long, device=args.device)
        
        edge_attr = getattr(graph, 'edge_attr', None)
        if edge_attr is not None and args.use_edge_features:
            edge_attr = edge_attr.to(args.device)

        if edge_index.shape[1] == 0: continue

        label = graph.y.long()[0]
        
        if hasattr(graph, 'sample_id'):
            sampleid = graph.sample_id.max().int().item()
        elif hasattr(graph, '_SAMPLE'):
            sampleid = graph._SAMPLE.max().int().item()
        else:
            continue

        if sampleid not in correct_lines or sampleid in visited_sampleids: continue

        prob = model(x, edge_index, batch, edge_attr=edge_attr, edge_types=edge_type)
        exp_prob_label = F.one_hot(torch.argmax(prob, dim=-1), 2)
        if label != args.positive_class_id or prob[0][args.positive_class_id] < prob[0][1 - args.positive_class_id]:
            continue
        print(f"解释样本: {sampleid}")

        try:
            edge_mask = explainer(x, edge_index)
            if isinstance(edge_mask, tuple): edge_mask = edge_mask[0]
            
            edge_weight = 1 - edge_mask[torch.argmax(exp_prob_label, dim=-1)]
            graph.__setitem__("edge_weight", torch.Tensor(edge_weight.detach().cpu()))
            graph.__setitem__("pred", exp_prob_label[0][args.positive_class_id])
            graph_exp_list.append(graph.detach().clone().cpu())
            visited_sampleids.add(sampleid)
        except Exception as e:
            print(f"Error {sampleid}: {e}")
            continue

    return graph_exp_list

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--num_heads", default=4, type=int, help="Heads for GAT")
    parser.add_argument("--gatv2_heads", default=4, type=int, help="Heads for GATv2")
    parser.add_argument("--gatv2_concatenate", action='store_true', default=True)
    parser.add_argument("--use_edge_features", action='store_true')
    parser.add_argument("--use_hierarchical_pooling", action='store_true')
    parser.add_argument("--pool_ratio", default=0.8, type=float)
    parser.add_argument("--hierarchical_pool_type", default="topk", type=str)

    # [关键修改] 增加了 Transformer 和 GatedGraphConv 选项
    parser.add_argument("--gnn_model", default="RGCN", type=str, 
                       choices=["GCN", "GCNConv", "GatedGraph", "GatedGraphConv", "GraphConv", "GAT", "GATv2", "RGCN", "RGAT", "Transformer"], 
                       help="GNN core.")
    
    parser.add_argument("--num_relations", default=20, type=int, help="Max edge types")     
    parser.add_argument("--use_focal", action="store_true")
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    
    parser.add_argument("--virtual_node", action="store_true")
    parser.add_argument("--use_jk", action="store_true")
    parser.add_argument("--jk_mode", type=str, default="mean")
    parser.add_argument("--dropedge_p", type=float, default=0.0)
    parser.add_argument("--use_graphnorm", action="store_true")
    parser.add_argument("--pna_deg", type=str, default=None)
    parser.add_argument("--edge_attr_dim", type=int, default=0)
    parser.add_argument("--role_feat_dim", type=int, default=0)
    
    parser.add_argument('--cuda_id', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--deterministic', action='store_true')

    parser.add_argument("--model_checkpoint_dir", default="saved_models", type=str)
    parser.add_argument("--gnn_hidden_size", default=256, type=int)
    parser.add_argument("--gnn_feature_dim_size", default=769, type=int)
    parser.add_argument("--residual", action='store_true', default=True)
    parser.add_argument("--num_gnn_layers", default=2, type=int)
    parser.add_argument("--num_ggnn_steps", default=3, type=int)
    parser.add_argument("--ggnn_aggr", default="add", type=str)
    parser.add_argument("--gin_eps", default=0., type=float)
    parser.add_argument("--gin_train_eps", action='store_true')
    parser.add_argument("--gconv_aggr", default="mean", type=str)
    parser.add_argument("--dropout_rate", default=0.3, type=float)
    parser.add_argument("--num_classes", default=2, type=int)
    parser.add_argument("--positive_class_id", default=1, type=int)

    parser.add_argument("--num_train_epochs", default=50, type=float)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--learning_rate", default=1e-3, type=float)
    parser.add_argument("--max_steps", default=-1, type=int)
    parser.add_argument("--warmup_steps", default=-1, type=int)
    parser.add_argument('--logging_steps', type=int, default=-1)
    parser.add_argument('--save_steps', type=int, default=-1)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--do_train", action='store_true')
    parser.add_argument("--do_test", action='store_true')
    parser.add_argument("--do_explain", action='store_true')

    parser.add_argument("--ipt_method", default="gnnexplainer", type=str)
    parser.add_argument("--ipt_update", action='store_true')
    parser.add_argument("--KM", default=8, type=int)
    parser.add_argument("--cfexp_L1", action='store_true')
    parser.add_argument("--cfexp_alpha", default=0.9, type=float)
    parser.add_argument("--hyper_para", action='store_true')
    parser.add_argument("--case_sample_ids", nargs='+')
    
    parser.add_argument('--overwrite_explain', action='store_true')
    parser.add_argument('--cfexp_epochs', type=int, default=800)
    parser.add_argument('--cfexp_lr', type=float, default=5e-2)
    parser.add_argument('--explain_cache_tag', type=str, default='')
    parser.add_argument('--exp_edge_thresh', type=float, default=-1.0)

    parser.add_argument("--decision_threshold", type=float, default=0.50)
    parser.add_argument("--auto_threshold", action="store_true")
    parser.add_argument("--auto_threshold_beta", type=float, default=1.0)

    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--class_weight_gamma", type=float, default=1.0)

    parser.add_argument("--calibrate_temp", action="store_true")
    
    parser.add_argument('--exp_top_lines', type=int, default=3)
    parser.add_argument('--exp_line_agg', type=str, default='sum', choices=['sum', 'max'])
    
    parser.add_argument("--gnnexplainer_epochs", default=1000, type=int)
    parser.add_argument("--gnnexplainer_lr", default=0.05, type=float)
    
    parser.add_argument("--graph_pooling", default="attn", type=str, choices=["mean", "sum", "max", "attn", "set2set", "unet"])

    parser.add_argument("--mask_mode", default="aligned", type=str, choices=["aligned", "all_ones", "random"])
    parser.add_argument("--mask_seed", default=42, type=int)

    args = parser.parse_args()

    device = torch.device("cuda:" + str(args.cuda_id) if torch.cuda.is_available() else "cpu")
    args.device = device
    args.model_checkpoint_dir = str(utils.cache_dir() / f"{args.model_checkpoint_dir}" / args.gnn_model)

    need_det = args.deterministic
    if args.graph_pooling.lower() == 'attn' and not args.deterministic:
        need_det = False
    set_seed(args.seed, deterministic=need_det)

    args.start_epoch = 0
    args.start_step = 0

    def _dataset_processed_file(partition: str) -> str:
        if args.mask_mode == "aligned":
            suffix = ""
        elif args.mask_mode == "random":
            suffix = f"_random_{args.mask_seed}"
        else:
            suffix = f"_{args.mask_mode}"
        return str(utils.processed_dir() / "vul_graph_dataset" / f"{partition}_processed_target{suffix}" / "data.pt")

    encoder = None
    tokenizer = None
    needs_build = any(not os.path.exists(_dataset_processed_file(p)) for p in ["train", "val", "test"])
    if needs_build:
        model_name_or_path = str(utils.processed_dir() / "graphcodebert-base")
        print(f"[info] processed graph dataset not found, loading encoder/tokenizer from: {model_name_or_path}")
        config = RobertaConfig.from_pretrained(model_name_or_path, local_files_only=True)
        tokenizer = RobertaTokenizer.from_pretrained(model_name_or_path, local_files_only=True)
        encoder = RobertaForSequenceClassification.from_pretrained(
            model_name_or_path,
            config=config,
            local_files_only=True,
        )

    model = EnhancedDetector(args)
    model.to(args.device)

    train_dataset = VulGraphDataset(root=str(utils.processed_dir() / "vul_graph_dataset"), partition='train', mask_mode=args.mask_mode, mask_seed=args.mask_seed, encoder=encoder, tokenizer=tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate, num_workers=4, pin_memory=True)
    
    valid_dataset = VulGraphDataset(root=str(utils.processed_dir() / "vul_graph_dataset"), partition='val', mask_mode=args.mask_mode, mask_seed=args.mask_seed, encoder=encoder, tokenizer=tokenizer)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate, num_workers=4, pin_memory=True)
    
    test_dataset = VulGraphDataset(root=str(utils.processed_dir() / "vul_graph_dataset"), partition='test', mask_mode=args.mask_mode, mask_seed=args.mask_seed, encoder=encoder, tokenizer=tokenizer)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate, num_workers=4, pin_memory=True)

    check_dataset_stats(train_dataloader, "Training")
    check_dataset_stats(valid_dataloader, "Validation")
    check_dataset_stats(test_dataloader, "Test")

    print("\n=== Model Diagnosis ===")
    sample_batch = next(iter(train_dataloader))
    model_diagnosis(model, args.device)

    if args.do_train:
        train(args, train_dataloader, valid_dataloader, test_dataloader, model)

    if args.do_test:
        checkpoint_prefix = 'checkpoint-best-f1/model.bin'
        model_checkpoint_dir = os.path.join(args.model_checkpoint_dir, '{}'.format(checkpoint_prefix))
        checkpoint_state = torch.load(model_checkpoint_dir, map_location=args.device)
        load_result = model.load_state_dict(checkpoint_state, strict=False)
        if load_result.missing_keys:
            print(f"[warn] checkpoint missing keys, using current model defaults: {load_result.missing_keys}")
        if load_result.unexpected_keys:
            print(f"[warn] checkpoint has unexpected keys that were ignored: {load_result.unexpected_keys}")
        model.to(args.device)

        if getattr(args, "calibrate_temp", False):
            args.temperature = calibrate_temperature(args, valid_dataloader, model)
        else:
            args.temperature = 1.0

        if getattr(args, "auto_threshold", False):
            best_thr = tune_decision_threshold(args, valid_dataloader, model, start=0.20, end=0.80, steps=31)
            args.decision_threshold = best_thr
            print(f"[info] using tuned threshold={best_thr:.3f} for test set")

        test_result = evaluate(args, test_dataloader, model)
        print("***** Test results *****")
        for key in sorted(test_result.keys()):
            print("  {} = {}".format(key, str(round(test_result[key], 4))))

        if args.do_explain:
            correct_lines = get_dep_add_lines_bigvul()
            explain_dir = str(utils.cache_dir() / "explanations" / f"{args.gnn_model}")
            os.makedirs(explain_dir, exist_ok=True)

            tag = f"_tag{args.explain_cache_tag}" if getattr(args, "explain_cache_tag", "") else ""
            ipt_save = os.path.join(
                explain_dir,
                f"{args.gnn_model}_{args.graph_pooling}_{args.ipt_method}_{args.KM}.pt"
            )

            print("Size of test dataset:", len(test_dataset))
            model.eval()
            for p in model.parameters(): p.requires_grad = False

            need_recompute = args.overwrite_explain or args.ipt_update or (not os.path.isfile(ipt_save))
            if (not need_recompute) and os.path.isfile(ipt_save):
                if len(torch.load(ipt_save, map_location="cpu")) == 0:
                    print(f"[warn] cached explanation file is empty, recomputing: {ipt_save}")
                    need_recompute = True
            if need_recompute:
                print(f"[explain] recompute -> {ipt_save}")
                if args.ipt_method == "gnnexplainer":
                    graph_exp_list = gnnexplainer_run(args, model, test_dataset, correct_lines)
                elif args.ipt_method == "cfexplainer":
                    graph_exp_list = cfexplainer_run(args, model, test_dataset, correct_lines)
                else:
                    graph_exp_list = []
                    print(f"Explanation method {args.ipt_method} not fully implemented in main fix yet.")
                
                # 确保保存逻辑健壮
                if graph_exp_list:
                    torch.save(graph_exp_list, ipt_save)
                    print(f"[explain] saved {len(graph_exp_list)} explanations to {ipt_save}")
                else:
                     print(f"[Warn] No explanations generated, saving empty list to {ipt_save}")
                     torch.save([], ipt_save)
            else:
                print(f"[explain] use cached: {ipt_save}")

            eval_exp(ipt_save, model, correct_lines, args)

if __name__ == "__main__":
    main()
