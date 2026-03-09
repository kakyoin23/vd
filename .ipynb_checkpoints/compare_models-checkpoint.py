import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch_geometric.utils import add_remaining_self_loops, coalesce
import torch_scatter
from sklearn.metrics import precision_score, recall_score, f1_score, fbeta_score

from models.vul_detector import Detector
from graph_dataset import VulGraphDataset, collate
from helpers import utils
from main import train


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


def evaluate(args, eval_dataloader, model, threshold=None, return_details=False):
    model.eval()
    all_probs, all_labels, all_preds_list = [], [], []
    thr = args.decision_threshold if threshold is None else threshold
    temp = getattr(args, "temperature", 1.0)

    with torch.no_grad():
        for batch_data in eval_dataloader:
            batch_data.to(args.device)
            x, edge_index, batch = batch_data.x, batch_data.edge_index.long(), batch_data.batch
            edge_index, _ = add_remaining_self_loops(edge_index, num_nodes=x.shape[0])
            edge_index = coalesce(edge_index)
            labels = torch_scatter.segment_csr(batch_data._VULN, batch_data.ptr).long()
            labels[labels != 0] = 1

            logits = model(x, edge_index, batch) / temp
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
    print(f"[tune] best thr={best_thr:.3f}  (F{beta}={best_score:.4f}, "
          f"P={best_pack['binary_precision']:.4f}, R={best_pack['binary_recall']:.4f}, "
          f"ACC={best_pack['eval_acc']:.4f})")
    return best_thr


def compare_models(args):
    model_names = ["SAGEConv", "GCNConv", "GATv2Conv", "GINConv", "GraphConv"]
    summary = {}

    # 数据集
    train_dataset = VulGraphDataset(root=str(utils.processed_dir() / "vul_graph_dataset"), partition='train')
    valid_dataset = VulGraphDataset(root=str(utils.processed_dir() / "vul_graph_dataset"), partition='val')
    test_dataset  = VulGraphDataset(root=str(utils.processed_dir() / "vul_graph_dataset"), partition='test')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,  collate_fn=collate,
                              num_workers=4, pin_memory=True, persistent_workers=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate,
                              num_workers=4, pin_memory=True, persistent_workers=True)
    test_loader  = DataLoader(test_dataset,  batch_size=args.batch_size, shuffle=False, collate_fn=collate,
                              num_workers=4, pin_memory=True, persistent_workers=True)

    for model_name in model_names:
        print(f"\nTraining and evaluating model: {model_name}")

        # 每个模型单独的 checkpoint 目录
        args.gnn_model = model_name
        args.model_checkpoint_dir = str(utils.cache_dir() / f"{args.model_checkpoint_root}" / args.gnn_model)

        # 初始化模型
        model = Detector(args).to(args.device)

        # 训练
        if args.do_train:
            train(args, train_loader, valid_loader, test_loader, model)

        # 加载 best-acc checkpoint
        ckpt_path = os.path.join(args.model_checkpoint_dir, 'checkpoint-best-acc', 'model.bin')
        model.load_state_dict(torch.load(ckpt_path, map_location=args.device))
        model.to(args.device)

        # 自动调阈值（可选）
        if args.auto_threshold:
            best_thr = tune_decision_threshold(args, valid_loader, model, start=0.20, end=0.80, steps=31)
            args.decision_threshold = best_thr
            print(f"[info] using tuned threshold={best_thr:.3f} for test set")

        # 测试
        test_res = evaluate(args, test_loader, model)
        print(f"***** Test results for {model_name} *****")
        for k in sorted(test_res.keys()):
            print(f"  {k} = {test_res[k]}")

        summary[model_name] = {
            "F1": test_res["binary_f1"],
            "Precision": test_res["binary_precision"],
            "Recall": test_res["binary_recall"],
            "Accuracy": test_res["eval_acc"],
            "Best Threshold": test_res["threshold"],
        }

    print("\n***** Model Comparison Summary *****")
    for name, res in summary.items():
        print(f"Model: {name}")
        print(f"  Best F1: {res['F1']}  Threshold: {res['Best Threshold']}")
        print(f"  Precision: {res['Precision']}  Recall: {res['Recall']}")
        print(f"  Accuracy: {res['Accuracy']}")
        print("-"*40)


def main():
    parser = argparse.ArgumentParser()
    # 设备/随机性
    parser.add_argument('--cuda_id', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--deterministic', action='store_true')
    # 数据/模型通用
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_train_epochs', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=5e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--adam_epsilon', type=float, default=1e-8)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--save_steps', type=int, default=-1)
    parser.add_argument('--warmup_steps', type=int, default=-1)
    parser.add_argument('--logging_steps', type=int, default=-1)
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_test', action='store_true')
    # Detector 需要的结构参数（与 models/vul_detector.py 对齐）
    parser.add_argument('--gnn_hidden_size', type=int, default=256)
    parser.add_argument('--gnn_feature_dim_size', type=int, default=768)
    parser.add_argument('--num_gnn_layers', type=int, default=2)
    parser.add_argument('--dropout_rate', type=float, default=0.2)
    parser.add_argument('--residual', action='store_true')
    parser.add_argument('--graph_pooling', type=str, default='mean', choices=['sum','mean','max','attn','set2set'])
    parser.add_argument('--gconv_aggr', type=str, default='mean')   # 给 GraphConv/SAGEConv 用
    parser.add_argument('--gin_eps', type=float, default=0.0)
    parser.add_argument('--gin_train_eps', action='store_true')
    parser.add_argument('--num_ggnn_steps', type=int, default=3)
    parser.add_argument('--ggnn_aggr', type=str, default='add')
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--positive_class_id', type=int, default=1)
    # 判别阈值 & 校准
    parser.add_argument('--decision_threshold', type=float, default=0.5)
    parser.add_argument('--auto_threshold', action='store_true')
    parser.add_argument('--auto_threshold_beta', type=float, default=1.0)
    parser.add_argument('--calibrate_temp', action='store_true')  # 这里只留占位，不在此脚本内执行温度寻优
    # 类权重/平滑（train() 内会读取）
    parser.add_argument('--label_smoothing', type=float, default=0.05)
    parser.add_argument('--class_weight_gamma', type=float, default=0.8)
    # checkpoint 根目录（每种 GNN 会自动拼上子目录）
    parser.add_argument('--model_checkpoint_root', type=str, default='saved_models')

    args = parser.parse_args()

    # 设备 & 随机种
    args.device = torch.device("cuda:"+str(args.cuda_id) if torch.cuda.is_available() else "cpu")
    # 使用 attn 时建议不强制严格确定性；这里默认 graph_pooling=mean，无需特殊处理
    set_seed(args.seed, deterministic=not (args.graph_pooling.lower() == 'attn' and not args.deterministic))

    # 与 main.train 对齐需要的字段
    args.start_epoch = 0
    args.start_step = 0

    compare_models(args)


if __name__ == "__main__":
    main()
