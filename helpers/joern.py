import json
import random
import shutil
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from . import utils
import scipy.sparse as sparse
from graphviz import Digraph


def nodelabel2line(label: str):
    """Given a node label, return the line number.

    Example:
    s = "METHOD_1.0: static long main()..."
    nodelabel2line(s)
    >>> '1.0'
    """
    try:
        return str(int(label))
    except:
        return label.split(":")[0].split("_")[-1]


def randcolor():
    """Generate random color."""
    def r():
        return random.randint(0, 255)
    return "#%02X%02X%02X" % (r(), r(), r())


def get_digraph(nodes, edges, edge_label=True, graph_type="full"):
    """Plot digraph given nodes and edges list - 支持完整CPG可视化"""
    dot = Digraph(comment=f"CPG Graph - {graph_type}")

    nodes = [n + [nodelabel2line(n[1])] for n in nodes]
    colormap = {"": "white"}
    for n in nodes:
        if n[2] not in colormap:
            colormap[n[2]] = randcolor()

    for n in nodes:
        style = {"style": "filled", "fillcolor": colormap[n[2]]}
        dot.node(str(n[0]), str(n[1]), **style)
    
    for e in edges:
        style = {"color": "black"}
        if e[2] == "CALL":
            style["style"] = "solid"
            style["color"] = "purple"
        elif e[2] == "AST":
            style["style"] = "solid"
            style["color"] = "black"
        elif e[2] == "CFG":
            style["style"] = "solid"
            style["color"] = "red"
        elif e[2] == "CDG":
            style["style"] = "solid"
            style["color"] = "blue"
        elif e[2] == "REACHING_DEF":
            style["style"] = "solid"
            style["color"] = "orange"
        elif e[2] == "REF":
            style["style"] = "dashed"
            style["color"] = "green"
        elif e[2] == "CONTAINS":
            style["style"] = "dotted"
            style["color"] = "gray"
        elif e[2] == "SOURCE_FILE":
            style["style"] = "dotted"
            style["color"] = "lightgray"
        elif e[2] == "DOMINATE":
            style["style"] = "dotted"
            style["color"] = "brown"
        elif e[2] == "POST_DOMINATE":
            style["style"] = "dotted"
            style["color"] = "pink"
        elif e[2] == "EVAL_TYPE":
            style["style"] = "dashed"
            style["color"] = "cyan"
        elif "DDG" in e[2]:
            style["style"] = "dashed"
            style["color"] = "darkgreen"
        else:
            # 为未知边类型提供默认样式
            style["style"] = "solid"
            style["color"] = "black"
        style["penwidth"] = "1"
        if edge_label:
            dot.edge(str(e[0]), str(e[1]), e[2], **style)
        else:
            dot.edge(str(e[0]), str(e[1]), **style)
    return dot


def run_joern(filepath: str, verbose: int):
    """Extract graph using Joern - 增强调试信息"""
    script_file = utils.external_dir() / "get_func_graph.scala"
    filename = utils.processed_dir() / filepath
    params = f"filename={filename}"
    command = f"joern --script {script_file} --params='{params}'"
    command = str(utils.external_dir() / "joern-cli" / command)
    
    if verbose > 1:
        print(f"[PYTHON DEBUG] Starting Joern for: {filepath}")
        print(f"[PYTHON DEBUG] Absolute path: {filename}")
        print(f"[PYTHON DEBUG] Command: {command}")
        print(f"[PYTHON DEBUG] Source file exists: {os.path.exists(filename)}")
    
    # 运行 Joern
    result = utils.subprocess_cmd(command, verbose=verbose)
    
    if verbose > 1:
        print(f"[PYTHON DEBUG] Joern process completed with result: {result}")
    
    # 检查输出文件
    expected_nodes = str(filename) + ".nodes.json"
    expected_edges = str(filename) + ".edges.json"
    
    if verbose > 1:
        print(f"[PYTHON DEBUG] Checking for output files:")
        print(f"[PYTHON DEBUG]   {expected_nodes} - exists: {os.path.exists(expected_nodes)}")
        print(f"[PYTHON DEBUG]   {expected_edges} - exists: {os.path.exists(expected_edges)}")
        
        if os.path.exists(expected_nodes):
            node_size = os.path.getsize(expected_nodes)
            print(f"[PYTHON DEBUG]   Nodes file size: {node_size} bytes")
        if os.path.exists(expected_edges):
            edge_size = os.path.getsize(expected_edges)
            print(f"[PYTHON DEBUG]   Edges file size: {edge_size} bytes")
    
    try:
        shutil.rmtree(utils.storage_dir().parent / "workspace" / filename.name)
        if verbose > 1:
            print(f"[PYTHON DEBUG] Cleaned up workspace")
    except Exception as E:
        if verbose > 1:
            print(f"[PYTHON DEBUG] Workspace cleanup failed: {E}")
        pass


def get_node_edges(filepath: str, verbose=0, graph_type="full"):
    """Get node and edges given filepath - 支持完整CPG和子图类型
    
    Args:
        filepath: 文件路径
        verbose: 详细级别
        graph_type: 图类型 - "full", "pdg", "ast", "cfg", "cdg", "ddg"
    """
    # 修复：使用与 run_joern 相同的路径逻辑
    if not Path(filepath).is_absolute():
        filename = utils.processed_dir() / filepath
    else:
        filename = Path(filepath)
    
    outfile = filename

    # 添加调试信息
    if verbose > 1:
        print(f"[DEBUG] Looking for CPG files for: {filepath}")
        print(f"[DEBUG] Using path: {outfile}")

    try:
        edges_path = str(outfile) + ".edges.json"
        with open(edges_path, "r") as f:
            edges = json.load(f)
            edges = pd.DataFrame(edges, columns=["innode", "outnode", "etype", "dataflow"])
            edges = edges.fillna("")
        if verbose > 1:
            print(f"[DEBUG] Successfully loaded edges from: {edges_path}")
    except FileNotFoundError:
        if verbose > 0:
            print(f"[ERROR] Edges file not found: {edges_path}")
        # 尝试其他可能的位置
        possible_paths = [
            filepath + ".edges.json",
            str(utils.processed_dir() / filepath) + ".edges.json",
            str(Path(filepath).absolute()) + ".edges.json",
        ]
        for path in possible_paths:
            if os.path.exists(path):
                if verbose > 0:
                    print(f"[INFO] Found edges file at: {path}")
                with open(path, "r") as f:
                    edges = json.load(f)
                    edges = pd.DataFrame(edges, columns=["innode", "outnode", "etype", "dataflow"])
                    edges = edges.fillna("")
                break
        else:
            return None, None
    except Exception as E:
        if verbose > 0:
            print(f"[ERROR] Failed to load edges: {E}")
        return None, None

    try:
        nodes_path = str(outfile) + ".nodes.json"
        with open(nodes_path, "r") as f:
            nodes = json.load(f)
            nodes = pd.DataFrame.from_records(nodes)
            if "controlStructureType" not in nodes.columns:
                nodes["controlStructureType"] = ""
            nodes = nodes.fillna("")
            try:
                nodes = nodes[
                    ["id", "_label", "name", "code", "lineNumber", "controlStructureType"]
                ]
            except Exception as E:
                if verbose > 1:
                    utils.debug(f"Failed {filepath}: {E}")
                return None
        if verbose > 1:
            print(f"[DEBUG] Successfully loaded nodes from: {nodes_path}")
    except FileNotFoundError:
        if verbose > 0:
            print(f"[ERROR] Nodes file not found: {nodes_path}")
        return None, None
    except Exception as E:
        if verbose > 0:
            print(f"[ERROR] Failed to load nodes: {E}")
        return None, None

    # Assign line number to local variables
    try:
        with open(filepath, "r") as f:
            code = f.readlines()
    except FileNotFoundError:
        # 如果原始文件不存在，尝试使用绝对路径
        try:
            with open(str(outfile), "r") as f:
                code = f.readlines()
        except FileNotFoundError:
            if verbose > 0:
                print(f"[ERROR] Source file not found: {filepath}")
            return None, None

    lmap = assign_line_num_to_local(nodes, edges, code)
    nodes.lineNumber = nodes.apply(
        lambda x: lmap[x.id] if x.id in lmap else x.lineNumber, axis=1
    )
    nodes = nodes.fillna("")

    # Assign node name to node code if code is null
    nodes.code = nodes.apply(lambda x: "" if x.code == "<empty>" else x.code, axis=1)
    nodes.code = nodes.apply(lambda x: x.code if x.code != "" else x["name"], axis=1)

    # Assign node label for printing in the graph
    nodes["node_label"] = (
        nodes._label + "_" + nodes.lineNumber.astype(str) + ": " + nodes.code
    )

    # Filter by node type
    nodes = nodes[nodes._label != "COMMENT"]
    nodes = nodes[nodes._label != "FILE"]

    # 根据graph_type过滤边
    if graph_type != "full":
        edges = rdg(edges, graph_type)

    # Remove nodes not connected to line number nodes (maybe not efficient)
    edges = edges.merge(
        nodes[["id", "lineNumber"]].rename(columns={"lineNumber": "line_out"}),
        left_on="outnode",
        right_on="id",
    )
    edges = edges.merge(
        nodes[["id", "lineNumber"]].rename(columns={"lineNumber": "line_in"}),
        left_on="innode",
        right_on="id",
    )
    edges = edges[(edges.line_out != "") | (edges.line_in != "")]

    # Uniquify types
    edges.outnode = edges.apply(
        lambda x: f"{x.outnode}_{x.innode}" if x.line_out == "" else x.outnode, axis=1
    )
    typemap = nodes[["id", "name"]].set_index("id").to_dict()["name"]

    linemap = nodes.set_index("id").to_dict()["lineNumber"]
    for e in edges.itertuples():
        if type(e.outnode) == str:
            lineNum = linemap[e.innode]
            node_label = f"TYPE_{lineNum}: {typemap[int(e.outnode.split('_')[0])]}"
            new_row = {"id": e.outnode, "node_label": node_label, "lineNumber": lineNum}
            nodes = pd.concat([nodes, pd.DataFrame([new_row])], ignore_index=True)

    if verbose > 1:
        print(f"[DEBUG] Successfully processed CPG: {len(nodes)} nodes, {len(edges)} edges")
        if graph_type == "full":
            analyze_cpg_structure(nodes, edges)

    return nodes, edges


def plot_node_edges(filepath: str, lineNumber: int = -1, filter_edges=[], graph_type="full"):
    """Plot node edges given filepath - 支持完整CPG"""
    nodes, edges = get_node_edges(filepath, graph_type=graph_type)

    if len(filter_edges) > 0:
        edges = edges[edges.etype.isin(filter_edges)]

    # Draw graph
    if lineNumber > 0:
        nodesforline = set(nodes[nodes.lineNumber == lineNumber].id.tolist())
    else:
        nodesforline = set(nodes.id.tolist())

    edges_new = edges[
        (edges.outnode.isin(nodesforline)) | (edges.innode.isin(nodesforline))
    ]
    nodes_new = nodes[
        nodes.id.isin(set(edges_new.outnode.tolist() + edges_new.innode.tolist()))
    ]
    dot = get_digraph(
        nodes_new[["id", "node_label"]].to_numpy().tolist(),
        edges_new[["outnode", "innode", "etype"]].to_numpy().tolist(),
        graph_type=graph_type
    )
    dot.render("/tmp/tmp.gv", view=True)


def full_run_joern(filepath: str, verbose=0, graph_type="full"):
    """Run full Joern extraction and save output - 支持完整CPG"""
    try:
        run_joern(filepath, verbose)
        nodes, edges = get_node_edges(filepath, verbose=verbose, graph_type=graph_type)
        return {"nodes": nodes, "edges": edges}
    except Exception as E:
        if verbose > 0:
            utils.debug(f"Failed {filepath}: {E}")
        return None


def full_run_joern_from_string(code: str, dataset: str, iid: str, verbose=0, graph_type="full"):
    """Run full joern from a string instead of file - 支持完整CPG"""
    savedir = utils.get_dir(utils.interim_dir() / dataset)
    savepath = savedir / f"{iid}.c"
    with open(savepath, "w") as f:
        f.write(code)
    return full_run_joern(savepath, verbose, graph_type=graph_type)


def neighbour_nodes(nodes, edges, nodeids: list, hop: int = 1, intermediate=True):
    """Given nodes, edges, nodeid, return hop neighbours."""
    nodes_new = (
        nodes.reset_index(drop=True).reset_index().rename(columns={"index": "adj"})
    )
    id2adj = pd.Series(nodes_new.adj.values, index=nodes_new.id).to_dict()
    adj2id = {v: k for k, v in id2adj.items()}

    arr = []
    for e in zip(edges.innode.map(id2adj), edges.outnode.map(id2adj)):
        arr.append([e[0], e[1]])
        arr.append([e[1], e[0]])

    arr = np.array(arr)
    shape = tuple(arr.max(axis=0)[:2] + 1)
    coo = sparse.coo_matrix((np.ones(len(arr)), (arr[:, 0], arr[:, 1])), shape=shape)

    def nodeid_neighbours_from_csr(nodeid):
        return [
            adj2id[i]
            for i in csr[
                id2adj[nodeid],
            ]
            .toarray()[0]
            .nonzero()[0]
        ]

    neighbours = defaultdict(list)
    if intermediate:
        for h in range(1, hop + 1):
            csr = coo.tocsr()
            csr **= h
            for nodeid in nodeids:
                neighbours[nodeid] += nodeid_neighbours_from_csr(nodeid)
        return neighbours
    else:
        csr = coo.tocsr()
        csr **= hop
        for nodeid in nodeids:
            neighbours[nodeid] += nodeid_neighbours_from_csr(nodeid)
        return neighbours


def rdg(edges, gtype):
    """Reduce graph given type - 扩展支持更多图类型"""
    if gtype == "reftype":
        return edges[(edges.etype == "EVAL_TYPE") | (edges.etype == "REF")]
    if gtype == "ast":
        return edges[(edges.etype == "AST")]
    if gtype == "pdg":
        return edges[(edges.etype == "REACHING_DEF") | (edges.etype == "CDG")]
    if gtype == "cfg":
        return edges[(edges.etype == "CFG")]
    if gtype == "cdg":
        return edges[(edges.etype == "CDG")]
    if gtype == "ddg":
        return edges[(edges.etype == "REACHING_DEF")]
    if gtype == "cfgcdg":
        return edges[(edges.etype == "CFG") | (edges.etype == "CDG")]
    if gtype == "full":
        return edges  # 完整CPG，不过滤
    if gtype == "all":
        return edges[
            (edges.etype == "REACHING_DEF")
            | (edges.etype == "CDG")
            | (edges.etype == "AST")
            | (edges.etype == "EVAL_TYPE")
            | (edges.etype == "REF")
        ]
    return edges  # 默认返回所有边


def assign_line_num_to_local(nodes, edges, code):
    """Assign line number to local variable in CPG."""
    label_nodes = nodes[nodes._label == "LOCAL"].id.tolist()
    onehop_labels = neighbour_nodes(nodes, rdg(edges, "ast"), label_nodes, 1, False)
    twohop_labels = neighbour_nodes(nodes, rdg(edges, "reftype"), label_nodes, 2, False)
    node_types = nodes[nodes._label == "TYPE"]
    id2name = pd.Series(node_types.name.values, index=node_types.id).to_dict()
    node_blocks = nodes[
        (nodes._label == "BLOCK") | (nodes._label == "CONTROL_STRUCTURE")
    ]
    blocknode2line = pd.Series(
        node_blocks.lineNumber.values, index=node_blocks.id
    ).to_dict()
    local_vars = dict()
    local_vars_block = dict()
    for k, v in twohop_labels.items():
        types = [i for i in v if i in id2name and i < 1000]
        if len(types) == 0:
            continue
        assert len(types) == 1, "Incorrect Type Assumption."
        block = onehop_labels[k]
        assert len(block) == 1, "Incorrect block Assumption."
        block = block[0]
        local_vars[k] = id2name[types[0]]
        local_vars_block[k] = blocknode2line[block]
    nodes["local_type"] = nodes.id.map(local_vars)
    nodes["local_block"] = nodes.id.map(local_vars_block)
    local_line_map = dict()
    for row in nodes.dropna().itertuples():
        localstr = "".join((row.local_type + row.name).split()) + ";"
        try:
            ln = ["".join(i.split()) for i in code][int(row.local_block) :].index(
                localstr
            )
            rel_ln = row.local_block + ln + 1
            local_line_map[row.id] = rel_ln
        except:
            continue
    return local_line_map


def drop_lone_nodes(nodes, edges):
    """Remove nodes with no edge connections."""
    nodes = nodes[(nodes.id.isin(edges.innode)) | (nodes.id.isin(edges.outnode))]
    return nodes


def plot_graph_node_edge_df(
    nodes, edges, nodeids=[], hop=1, drop_lone_nodes=True, edge_label=True, graph_type="full"
):
    """Plot graph from node and edge dataframes - 支持完整CPG"""
    # Drop lone nodes
    if drop_lone_nodes:
        nodes = nodes[(nodes.id.isin(edges.innode)) | (nodes.id.isin(edges.outnode))]

    # Get subgraph
    if len(nodeids) > 0:
        nodeids = nodes[nodes.lineNumber.isin(nodeids)].id
        keep_nodes = neighbour_nodes(nodes, edges, nodeids, hop)
        keep_nodes = set(list(nodeids) + [i for j in keep_nodes.values() for i in j])
        nodes = nodes[nodes.id.isin(keep_nodes)]
        edges = edges[
            (edges.innode.isin(keep_nodes)) & (edges.outnode.isin(keep_nodes))
        ]

    dot = get_digraph(
        nodes[["id", "node_label"]].to_numpy().tolist(),
        edges[["outnode", "innode", "etype"]].to_numpy().tolist(),
        edge_label=edge_label,
        graph_type=graph_type
    )
    dot.render("/tmp/tmp.gv", view=True)


def analyze_cpg_structure(nodes, edges):
    """分析CPG结构 - 统计节点和边类型"""
    print("=== CPG结构分析 ===")
    
    # 节点类型统计
    node_types = nodes['_label'].value_counts()
    print("节点类型分布:")
    for node_type, count in node_types.items():
        print(f"  {node_type}: {count}")
    
    # 边类型统计
    edge_types = edges['etype'].value_counts()
    print("\n边类型分布:")
    for edge_type, count in edge_types.items():
        print(f"  {edge_type}: {count}")
    
    # 图规模统计
    print(f"\n图规模:")
    print(f"  节点总数: {len(nodes)}")
    print(f"  边总数: {len(edges)}")
    if len(nodes) > 0:
        print(f"  平均度: {len(edges) / len(nodes):.2f}")
    
    return node_types, edge_types


def get_full_cpg(filepath: str, verbose=0):
    """专门提取完整CPG的便捷函数"""
    return get_node_edges(filepath, verbose=verbose, graph_type="full")


def get_cpg_statistics(filepath: str, verbose=0):
    """获取CPG统计信息"""
    nodes, edges = get_full_cpg(filepath, verbose=verbose)
    if nodes is not None and edges is not None:
        return analyze_cpg_structure(nodes, edges)
    return None, None