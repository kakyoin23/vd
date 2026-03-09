import tempfile
import shutil
import os
import subprocess
from pathlib import Path
import concurrent.futures
from tqdm import tqdm

# === 1. 路径配置 ===
# 指向旧版 Joern v1.1 主目录
JOERN_BASE = Path("/root/autodl-tmp/counterfactual-vulnerability-detection-main/cfexplainer/storage/external/joern-cli11")
JOERN_MAIN = JOERN_BASE / "joern"  # 我们直接调用主程序

# 输入/输出路径
try:
    from helpers import utils
    SLICE_DIR = utils.processed_dir() / "bigvul/sliced_before"
    GRAPH_OUTPUT_DIR = utils.processed_dir() / "bigvul/graphs_before"
except ImportError:
    SLICE_DIR = Path("/root/autodl-tmp/counterfactual-vulnerability-detection-main/cfexplainer/storage/processed/bigvul/sliced_before")
    GRAPH_OUTPUT_DIR = Path("/root/autodl-tmp/counterfactual-vulnerability-detection-main/cfexplainer/storage/processed/bigvul/graphs_before")

GRAPH_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# === 2. 嵌入 Scala 脚本 (这就是你刚才提供的代码) ===
# 稍作修改以确保兼容性
SCALA_SCRIPT_CONTENT = """
@main def exec(filename: String, outDir: String, name: String) = {
   // 1. 导入代码 (解析)
   importCode.c(filename)
   
   // 2. 运行数据流分析
   run.ossdataflow
   
   // 3. 定义输出路径
   val nodesPath = outDir + "/" + name + ".nodes.json"
   val edgesPath = outDir + "/" + name + ".edges.json"
   
   // 4. 导出边 (Edges) - 包含变量信息
   cpg.graph.E.map(node=>List(node.inNode.id, node.outNode.id, node.label, node.propertiesMap.get("VARIABLE"))).toJson |> edgesPath
   
   // 5. 导出点 (Nodes)
   cpg.graph.V.map(node=>node).toJson |> nodesPath
   
   // 6. 清理工作区 (防止内存爆炸)
   delete
}
"""

# 将 Scala 脚本写入系统临时目录
SCALA_SCRIPT_PATH = Path("/tmp/generate_split_json.sc")
with open(SCALA_SCRIPT_PATH, "w") as f:
    f.write(SCALA_SCRIPT_CONTENT)

def generate_graph(file_path):
    """
    调用 Scala 脚本生成 .nodes.json 和 .edges.json
    """
    try:
        file_path = Path(file_path)
        file_id = file_path.stem
        
        # 这里的输出逻辑稍微变一下：
        # 原代码是 filename + ".nodes.json"，我们把它统一放到 output_dir 下
        # 每个样本一个文件夹
        sample_output_dir = GRAPH_OUTPUT_DIR / file_id
        
        # 1. 断点续传
        expected_nodes = sample_output_dir / f"{file_id}.nodes.json"
        if expected_nodes.exists() and expected_nodes.stat().st_size > 0:
            return "Skipped"
            
        # 2. 隔离环境 (每个进程一个临时目录，防止 cpg.bin 冲突)
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # 创建最终输出目录
            sample_output_dir.mkdir(parents=True, exist_ok=True)

            # === 配置 Java 11 环境 ===
            env_vars = os.environ.copy()
            env_vars['JAVA_HOME'] = '/usr/lib/jvm/java-11-openjdk-amd64' 
            env_vars['PATH'] = f"{env_vars['JAVA_HOME']}/bin:" + env_vars.get('PATH', '')
            env_vars['JAVA_OPTS'] = "-Xmx4G"

            # === 运行 Joern 脚本 ===
            # 参数说明:
            # filename: 输入的 .c 文件绝对路径
            # outDir: 输出目录
            # name: 文件名前缀 (比如 1001)
            cmd = [
                str(JOERN_MAIN),
                "--script", str(SCALA_SCRIPT_PATH),
                "--params", f"filename={file_path.absolute()},outDir={sample_output_dir.absolute()},name={file_id}"
            ]
            
            # cwd=temp_path 很重要，让 Joern 的 workspace 建在临时目录里
            subprocess.run(cmd, check=True, capture_output=True, env=env_vars, cwd=temp_path)
            
            # 检查是否生成成功
            if not expected_nodes.exists():
                return f"Error: Nodes file not generated"

        return "Success"

    except subprocess.CalledProcessError as e:
        err_msg = e.stderr.decode('utf-8', errors='ignore') or e.stdout.decode('utf-8', errors='ignore')
        return f"JoernError: {file_path.name}\n{err_msg[:300]}"
    except Exception as e:
        return f"Error: {str(e)}"

def main():
    print(f"Input: {SLICE_DIR}")
    print(f"Output: {GRAPH_OUTPUT_DIR}")
    
    slice_files = list(SLICE_DIR.glob("*.c"))
    print(f"Found {len(slice_files)} files.")
    
    # 建议先用少量线程测试，因为这个脚本比之前的 parse+export 更重一些
    MAX_WORKERS = 16 
    print(f"🚀 Starting with {MAX_WORKERS} workers...")
    
    success, fail, skip = 0, 0, 0

    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(generate_graph, f): f for f in slice_files}
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(slice_files)):
            res = future.result()
            if res == "Success": success += 1
            elif res == "Skipped": skip += 1
            else:
                fail += 1
                if fail <= 3: print(f"\n[FAIL] {res}\n")

    print(f"Done. Success:{success} Fail:{fail}")

if __name__ == "__main__":
    main()