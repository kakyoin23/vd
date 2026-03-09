import os
import shutil
import subprocess
import time
import platform
import multiprocessing
from pathlib import Path
from glob import glob
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# ================= 配置区 =================
# Joern 安装路径
JOERN_CLI_DIR = Path("/root/autodl-tmp/counterfactual-vulnerability-detection-main/cfexplainer/storage/external/joern-cli")

# Scala 脚本路径 (请确认你的脚本名字是 v4 还是 v5)
SCRIPT_PATH = "get_slice_v4.sc" 

# 尝试自动导入路径，如果失败则使用默认路径
try:
    from helpers import utils 
    SOURCE_DIR = utils.processed_dir() / "bigvul/before"
    TARGET_DIR = utils.processed_dir() / "bigvul/sliced_before"
except ImportError:
    SOURCE_DIR = Path("/root/autodl-tmp/data/source") 
    TARGET_DIR = Path("/root/autodl-tmp/data/target")

# 白名单文件路径
TARGET_IDS_FILE = Path("target_ids.txt")

# ================= 策略配置 =================
# 智能分批阈值：小于 50KB 的文件视为小文件
SMALL_FILE_THRESHOLD = 50 * 1024 

# 小文件批次大小 (追求速度)
BATCH_SIZE_FAST = 10 

# 大文件批次大小 (追求稳定，隔离 OOM)
BATCH_SIZE_SAFE = 1 

# 并发进程数 (建议 16-24，配合 1TB 内存)
MAX_WORKERS = 16 

# 单个 Worker 的内存限制 (Total = MAX_WORKERS * 32G)
JAVA_HEAP_SIZE = "32G"

# ================= 临时目录设置 =================
if platform.system() == "Linux" and Path("/dev/shm").exists():
    TEMP_WORK_DIR = Path("/dev/shm/temp_batches_smart_rerun")
    print(f"🚀 启用内存盘加速: {TEMP_WORK_DIR}")
else:
    TEMP_WORK_DIR = Path("temp_batches_smart_rerun")

def process_batch(batch_id, file_list):
    """
    处理单个 Batch 的核心逻辑
    """
    current_batch_dir = TEMP_WORK_DIR / f"batch_{batch_id}"
    
    # 清理并创建临时目录
    if current_batch_dir.exists():
        shutil.rmtree(current_batch_dir)
    current_batch_dir.mkdir(parents=True, exist_ok=True)
    
    file_map = {} 
    
    # 1. 拷贝文件到临时目录
    for src_path in file_list:
        fname = Path(src_path).name
        # 双重检查：防止多进程冲突，跳过已完成的文件
        target_file = TARGET_DIR / fname
        if target_file.exists() and target_file.stat().st_size > 0:
            continue
            
        shutil.copy(src_path, current_batch_dir / fname)
        file_map[fname] = src_path

    # 如果目录下没有文件（说明都被过滤了），直接返回
    if not list(current_batch_dir.glob("*.c")):
        shutil.rmtree(current_batch_dir)
        return "Skipped"

    try:
        joern_bin = JOERN_CLI_DIR / "joern"
        
        # 构造 Joern 命令
        cmd = f"{joern_bin} -J-Xmx{JAVA_HEAP_SIZE} --script {SCRIPT_PATH} --param inputDir='{str(current_batch_dir.resolve())}'"
        
        # 执行命令，设置超时 (30分钟)
        result = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT, timeout=1800)
        output_str = result.decode("utf-8", errors="ignore")
        
        # 解析结果
        results = {}
        for line in output_str.splitlines():
            if "###RESULT###" in line:
                parts = line.strip().split(":")
                if len(parts) >= 3:
                    fname_res = parts[-2].strip() 
                    lines_str = parts[-1].strip()
                    results[Path(fname_res).name] = lines_str

        # 写入切片结果
        for fname, src_path in file_map.items():
            lines_to_keep = set()
            target_path = TARGET_DIR / fname
            
            if fname in results and results[fname]:
                try:
                    line_parts = results[fname].split(',')
                    lines_to_keep = set(int(x) for x in line_parts if x.strip().isdigit())
                except ValueError:
                    pass
            
            # 读取源代码
            try:
                with open(src_path, 'r', errors='ignore') as f:
                    code_lines = f.readlines()
            except Exception:
                continue

            # 只有切出内容的才写入
            if lines_to_keep:
                with open(target_path, 'w') as f:
                    for i, ln in enumerate(code_lines):
                        if (i + 1) in lines_to_keep:
                            f.write(ln)
            else:
                # 关键：如果失败或为空，确保删除可能存在的空文件，以便下次重试
                if target_path.exists():
                    os.remove(target_path)

        return "Success"

    except subprocess.TimeoutExpired:
        # print(f"Batch {batch_id} timeout.")
        return "Timeout"
    except Exception as e:
        # print(f"Batch {batch_id} failed: {e}")
        return "Failed"
    finally:
        # 清理临时目录
        if current_batch_dir.exists():
            shutil.rmtree(current_batch_dir)

def main():
    # 1. 全局清理
    if TEMP_WORK_DIR.exists():
        try: shutil.rmtree(TEMP_WORK_DIR)
        except: pass

    # 2. 加载白名单
    whitelist = set()
    if TARGET_IDS_FILE.exists():
        print(f"✅ 检测到白名单文件: {TARGET_IDS_FILE}")
        with open(TARGET_IDS_FILE, "r") as f:
            whitelist = set(line.strip() for line in f if line.strip())
        print(f"   白名单包含 {len(whitelist)} 个样本 ID")
    else:
        print("⚠️ 未检测到 target_ids.txt，将处理所有源文件！(请确认是否符合预期)")

    os.makedirs(TARGET_DIR, exist_ok=True)
    
    # 3. 扫描源文件
    print(f"正在扫描源文件目录: {SOURCE_DIR} ...")
    all_files_raw = list(glob(str(SOURCE_DIR / "*.c")))
    
    files_to_process = []
    
    print("正在根据 白名单 和 完成状态 过滤任务...")
    for f_path in tqdm(all_files_raw):
        fid = Path(f_path).stem  # 获取文件名(不含后缀)作为 ID
        fname = Path(f_path).name # 获取完整文件名
        target_f = TARGET_DIR / fname
        
        # 规则1: 如果白名单存在，且ID不在白名单里 -> 跳过
        if whitelist and fid not in whitelist:
            continue
            
        # 规则2: 如果目标文件已存在且大小 > 0 -> 跳过 (已完成)
        if target_f.exists() and target_f.stat().st_size > 0:
            continue
            
        files_to_process.append(f_path)

    if not files_to_process:
        print("🎉 所有目标任务均已完成！无需补跑。")
        return

    # 4. 按大小排序 (从小到大)
    print("正在按文件大小排序 (优先处理小文件)...")
    files_to_process.sort(key=lambda x: os.path.getsize(x))

    # 5. 智能分批
    small_files = []
    large_files = []
    
    for f in files_to_process:
        if os.path.getsize(f) < SMALL_FILE_THRESHOLD:
            small_files.append(f)
        else:
            large_files.append(f)

    final_batches = []
    
    # 5.1 小文件打包
    for i in range(0, len(small_files), BATCH_SIZE_FAST):
        final_batches.append(small_files[i : i + BATCH_SIZE_FAST])
        
    # 5.2 大文件独立
    for i in range(0, len(large_files), BATCH_SIZE_SAFE):
        final_batches.append(large_files[i : i + BATCH_SIZE_SAFE])

    print(f"\n{'='*50}")
    print(f"🚀 任务启动配置")
    print(f"需处理任务数: {len(files_to_process)}")
    print(f"  - 小文件 (<50KB): {len(small_files)} 个 -> Batch Size {BATCH_SIZE_FAST}")
    print(f"  - 大文件 (>50KB): {len(large_files)} 个 -> Batch Size {BATCH_SIZE_SAFE}")
    print(f"总 Batch 数: {len(final_batches)}")
    print(f"并发 Worker 数: {MAX_WORKERS}")
    print(f"{'='*50}\n")

    # 6. 执行并发处理
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_batch, i, b): i for i, b in enumerate(final_batches)}
        for _ in tqdm(as_completed(futures), total=len(final_batches), desc="Processing"):
            pass
            
    print("\n✅ 所有任务队列执行完毕。")

if __name__ == "__main__":
    main()