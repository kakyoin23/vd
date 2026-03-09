import argparse
import os
import platform
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from glob import glob
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from tqdm import tqdm

# ================= 默认配置 =================
DEFAULT_JOERN_CLI_DIR = Path(
    "/root/autodl-tmp/counterfactual-vulnerability-detection-main/cfexplainer/storage/external/joern-cli"
)
DEFAULT_SCRIPT_PATH = Path("get_slice_v4.sc")
DEFAULT_TARGET_IDS_FILE = Path("target_ids.txt")

try:
    from helpers import utils

    DEFAULT_SOURCE_DIR = utils.processed_dir() / "bigvul/before"
    DEFAULT_TARGET_DIR = utils.processed_dir() / "bigvul/sliced_before"
except ImportError:
    DEFAULT_SOURCE_DIR = Path("/root/autodl-tmp/data/source")
    DEFAULT_TARGET_DIR = Path("/root/autodl-tmp/data/target")

SMALL_FILE_THRESHOLD = 50 * 1024
BATCH_SIZE_FAST = 10
BATCH_SIZE_SAFE = 1
DEFAULT_MAX_WORKERS = 16
DEFAULT_JAVA_HEAP_SIZE = "32G"
DEFAULT_TIMEOUT_SECONDS = 1800

if platform.system() == "Linux" and Path("/dev/shm").exists():
    DEFAULT_TEMP_WORK_DIR = Path("/dev/shm/temp_batches_smart_rerun")
else:
    DEFAULT_TEMP_WORK_DIR = Path("temp_batches_smart_rerun")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Joern slicing in parallel batches.")
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--target-dir", type=Path, default=DEFAULT_TARGET_DIR)
    parser.add_argument("--script-path", type=Path, default=DEFAULT_SCRIPT_PATH)
    parser.add_argument("--joern-cli-dir", type=Path, default=DEFAULT_JOERN_CLI_DIR)
    parser.add_argument("--target-ids-file", type=Path, default=DEFAULT_TARGET_IDS_FILE)
    parser.add_argument("--temp-work-dir", type=Path, default=DEFAULT_TEMP_WORK_DIR)
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument("--java-heap-size", type=str, default=DEFAULT_JAVA_HEAP_SIZE)
    parser.add_argument("--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--small-file-threshold", type=int, default=SMALL_FILE_THRESHOLD)
    parser.add_argument("--batch-size-fast", type=int, default=BATCH_SIZE_FAST)
    parser.add_argument("--batch-size-safe", type=int, default=BATCH_SIZE_SAFE)
    return parser.parse_args()


def parse_result_line(line: str) -> Tuple[str, str]:
    # format: ###RESULT###:filename:line1,line2,...
    if not line.startswith("###RESULT###:"):
        return "", ""
    payload = line[len("###RESULT###:") :]
    if ":" not in payload:
        return "", ""
    fname, lines_str = payload.split(":", 1)
    return Path(fname.strip()).name, lines_str.strip()


def process_batch(
    batch_id: int,
    file_list: List[str],
    temp_work_dir: str,
    target_dir: str,
    joern_cli_dir: str,
    script_path: str,
    java_heap_size: str,
    timeout_seconds: int,
) -> str:
    temp_root = Path(temp_work_dir)
    current_batch_dir = temp_root / f"batch_{batch_id}"
    target_dir_path = Path(target_dir)
    joern_bin = Path(joern_cli_dir) / "joern"

    if current_batch_dir.exists():
        shutil.rmtree(current_batch_dir)
    current_batch_dir.mkdir(parents=True, exist_ok=True)

    file_map: Dict[str, str] = {}

    for src_path in file_list:
        src = Path(src_path)
        fname = src.name
        target_file = target_dir_path / fname

        if target_file.exists() and target_file.stat().st_size > 0:
            continue

        shutil.copy(src, current_batch_dir / fname)
        file_map[fname] = str(src)

    if not list(current_batch_dir.glob("*.c")):
        shutil.rmtree(current_batch_dir)
        return "Skipped"

    cmd = [
        str(joern_bin),
        f"-J-Xmx{java_heap_size}",
        "--script",
        str(Path(script_path).resolve()),
        "--param",
        f"inputDir={str(current_batch_dir.resolve())}",
    ]

    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout_seconds,
            check=True,
            text=True,
            errors="ignore",
        )

        output_str = proc.stdout
        results: Dict[str, str] = {}
        for line in output_str.splitlines():
            fname_res, lines_str = parse_result_line(line.strip())
            if fname_res:
                results[fname_res] = lines_str

        for fname, src_path in file_map.items():
            lines_to_keep = set()
            target_path = target_dir_path / fname

            if fname in results and results[fname]:
                try:
                    line_parts = results[fname].split(",")
                    lines_to_keep = {int(x) for x in line_parts if x.strip().isdigit()}
                except ValueError:
                    lines_to_keep = set()

            try:
                with open(src_path, "r", errors="ignore") as f:
                    code_lines = f.readlines()
            except Exception:
                continue

            if lines_to_keep:
                with open(target_path, "w") as f:
                    for i, ln in enumerate(code_lines, start=1):
                        if i in lines_to_keep:
                            f.write(ln)
            else:
                if target_path.exists():
                    os.remove(target_path)

        return "Success"

    except subprocess.TimeoutExpired:
        return f"Timeout(batch={batch_id})"
    except subprocess.CalledProcessError as e:
        msg = ""
        if e.stdout:
            msg = e.stdout[-200:].replace("\n", " ")
        return f"Failed(batch={batch_id}, code={e.returncode}, msg={msg})"
    except Exception as e:
        return f"Failed(batch={batch_id}, msg={e})"
    finally:
        if current_batch_dir.exists():
            shutil.rmtree(current_batch_dir)


def load_whitelist(target_ids_file: Path) -> set:
    if target_ids_file.exists():
        print(f"✅ 检测到白名单文件: {target_ids_file}")
        with open(target_ids_file, "r") as f:
            whitelist = {line.strip() for line in f if line.strip()}
        print(f"   白名单包含 {len(whitelist)} 个样本 ID")
        return whitelist

    print("⚠️ 未检测到 target_ids.txt，将处理所有源文件！(请确认是否符合预期)")
    return set()


def discover_source_files(source_dir: Path) -> List[str]:
    files = [str(p) for p in source_dir.rglob("*.c")]
    if not files:
        # backward-compatible fallback
        files = list(glob(str(source_dir / "*.c")))
    return files


def split_batches(files_to_process: List[str], small_file_threshold: int, batch_size_fast: int, batch_size_safe: int) -> Tuple[List[str], List[str], List[List[str]]]:
    small_files: List[str] = []
    large_files: List[str] = []

    for f in files_to_process:
        if os.path.getsize(f) < small_file_threshold:
            small_files.append(f)
        else:
            large_files.append(f)

    final_batches: List[List[str]] = []
    for i in range(0, len(small_files), batch_size_fast):
        final_batches.append(small_files[i : i + batch_size_fast])
    for i in range(0, len(large_files), batch_size_safe):
        final_batches.append(large_files[i : i + batch_size_safe])

    return small_files, large_files, final_batches


def main():
    args = parse_args()

    if not args.script_path.exists():
        raise FileNotFoundError(f"Scala script not found: {args.script_path}")

    joern_bin = args.joern_cli_dir / "joern"
    if not joern_bin.exists():
        raise FileNotFoundError(f"Joern binary not found: {joern_bin}")

    if args.temp_work_dir.exists():
        try:
            shutil.rmtree(args.temp_work_dir)
        except Exception:
            pass

    whitelist = load_whitelist(args.target_ids_file)

    os.makedirs(args.target_dir, exist_ok=True)

    print(f"正在扫描源文件目录: {args.source_dir} ...")
    all_files_raw = discover_source_files(args.source_dir)

    files_to_process: List[str] = []
    print("正在根据 白名单 和 完成状态 过滤任务...")
    for f_path in tqdm(all_files_raw):
        fid = Path(f_path).stem
        fname = Path(f_path).name
        target_f = args.target_dir / fname

        if whitelist and fid not in whitelist:
            continue
        if target_f.exists() and target_f.stat().st_size > 0:
            continue

        files_to_process.append(f_path)

    if not files_to_process:
        print("🎉 所有目标任务均已完成！无需补跑。")
        return

    print("正在按文件大小排序 (优先处理小文件)...")
    files_to_process.sort(key=lambda x: os.path.getsize(x))

    small_files, large_files, final_batches = split_batches(
        files_to_process,
        args.small_file_threshold,
        args.batch_size_fast,
        args.batch_size_safe,
    )

    print(f"\n{'='*50}")
    print("🚀 任务启动配置")
    print(f"需处理任务数: {len(files_to_process)}")
    print(f"  - 小文件 (<{args.small_file_threshold}B): {len(small_files)} 个 -> Batch Size {args.batch_size_fast}")
    print(f"  - 大文件 (≥{args.small_file_threshold}B): {len(large_files)} 个 -> Batch Size {args.batch_size_safe}")
    print(f"总 Batch 数: {len(final_batches)}")
    print(f"并发 Worker 数: {args.max_workers}")
    print(f"Java Heap: {args.java_heap_size}, Timeout: {args.timeout_seconds}s")
    print(f"{'='*50}\n")

    summary: Dict[str, int] = {"Success": 0, "Skipped": 0, "Timeout": 0, "Failed": 0}

    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(
                process_batch,
                i,
                batch,
                str(args.temp_work_dir),
                str(args.target_dir),
                str(args.joern_cli_dir),
                str(args.script_path),
                args.java_heap_size,
                args.timeout_seconds,
            ): i
            for i, batch in enumerate(final_batches)
        }

        for fut in tqdm(as_completed(futures), total=len(final_batches), desc="Processing"):
            result = fut.result()
            if result.startswith("Success"):
                summary["Success"] += 1
            elif result.startswith("Skipped"):
                summary["Skipped"] += 1
            elif result.startswith("Timeout"):
                summary["Timeout"] += 1
                print(f"⚠️ {result}")
            else:
                summary["Failed"] += 1
                print(f"❌ {result}")

    print("\n✅ 所有任务队列执行完毕。")
    print(f"Summary: {summary}")


if __name__ == "__main__":
    main()
