import os
import shutil
import subprocess
from pathlib import Path
from glob import glob

# === 配置区 ===
JOERN_CLI_DIR = Path("/root/autodl-tmp/counterfactual-vulnerability-detection-main/cfexplainer/storage/external/joern-cli")
SCRIPT_PATH = "get_slice_v4.sc"

# 指向你刚才手动测试的那个文件夹
TEMP_INPUT_DIR = Path("debug_temp_input") 
TEMP_OUTPUT_DIR = Path("debug_output")

def debug_run():
    print(f"=== 🧪 开始调试模式 (使用现有文件) ===")
    
    # 1. 检查输入目录
    if not TEMP_INPUT_DIR.exists():
        print(f"❌ 错误：目录 {TEMP_INPUT_DIR} 不存在！请先确保里面有 vuln_test.c")
        return

    # 获取目录下现有的 .c 文件
    current_files = list(TEMP_INPUT_DIR.glob("*.c"))
    if not current_files:
        print(f"❌ 错误：在 {TEMP_INPUT_DIR} 里没找到 .c 文件")
        return
        
    print(f"📂 发现 {len(current_files)} 个现有文件，准备处理:")
    for f in current_files:
        print(f"  - {f.name}")

    # 2. 清理输出目录 (输入目录不动)
    if TEMP_OUTPUT_DIR.exists(): shutil.rmtree(TEMP_OUTPUT_DIR)
    TEMP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 3. 构造 Joern 命令
    joern_bin = JOERN_CLI_DIR / "joern"
    cmd = f"{joern_bin} --script {SCRIPT_PATH} --param inputDir='{str(TEMP_INPUT_DIR.resolve())}'"

    print(f"\n🚀 正在执行 Joern ...")
    print(f"执行命令: {cmd}")

    try:
        # 运行命令
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        stdout, _ = process.communicate()
        output_str = stdout.decode('utf-8', errors='ignore')

        print("\n=== 📜 Joern 输出分析 ===")
        
        # 提取结果行
        result_lines = [line for line in output_str.splitlines() if "###RESULT###" in line]
        
        if not result_lines:
            print("⚠️ 未找到 RESULT 输出！以下是完整日志片段：")
            print(output_str[-1000:]) # 打印最后1000字符
        else:
            print(f"捕获到 {len(result_lines)} 条结果，正在解析...")

        # 4. 验证解析逻辑 (使用修复后的 robust 写法)
        for line in result_lines:
            print(f"  [原始行] {line.strip()}")
            
            parts = line.strip().split(":")
            
            # === 关键解析逻辑 ===
            if len(parts) >= 3:
                # 兼容可能带路径的文件名
                lines_str = parts[-1].strip()
                fname_raw = ":".join(parts[1:-1]).strip()
                fname = Path(fname_raw).name 
                
                print(f"  ✅ 解析成功 -> 文件名: [{fname}] | 行号: [{lines_str}]")
                
                # 尝试写入
                original_file = TEMP_INPUT_DIR / fname
                target_file = TEMP_OUTPUT_DIR / fname
                
                if not original_file.exists():
                    print(f"     ⚠️ 警告：找不到对应的源文件 {original_file}")
                    continue

                if lines_str:
                    lines_to_keep = set(int(x) for x in lines_str.split(','))
                    with open(original_file, 'r', errors='ignore') as f:
                        content = f.readlines()
                    
                    with open(target_file, 'w') as f:
                        for i, c in enumerate(content):
                            if (i + 1) in lines_to_keep:
                                f.write(c)
                    
                    size = target_file.stat().st_size
                    print(f"     └─ 写入 {target_file} (大小: {size} bytes) -> {'✅ 成功' if size > 0 else '❌ 空文件'}")
                else:
                    with open(target_file, 'w') as f: pass
                    print(f"     └─ 无切片结果 (Empty Slice)，已生成空文件。")
            else:
                print(f"  ❌ 解析失败：格式不正确")

        print(f"\n=== 🏁 测试结束 ===")
        print(f"请检查目录 {TEMP_OUTPUT_DIR} 查看生成的文件")

    except Exception as e:
        print(f"❌ 发生异常: {e}")

if __name__ == "__main__":
    debug_run()