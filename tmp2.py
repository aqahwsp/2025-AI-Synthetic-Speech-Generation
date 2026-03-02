import os
import subprocess
import shutil

# --- 用户配置 ---
# 1. 存放 .dex 文件的文件夹路径
INPUT_DEX_DIR = 'D://00000000//code//'

# 2. 输出汇总 .txt 文件的文件夹路径
OUTPUT_TXT_DIR = 'D://00000000//code//'

# 3. JADX 可执行文件的路径 (重要：请根据你的实际路径修改!)
#    Windows 示例: 'C://path//to//jadx-1.4.7//bin//jadx.bat'
#    Linux/macOS 示例: '/path/to/jadx-1.4.7/bin/jadx'
JADX_EXECUTABLE_PATH = 'D://APP//jadx//bin//jadx.bat'

# 4. 存放反编译后源码的临时文件夹 (脚本会自动创建和删除)
TEMP_SOURCE_DIR = 'temp_decompiled_src_dex'

JAVA_HOME_PATH = 'C://Program Files//Java//jdk-21'
# --- 配置结束 ---


def decompile_dex_and_summarize(dex_file_path, output_txt_path):
    """
    主函数：使用 JADX 反编译 .dex 文件，然后将源码汇总到 .txt 文件中。
    """
    # 检查输入文件和反编译器是否存在
    if not os.path.exists(dex_file_path):
        print(f"错误: 目标文件 '{dex_file_path}' 不存在。")
        return
    if not os.path.exists(JADX_EXECUTABLE_PATH):
        print(f"错误: JADX 可执行文件 '{JADX_EXECUTABLE_PATH}' 不存在。")
        print("请检查 JADX_EXECUTABLE_PATH 变量是否已正确配置。")
        return

    # --- 1. 执行反编译 ---
    print(f"开始使用 JADX 反编译 {dex_file_path}...")

    # 清理旧的临时文件夹（如果存在）
    if os.path.exists(TEMP_SOURCE_DIR):
        shutil.rmtree(TEMP_SOURCE_DIR)

    # JADX 会自动创建输出目录，所以我们不需要手动创建
    # os.makedirs(TEMP_SOURCE_DIR)

    # 构建并执行命令行指令
    # -d 指定输出目录
    # --show-bad-code 继续处理有问题的代码
    # --no-res 不反编译资源文件
    command = [
        JADX_EXECUTABLE_PATH,
        '-d',
        TEMP_SOURCE_DIR,
        '--show-bad-code',
        '--no-res',
        dex_file_path
    ]

    try:
        # 复制当前的环境变量
        process_env = os.environ.copy()
        # 在复制的环境变量中明确设置 JAVA_HOME
        process_env['JAVA_HOME'] = JAVA_HOME_PATH

        # 使用 subprocess.run 执行命令, 并传入我们修改过的环境变量
        # 设置 shell=False (更安全)，因为命令和参数是列表形式
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            encoding='utf-8',
            env=process_env  # 将包含 JAVA_HOME 的环境传递给子进程
        )
        print(f"反编译成功！源代码已输出到临时文件夹 '{TEMP_SOURCE_DIR}'。")
        # print(f"JADX 输出:\n{result.stdout}") # 如果需要调试可以取消注释
    except subprocess.CalledProcessError as e:
        print("--- 反编译失败！ ---")
        print(f"错误码: {e.returncode}")
        print(f"错误输出 (stderr):\n{e.stderr}")
        print(f"标准输出 (stdout):\n{e.stdout}")
    except FileNotFoundError:
        print(f"错误: 命令 '{JADX_EXECUTABLE_PATH}' 未找到。请确保路径正确。")
        return
    except Exception as e:
        print(f"执行 JADX 时发生未知错误: {e}")
        return

    # --- 2. 汇总源代码到 .txt 文件 ---
    if not os.path.exists(TEMP_SOURCE_DIR):
        print(f"错误：临时目录 '{TEMP_SOURCE_DIR}' 不存在，可能反编译失败了。")
    else:
        print(f"开始汇总源代码到 {output_txt_path}...")
        try:
            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_txt_path), exist_ok=True)

            with open(output_txt_path, 'w', encoding='utf-8') as summary_file:
                # 遍历 JADX 生成的所有源文件
                for root, dirs, files in os.walk(TEMP_SOURCE_DIR):
                    for file in files:
                        if file.endswith('.java'):
                            file_path = os.path.join(root, file)
                            # 写入文件头，标明源文件路径
                            summary_file.write(
                                f'--- Source file: {os.path.relpath(file_path, TEMP_SOURCE_DIR)} ---\n\n')
                            try:
                                with open(file_path, 'r', encoding='utf-8', errors='ignore') as source_file:
                                    summary_file.write(source_file.read())
                                    summary_file.write('\n\n')
                            except Exception as e:
                                summary_file.write(f"--- Error reading file {file_path}: {e} ---\n\n")
            print(f"源代码汇总成功！输出文件: {output_txt_path}")
        except Exception as e:
            print(f"汇总过程中发生错误: {e}")

    # --- 3. 清理临时文件夹 ---
    if os.path.exists(TEMP_SOURCE_DIR):
        shutil.rmtree(TEMP_SOURCE_DIR)
        print(f"临时文件夹 '{TEMP_SOURCE_DIR}' 已被删除。")


# --- 运行主函数 ---
if __name__ == "__main__":
    if not os.path.exists(INPUT_DEX_DIR):
        print(f"错误：输入目录 '{INPUT_DEX_DIR}' 不存在。请检查配置。")
    else:
        file_counter = 0
        for root, dirs, files in os.walk(INPUT_DEX_DIR):
            for file in files:
                if file.endswith('.dex'):
                    file_counter += 1
                    input_file_path = os.path.join(root, file)
                    # 构建输出文件名，例如：code_1_classes.dex.txt
                    output_filename = f"code_{file_counter}_{os.path.basename(file)}.txt"
                    output_file_path = os.path.join(OUTPUT_TXT_DIR, output_filename)

                    print(f"\n--- 处理文件 {file_counter}: {input_file_path} ---")
                    decompile_dex_and_summarize(input_file_path, output_file_path)
        print(f"\n--- 全部处理完成，共处理了 {file_counter} 个 .dex 文件。 ---")