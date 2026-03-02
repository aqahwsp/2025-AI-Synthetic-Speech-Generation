import os
import subprocess
import zipfile
import shutil
import pathlib

# --- 用户配置 ---
# 1. 要反编译的 .jar 文件
INPUT_JAR_PATH0 = 'D://Download//00000000000000000000000000000//'

# 2. 输出的汇总 .txt 文件名
OUTPUT_TXT_PATH0 = 'D://Download//00000000000000000000000000000//code'

# 3. 反编译器 .jar 文件的路径 (假设它和脚本在同一个文件夹)
DECOMPILER_JAR_PATH = 'vineflower.jar'

# 4. 存放反编译后源码的临时文件夹 (脚本会自动创建和删除)
TEMP_SOURCE_DIR = 'temp_decompiled_src'

# --- 配置结束 ---


def decompile_and_summarize(INPUT_JAR_PATH,OUTPUT_TXT_PATH):
    """
    主函数：反编译 .jar 文件，然后将源码汇总到 .txt 文件中。
    """
    # 检查输入文件和反编译器是否存在
    if not os.path.exists(INPUT_JAR_PATH):
        print(f"错误: 目标文件 '{INPUT_JAR_PATH}' 不存在。")
        return
    if not os.path.exists(DECOMPILER_JAR_PATH):
        print(f"错误: 反编译器 '{DECOMPILER_JAR_PATH}' 不存在。")
        print("请下载 Vineflower 并将其与此脚本放在同一文件夹下。")
        return

    # --- 1. 执行反编译 ---
    print(f"开始使用 {DECOMPILER_JAR_PATH} 反编译 {INPUT_JAR_PATH}...")

    # 清理旧的临时文件夹（如果存在）
    if os.path.exists(TEMP_SOURCE_DIR):
        shutil.rmtree(TEMP_SOURCE_DIR)

    # 创建新的临时文件夹
    os.makedirs(TEMP_SOURCE_DIR)

    # 构建并执行命令行指令
    command = [
        r'C:\Program Files\Java\jdk-21\bin\java',
        '-jar',
        DECOMPILER_JAR_PATH,
        INPUT_JAR_PATH,
        TEMP_SOURCE_DIR
    ]

    try:
        # shell=True 在某些环境下是必要的，但要注意安全风险
        # 这里我们控制了所有参数，所以是安全的
        subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"反编译成功！源代码已输出到临时文件夹 '{TEMP_SOURCE_DIR}'。")
    except subprocess.CalledProcessError as e:
        print("--- 反编译失败！ ---")
        print(f"错误码: {e.returncode}")
        print(f"错误输出:\n{e.stderr}")
        # 清理失败后创建的文件夹
        shutil.rmtree(TEMP_SOURCE_DIR)
        return
    except FileNotFoundError:
        print("错误: 'java' 命令未找到。请确保您已正确安装 Java 并将其添加到了系统 PATH。")
        return

    # --- 2. 汇总源代码到 .txt 文件 ---
    # 检查临时目录是否存在，防止因反编译失败而出错
    if not os.path.exists(TEMP_SOURCE_DIR):
        print(f"错误：临时目录 '{TEMP_SOURCE_DIR}' 不存在，可能反编译失败了。")
    else:
        print(f"开始汇总源代码到 {OUTPUT_TXT_PATH}...")
        try:
            # 使用 'w' 模式打开文件，如果文件已存在则覆盖
            with open(OUTPUT_TXT_PATH, 'w', encoding='utf-8') as summary_file:
                # os.walk 会遍历一个目录下的所有文件夹和文件
                for root, dirs, files in os.walk(TEMP_SOURCE_DIR):
                    for file in files:
                        # 我们只关心 .java 文件
                        if file.endswith('.java'):
                            file_path = os.path.join(root, file)
                            summary_file.write(f'--- Source file: {file_path} ---\n\n')
                            try:
                                # 读取每个 .java 文件的内容并写入汇总文件
                                with open(file_path, 'r', encoding='utf-8', errors='ignore') as source_file:
                                    summary_file.write(source_file.read())
                                    summary_file.write('\n\n')
                            except Exception as e:
                                summary_file.write(f"--- Error reading file {file_path}: {e} ---\n\n")
            print("源代码汇总成功！")
        except Exception as e:
            print(f"汇总过程中发生错误: {e}")


# --- 运行主函数 ---
if __name__ == "__main__":
    i=0
    for root, dirs, files in os.walk(INPUT_JAR_PATH0):
        for file in files:
            # 我们只关心 .java 文件
            if file.endswith('.jar'):
                i+=1
                decompile_and_summarize(os.path.join(root, file),OUTPUT_TXT_PATH0+str(i)+'.txt')


