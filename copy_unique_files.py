import os
import shutil
import argparse
from pathlib import Path

def copy_unique_files(source_dir: str, target_dir: str, overwrite: bool = False):
    """
    将源目录(A)中有但目标目录(B)中没有的文件拷贝到目标目录，保留目录结构
    跳过.safetensors后缀的文件不拷贝
    
    Args:
        source_dir: 源目录路径（A目录）
        target_dir: 目标目录路径（B目录）
        overwrite: 若目标目录已存在同名文件，是否覆盖（默认False）
    """
    # 规范化路径（处理绝对/相对路径、末尾斜杠等问题）
    source_path = Path(source_dir).resolve()
    target_path = Path(target_dir).resolve()

    # 校验目录是否存在
    if not source_path.exists():
        raise FileNotFoundError(f"源目录不存在: {source_path}")
    if not target_path.exists():
        print(f"目标目录不存在，自动创建: {target_path}")
        target_path.mkdir(parents=True, exist_ok=True)

    # 统计变量
    copied_count = 0
    skipped_count = 0
    skipped_safetensors = 0  # 新增：统计跳过的.safetensors文件数
    error_count = 0

    print(f"开始对比目录：")
    print(f"源目录(A): {source_path}")
    print(f"目标目录(B): {target_path}")
    print("-" * 50)

    # 遍历源目录的所有文件（递归）
    for source_file in source_path.rglob("*"):
        # 跳过目录（只处理文件）
        if source_file.is_dir():
            continue
        
        # 核心改动：跳过.safetensors后缀的文件
        if source_file.suffix.lower() == ".safetensors":
            print(f"🚫 跳过.safetensors文件: {source_file}")
            skipped_safetensors += 1
            continue

        # 计算文件相对于源目录的相对路径（用于保留层级）
        rel_path = source_file.relative_to(source_path)
        # 目标文件的完整路径
        target_file = target_path / rel_path

        try:
            # 检查目标文件是否存在
            if not target_file.exists():
                # 创建目标目录（如果不存在）
                target_file.parent.mkdir(parents=True, exist_ok=True)
                # 拷贝文件
                shutil.copy2(source_file, target_file)  # copy2 保留文件元数据
                print(f"✅ 拷贝文件: {source_file} -> {target_file}")
                copied_count += 1
            else:
                if overwrite:
                    # 覆盖已存在的文件
                    shutil.copy2(source_file, target_file)
                    print(f"🔄 覆盖文件: {source_file} -> {target_file}")
                    copied_count += 1
                else:
                    print(f"⏩ 跳过已存在文件: {target_file}")
                    skipped_count += 1
        except Exception as e:
            print(f"❌ 拷贝失败: {source_file} -> {target_file} | 错误: {str(e)}")
            error_count += 1

    # 输出统计结果（新增skipped_safetensors统计）
    print("-" * 50)
    print(f"执行完成！")
    print(f"✅ 成功拷贝/覆盖: {copied_count} 个文件")
    print(f"⏩ 跳过已存在文件: {skipped_count} 个文件")
    print(f"🚫 跳过.safetensors文件: {skipped_safetensors} 个文件")  # 新增统计项
    print(f"❌ 拷贝失败: {error_count} 个文件")

def main():
    # 命令行参数解析
    parser = argparse.ArgumentParser(description='将A目录中有但B目录中没有的文件拷贝到B目录（保留层级，跳过.safetensors文件）')
    parser.add_argument('--source', '-s', required=True, help='源目录路径（A目录）')
    parser.add_argument('--target', '-t', required=True, help='目标目录路径（B目录）')
    parser.add_argument('--overwrite', '-o', action='store_true', help='是否覆盖目标目录已存在的文件（默认不覆盖）')
    
    args = parser.parse_args()

    # 执行拷贝逻辑
    try:
        copy_unique_files(args.source, args.target, args.overwrite)
    except Exception as e:
        print(f"程序执行失败: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()