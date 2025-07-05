import os
import shutil
from pathlib import Path

def main():
    output_dir = Path("depth_output")
    deleted_folders = []
    failed_deletions = []

    if not output_dir.exists() or not output_dir.is_dir():
        print(f"错误: 目录 '{output_dir}' 不存在或不是目录")
        return

    for item in output_dir.iterdir():
        if item.is_dir() and item.name.startswith("output_"):
            qa_path = item / "qa.jsonl"
            if qa_path.exists() and qa_path.is_file():
                pass
            else:
                try:
                    shutil.rmtree(item)
                    deleted_folders.append(str(item))
                    print(f"已删除: {item}")
                except Exception as e:
                    failed_deletions.append((str(item), str(e)))
                    print(f"删除失败: {item} - 错误: {e}")

    print("\n===== 操作总结 =====")
    print(f"已成功删除 {len(deleted_folders)} 个文件夹")
    print(f"删除失败 {len(failed_deletions)} 个文件夹")
    
    if failed_deletions:
        print("\n删除失败的文件夹列表:")
        for folder, error in failed_deletions:
            print(f"- {folder}: {error}")

if __name__ == "__main__":
    main(