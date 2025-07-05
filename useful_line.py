import random
import os
import json
from pathlib import Path

def main():
    output_dir = Path("output")
    total_folders = 0
    folders_with_qa = 0
    folders_without_qa = 0
    total_data_lines = 0

    for item in output_dir.iterdir():
        if item.is_dir() and item.name.startswith("output_"):
            total_folders += 1
            qa_path = item / "qa.jsonl"
            if qa_path.exists() and qa_path.is_file():
                folders_with_qa += 1
                with open(qa_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                json.loads(line)
                                total_data_lines += 1
                            except json.JSONDecodeError:
                                continue
            else:
                folders_without_qa += 1

    print(f"总文件夹数: {total_folders}")
    print(f"有qa.jsonl的文件夹数: {folders_with_qa}")
    print(f"没有qa.jsonl的文件夹数: {folders_without_qa}")
    print(f"qa.jsonl中有效JSON行总数: {total_data_lines}")

if __name__ == "__main__":
    main() 