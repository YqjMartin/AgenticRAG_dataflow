import os
import json
import random

def get_used_ids(output_root="./output"):
    """从已存在的 ./output/output_{id} 中提取已使用过的 id"""
    used_ids = set()
    if not os.path.exists(output_root):
        return used_ids

    for name in os.listdir(output_root):
        if name.startswith("output_"):
            used_ids.add(name.replace("output_", ""))
    return used_ids

def reservoir_sample_skip_used(input_file, output_file, sample_size, output_root="./output"):
    used_ids = get_used_ids(output_root)
    print(f"[Info] Found {len(used_ids)} used IDs from {output_root}")

    reservoir = []
    with open(input_file, 'r', encoding='utf-8') as infile:
        for i, line in enumerate(infile):
            if i % 1000 == 0:
                print(f"[Progress] Read {i} lines")

            try:
                item = json.loads(line)
                item_id = str(item.get("id", "")).strip()
                if not item_id or item_id in used_ids:
                    continue
            except json.JSONDecodeError:
                continue  # 跳过格式不对的行

            if len(reservoir) < sample_size:
                reservoir.append(line)
            else:
                r = random.randint(0, i)
                if r < sample_size:
                    reservoir[r] = line

    print(f"[Result] Sampled {len(reservoir)} items after filtering used IDs.")

    with open(output_file, 'w', encoding='utf-8') as outfile:
        outfile.writelines(reservoir)

    print(f"[Done] Wrote sample to {output_file}")

if __name__ == "__main__":
    input_jsonl = '/mnt/public/data/lh/yqj/TaskCraft/src/tools/FlashRAG/retrieval_corpus/wiki18_100w.jsonl'
    output_jsonl = './mydata.jsonl'
    total_lines = 21015324
    reservoir_sample_skip_used(input_jsonl, output_jsonl, sample_size=1000)
