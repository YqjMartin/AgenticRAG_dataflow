import os
import json
import logging
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from taskcraft import *

os.makedirs("./Logs", exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file_path = f"./Logs/{timestamp}.log"

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.handlers.clear()

file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
file_handler.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

logging.info("日志系统已初始化")

total_num = 0

def process_item(doc_id):
    base_dir = f"./output/output_{doc_id}"
    qa_path = os.path.join(base_dir, "qa.jsonl")
    context_path = os.path.join(base_dir, "readed_context.jsonl")
    output_dir = f"./depth_output/output_{doc_id}"
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(qa_path) or not os.path.exists(context_path):
        logging.warning(f"[{doc_id}] 缺少 qa.jsonl 或 readed_context.jsonl 跳过。")
        return None

    try:
        with open(context_path, "r", encoding="utf-8") as f:
            context_json = json.loads(f.readline())
            golden_doc = context_json.get("text", "")
            identifier = golden_doc.strip().split("\n")[0].strip("\"“”") 
    except Exception as e:
        logging.error(f"[{doc_id}] 读取 context 失败: {e}")
        return None

    try:
        with open(qa_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception as e:
        logging.error(f"[{doc_id}] 读取 qa.jsonl 失败: {e}")
        return None

    results = []
    for idx, line in enumerate(lines):
        #try:
        qa = json.loads(line)
        question = qa["question"]
        answers = qa["golden_answer"]
        if not isinstance(answers, list) or not answers:
            logging.warning(f"[{doc_id}] 第 {idx} 个问题缺少有效答案，跳过。")
            continue

        answer = answers[0]  # 使用第一个答案(原始答案)
        output_path = os.path.join(output_dir, f"depth_extend_{idx}.json")
        if os.path.exists(output_path):
            logging.info(f"[{doc_id}] 问题 {idx} 已存在，跳过。")
            continue

        full_result, final_question, final_answer = depth_extend(
            query=question,
            golden_answer=answer,
            identifier=identifier,
            golden_doc=golden_doc,
            trajectory=None,
            model_id="gpt-4o",
            extended_attempts=3,
            max_merge_retry=3,
            max_hops=2,
            max_backward_step=4,
            max_verify_step=4
        )

        if full_result and final_question and final_answer:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump({
                    "question": final_question,
                    "answer": final_answer,
                    "full_result": full_result
                }, f, ensure_ascii=False, indent=4)
            print("full_result:", full_result)
            print("final_question:", final_question)
            print("final_answer:", final_answer)
            total_num += 1

        results.append(doc_id)

        # except Exception as e:
        #     logging.error(f"[{doc_id}] 处理问题 {idx} 失败: {e}")
        #     continue

    return doc_id if results else None

all_doc_ids = [
    name.replace("output_", "") for name in os.listdir("./output")
    if name.startswith("output_") and os.path.isdir(os.path.join("./output", name))
][100:200]

# selected_indices = [10, 20, 30, 50, 100, 110, 200, 300, 400, 500]
# all_doc_ids = [all_doc_ids[i] for i in selected_indices if i < len(all_doc_ids)]

max_workers = 10

with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = [executor.submit(process_item, doc_id) for doc_id in all_doc_ids]
    for future in tqdm(as_completed(futures), total=len(futures), desc="Processing", dynamic_ncols=True):
        result = future.result()
        if result:
            logging.info(f"[成功] 完成处理 doc_id={result}")
        else:
            logging.info("[失败]")

print("所有成功的数据数量:", total_num)
