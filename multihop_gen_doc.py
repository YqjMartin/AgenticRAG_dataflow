import json
import os
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from datetime import datetime
from taskcraft import *

os.makedirs("./Logs", exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file_path = f"./Logs/{timestamp}.log"

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
def process_item(line):
    if not line.strip() or line.startswith("//") or line.startswith("#"):
        logging.info("Skipping empty line or comment.")
        return None

    try:
        item = json.loads(line)
    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse JSON line: {e}")
        return None

    id = item.get("id", "")
    contents = item.get("contents", "")
    if not contents or not id:
        logging.warning("Skipping item due to missing 'id' or 'contents'.")
        return None

    output_dir = f"./depth_output/output_{id}"
    if os.path.exists(output_dir):
        logging.info(f"Skipping id={id} because output folder already exists.")
        return None

    try:
        full_result, final_question, final_answer = depth_extend(
            golden_doc=contents,
            trajectory=None,
            model_id="gpt-4o",
            extended_attempts=5,
            max_merge_retry=5,
            max_hops=2,
            max_backward_step=4,
            max_verify_step=4
        )
        print("full_result:", full_result)
        print("final_question:", final_question)
        print("final_answer:", final_answer)
        if full_result and final_question and final_answer:
            output_path = f"./depth_output/output_{id}/output.jsonl"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump({
                    "question": final_question,
                    "answer": final_answer,
                    "full_result": full_result
                }, f, ensure_ascii=False, indent=4)
            total_num += 1
            return id
    except Exception as e:
        logging.error(f"Processing id={id} failed: {e}")
        return None

# 控制并发线程数
max_workers = 10

with open("mydata.jsonl", "r", encoding="utf-8") as input_file:
    lines = input_file.readlines()[:100]

with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = [executor.submit(process_item, line) for line in lines]
    for future in tqdm(as_completed(futures), total=len(futures), desc="Processing", dynamic_ncols=True):
        result = future.result()
        if result:
            logging.info(f"[Success] Finished processing id={result}")

print(total_num)