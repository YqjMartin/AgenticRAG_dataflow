import json
import os
from tqdm import tqdm
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from taskcraft.src.tools import *
from taskcraft.src.utils import CUSTOM_ROLE_CONVERSIONS, run_llm_prompt
from oagents import OpenAIServerModel
from typing import Dict

gloden_doc_prompt = """You are given the following document that contains relevant information to help answer a question.
Document:
\"\"\"
{golden_doc}
\"\"\"
Question:
{new_query}
Please answer the question using ONLY the information in the provided document. Return the final answer directly, with no explanation.
"""

def process_one_item(data: Dict, model, verify_agent) -> Dict:
    result = {"id": data["id"], "original_data": data}
    question = data["question"]
    golden_answer = data["golden_answers"][0]

    decomposition = data["metadata"].get("question_decomposition", [])
    if len(decomposition) < 2:
        result["verification_result"] = "invalid"
        return result

    golden_doc = decomposition[0]["support_paragraph"]["paragraph_text"]
    now_doc = decomposition[1]["support_paragraph"]["paragraph_text"]

    try:
        # Step 1: search-only
        score = verify_agent.forward_agent(question, golden_answer)
        if score.get("llm_score", 0) > 0:
            result["verification_result"] = "only_need_search"
            return result

        # Step 2: golden_doc only
        answer1 = run_llm_prompt(model, gloden_doc_prompt.format(
            golden_doc=golden_doc, new_query=question), developer_prompt=None, return_json=True)
        score1 = verify_agent.recall_score(golden_answer, answer1, model, num_parallel_predictions=1)
        if score1 > 0:
            result["verification_result"] = "only_need_golden_doc"
            return result

        # Step 3: now_doc only
        answer2 = run_llm_prompt(model, gloden_doc_prompt.format(
            golden_doc=now_doc, new_query=question), developer_prompt=None, return_json=True)
        score2 = verify_agent.recall_score(golden_answer, answer2, model, num_parallel_predictions=1)
        if score2 > 0:
            result["verification_result"] = "only_need_source_doc"
            return result

        # Step 4: both
        merged_doc = golden_doc + "\n" + now_doc
        answer3 = run_llm_prompt(model, gloden_doc_prompt.format(
            golden_doc=merged_doc, new_query=question), developer_prompt=None, return_json=True)
        score3 = verify_agent.recall_score(golden_answer, answer3, model, num_parallel_predictions=1)
        if score3 > 0:
            result["verification_result"] = "need_both_doc"
        else:
            result["verification_result"] = "cannot_answer"
    except Exception as e:
        result["verification_result"] = "error"
        result["error"] = str(e)

    return result

def verify_multihop_data_parallel(input_jsonl_path, output_result_path, model, verify_agent, max_workers=4):
    os.makedirs(os.path.dirname(output_result_path), exist_ok=True)
    stats = {
        "only_need_search": 0,
        "only_need_golden_doc": 0,
        "only_need_source_doc": 0,
        "need_both_doc": 0,
        "cannot_answer": 0,
        "invalid": 0,
        "error": 0
    }

    processed_ids = set()
    if os.path.exists(output_result_path):
        with open(output_result_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    processed_ids.add(data["id"])
                except:
                    continue

    results = []

    with open(input_jsonl_path, "r", encoding="utf-8") as f:
        lines = []
        for line in f:
            data = json.loads(line)
            if data["id"] not in processed_ids:
                lines.append(data)
        lines = lines[:100]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_one_item, line, model, verify_agent) for line in lines]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Verifying"):
            res = future.result()
            results.append(res)
            stats[res["verification_result"]] += 1

    # 写入结果
    with open(output_result_path, "a", encoding="utf-8") as fout:
        for item in results:
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")

    # 写入统计
    with open(output_result_path.replace(".jsonl", "_stats.json"), "a", encoding="utf-8") as stat_file:
        stat_file.write('\n')
        json.dump(stats, stat_file, indent=2, ensure_ascii=False)

    print("==== Verification Summary ====")
    for k, v in stats.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    input_jsonl_path = '/mnt/public/data/lh/yqj/musique/dev.jsonl'
    output_result_path = './verify_results/musique_result.jsonl'

    model = OpenAIServerModel(
        model_id="gpt-4o",
        custom_role_conversions=CUSTOM_ROLE_CONVERSIONS,
        max_completion_tokens=8192,
        api_key=os.environ.get("OPENAI_API_KEY"),
        api_base=os.environ.get("OPENAI_API_BASE"),
    )

    verify_agent = VerifyAgent(model, "verify")

    verify_multihop_data_parallel(
        input_jsonl_path=input_jsonl_path,
        output_result_path=output_result_path,
        model=model,
        verify_agent=verify_agent,
        max_workers=20 
    )


