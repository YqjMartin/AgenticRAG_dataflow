import json
import os
from tqdm import tqdm
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from concurrent.futures import ThreadPoolExecutor, as_completed
from taskcraft.src.tools import *
from taskcraft.src.utils import CUSTOM_ROLE_CONVERSIONS, run_llm_prompt
from oagents import OpenAIServerModel
from typing import Dict

golden_doc_prompt = """You are given the following document that contains relevant information to help answer a question.
Document:
\"\"\"
{golden_doc}
\"\"\"
Question:
{new_query}
Please answer the question using ONLY the information in the provided document. Return the final answer directly, with no explanation.
"""
reasoning_question = """Please solve the following problem and return result. Ensure responses are as concise as possible, focusing only on key information while omitting redundant details. Please return the result in JSON format with keys 'answer_list': List[str] the list of answers.

The problem is:
{problem}
"""
def process_musique_item(data: Dict, model, verify_agent) -> Dict:
    result = {"id": data["id"], "original_data": data}
    question = data["question"]
    golden_answer = data["golden_answers"][0]

    decomposition = data["metadata"].get("question_decomposition", [])
    if len(decomposition) < 2:
        result["verification_result"] = "invalid" #没有相关的doc
        return result

    golden_doc = decomposition[0]["support_paragraph"]["paragraph_text"]
    now_doc = decomposition[1]["support_paragraph"]["paragraph_text"]

    try:
        # Step 1: reasoning only
        reasoning_result = verify_agent.forward_llm(reasoning_question.format(problem=question))
        score = verify_agent.recall_score(golden_answer, reasoning_result, model, num_parallel_predictions=1)
        if score > 0:
            result["verification_result"] = "only_reasoning"
            return result
        # if score['agent_score'] > 0: # agent score > 0 有可能是只检索到一个doc就可以得到的结果，没有实际意义
        #     result["verification_result"] = "agent_can_do"
        #     return result

        # Step 2: golden_doc only
        answer1 = run_llm_prompt(model, golden_doc_prompt.format(
            golden_doc=golden_doc, new_query=question), developer_prompt=None, return_json=True)
        score1 = verify_agent.recall_score(golden_answer, answer1, model, num_parallel_predictions=1)
        if score1 > 0:
            result["verification_result"] = "only_need_golden_doc"
            return result

        # Step 3: now_doc only
        answer2 = run_llm_prompt(model, golden_doc_prompt.format(
            golden_doc=now_doc, new_query=question), developer_prompt=None, return_json=True)
        score2 = verify_agent.recall_score(golden_answer, answer2, model, num_parallel_predictions=1)
        if score2 > 0:
            result["verification_result"] = "only_need_source_doc"
            return result

        # Step 4: both
        merged_doc = golden_doc + "\n" + now_doc
        answer3 = run_llm_prompt(model, golden_doc_prompt.format(
            golden_doc=merged_doc, new_query=question), developer_prompt=None, return_json=True)
        score3 = verify_agent.recall_score(golden_answer, answer3, model, num_parallel_predictions=1)
        if score3 > 0:
            result["verification_result"] = "need_multihop"
        else:
            result["verification_result"] = "cannot_answer"

    except Exception as e:
        logging.error(f"处理项目 {data['id']} 时出错: {e}", exc_info=True)
        result["verification_result"] = "error"
        result["error"] = str(e)

    return result


def process_hotpotqa_item(data: Dict, model, verify_agent) -> Dict:
    result = {"id": data["id"], "original_data": data}
    #print("id:",data["id"])
    question = data["question"]
    golden_answer = data["golden_answers"][0]

    # 解析类HotpotQA的格式
    context = data['metadata']["context"]
    supporting_titles = data['metadata']["supporting_facts"]["title"]
    #print("supports:", supporting_titles)
    supporting_sent_ids = data['metadata']["supporting_facts"]["sent_id"]

    # 构造 golden_docs: List[str]
    sent_field = "sentences" if "sentences" in context else "content"
    title_to_sents = dict(zip(context["title"], context[sent_field]))
    #print("contexts:",title_to_sents.keys())

    golden_docs = []
    for title, sent_id in zip(supporting_titles, supporting_sent_ids):
        try:
            sentence = title_to_sents[title][sent_id]
            # print("title:", title, "sent_id:", sent_id)
            # print("sentence:", sentence)
            golden_docs.append(f"{title}: {sentence}")
        except Exception as e:
            logging.warning(f"无法从 title {title} 中提取第 {sent_id} 个句子: {e}", exc_info=True)

    if not golden_docs:
        result["verification_result"] = "invalid" #没有相关的上下文
        return result

    # 构造非支持性文档
    # non_support_titles = [t for t in context["title"] if t not in supporting_titles]
    # now_doc = ""
    # if non_support_titles:
    #     sampled_title = random.choice(non_support_titles)
    #     sampled_sents = title_to_sents[sampled_title]
    #     now_doc = "\n".join([f"{sampled_title}: {s}" for s in sampled_sents[:3]])

    try:
        # Step 1 reasoning only
        reasoning_result = verify_agent.forward_llm(reasoning_question.format(problem=question))
        score_reasoning = verify_agent.recall_score(golden_answer, reasoning_result, model, num_parallel_predictions=1)
        if score_reasoning > 0:
            result["verification_result"] = "only_reasoning"
            return result

        # Step 2 single-hop 文档逐个验证
        for idx, doc in enumerate(golden_docs):
            answer = run_llm_prompt(model, golden_doc_prompt.format(
                golden_doc=doc, new_query=question), developer_prompt=None, return_json=True)
            score = verify_agent.recall_score(golden_answer, answer, model, num_parallel_predictions=1)
            if score > 0:
                result["verification_result"] = "only_need_single_hop"
                return result

        # Step 3: 合并所有 golden_docs 进行验证
        merged_doc = "\n".join(golden_docs)
        answer_all = run_llm_prompt(model, golden_doc_prompt.format(
            golden_doc=merged_doc, new_query=question), developer_prompt=None, return_json=True)
        score_all = verify_agent.recall_score(golden_answer, answer_all, model, num_parallel_predictions=1)
        if score_all > 0:
            result["verification_result"] = "need_multihop"
        else:
            result["verification_result"] = "cannot_answer"

    except Exception as e:
        logging.error(f"处理项目 {data['id']} 时出错: {e}", exc_info=True)
        result["verification_result"] = "error"
        result["error"] = str(e)

    return result


def verify_multihop_data_parallel(input_jsonl_path, output_result_path, model, verify_agent, max_workers=4, is_hotpotqa=False):
    os.makedirs(os.path.dirname(output_result_path), exist_ok=True)
    if is_hotpotqa:
        stats = {
            "only_reasoning": 0,
            "only_need_single_hop": 0,
            "need_multihop": 0,
            "cannot_answer": 0,
            "invalid": 0,
            "error": 0
        }
    else:
        stats = {
            "only_reasoning": 0,
            "only_need_golden_doc": 0,
            "only_need_source_doc": 0,
            "need_multihop": 0,
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
        lines = lines[:1000]

    if not is_hotpotqa:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_musique_item, line, model, verify_agent) for line in lines]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Verifying"):
                res = future.result()
                results.append(res)
                stats[res["verification_result"]] += 1
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_hotpotqa_item, line, model, verify_agent) for line in lines]
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
    input_jsonl_path = '/mnt/public/data/lh/yqj/FlashRAG/musique/dev.jsonl'
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
        max_workers=20,
        is_hotpotqa=False
    )