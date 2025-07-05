# -*- coding: utf-8 -*-
# @Project      : taskcraft
# @File         : gen_depth_based_task.py
# @Author       : Dingfeng Shi <shidingfeng@outlook.com>, Jingyi Cao <224040283@link.cuhk.edu.cn>, Qianben Chen <chenqianben@oppo.com>
# @LastUpdated  : 2025/6/11
# @LICENSE      : Apache License 2.0

import argparse
import logging
import os
from pathlib import Path
from typing import Optional, List, Dict, Tuple

from taskcraft.src.tools import *
from taskcraft.src.utils import CUSTOM_ROLE_CONVERSIONS, run_llm_prompt, load_yaml, write_json
from oagents import OpenAIServerModel
import types
import json

# load prompt templates
verify_prompt_yaml_path = Path(__file__).parent / "prompts/verify_prompts.yaml"
verify_prompt_template = load_yaml(verify_prompt_yaml_path)
depth_prompt_yaml_path = Path(__file__).parent / "prompts/depth_prompts.yaml"
depth_prompt_templates = load_yaml(depth_prompt_yaml_path)
general_prompt_yaml_path = Path(__file__).parent / "prompts/general_prompts.yaml"
general_prompt_templates = load_yaml(general_prompt_yaml_path)


class DepthExtend:
    def __init__(self, model, search_agent='SearchAgent', verify_agent='VerifyAgent'):
        self.model = model
        self.search_agent = search_agent
        self.verify_agent = verify_agent
        if self.search_agent in globals():
            self.search_agent = globals()[self.search_agent]
        else:
            raise ImportError(f"The agent: {self.search_agent} not found, please check the import path.")

        if self.verify_agent in globals():
            self.verify_agent = globals()[self.verify_agent]
        else:
            raise ImportError(f"The agent: {self.verify_agent} not found, please check the import path.")

        self.prompt_templates = {
            "augmented_question": depth_prompt_templates['augmented_question_prompt'],
            "backward_task_prompt": depth_prompt_templates['backward_task_prompt'],
            "generate_query_prompt": depth_prompt_templates['generate_query_prompt'],
            "is_superset_valid": depth_prompt_templates['is_superset_valid'],
            "rag_backward_task_prompt": depth_prompt_templates['rag_backward_task_prompt'],
            "rag_generate_query_prompt": depth_prompt_templates['rag_generate_query_prompt'],
        }

    # element --> search(element) --> superset --> id --> Query
    def backward(self, element, original_doc, max_step=10, max_retries=3, **kwargs):
        backward_agent = self.search_agent(self.model, 'backward_agent', max_step=max_step, **kwargs)

        # 1. Generate relation and superset 仅传入原始的doc时最简单的了，不然每一次都要更新retriever，修改smolagent的流程太复杂了，还是之后验证更加正确一些
        #try:
        backward_question = self.prompt_templates['rag_backward_task_prompt'].format(element=element, original_doc=original_doc)
        backward_result = backward_agent(backward_question, return_json=True,
                                        max_retries=max_retries)  # agent_result, agent_trajectory
        if isinstance(backward_result, dict) and "error" in backward_result:
            return backward_result
        backward_result = backward_result["agent_result"]  # identifier, relation, source_docs
        # print("backward_result:", backward_result)
        # print("backward_result type:", type(backward_result))
        backward_result = json.loads(backward_result) if isinstance(backward_result, str) else backward_result
        # print("backward_result identifier:", backward_result['identifier'])
        # print("backward_result identifier type:", type(backward_result['identifier']))
        # except json.JSONDecodeError as e:
        #     print("[JSON ERROR]", str(e))
        #     print("backward_result:", backward_result)
        #     print("backward_result type:", type(backward_result))
        #     raise e

        # 2. Check if superset is valid 语义验证合理性
        developer_prompt = self.prompt_templates['is_superset_valid']
        prompt = f'''
                Given superset: {backward_result['identifier']}\n
                Given relationship: {backward_result['relation']}\n
                Given subset: {element}\n
                '''
        query_check = run_llm_prompt(self.model, prompt, developer_prompt, return_json=True, max_retries=max_retries)
        if "error" in query_check:
            return query_check
        if query_check == "invalid":
            backward_result["error"] = "error superset"
            return backward_result

        # 3. Generate question based on superset and relation 
        prompt = self.prompt_templates['rag_generate_query_prompt'].format(identifier=backward_result['identifier'], relation=backward_result['relation'], answer=element)

        success = False
        for _ in range(3):
            try:
                query = run_llm_prompt(self.model, prompt, developer_prompt=None, return_json=True,
                                        max_retries=max_retries)  # new_query
                if "error" in query:
                    return query
                query = query['new_query']
                success = True
                break
            except Exception as e:
                logging.warning("[Failed]: Exception occurred while generating query: " + str(e))
                continue

        if success:
            return {
                'now_query': query, 
                'element': element,
                'identifier': backward_result['identifier'], 
                'relation': backward_result['relation'],
                'source_doc': backward_result['source_doc'],
            }
        else:
            logging.warning("[Failed]: Fail to generate a new query based on the superset.")
            return None

    def verify(self, query, golden_answer, max_step=10, max_retries=3, **kwargs):
        verify_agent = self.verify_agent(self.model, "verify_agent", search_agent=self.search_agent,
                                         max_step=max_step, **kwargs)
        verify_result = verify_agent(query, golden_answer, max_retries=max_retries)
        return verify_result

def generate_initial_qa_with_identifier(paragraph, model, max_retries=3):
    gen_initial_qa_with_identifier_prompt = f"""
You are an expert question writer and dataset curator. Given a paragraph, your job is to create a high-quality QA pair, and identify the key element (called `identifier`) that the question revolves around. This identifier will be used to later create multi-hop reasoning questions.

### Requirements:
1. The question must be natural, well-formed, and answerable using ONLY the content in the paragraph.
2. The answer must be concise and unambiguous (e.g., a name, date, number, or short phrase from the paragraph).
3. The identifier should be a key phrase or entity in the question — the anchor the question depends on.
4. The identifier must appear in the question and be different from the answer.
5. The identifier will be used in future steps to generate more complex multi-hop questions.

### Output format:
{{
  "question": "<your question>",
  "answer": "<short answer from paragraph>",
  "identifier": "<the core entity or phrase in the question to be later replaced>"
}}

### Example:
Input:
Paragraph: Beethoven's Symphony No. 5 in C Minor, "Fate," is iconic. Its four-note opening ("da-da-da-DUM") symbolizes fate knocking. Composed amid his hearing loss, it blends drama and triumph. Premiered in 1808, it's a masterpiece of resilience.
Output:
{{
  "question": "What is the minor key of Beethoven's Symphony No. 5?",
  "answer": "C Minor",
  "identifier": "Beethoven's Symphony No. 5"
}}

### Input:
{paragraph}
"""
    prompt = gen_initial_qa_with_identifier_prompt
    for _ in range(max_retries):
        try:
            result = run_llm_prompt(model, prompt, return_json=True)
            if (
                isinstance(result, dict) and
                "question" in result and
                "answer" in result and
                "identifier" in result
            ):
                return result["question"], result["answer"], result["identifier"]
        except Exception as e:
            logging.warning(f"[Warning] Failed to parse QA+identifier: {str(e)}")
    return None, None, None


def process_single_task(args):
    # initialze model
    model = OpenAIServerModel(
        args.model_id,
        custom_role_conversions=CUSTOM_ROLE_CONVERSIONS,
        max_completion_tokens=8192,
        api_key=os.environ.get("OPENAI_API_KEY"),
        api_base=os.environ.get("OPENAI_API_BASE"),
    )
    module = DepthExtend(model, search_agent=args.search_agent, verify_agent=args.verify_agent)

    full_results = []
    valid_hop = 1
    for hop in range(args.extended_attempts):
        try:
            # initial qa
            if hop == 0:
                if not args.have_original_qa:
                    q, a, idf = generate_initial_qa_with_identifier(args.golden_doc, model)
                    full_query = q
                    args.golden_answer = a
                    last_identifier = idf
                    print("initial question:",q,"\ninitial answer:",a,"\ninitial identifier:",idf)
                    full_results.append({
                        'initial_query': full_query,
                        'initial_answer': args.golden_answer,
                        'initial_trajectory': args.trajectory,
                        'initial_identifier': last_identifier,
                        'initial_doc': args.golden_doc,
                        'valid_hop': 1
                    })
                else:
                    full_query = args.query
                    last_identifier = args.identifier
                    full_results.append({
                        'initial_query': full_query,
                        'initial_answer': args.golden_answer,
                        'initial_trajectory': args.trajectory,
                        'initial_identifier': last_identifier,
                        'initial_doc': args.golden_doc,
                        'valid_hop': 1
                    })
            
            logging.info(f"The {hop + 1} attempt to extend the query")

            # 方法的核心就是想要用一个用一个更加宽泛的indentifier来替换当前的identifier，实现从一个更加宽泛的indentifier->原本query的indentifier->原始问题的答案
            # 以multihop=2为例，核心是三个部分，称为：core_query(就是原始的query),auxiliary_query(就是一个query，通过这个query能够查找到core_query的indentifier)，和原本的indentifier。有了上面三个部分之后，让LLM用auxiliary_query替换core_query中的identifier，得到一个新的query，也就增加了一个hop了

            # step1 backward
            backward_result = module.backward(last_identifier, args.golden_doc, max_step=args.max_backward_step)
            if isinstance(backward_result, dict) and "error" in backward_result:
                logging.warning(f"[Failed]: The extended task in the {hop} attempt failed in backward.")
                continue
            now_query = backward_result["now_query"]
            now_doc = backward_result["source_doc"]
            logging.info(f"The generated intermediate task in the {hop + 1} attempt: {now_query}")
            # 区分：now_query指向last_indentifier->和full_query merge之后得到new_query->经过验证->更新full_query

            # step2 query merge
            success = False
            for _ in range(args.max_merge_retry):
                #try:
                prompt = depth_prompt_templates['rag_merge_query_prompt'].format(core_query=args.query, golden_answer=args.golden_answer, auxiliary_query=now_query, auxiliary_answer=last_identifier) #修改prompt OK
                new_query = run_llm_prompt(model, prompt, developer_prompt=None, return_json=True)  # analysis, new_query
                if isinstance(new_query, dict) and "error" in new_query: #？
                    logging.warning(f"[Failed]: Fail to merge a new query.")
                    continue
                new_query = new_query['new_query']

                # 能否不搜索就得到结论
                self_search_result = module.verify(new_query, args.golden_answer, max_step=args.max_verify_step)
                if isinstance(self_search_result, dict) and "error" in self_search_result:
                    logging.warning(f"[Failed]: The merged query fail the agentic verification. Due to: {self_search_result['error']}")
                    continue
                llm_score = self_search_result["llm_score"]

                # 给出一条gloden_doc能否解决
                gloden_doc_prompt = """You are given the following document that contains relevant information to help answer a question.
                Document:
                \"\"\"
                {golden_doc}
                \"\"\"
                Question:
                {new_query}
                Please answer the question using ONLY the information in the provided document. Return the final answer directly, with no explanation.
                """
                verify_agent = module.verify_agent(model, "verify_agent", search_agent=module.search_agent, max_step=5)
                # 原本的doc
                other_answer = run_llm_prompt(model, gloden_doc_prompt.format(golden_doc=args.golden_doc, new_query=new_query), developer_prompt=None, return_json=True) 
                gloden_doc_result = verify_agent.recall_score(args.golden_answer, other_answer, model, num_parallel_predictions=1)
                if gloden_doc_result > 0:
                    logging.warning(f"[Failed]: Only need original doc to solve the problem.")
                    continue

                # 检索到的doc
                other_answer = run_llm_prompt(model, gloden_doc_prompt.format(golden_doc=now_doc, new_query=new_query), developer_prompt=None, return_json=True) 
                now_doc_result = verify_agent.recall_score(args.golden_answer, other_answer, model, num_parallel_predictions=1)
                if now_doc_result > 0:
                    logging.warning(f"[Failed]: Only need source doc to solve the problem.")
                    continue
                
                # 给出所有doc
                other_answer = run_llm_prompt(model, gloden_doc_prompt.format(golden_doc=args.golden_doc+'\n'+now_doc, new_query=new_query), developer_prompt=None, return_json=True) 
                full_doc_result = verify_agent.recall_score(args.golden_answer, other_answer, model, num_parallel_predictions=1)
                if full_doc_result <= 0:
                    logging.warning(f"[Failed]: This question can't resove.")
                    continue

                success = True
                break

            if not success:
                logging.warning(f"[Failed]: Fail to merge a new query in {args.max_merge_retry} tries.")
                continue

            valid_hop += 1
            logging.info(f"[Success]: The extended task in the {hop + 1} attempt: {new_query}")

            full_results.append({
                "now_query": now_query,
                "now_answer": last_identifier,
                "valid_hop": vaild_hop,
                "now_doc": now_doc,
            })

            full_results.append({
                "new_query": new_query,
                "golden_answer": args.golden_answer,
                "valid_hop": valid_hop,
                "last_doc_result": golden_doc_result,
                "now_doc_result": now_doc_result,
                "full_doc_result": full_doc_result,
            })

            # 更新相关参数
            last_identifier = backward_result['identifier']
            full_query = new_query
            args.golden_doc = args.golden_doc + '\n' + now_doc

            if valid_hop >= args.max_hops:
                logging.info(f"Reach the max hops: {args.max_hops}, stop extending.")
                break
        except Exception as e:
            logging.warning("[Failed]: Exception occurred while generate new question: " + str(e))
            continue 

    if len(full_results) <= 1:
        logging.warning(f"Failed to extend the query in {args.extended_attempts} tries.")
        return None, None, None

    # return final question and answer
    return full_results, full_results[-1]["query"], full_results[-1]["answer"]

def depth_extend(
        golden_doc: str,
        query: str = None,
        golden_answer: str = None,
        identifier: str = None,
        have_original_qa: bool = False,
        trajectory: Optional[List] = None,
        model_id: str = "gpt-4.1",
        extended_attempts: int = 4,
        max_merge_retry: int = 3,
        max_hops=2,
        max_backward_step=10,
        max_verify_step=10,
        search_agent='SearchAgent',
        verify_agent='VerifyAgent'
):
    args = types.SimpleNamespace(
        query=query,
        golden_answer=golden_answer,
        identifier=identifier,
        golden_doc=golden_doc,
        have_original_qa=have_original_qa,
        trajectory=trajectory,
        model_id=model_id,
        extended_attempts=extended_attempts,
        max_merge_retry=max_merge_retry,
        max_hops=max_hops,
        max_backward_step=max_backward_step,
        max_verify_step=max_verify_step,
        search_agent=search_agent,
        verify_agent=verify_agent,
    )
    full_result, final_question, final_answer = process_single_task(args)
    return full_result, final_question, final_answer

# def process_single_task(args):
#     # initialze model
#     model = OpenAIServerModel(
#         args.model_id,
#         custom_role_conversions=CUSTOM_ROLE_CONVERSIONS,
#         max_completion_tokens=8192,
#         api_key=os.environ.get("OPENAI_API_KEY"),
#         api_base=os.environ.get("OPENAI_API_BASE"),
#     )
#     module = DepthExtend(model, search_agent=args.search_agent, verify_agent=args.verify_agent)

#     # check identifier, if it is None, should be initialized
#     if args.identifier is None:
#         logging.info("Identifier is not provided, generating identifier...")
#         developer_prompt = general_prompt_templates["get_identifier_prompt"]
#         prompt = f"""Now process this question: {args.query}"""
#         identifier_result = run_llm_prompt(model, prompt, developer_prompt=developer_prompt, return_json=True)
#         args.identifier = identifier_result["content_identifier"]

#     full_results = []
#     full_query = args.query
#     last_identifier = args.identifier
#     full_results.append({
#         'initial_query': full_query,
#         'initial_answer': args.golden_answer,
#         'initial_trajectory': args.trajectory,
#         'initial_identifier': last_identifier,
#         'initial_doc': args.golden_doc,
#         'valid_hop': 1
#     })

#     valid_hop = 1
#     for hop in range(args.extended_attempts):
#         try:
#             logging.info(f"The {hop + 1} attempt to extend the query")

#             # 方法的核心就是想要用一个用一个更加宽泛的indentifier来替换当前的identifier，实现从一个更加宽泛的indentifier->原本query的indentifier->原始问题的答案
#             # 以multihop=2为例，核心是三个部分，称为：core_query(就是原始的query),auxiliary_query(就是一个query，通过这个query能够查找到core_query的indentifier)，和原本的indentifier。有了上面三个部分之后，让LLM用auxiliary_query替换core_query中的identifier，得到一个新的query，也就增加了一个hop了

#             # step1 backward
#             backward_result = module.backward(last_identifier, args.golden_doc,max_step=args.max_backward_step)
#             if isinstance(backward_result, dict) and "error" in backward_result:
#                 logging.warning(f"[Failed]: The extended task in the {hop} attempt failed in backward.")
#                 continue
#             now_query = backward_result["now_query"]
#             now_doc = backward_result["source_doc"]
#             logging.info(f"The generated intermediate task in the {hop + 1} attempt: {now_query}")


#             # step2 check now query result 判断能否新的获得到当前的identifier
#             # 区分：now_query指向last_indentifier->和full_query merge之后得到new_query->经过验证->更新full_query
#             # logging.info(f"start agentic verify...")
#             # backward_verify_result = module.verify(now_query, last_identifier, max_step=args.max_verify_step)
#             # if isinstance(backward_verify_result, dict) and "error" in backward_verify_result:
#             #     logging.warning(f"[Failed]: The intermediate query fail the agentic verification. Due to: {backward_verify_result['error']}")
#             #     continue

#             # # if agent can not answer correcly, then the agent trajectory is wrong
#             # if backward_verify_result["agent_score"] <= 0:
#             #     logging.warning(f"[Failed]: The extended task in the {hop} attempt failed because llm can solve it.")
#             #     continue

#             # step3 query merge
#             success = False
#             for _ in range(args.max_merge_retry):
#                 #try:
#                 prompt = depth_prompt_templates['rag_merge_query_prompt'].format(core_query=args.query, golden_answer=args.golden_answer, auxiliary_query=now_query, auxiliary_answer=last_identifier) #修改prompt OK
#                 new_query = run_llm_prompt(model, prompt, developer_prompt=None, return_json=True)  # analysis, new_query
#                 if isinstance(new_query, dict) and "error" in new_query: #？
#                     logging.warning(f"[Failed]: Fail to merge a new query.")
#                     continue
#                 new_query = new_query['new_query']

#                 # judge new query
#                 # query_compare_prompt = depth_prompt_templates['query_compare_prompt'].format(
#                 #     last_identifier=last_identifier,
#                 #     new_task=new_query,
#                 #     full_query=full_query,
#                 #     golden_answer=args.golden_answer
#                 # )
#                 # query_judge = run_llm_prompt(model, query_compare_prompt, developer_prompt=None, return_json=True)  # analysis, is_valid
#                 # if isinstance(query_judge, dict) and "error" in query_judge: #？
#                 #     logging.warning(f"[Failed]: Fail to judge the merged query. The merged query is: {new_query}")
#                 #     continue
#                 # if not query_judge["is_valid"]:
#                 #     logging.warning(
#                 #         f"\n[Failed]: The merged query expression do not pass semantic analysis. \n[Core query]: {full_query}. \n[Auxiliary query]: {now_query}.\n [Merged query]: {new_query}\n")
#                 #     continue
#                 # 上面是直接用LLM来判断新query是否满足要求，可能对于他的环境来说可以吧，但是我觉得可以分为: 
#                 # 1.LLM自行搜索能否解决（因为用的是之前生成的问题，而之前的问题都有过验证LLM是不能在不搜索的情况下解决的，故省略，如果是没有经过验证的问题，那么需要这个验证步骤）; 
#                 # 2.对LLM给出一条gloden_doc能否解决（理想的one-hop的情况）,也要做一下如果给出所有gloden_doc的话，LLM能否解决问题，这是一个上限？或者是删除？

#                 # 自行搜索
#                 self_search_result = module.verify(new_query, args.golden_answer, max_step=args.max_verify_step)
#                 if isinstance(self_search_result, dict) and "error" in self_search_result:
#                     logging.warning(f"[Failed]: The merged query fail the agentic verification. Due to: {self_search_result['error']}")
#                     continue
#                 llm_score = self_search_result["llm_score"]

#                 # 给出一条gloden_doc能否解决
#                 gloden_doc_prompt = """You are given the following document that contains relevant information to help answer a question.
#                 Document:
#                 \"\"\"
#                 {golden_doc}
#                 \"\"\"
#                 Question:
#                 {new_query}
#                 Please answer the question using ONLY the information in the provided document. Return the final answer directly, with no explanation.
#                 """
#                 verify_agent = module.verify_agent(model, "verify_agent", search_agent=module.search_agent, max_step=5)
#                 # 原本的doc
#                 other_answer = run_llm_prompt(model, gloden_doc_prompt.format(golden_doc=args.golden_doc, new_query=new_query), developer_prompt=None, return_json=True) 
#                 gloden_doc_result = verify_agent.recall_score(args.golden_answer, other_answer, model, num_parallel_predictions=1)
#                 if gloden_doc_result > 0:
#                     logging.warning(f"[Failed]: Only need original doc to solve the problem.")
#                     continue

#                 # 检索到的doc
#                 other_answer = run_llm_prompt(model, gloden_doc_prompt.format(golden_doc=now_doc, new_query=new_query), developer_prompt=None, return_json=True) 
#                 now_doc_result = verify_agent.recall_score(args.golden_answer, other_answer, model, num_parallel_predictions=1)
#                 if now_doc_result > 0:
#                     logging.warning(f"[Failed]: Only need source doc to solve the problem.")
#                     continue
                
#                 # 给出所有doc
#                 other_answer = run_llm_prompt(model, gloden_doc_prompt.format(golden_doc=args.golden_doc+'\n'+now_doc, new_query=new_query), developer_prompt=None, return_json=True) 
#                 full_doc_result = verify_agent.recall_score(args.golden_answer, other_answer, model, num_parallel_predictions=1)
#                 if full_doc_result <= 0:
#                     logging.warning(f"[Failed]: This question can't resove.")
#                     continue

#                 success = True
#                 break
#                 # except Exception as e:
#                 #     logging.warning("[Failed]: Exception occurred while merging query: " + str(e))
#                 #     continue

#             if not success:
#                 logging.warning(f"[Failed]: Fail to merge a new query in {args.max_merge_retry} tries.")
#                 continue

#             valid_hop += 1
#             logging.info(f"[Success]: The extended task in the {hop + 1} attempt: {new_query}")

#             full_results.append({
#                 "query": now_query,
#                 "answer": last_identifier,
#                 "valid_hop": vaild_hop,
#                 # "trajectory": backward_verify_result['agent_trajectory'],
#                 # "backward_agent_score": backward_verify_result["agent_score"], # now_query 能否得到 last_identifier 的分数
#                 # "backward_llm_score": backward_verify_result["llm_score"],
#                 "now_doc": now_doc,
#             })

#             full_results.append({
#                 "query": new_query,
#                 "answer": args.golden_answer,
#                 "valid_hop": valid_hop,
#                 "last_doc_result": golden_doc_result,
#                 "now_doc_result": now_doc_result,
#                 "full_doc_result": full_doc_result,
#             })

#             # 更新相关参数
#             last_identifier = backward_result['identifier']
#             full_query = new_query
#             args.golden_doc = args.golden_doc + '\n' + now_doc

#             if valid_hop >= args.max_hops:
#                 logging.info(f"Reach the max hops: {args.max_hops}, stop extending.")
#                 break
#         except Exception as e:
#             logging.warning("[Failed]: Exception occurred while generate new question: " + str(e))
#             continue 

#     if len(full_results) <= 1:
#         logging.warning(f"Failed to extend the query in {args.extended_attempts} tries.")
#         return None, None, None

#     # return final question and answer
#     return full_results, full_results[-1]["query"], full_results[-1]["answer"]

#     # # postprocess trajectory
#     # solve_trajectory = {
#     #     "question": full_results[-1]["query"],
#     #     "golden_answer": full_results[-1]["answer"],
#     #     "trajectory": [],
#     # }
#     # solve_trajectory["trajectory"].append({
#     #     "sub_query": full_results[0]["initial_query"],
#     #     "sub_answer": full_results[0]["initial_answer"],
#     #     "sub_trajectory": full_results[0].get("trajectory", None) # None if the initial query does not have a trajectory
#     # })
#     # # full_results[0]是初始的问题数据，之后full_results[1]是生成的中间问题的步骤（这个包括用toolagent backward的trajectory，full_results[2]是生成的最终问题的步骤
#     # for i in range(1, len(full_results), 2):
#     #     item = full_results[i]
#     #     assert item["trajectory"] is not None
#     #     solve_trajectory["trajectory"].append({
#     #         "sub_query": item["query"],
#     #         "sub_answer": item["answer"],
#     #         "sub_trajectory": item["trajectory"]
#     #     })
#     # solve_trajectory["trajectory"] = solve_trajectory["trajectory"][::-1] #翻转