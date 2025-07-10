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

def _clean_json_block(item: str) -> str:
    return item.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()

class MultihopExtend:
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

        # self.prompt_templates = {
        #     "augmented_question": depth_prompt_templates['augmented_question_prompt'],
        #     "backward_task_prompt": depth_prompt_templates['backward_task_prompt'],
        #     "generate_query_prompt": depth_prompt_templates['generate_query_prompt'],
        #     "is_superset_valid": depth_prompt_templates['is_superset_valid'],
        #     "rag_backward_task_prompt": depth_prompt_templates['rag_backward_task_prompt'],
        #     "rag_generate_query_prompt": depth_prompt_templates['rag_generate_query_prompt'],
        # }

    def forward(self, original_question, original_answer, original_doc, max_step=10, max_retries=3, **kwargs):
        """
        以answer为锚点，向后扩展生成新的问题
        """
        forward_agent = self.search_agent(self.model, 'forward_agent', max_step=max_step, **kwargs)

        # 生成下一个QA
        forward_question = depth_prompt_templates['rag_forward_prompt'].format(original_answer=original_answer, original_question=original_question, original_doc=original_doc)
        forward_result = forward_agent(forward_question, return_json=True, max_retries=max_retries)
        if isinstance(forward_result, dict) and "error" in forward_result:
            raise Exception("error:", forward_result['error'])
        forward_result = forward_result["agent_result"]
        if isinstance(forward_result, str):
            forward_result = json.loads(_clean_json_block(forward_result))
        return forward_result

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
            logging.warning(e)
            continue
    raise Exception("Fail to generate QA")


def process_single_task(args):
    # initialze model
    model = OpenAIServerModel(
        args.model_id,
        custom_role_conversions=CUSTOM_ROLE_CONVERSIONS,
        max_completion_tokens=8192,
        api_key=os.environ.get("OPENAI_API_KEY"),
        api_base=os.environ.get("OPENAI_API_BASE"),
    )
    module = MultihopExtend(model, search_agent=args.search_agent, verify_agent=args.verify_agent)

    full_results = []
    hop = 1
    initial_question = ""
    initial_answer = ""
    final_question = ""
    final_answer = ""
    final_doc = ""
    first_doc_result = -1
    new_doc_result = -1
    full_doc_result = -1
    is_success = False

    for att_now in range(args.extended_attempts):
        try:
            # initial qa
            if att_now == 0:
                if not args.have_original_qa:
                    q, a, idf = generate_initial_qa_with_identifier(args.golden_doc, model)
                    initial_question = q
                    initial_answer = a 
                    final_question = initial_question
                    final_answer = initial_answer
                    full_results.append({
                        'initial_question': initial_question,
                        'initial_answer': initial_answer,
                        'initial_trajectory': args.trajectory,
                        'initial_doc': args.golden_doc,
                        'hop': 1
                    })
                else:
                    initial_question = args.query
                    initial_answer = args.golden_answer
                    final_question = initial_question
                    final_answer = initial_answer
                    full_results.append({
                        'initial_question': initial_question,
                        'initial_answer': initial_answer,
                        'initial_trajectory': args.trajectory,
                        'initial_doc': args.golden_doc,
                        'hop': 1
                    })
                final_doc = args.golden_doc
            # step1 forward
            forward_result = module.forward(final_question, final_answer, final_doc)
            new_question = forward_result['new_question']
            new_answer = forward_result['new_answer']
            new_doc = forward_result['source_doc']

            # step2 query merges
            prompt = depth_prompt_templates['forward_merge_prompt'].format(original_question=final_question, original_answer=final_answer, original_doc=final_doc,
                                                                           new_question=new_question, new_answer=new_answer, new_doc=new_doc)
            final_qa = run_llm_prompt(model, prompt, developer_prompt=None, return_json=True)
            if isinstance(final_qa, str):
                final_qa = json.loads(_clean_json_block(final_qa))

            # 验证
            gloden_doc_prompt = """You are given the following document that contains relevant information to help answer a question.
            Document:
            \"\"\"
            {doc}
            \"\"\"
            Question:
            {question}
            Please answer the question using ONLY the information in the provided document. Return the final answer directly, with no explanation.
            """

            verify_agent = module.verify_agent(model, "verify_agent", search_agent=module.search_agent, max_step=5)
            # 原本的doc
            other_answer = run_llm_prompt(model, gloden_doc_prompt.format(doc=final_doc, question=final_qa['final_question']), developer_prompt=None, return_json=True) 
            first_doc_result = verify_agent.recall_score(final_qa['final_answer'], other_answer, model, num_parallel_predictions=1)
            if first_doc_result > 0:
                logging.warning("[Failed]: Only need original doc to solve the problem.")
                continue

            # 检索到的doc
            other_answer = run_llm_prompt(model, gloden_doc_prompt.format(doc=new_doc, question=final_qa['final_question']), developer_prompt=None, return_json=True) 
            new_doc_result = verify_agent.recall_score(final_qa['final_answer'], other_answer, model, num_parallel_predictions=1)
            if new_doc_result > 0:
                logging.warning("[Failed]: Only need source doc to solve the problem.")
                continue
            
            # 给出所有doc
            other_answer = run_llm_prompt(model, gloden_doc_prompt.format(doc=final_doc + '\n' + new_doc, question=final_qa['final_question']), developer_prompt=None, return_json=True) 
            full_doc_result = verify_agent.recall_score(final_qa['final_answer'], other_answer, model, num_parallel_predictions=1)
            if full_doc_result <= 0:
                logging.warning("[Failed]: This question can't resove.")
                continue

            hop += 1
            is_success = True
            logging.info(f"[Success] {final_question}")

            full_results.append({
                'initial_question': initial_question,
                'initial_answer': initial_answer,
                'initial_trajectory': args.trajectory,
                'initial_doc': args.golden_doc,
                'new_question': new_question,
                'new_answer': new_answer,
                'new_doc': new_doc,
                'final_question': final_qa['final_question'],
                'final_answer': final_qa['final_answer'],
                'first_doc_result': first_doc_result,
                'new_doc_result': new_doc_result,
                'full_doc_result': full_doc_result,
                'is_success': is_success,
                'hop': hop
            })

            # 更新相关参数
            final_question = final_qa['final_question']
            final_answer = final_qa['final_answer']
            final_doc = final_doc + '\n' + new_doc

            if hop >= args.max_hops:
                break

        except Exception as e:
            logging.warning(f"[Failed] in attempt:{att_now} " + str(e))
            continue 

    if len(full_results) <= 1:
        logging.warning(f"Failed to extend the query in {args.extended_attempts} tries.")
        #return None, None, None
    
    full_results.append({
        'initial_question': initial_question,
        'initial_answer': initial_answer,
        'initial_trajectory': args.trajectory,
        'initial_doc': args.golden_doc,
        'new_question': new_question,
        'new_answer': new_answer,
        'new_doc': new_doc,
        'final_question': final_qa['final_question'],
        'final_answer': final_qa['final_answer'],
        'first_doc_result': first_doc_result,
        'new_doc_result': new_doc_result,
        'full_doc_result': full_doc_result,
        'is_success': is_success,
        'hop': hop
    })

    return full_results, full_results[-1]["final_question"], full_results[-1]["final_answer"], is_success

def multihop_extend(
        golden_doc: str,
        have_original_qa: bool,
        query: str = None,
        golden_answer: str = None,
        identifier: str = None,
        trajectory: Optional[List] = None,
        model_id: str = "gpt-4.1",
        extended_attempts: int = 5,
        max_hops=2,
        search_agent='SearchAgent',
        verify_agent='VerifyAgent'
):
    args = types.SimpleNamespace(
        golden_doc=golden_doc,
        have_original_qa=have_original_qa,
        query=query,
        golden_answer=golden_answer,
        identifier=identifier,
        trajectory=trajectory,
        model_id=model_id,
        extended_attempts=extended_attempts,
        max_hops=max_hops,
        search_agent=search_agent,
        verify_agent=verify_agent,
    )
    full_result, final_question, final_answer, is_success = process_single_task(args)
    return full_result, final_question, final_answer, is_success