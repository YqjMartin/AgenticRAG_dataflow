# -*- coding: utf-8 -*-
# @Project      : taskcraft
# @File         : gen_width_based_task.py
# @Author       : Jingyi Cao <224040283@link.cuhk.edu.cn>
# @LastUpdated  : 2025/6/11
# @LICENSE      : Apache License 2.0

import json
import os
from pathlib import Path
from typing import List

from oagents import OpenAIServerModel
from src.tools import *
from src.utils import CUSTOM_ROLE_CONVERSIONS, load_yaml

MAX_RETRIES = 3

# load prompt templates
verify_prompt_yaml_path = Path(__file__).parent / "prompts/verify_prompts.yaml"
verify_prompt_template = load_yaml(verify_prompt_yaml_path)
width_prompt_yaml_path = Path(__file__).parent / "prompts/width_prompts.yaml"
width_prompt_templates = load_yaml(width_prompt_yaml_path)
general_prompt_yaml_path = Path(__file__).parent / "prompts/general_prompts.yaml"
general_prompt_templates = load_yaml(general_prompt_yaml_path)


def retry_predict(model, prompt, developer_prompt=None):
    messages = []
    if developer_prompt:
        developer_text = {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": developer_prompt
                }
            ]
        }
        messages.append(developer_text)

    messages.append({
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": f"{prompt}",

            }
        ],
    }, )

    for _ in range(MAX_RETRIES):
        try:
            response = model(messages)
            if response.content:
                break
        except Exception as e:
            print(f"API Error: {e}, retrying...")

    return response.content

def check_queries(model_id, qa_batch):
    """
    Check if synthesized complex questions meet quality standards through two validation steps:
    1. Verify if complex questions can be decomposed back to original questions
    2. Verify if the LLM can answer the complex questions correctly

    :param model: model instance
    :param qa_batch: list of complex questions to validate (batch of 10)
    :return: list of validated complex questions
    """
    model = OpenAIServerModel(
        model_id,
        custom_role_conversions=CUSTOM_ROLE_CONVERSIONS,
        max_completion_tokens=8192,
        api_key=os.environ.get("OPENAI_API_KEY"),
        api_base=os.environ.get("OPENAI_API_BASE"),
    )
    # Step 1: Check if complex questions can be decomposed to original questions
    developer_prompt_step1 = width_prompt_templates['check_prompt_1']

    input_batch = []
    for q in qa_batch:
        input_batch.append({
            "index": q["index"],
            "complex_question": q["question"],
            "original_questions": q["original_questions"]
        })

    prompt_step1 = f"""
    Here are the base questions to process:
    {json.dumps(input_batch, indent=2, ensure_ascii=False)}
    Each dictionary contains: index (unique ID), complex_question (original complex question), 
    and original_questions (list of original questions).
    """

    validated_questions = []

    for _ in range(3):  # Retry up to 3 times
        try:
            response = retry_predict(model, prompt_step1, developer_prompt_step1)
            response = json.loads(response)

            for item in response:
                if item.get("state", 0) == 1:
                    # Find the original question in qa_batch
                    for q in qa_batch:
                        if q["index"] == item["index"]:
                            validated_questions.append(q)
                            break
            break
        except Exception as e:
            print(f"Error in step 1 validation: {e}")
            continue

    print(f"After decomposition check: {len(validated_questions)} questions remain")

    # Step 2: Verify if LLM can answer the complex questions
    developer_prompt_step2 = width_prompt_templates['check_prompt_2']
    input_batch = []
    for q in validated_questions:
        input_batch.append({
            "index": q["index"],
            "complex_question": q["question"],
        })
    prompt_step2 = f"""
    Please answer these research questions:
    {json.dumps(input_batch, indent=2, ensure_ascii=False)}
    """

    final_questions = []
    verify_agent = VerifyAgent(model, "verify")
    for _ in range(3):  # Retry up to 3 times
        try:
            response = retry_predict(model, prompt_step2, developer_prompt_step2)
            response = json.loads(response)

            for item in response:
                # Only keep questions where LLM has medium-high confidence (3+)
                score = verify_agent.recall_score(validated_questions[item["index"]]["golden_answers"], item["llm_answer"], model)

                if score < 1:
                    # llm回答不相似
                    for q in validated_questions:
                        if q["index"] == item["index"]:
                            final_questions.append(q)
                            break
            break
        except Exception as e:
            print(f"Error in step 2 validation: {e}")
            continue

    print(f"After LLM answer check: {len(final_questions)} questions remain")
    return final_questions

def width_extend(qa_batch, model_id="gpt-4.1") -> List[dict]:
    """
    Group and merge queries to generate higher-level questions
    :param model: model instance
    :param qa_batch: list of questions and answers (batch of 10)
    :return: list of validated grouped queries
    """
    model = OpenAIServerModel(
        model_id,
        custom_role_conversions=CUSTOM_ROLE_CONVERSIONS,
        max_completion_tokens=8192,
        api_key=os.environ.get("OPENAI_API_KEY"),
        api_base=os.environ.get("OPENAI_API_BASE"),
    )

    developer_prompt = width_prompt_templates['merge_prompt']

    # Create input data without golden_answers
    input_batch = []
    for i, q in enumerate(qa_batch):
        q.update({'index': i})
        input_batch.append(q)

    prompt = f"""
    Here are the base questions to process:
    {json.dumps(input_batch, indent=2, ensure_ascii=False)}
    Each dictionary contains: index (unique ID), question (original question), and content_identifier (identifier).
    """
    for _ in range(3):  # Retry up to 3 times
        try:
            response = retry_predict(model, prompt, developer_prompt)
            response = json.loads(response)


            merged_results = []
            for item in response:
                # Validate required fields
                if not all(k in item for k in ["question", "index"]):
                    continue

                # Get golden_answers for the indices
                golden_answers = []
                original_questions = []
                for idx in item["index"]:
                    golden_answers.append(input_batch[idx]["golden_answer"])
                    original_questions.append(input_batch[idx]["question"])

                merged_results.append({
                    "question": item["question"],
                    "content_identifier": item["content_identifier"],
                    "golden_answers": golden_answers,
                    "qa_index": item["index"],
                    "original_questions": original_questions
                })
            for index, query in enumerate(merged_results):
                query["index"] = index
            # Check if synthesized complex questions meet quality standards
            checked_results=[]
            validation_batches = [merged_results[i:i + 10] for i in range(0, len(merged_results), 10)]
            for batch in validation_batches:
                validated = check_queries(model_id, batch)
                checked_results.extend(validated)



            validated=[]
            for item in checked_results:
                validated.append({
                    "question": item["question"],
                    "content_identifier": item["content_identifier"],
                    "golden_answers": golden_answers,
                    "index": item["qa_index"],
                    "original_questions": original_questions
                })
            print(f"validated questions: {validated}")
            return validated

        except json.JSONDecodeError:
            print("Failed to parse JSON response, retrying...")
        except KeyError as e:
            print(f"Missing required field in response: {e}, retrying...")
        except Exception as e:
            print(f"Error: {e}, retrying...")

    return []  # Return empty list after all retries fail

if __name__ == '__main__':
    # from taskcraft import width_extend
    width_extend(
        [{"question": "According to the paper 'TaskCraft: Automated Generation of Agentic Tasks', what methods in this paper use to expand atomic tasks into structurally and hierarchically complex challenges?",
     "content_identifier": "TaskCraft: Automated Generation of Agentic Tasks",
          "golden_answer": "depth-based and width-based extensions"},
         {"question": "According to the paper '[2506.10055] TaskCraft: Automated Generation of Agentic Tasks', what is the size of the synthetic agentic task dataset produced by TaskCraft and how does its difficulty vary?",
          "content_identifier": "TaskCraft: Automated Generation of Agentic Tasks",
          "golden_answer": "36,000 agentic tasks with varying difficulty"}]
    )
