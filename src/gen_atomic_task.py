# -*- coding: utf-8 -*-
# @Project      : taskcraft
# @File         : gen_atomic_task.py
# @Author       : Dingfeng Shi <shidingfeng@outlook.com>, Jingyi Cao <224040283@link.cuhk.edu.cn>, Qianben Chen <chenqianben@oppo.com>
# @LastUpdated  : 2025/6/11
# @LICENSE      : Apache License 2.0

import argparse
import concurrent.futures
import os
import time
import types
from functools import partial
from pathlib import Path
from typing import List, Tuple, Literal
import langid
import logging
import json
import requests
from dotenv import load_dotenv
from tqdm import tqdm

from .data_processor import (
    identify_address_type,
    find_main_title_enhanced,
    extract_html_content,
    get_content_identifier,
    split_text_intelligently,
    get_image_from_pdf
)
from .utils import (
    CUSTOM_ROLE_CONVERSIONS,
    load_yaml,
    run_llm_prompt,
    write_jsonl
)
from .tools import (
    CrawlerReadTool,
    SimpleCrawler,
    VisualInspectorTool,
    VerifyAgent
)
from oagents import OpenAIServerModel

load_dotenv(override=True)

# load prompt templates
atomic_prompt_yaml_path = Path(__file__).parent / "prompts/atomic_prompts.yaml"
atomic_prompt_templates = load_yaml(atomic_prompt_yaml_path)
general_prompt_yaml_path = Path(__file__).parent / "prompts/general_prompts.yaml"
general_prompt_templates = load_yaml(general_prompt_yaml_path)

######################################## STEP 1 ########################################
def trans_language(
        model,
        bad_conclusions: List[str],
        language: str
) -> List[str]:
    """
    Translate the given bad conclusions into the specified language.
    
    Args:
        model: The model to use for translation.
        bad_conclusions: A list of conclusions that need translation.
        language: The target language for translation.
        
    Returns:
        A list of translated conclusions.
    """
    developer_prompt = atomic_prompt_templates["trans_language_prompt"].format(language=language)
    all_valid_conclusions = []
    remaining_bad = bad_conclusions.copy()
    for _ in range(3):
        prompt = f"Conclusions to translate: {remaining_bad}"
        response = run_llm_prompt(model, prompt, developer_prompt, return_json=True)
        translated = response.get('conclusions', [])
        if not isinstance(translated, list):
            continue

        valid_conclusions = []
        bad_conclusions = []
        for original, translation in zip(remaining_bad, translated):
            lang, _ = langid.classify(translation)
            if lang == language:
                valid_conclusions.append(translation)
            else:
                bad_conclusions.append(original)

        all_valid_conclusions.extend(valid_conclusions)
        remaining_bad = bad_conclusions
        if not remaining_bad:
            break
    return all_valid_conclusions

def download_if_url(input_str, save_path):
    if input_str.startswith("http://") or input_str.startswith("https://"):
        logging.warning("try to directly process file with URL, which may be not fully support.")
        logging.info("start downloading file from URL: %s", input_str)
        print("start downloading file from URL: %s", input_str)

        with requests.Session() as session:
            response = session.get(input_str, stream=True)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))
            with open(save_path, "wb") as f, tqdm(
                    desc="Downloading",
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
            ) as progress_bar:
                for chunk in response.iter_content(chunk_size=1024):
                    f.write(chunk)
                    progress_bar.update(len(chunk))

        logging.info("successfully downloaded file from URL: %s", input_str)
        print("successfully downloaded file from URL: %s", input_str)
        return True
    return False

def get_initial_conclusion(
        args,
        model,
        content_identifier,
        language,
        prompt_input,
):
    """
    
    This function retrieves initial conclusions from the provided document section
    based on the specified model and language.

    Args:
        model: The model used for processing the content.
        section: The section of the document to be processed.
        language: The target language for the conclusions.
    Returns:
        A list of validated conclusions, each containing the conclusion text, 
        its relationship (R), and the content identifier.
    """
    if args.modal == "single":
        developer_prompt = atomic_prompt_templates["get_initial_conclusion_for_default_prompt"]
        prompt = f"""
        The document content to be processed is as follows: {prompt_input}
        """
    elif args.modal == "multi":
        current_description = prompt_input['description']
        current_analysis = prompt_input['analysis']
        developer_prompt = atomic_prompt_templates["get_initial_conclusion_for_image_prompt"]
        prompt = f"""
        The following is the description of the image with the caption "{current_description}" in the article titled "{content_identifier}":
        {current_analysis}
        """

    response = run_llm_prompt(model, prompt, developer_prompt, return_json=True)

    validated_conclusions = []
    for item in response:
        conclusion = item['conclusion']
        R = item['R']
        validated_conclusions.append({
            'conclusion': conclusion,
            'R': R,
        })
    return validated_conclusions


def get_candidate_tasks(args, model, content_identifier, language, readed_context):
    if args.modal == "single":
        chunk_size = args.chunk_size if hasattr(args, "chunk_size") else 16384
        sub_readed_context_list = split_text_intelligently(readed_context[0]["text"], chunk_size=chunk_size,
                                                           chunk_overlap=0)
    elif args.modal == "multi":
        sub_readed_context_list = readed_context

    all_conclusions = []
    futures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        for sub_ctx in sub_readed_context_list:
            futures.append(executor.submit(get_initial_conclusion, args, model, content_identifier, language, sub_ctx))
        for future, sub_readed_context in zip(concurrent.futures.as_completed(futures), sub_readed_context_list):
            result = future.result()
            if args.modal == "multi":
                for item in result:
                    item["R"] = f"{item['R']}, in {sub_readed_context['description']}."
            all_conclusions.extend(result)
            if args.max_candiated_conclusions > 0 and len(all_conclusions) >= args.max_candiated_conclusions:
                break

    if args.max_candiated_conclusions < 0:
        return all_conclusions
    return all_conclusions[:args.max_candiated_conclusions]


def read_indentifier_language(
        args,
        model,
        input_reader,
):
    """
    This function reads conclusions from a given input file or URL and returns the content identifier, conclusions, and language.

    Args:
        args: Command line arguments containing parameters like max_pdf_pages and chunk_size.
        model: The model used for processing the content.
        input_reader: A function to read content from the input file or URL.
    Returns:
        A tuple containing:
            - content_identifier: The identifier for the content.
            - all_conclusions: A list of conclusions extracted from the content.
            - language: The language of the extracted conclusions.
    """
    # get content_identifier
    content_format, content_type = identify_address_type(args.input)
    logging.info(f"The recognized content type is: {content_type}, content format is: {content_format}")
    print(f"The recognized content type is: {content_type}, content format is: {content_format}")
    if content_type == 'file':
        # check if the file exists
        if not os.path.exists(args.input):
            # if it is a web file
            if download_if_url(args.input,os.path.join(args.tmp_dir,'tmp')):
                args.input = os.path.join(args.tmp_dir, 'tmp')
            else:
                raise Exception("The input file does not exist or is not a valid URL.")

        if content_format == 'pdf':
            max_num = args.max_pdf_pages if hasattr(args, "max_pdf_pages") else 10
            content_identifier, readed_context = find_main_title_enhanced(args.input, model,
                                                                          max_num=max_num)  # only read first page to improve efficiency
            if not content_identifier or not readed_context:
                logging.error("Can not read file successfully")
                return "", "", "", content_format, content_type
        elif content_format == 'text':
            with open(args.input, 'r', encoding='utf-8') as f:
                readed_context = f.read()
            content_identifier = get_content_identifier(args.input, content_format)
        elif content_format == 'image':
            if args.modal == "single":
                logging.warning("The input file is an image, automatically set the modal to 'multi'.")
                args.modal = "multi"
            return args.input, "en", "", content_format, content_type
        else:
            logging.warning(f"Unsupported file format: {content_format}. It may not be processed correctly.")
            return "", "", "", content_format, content_type
    elif content_type == 'url':
        readed_context = input_reader(args.input)
        if 'Sever Error' in readed_context or 'BalanceError' in readed_context:
            return "", "", "", content_format, content_type
        content_identifier = get_content_identifier(args.input, content_format)
    elif content_type == 'command':
        developer_prompt = general_prompt_templates["get_identifier_prompt"]
        prompt = f"""Now process this question: {args.input}"""
        identifier_result = run_llm_prompt(model, prompt, developer_prompt=developer_prompt, return_json=True)
        content_identifier = identifier_result["content_identifier"]
        logging.info(f"Get content identifier from the input text: {content_identifier}")
        print(f"Get content identifier from the input text: {content_identifier}")
        readed_context = args.input

    language, _ = langid.classify(readed_context)
    return content_identifier, language, readed_context, content_format, content_type


def select_images(model, image_data_list):
    """
    从图像数据列表中筛选出最有价值的前5个图像标题
    """
    titles = []
    for idx, image_data in enumerate(image_data_list):
        titles.append({
            "index": idx,
            "description": image_data.get("description", "")
        })
    system_prompt = atomic_prompt_templates["select_images_prompt"]
    prompt = f'''
            Titles to be processed are as follows:
            {titles}
            '''
    images = run_llm_prompt(model, prompt, system_prompt, return_json=True)
    return images


def analyze_image(args, model, image_path, description):
    """分析图像的主函数"""
    text_limit = args.text_limit if hasattr(args, "text_limit") else 1000000
    inspector = VisualInspectorTool(model=model, text_limit=text_limit)
    question = atomic_prompt_templates["analyze_image_prompt"].format(description=description)
    result = inspector.forward(file_path=image_path, question=question)
    return result


######################################## STEP 2 ########################################
def polish_readed_context(args, model, language, origin_readed_contexts, content_format):
    if args.modal == "single":
        readed_context = [{"text": origin_readed_contexts}]
    elif args.modal == "multi":
        if content_format not in ["pdf", "image"]:
            logging.warning(
                f'Not support data format "{content_format}" for multi-modal mode. We now support ["pdf", "jpg", "png", "bmp","jpeg"].'
                f' So, set to single modal mode.')
            # set to single modal
            args.modal = 'single'
            readed_context = [{"text": origin_readed_contexts}]
        else:
            if content_format == 'pdf':
                image_data_list = get_image_from_pdf(args.input, args.tmp_dir)
                indexs_titles = select_images(model, image_data_list)
                selected_data_list = []
                for index_title in indexs_titles:
                    item = image_data_list[int(index_title["id"])]
                    item["title"] = index_title["title"]
                    selected_data_list.append(item)
            else:
                image_data_list = {
                    "image_path": args.input,
                    "description": args.input,
                    "text": "",
                    "title": os.path.splitext(args.input)[0].split("/")[-1]
                }
                selected_data_list = [image_data_list]

            readed_context = []
            for img_data in selected_data_list:
                analysis_result = analyze_image(args, model, img_data["image_path"], img_data["description"])
                if analysis_result:
                    readed_context.append({
                        **img_data,
                        "analysis": analysis_result,
                    })
                else:
                    continue
    outfile = os.path.join(args.tmp_dir, "readed_context.jsonl")
    write_jsonl(outfile, readed_context, "w")
    return readed_context


######################################## STEP 3 ########################################
def clean_qa(model, input_qa, identifier):
    developer_prompt = atomic_prompt_templates["clean_qa_prompt"]
    prompt = f"The data need to be processed is as follows: {input_qa}"
    response = run_llm_prompt(model, prompt, developer_prompt, return_json=True)
    new_data = {
        "question": response["question"],
        "answer": response["refined_answer"],
        "identifier": identifier,
    }
    return new_data


def generate_initial_question(
        model,
        conclusion_info,
        content_identifier,
):
    """
    Generate an initial question based on the provided conclusion information.

    Args:
        model: The model to use for generating the question.
        conclusion_info: A dictionary containing the conclusion, its identifier, and the relationship (R).
        content_identifier: A unique identifier for the content being processed.
        language: The language of the conclusion.
    Returns:
        A dictionary containing the generated question, the original answer, and the content identifier.
    """

    current_conclusion = conclusion_info['conclusion']
    R = conclusion_info['R']
    system_prompt = atomic_prompt_templates["generate_initial_question_prompt"]
    prompt = f"""
Data to be Processed:
ID: {content_identifier}
R: {R}
A: {current_conclusion}
"""
    parsed_result = run_llm_prompt(model, prompt, system_prompt, return_json=True)
    if not parsed_result or not isinstance(parsed_result, dict):
        return None
    question = parsed_result["Q"]

    input_qa = {
        "question": question,
        "original_answer": current_conclusion,
    }
    cleaned_result = clean_qa(model, input_qa, content_identifier)
    return cleaned_result

def filter_question(args, model, initial_qa):
    question = initial_qa["question"]
    golden_answer = initial_qa["answer"]
    verify_agent = VerifyAgent(model, "verify", debug=args.debug)

    for i in range(3):
        try:
            verify_result = verify_agent(question, golden_answer)

            if "error" in verify_result:
                logging.warning(f"Error in the {i+1}th try verified process: {verify_result['error']}")
                continue
            # agent_step_number is 0, skip this conclusion
            if verify_result["agent_step_number"] == 0:
                logging.warning(f"Agent step number is 0 in {i+1}th try: {initial_qa}")
                continue
            # if llm's answer is correct, then this is not an atomic conclusion
            if verify_result["llm_score"] > 0:
                logging.warning(f"LLM can answer this question directly in {i+1}th try: {initial_qa}")
                continue
            # if agent's answer is 0, then this is not a valid conclusion
            if verify_result["agent_score"] == 0:
                logging.warning(f"Agent cannot solve this question in {i+1}th try: {initial_qa}")
                continue

            qa = {
                "question": initial_qa["question"],
                "golden_answer": initial_qa["answer"],
                "content_identifier": initial_qa["identifier"],
                "model_id": args.model_id,
                "agent_result": verify_result["agent_result"],
                "agent_trajectory": verify_result["agent_trajectory"],
                "agent_score": verify_result["agent_score"],
                "agent_step_number": verify_result["agent_step_number"],
                "llm_result": verify_result["llm_result"],
                "llm_score": verify_result["llm_score"],
            }
            return qa

        except Exception as e:
            logging.error(f"Error in the verified process in {i+1}th try: {e}")
            continue
    return None

def generate_initial_questions(
        args,
        model,
        conclusions,
        content_identifier,
        num_workers,
):
    """
    Generate initial research questions based on the original conclusions (parallel version).

    Args:
        model: The prediction model to use for generating questions.
        conclusions: A list of original conclusions to process.
        content_identifier: A unique identifier for the content being processed.
        num_workers: The maximum number of parallel workers to use.

    Returns:
        A list of all processed results.
    """
    all_results = []
    process_func = partial(
        generate_initial_question, 
        model, 
        content_identifier=content_identifier
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_func, conclusion_info) for conclusion_info in conclusions]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            all_results.append(result)
            if args.max_candidate_atomic > 0 and len(all_results) >= args.max_candidate_atomic:
                break

    if args.max_candidate_atomic < 0:
        return all_results

    return all_results[:args.max_candidate_atomic]

def filter_questions(args, model, initial_qas, num_workers):
    # wrap func
    process_func = partial(
        filter_question,
        args,
        model,
    )

    selected_conclusions = []
    with concurrent.futures.ThreadPoolExecutor(num_workers) as executor:
        futures = {
            executor.submit(process_func, initial_qa): \
                index for index, initial_qa in enumerate(initial_qas, start=1)
        }
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result is not None:
                selected_conclusions.append(result)
    return selected_conclusions

def more_optional_answer(args, model, filtered_question):
    original_answer = filtered_question["golden_answer"]
    system_prompt = atomic_prompt_templates["expand_answer_prompt"]
    prompt = f"""
The original answer is: {original_answer}
Please list all possible textual expressions that have the same meaning or refer to the same entity, especially in different formats (e.g., dates, names, abbreviations).
Respond with a JSON list of strings. Do not explain.
"""
    try:
        response = run_llm_prompt(model, prompt, system_prompt, return_json=True)
        print(f"Response from LLM: {response}")
        optional_answers = response

        if original_answer not in optional_answers:
            optional_answers.insert(0, original_answer)

        result = filtered_question.copy()
        result["optional_answers"] = optional_answers
        return result
    except Exception as e:
        logging.warning(f"[Error]: Failed to expand answer for: {original_answer}, error: {e}")
        return filtered_question

def more_optional_answers(args, model, filtered_questions, num_workers):
    from functools import partial
    import concurrent.futures

    process_func = partial(
        more_optional_answer, 
        args, 
        model
    )

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_func, q) for q in filtered_questions]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result is not None:
                results.append(result)
    return results

def filter_atomic_tasks(
        args,
        model,
        conclusions,
        content_identifier,
        language,
        start_time
):
    """
    This function processes the initial conclusions to generate atomic conclusions and questions.
    It filters out non-atomic conclusions and writes the results to a file.
    
    Args:
        args: Command line arguments containing parameters like tmp_dir and num_workers.
        model: The model used for processing the conclusions.
        conclusions: A list of initial conclusions to process.
        content_identifier: A unique identifier for the content being processed.
        language: The language of the conclusions.
    Returns:
        A list of filtered questions and their corresponding atomic conclusions.
    """
    logging.info(f"The number of canditate conclusions: {len(conclusions)}")
    print(f"The number of canditate conclusions: {len(conclusions)}")
    if len(conclusions) == 0:
        print("No candidate conclusions")
        return []

    # ------------------------ Generate Initial Questions ------------------------
    initial_qas = generate_initial_questions(
        args,
        model,
        conclusions,
        content_identifier,
        num_workers=args.num_workers
    )
    #print("initial_qas:", initial_qas)
    # ------------------------ Filter Questions ------------------------
    filtered_questions = filter_questions(
        args,
        model,
        initial_qas,
        num_workers=args.num_workers
    )
    #print("filtered_questions:", filtered_questions)
    if len(filtered_questions) == 0:
        print("No filtered quesrtions")
        return []
    else:
        print("filtered_questions len:", len(filtered_questions))
        # print("question:", filtered_questions[0]["question"])
        # print("gloden answer:", filtered_questions[0]["golden_answer"])

    #------------------------ More Optional Answer ------------------------
    final_questions = more_optional_answers(
        args,
        model,
        filtered_questions,
        num_workers=args.num_workers
    )
    print("final_questions len:", len(final_questions))
    print("final_questions type:", type(final_questions))

    #------------------------ Write Results ------------------------
    generate_metadata = os.path.join(args.tmp_dir, 'generate_metadata.txt')
    qa_file = os.path.join(args.tmp_dir, 'qa.jsonl')
    with open(generate_metadata, 'a', encoding='utf-8') as file:
        file.write(f"\n=== Final Filtered Atomic Conclusion Statistics ===")
        file.write(f"\nContent Identifier: {content_identifier}")
        file.write(f"\nOriginal conclusion count: {len(initial_qas)}, after filtering atomic conclusions: {len(final_questions)}")
        file.write(f"\nTime taken for atomic conclusion generation: {time.time() - start_time} seconds")
        file.write('\n')
    with open(qa_file, 'a', encoding='utf-8') as file:
        for i, q in enumerate(final_questions, 1):
            if "question" not in q or "optional_answers" not in q:
                print(f"Warning: The {i}th element in filtered_question is incorrectly formatted, skipping this element.")
                continue
            qa_dict = {
                "question": q["question"],
                "golden_answer": q["optional_answers"]
            }
            file.write(json.dumps(qa_dict, ensure_ascii=False) + "\n")
        
    return final_questions


def gen_atomic_tasks(
        input: str,
        tmp_dir: str = "output",
        modal: Literal["single", "multi"] = "single",
        max_candiated_conclusions: int = 20,
        max_candidate_atomic: int = 10,
        model_id: str = "gpt-4.1",
        num_workers: int = 1,
        debug: bool = False,
        max_completion_tokens: int = 8192,
        chunk_size: int = 16384,
        max_pdf_pages: int = 10,
        text_limit: int = 1000000,
        return_readed_context: bool = False,
):
    """
    This function generates atomic tasks from a given input file or URL.
    Args:
        input: the path of input file
        tmp_dir: the temporal directory to save intermediate results (e.g., image from PDF)
        modal: the mode of operation, could be "single" or "multi". When "single", only text is processed. When "multi",
               images are also processed.x
        max_candiated_conclusions: the maximum number of candidate conclusions to be extracted from the input file.
                                   If -1, all conclusions will be processed.
        max_candidate_atomic: the maximum number of atomic tasks to be validated with agent.
                              If -1, all candidates will be validated
        model_id: the model ID to be used for processing, e.g., "gpt-4.1".
        num_workers: the number of parallel workers to be used for processing.
        debug: if True, debug mode is enabled, which will print more information.
        max_completion_tokens: the maximum number of completion tokens for the model.
        chunk_size: the size of text chunks to be processed at once for the text in the input file, default is 16384.
        max_pdf_pages: the maximum number of PDF pages to be processed, default is 10.
        text_limit: the maximum text length to be processed for the crawler, default is 1000000.
        return_readed_context: bool, if True, the function will return the read context.

    Returns:
        Dict: { "candidate_tasks": List[Dict],
                "atomic_tasks": List[Dict],
                "language": str,
                "readed_context": List[Dict] (optional)}
        each item in atmoic_task is a dictionary with keys:
            question, golden_answer, content_identifier, model_id, agent_result, agent_trajectory, agent_score,
            agent_step_number, llm_result, llm_score.
    """
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir, exist_ok=True)

    start_time = time.time()
    crawler = SimpleCrawler(serpapi_key=os.getenv("SERP_API_KEY"))
    input_reader = CrawlerReadTool(crawler, read_type='jina_read')
    model = OpenAIServerModel(
        model_id,
        custom_role_conversions=CUSTOM_ROLE_CONVERSIONS,
        max_completion_tokens=max_completion_tokens,
        api_key=os.environ.get("OPENAI_API_KEY"),
        api_base=os.environ.get("OPENAI_API_BASE"),
    )

    args = types.SimpleNamespace(
        input=input,
        tmp_dir=tmp_dir,
        modal=modal,
        max_candiated_conclusions=max_candiated_conclusions,
        max_candidate_atomic=max_candidate_atomic,
        model_id=model_id,
        num_workers=num_workers,
        debug=debug,
        max_completion_tokens=max_completion_tokens,
        chunk_size=chunk_size,
        max_pdf_pages=max_pdf_pages,
        text_limit=text_limit
    )

    logging.info(f"Start to read content identifier and language from input file: {input}")
    print(f"Start to read content identifier and language from input file: {input}")
    try:
        content_identifier, language, readed_context, content_format, content_type = read_indentifier_language(
            args,
            model,
            input_reader
        )
        # print("content_identifier:", content_identifier)
        # print("language:", language)
        # print("readed_context:", readed_context)
        # print("content_format:", content_format)
        # print("content_type:", content_type)
    except Exception as e:
        logging.error(f"Error reading content identifier and language: {e}")
        return None

    # step2: in "multi_modal" mode, extract image analysis
    logging.info("Start to polish context.")
    print("Start to polish context.")
    try:
        readed_context = polish_readed_context(args, model, language, readed_context, content_format)
    except Exception as e:
        logging.error(f"Failed to polish readed context: {e}")
        return None

    if not readed_context:
        logging.warning("failed to get context, please retry or check the input file or URL.")
        return None

    # step3: get conclusions
    try:
        logging.info("Start to get candidate tasks from the context.")
        print("Start to get candidate tasks from the context.")
        candidate_tasks = get_candidate_tasks(
            args,
            model,
            content_identifier,
            language,
            readed_context
        )
        #print("candidate_tasks:", candidate_tasks)
    except Exception as e:
        logging.error(f"Failed to get candidate conclusions: {e}")
        return None

    if not candidate_tasks:
        logging.error("Failed to get candidate conclusions from the context.")
        return None

    # step4: get atomic task based on conclusions and content identifier
    logging.info("Start to generate atomic tasks from the candidate conclusions.")
    print("Start to generate atomic tasks from the candidate conclusions.")
    try:
        filtered_atomic_tasks = filter_atomic_tasks(
            args,
            model,
            candidate_tasks,
            content_identifier,
            language,
            start_time
        )
        print("filtered_atomic_tasks:", filtered_atomic_tasks)
    except Exception as e:
        logging.error(f"Failed to generate atomic tasks: {e}")
        return None

    if not filtered_atomic_tasks:
        logging.warning("Did not get atomic tasks from the conclusions.")
        return None

    ret = {"candidate_tasks": candidate_tasks,
           "atomic_tasks": filtered_atomic_tasks,
           "language": language}

    if return_readed_context:
        ret.update({"readed_context": readed_context})

    return ret


###############################################################################################################################################
if __name__ == "__main__":
    # ret = gen_atomic_tasks('test/2505.00582.pdf', modal='multi')
    ret = gen_atomic_tasks(
        input='"2012 PDC World Youth Championship"\n2012 PDC World Youth Championship The 2012 PDC World Youth Championship was the second edition of the PDC World Youth Championship, a tournament organised by the Professional Darts Corporation for darts players aged between 14 and 21. The knock-out stages from the last 64 to the semi-finals were played in Crawley on 4 November 2011 and weren\'t broadcast live on TV. The final took place on 17 May 2012, before the final of the 2012 Premier League Darts, which was shown live on Sky Sports. The two finalists became PDC Pro Tour card holders for 2012 and 2013, and also', 
        tmp_dir="output", 
        modal='single',
        max_candiated_conclusions=3,
        max_candidate_atomic=1,
        model_id="gpt-4.1",
        num_workers=1,
        debug=True,
        max_completion_tokens=1024,
        chunk_size=1024,
        max_pdf_pages=1,
        return_readed_context=True)
