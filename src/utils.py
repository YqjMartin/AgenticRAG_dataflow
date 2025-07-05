# -*- coding: utf-8 -*-
# @Project      : taskcraft
# @File         : utils.py
# @Author       : Qianben Chen <chenqianben@oppo.com>
# @LastUpdated  : 2025/6/11
# @LICENSE      : Apache License 2.0


import os
import json
import yaml
import re


CUSTOM_ROLE_CONVERSIONS = {"tool-call": "assistant", "tool-response": "user"}
AUTHORIZED_IMPORTS = [
    "pdfplumber",
    "requests",
    "zipfile",
    "os",
    "pandas",
    "numpy",
    "sympy",
    "json",
    "bs4",
    "pubchempy",
    "xml",
    "yahoo_finance",
    "Bio",
    "sklearn",
    "scipy",
    "pydub",
    "io",
    "PIL",
    "chess",
    "PyPDF2",
    "pptx",
    "torch",
    "datetime",
    "fractions",
    "csv",
    "random",
    "re",
    "sys",
    "shutil",
    "json"
]


def read_jsonl(infile):
    data = []
    with open(infile, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def write_jsonl(outfile, data, mode='w'):
    with open(outfile, mode, encoding='utf-8') as f:
        for line in data:
            f.write(json.dumps(line, ensure_ascii=False) + '\n')

def read_json(infile):
    with open(infile, 'r', encoding='utf-8') as f:
        return json.load(f)

def write_json(outfile, data, mode='w'):
    with open(outfile, mode, encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def read_txt(infile):
    with open(infile, 'r', encoding='utf-8') as f:
        return f.read()

def write_txt(outfile, data, mode='w'):
    with open(outfile, mode, encoding='utf-8') as f:
        f.write(data)

def load_yaml(infile):
    with open(infile, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def write_yaml(outfile, data, mode='w'):
    with open(outfile, mode, encoding='utf-8') as f:
        yaml.dump(data, f, allow_unicode=True)

def safe_json_loads(text):
    text = text.lstrip("```json").rstrip("```").strip()
    try:
        return json.loads(text)
    except Exception as e:
        return str(text)

def run_llm_prompt(model, prompt, developer_prompt=None, only_return_msg=False, return_json=False, max_retries=3):
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

    if only_return_msg:
        return messages
    else:
        last_error = None
        for _ in range(max_retries):
            try:
                response = model(messages)
                return safe_json_loads(response.content) if return_json else response.content
            except Exception as e:
                last_error = f"[run_llm_prompt] error: {e}"
        raise Exception(str(last_error))
        

def run_llm_msg(model, msg, prompt, developer_prompt=None, only_return_msg=False, return_json=False, max_retries=3):
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

    for id, prompt in enumerate(msg):
        if id % 2 == 0:
            messages.append({"role": "user", "content": [
                {"type": "text", "text": f"{prompt}"}]})
        else:
            messages.append({"role": "assistant", "content": [
                {"type": "text", "text": f"{prompt}"}]})
    
    if only_return_msg:
        return messages
    else:
        last_error = None
        for _ in range(max_retries):
            try:
                response = model(messages)
                return safe_json_loads(response.content) if return_json else response.content
            except Exception as e:
                # print(f"API Error: {e}, retrying...")
                last_error = f"[run_llm_msg] error: {e}"
        raise Exception(str(last_error))


def extract_answer(output_text):
    """Helper function to extract answer from output text"""
    answer_match = re.search(r'<answer>(.*?)</answer>', output_text, re.DOTALL)
    return answer_match.group(1).strip() if answer_match else None
