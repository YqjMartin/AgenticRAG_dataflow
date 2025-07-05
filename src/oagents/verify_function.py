#load test case 
import json
import os
from openai import OpenAI
# from browser_env.utils import pil_to_b64, pil_to_vertex
import numpy as np
from PIL import Image
import requests
import re
import openai
import asyncio  # 添加异步支持

KEY = ""
URL = ""  # 智慧地球
MODEL = "gpt-4.1"

client = openai.OpenAI(api_key=KEY, base_url=URL, timeout=600.0, max_retries=3)

def exact_content(response):
    content_list=[]
    stack=[]
    extracted_dict={
        'score':0,
        'analysis':''
    }
    for i, char in enumerate(response):
        if char == '{':
            if not stack:  # 如果是第一个左括号，记录开始位置
                start = i
            stack.append(char)
        elif char == '}':
            if stack:  # 有匹配的左括号
                stack.pop()
                if not stack and start is not None:  # 括号对完全匹配
                    # 包含大括号的完整内容
                    content_list.append(response[start:i+1])

    # 尝试解析每一对大括号中的内容
    extraction_success = False
    for content in content_list:
        # print(content)
        try:
            # 去除可能存在的多余逗号
            cleaned_content = re.sub(r',\s*\}', '}', content)

            # 修复缺少引号的键
            cleaned_content = re.sub(r'(?<=\{)\s*([^"]+?)\s*:', r'"\1":', cleaned_content)

            # 处理没有多余逗号的情况
            cleaned_content = re.sub(r'(?<=\})\s*$', '', cleaned_content)

            extracted_dict = json.loads(cleaned_content)

            if extraction_success:
                break

        except json.JSONDecodeError:
            continue
    return extracted_dict
def evaluate_answer(text, system_prompt, mode='PRM'):
    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": text  # Use the passed text value
        }
    ]
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            max_tokens=256,
            top_p=1.0
        )
        content = response.choices[0].message.content
            
        # if mode=='PRM':
        #     # Extract PRM score
        #     score_match = re.search(r'PRM score:\s*(\d+)', content)
        #     if score_match:
        #         # score=float(score_match.group(1))
        #         return float(score_match.group(1)),content
        # elif mode=='ORM':
        #     # Extract ORM status
        #     score_match = re.search(r'ORM score:\s*(\d+)', content)
        #     if score_match:
        #         return float(score_match.group(1)),content
        # elif mode=='ORM':
        if mode=='ORM' or mode=='PRM':
            # Extract ORM status
            result=exact_content(content)
            return result['score'],result['analysis']
        elif mode=='list-wise':
            result=exact_content(content)
            return result['index'],result['thought']
        else:
            pass

        
        return 0.0,content
    except Exception as e:
        return 0.0,f"Error processing item: {e}"


# Example usage of the new function
if __name__ == "__main__":
    test_text = "this is a test text"
    result =evaluate_answer(test_text)
    print(result)




