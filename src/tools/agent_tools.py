# @Project      : taskcraft
# @File         : agent_tools.py
# @Author       : Qianben Chen <chenqianben@oppo.com>
# @LastUpdated  : 2025/6/11
# @LICENSE      : Apache License 2.0

import logging
import os
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Set, Tuple, TypedDict, Union
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .visual_inspector_tool import VisualInspectorTool
from .async_web_crawler_tool import SimpleCrawler, CrawlerSearchTool, CrawlerReadTool, CrawlerArchiveSearchTool
from .text_inspector_tool import TextInspectorTool
from .rag_tool import RAGTool
from taskcraft.src.utils import safe_json_loads, run_llm_prompt, run_llm_msg, load_yaml, write_yaml
from oagents import ToolCallingAgent
from oagents import ActionStep, ReflectionStep, AgentMemory, PlanningStep, SystemPromptStep, TaskStep, ToolCall

# load prompt templates
verify_prompt_yaml_path = Path(__file__).parent.parent / "prompts/verify_prompts.yaml"
verify_prompt_templates = load_yaml(verify_prompt_yaml_path)


class BaseAgent:
    def __init__(self, model, name):
        self.model = model
        self.name = name

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class SearchAgent(BaseAgent):
    def __init__(self, model, name, **kwargs):
        super().__init__(model, name)
        crawler = SimpleCrawler(serpapi_key=os.environ.get("SERPAPI_API_KEY"))
        search = CrawlerSearchTool(crawler)  # inputs: query
        read = CrawlerReadTool(crawler)  # inputs: url
        archive_search = CrawlerArchiveSearchTool(crawler)
        # text_inspect = TextInspectorTool(model, kwargs.get("text_limit", 100000))
        visual_inspect = VisualInspectorTool(model, kwargs.get("text_limit", 100000))
        rag_tool = RAGTool()  # inputs: query
        #tools = [search, read, archive_search, visual_inspect]
        tools = [rag_tool]

        self.agent_fn = ToolCallingAgent(
            model=model,
            tools=tools,
            max_steps=kwargs.get("max_step", 20),
            verbosity_level=2, # DEBUG
            reflection_interval=kwargs.get("reflection_interval", 5),
            name=name,
#             description="""
# A team member who will answer your questions by searching the internet and browsing the web and local wiki knowledge base.  
# You can ask him any questions that require researching information or browsing webpages.  
# Please provide as much background information as possible, especially when you need to search for content within a specific time frame!  
# Do not hesitate to assign complex search tasks to him, such as finding differences between two webpages.  
# If you need to use the search tool and have more than three keywords, please conduct multiple separate searches.  
# Your request must be a complete sentence, not a Google-style search term! For example, "Help me find information about... (…)" rather than just a few keywords.
# """.strip(),
            description="""
This RAG-powered system answers queries by retrieving information from a local wiki knowledge base, then generating responses. It can address any questions requiring information research within the local repository. Please provide comprehensive background details, especially when specifying a particular time frame for the search! It is capable of handling complex retrieval tasks, such as identifying differences between local documents or knowledge entries. When using the retrieval tool, structure your request as a complete sentence—avoid fragmentary search terms! For example, use "Help me find local information about..." instead of isolated keywords. Note: This system does not access external webpages and relies solely on the local knowledge base for content retrieval.""", #使用不同的工具时需要修改description
            provide_run_summary=False,
            prompts_type="w_think_reflect",
            debug=kwargs.get("debug", False),
        )

    def forward(self, task, return_json=False, max_retries=3):
        last_error = None
        for _ in range(max_retries):
            try:
                result = self.agent_fn.run(task)
                if return_json and isinstance(result, str):
                    result = safe_json_loads(result)
                elif not return_json and isinstance(result, dict):
                    result = str(result)

                traj = self.capture_trajectory()
                step_num = self.traj_step_num()
                return {
                    "agent_result": result,
                    "agent_trajectory": traj,
                    "traj_step_num": step_num
                }
            except Exception as e:
                last_error = e
                continue
        return {"error": str(last_error)}

    # Record the trajectory of the agent
    def capture_trajectory(self):
        if not hasattr(self, 'agent_fn'):
            raise ValueError("[capture_trajectory] agent_fn is not defined.")
        if not isinstance(self.agent_fn, ToolCallingAgent):
            raise ValueError("[capture_trajectory] agent_fn must be an instance of ToolCallingAgent.")
        trajectory = []
        for step_num, step in enumerate(self.agent_fn.memory.steps):
            if isinstance(step, TaskStep):
                # traj = {"name": "task", "value": step.task}
                continue
            elif isinstance(step, PlanningStep):
                traj = {"name": "facts", "value": step.facts, "think": step.facts_think}
                trajectory.append(traj)
                traj = {"name": "plan", "value": step.plan, "think": step.plan_think}
                trajectory.append(traj)
            elif isinstance(step, ReflectionStep):
                traj = {"name": "reflection", "value": step.history_trajectory_score,
                        "think": step.history_trajectory_analysis}
                trajectory.append(traj)
                traj = {"name": "facts", "value": step.facts, "think": step.facts_think}
                trajectory.append(traj)
                traj = {"name": "plan", "value": step.plan, "think": step.plan_think}
                trajectory.append(traj)
            elif isinstance(step, ActionStep):
                traj = {"name": "action", "value": step.model_output,
                        "obs": step.action_output if step.action_output else step.observations,
                        "think": step.action_think}
                trajectory.append(traj)
            else:
                raise ValueError("[capture_trajectory] Unknown Step:", step)
        return trajectory

    def traj_step_num(self):
        return self.agent_fn.step_number


class VerifyAgent(BaseAgent):
    def __init__(self, model, name, search_agent=SearchAgent, **kwargs):
        super().__init__(model, name)
        self.forward_llm = partial(run_llm_prompt, model, developer_prompt=None, only_return_msg=False,
                                   return_json=False, max_retries=1)

        self.forward_agent = search_agent(model, name, **kwargs)
        self.judge_model = kwargs.get("judge_model", model)
        self.prompt_templates = {
            "score_prompt_single": verify_prompt_templates['score_prompt_single'],
        }

    def forward(self, query, golden_answer, max_retries=3, metric='recall'):
        assert metric in ['recall', 'acc'], f'evaluation metric must be "recall" or "acc", now is "{metric}"'

        last_error = None

        # augmentation
        arguement_query = """Please solve the following problem and return relevant results that meet the query requirements. Ensure responses are as concise as possible, focusing only on key information while omitting redundant details. Please return the result in JSON format with keys 'answer_list': List[str] the list of answers.

        The task is:
        """
        arguement_query += query

        for _ in range(max_retries):
            try:
                llm_result = self.forward_llm(arguement_query)
                if isinstance(llm_result, dict) and "error" in llm_result: #永远不会执行的无意义代码
                    continue
                llm_score = self.recall_score(golden_answer, llm_result, self.judge_model)

                # if llm can answer correctly, then it is not an atomic conclusion
                if llm_score >= 1:
                    return {
                        "error": "\nLLM can solve this question.\n This question will be omit.\n"
                    }

                agent_result_dict = self.forward_agent(arguement_query, return_json=True)
                if isinstance(agent_result_dict, dict) and "error" in agent_result_dict:
                    continue
                agent_result = agent_result_dict["agent_result"]['answer_list']
                agent_trajectory = agent_result_dict["agent_trajectory"]
                agent_step_num = agent_result_dict["traj_step_num"]

                # recall的分数就是用run_llm_prompt去判断模型的得分，如果完全正确则分数为2，存在额外的答案为1，完全错误为0
                # acc的分数包括两个，一个是是否包含了正确的答案（recall_answer），另一个是模型一共平均输出了多少个答案（valid_infor_num）,感觉是没写清楚的产物
                if metric == 'recall':
                    agent_score = self.recall_score(golden_answer, agent_result, self.judge_model)
                    return {
                        "agent_score": agent_score,
                        "llm_score": llm_score,
                        "agent_result": agent_result,
                        "llm_result": llm_result,
                        "agent_step_number": agent_step_num,
                        "agent_trajectory": agent_trajectory
                    }
                elif metric == 'acc':
                    recall_score, valid_infor_num = self.recall_score(golden_answer, agent_result, self.judge_model)
                    return {
                        "agent_score": recall_score,
                        "llm_score": llm_score,
                        "valid_infor_num": valid_infor_num,
                        "agent_result": agent_result,
                        "llm_result": llm_result,
                        "agent_step_number": agent_step_num,
                        "agent_trajectory": agent_trajectory
                    }

            except Exception as e:
                last_error = e
                continue
        return {"error": last_error}

    @staticmethod
    def _run_parallel_predictions(judge_model, prompt, developer_prompt, num_parallel_predictions=1):
        with ThreadPoolExecutor(max_workers=num_parallel_predictions) as executor:
            futures = [
                executor.submit(run_llm_prompt, judge_model, prompt, developer_prompt=developer_prompt,
                                return_json=True, max_retries=1)
                for _ in range(num_parallel_predictions)
            ]
            results = [future.result() for future in futures if future is not None]

        return results

    @staticmethod
    def recall_score(golden_answer, other_answer, judge_model, num_parallel_predictions=1):
        developer_prompt = verify_prompt_templates["score_prompt_single"]
        prompt = f"""The inputs are as follows:
    Golden Answer: {golden_answer}
    Other Answer: {other_answer}
    """.strip()
        try:
            results = VerifyAgent._run_parallel_predictions(judge_model, prompt, developer_prompt,
                                                            num_parallel_predictions)

            valid_scores = [score["answer_score"] for score in results if score is not None]
            avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else None

            if avg_score is not None:
                return avg_score
        except Exception as e:
            logging.error(f"[compare_answer] error: {e}")
        return None

    @staticmethod
    def acc_score(golden_answer, agent_answer, judge_model, num_parallel_predictions=1):
        developer_prompt = verify_prompt_templates["acc_prompt_single"]
        prompt = f"""The inputs are as follows:
            Golden Answer: {golden_answer}
            Other Answer: {agent_answer}
            """.strip()

        try:
            results = VerifyAgent._run_parallel_predictions(judge_model, prompt, developer_prompt,
                                                            num_parallel_predictions)
            recall_answer = [score["recall_answer"] for score in results if score is not None]
            valid_infor_num = [score["valid_num"] for score in results if score is not None]
            recall_answer = (sum(recall_answer) / len(recall_answer)) > 0.5 if recall_answer else False
            valid_infor_num = sum(valid_infor_num) / len(valid_infor_num) if valid_infor_num else None

            if recall_answer is not None and valid_infor_num is not None:
                return recall_answer, valid_infor_num
        except Exception as e:
            logging.error(f"[compare_answer] error: {e}")
        return None, None
