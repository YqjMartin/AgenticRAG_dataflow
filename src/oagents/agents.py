#!/usr/bin/env python
# coding=utf-8

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import importlib
import inspect
import json
import os
import re
import sys
import tempfile
import textwrap
import time
from collections import deque
from logging import getLogger
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Set, Tuple, TypedDict, Union
import heapq
from collections import deque
import jinja2
import yaml
from huggingface_hub import create_repo, metadata_update, snapshot_download, upload_folder
from jinja2 import StrictUndefined, Template
from rich.console import Group
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text

from .verify_function import evaluate_answer
import uuid

from .agent_types import AgentAudio, AgentImage, AgentType, handle_agent_output_types
from .utils import safe_json_loads
from .default_tools import TOOL_MAPPING, FinalAnswerTool
from .e2b_executor import E2BExecutor
from .local_python_executor import (
    BASE_BUILTIN_MODULES,
    LocalPythonInterpreter,
    fix_final_answer_code,
)
from .memory import ActionStep, ReflectionStep, AgentMemory, PlanningStep, SystemPromptStep, TaskStep, ToolCall
from .models import (
    ChatMessage,
    MessageRole,
    Model,
)
from .monitoring import (
    YELLOW_HEX,
    AgentLogger,
    LogLevel,
    Monitor,
)
from .tools import Tool
from .utils import (
    AgentError,
    AgentExecutionError,
    AgentGenerationError,
    AgentMaxStepsError,
    AgentParsingError,
    make_init_file,
    parse_code_blobs,
    parse_json_tool_call,
    truncate_content,
)

from.knowledge_retrieval import AKB_Manager

KNOWLEDGE_DATABASE_PATH = ["../../src/oagents/knowledge_base/knowledge_database.json"]

logger = getLogger(__name__)


def take_a_breath():
    pass
    # time.sleep(1)
    # input("\n\nPress 'Enter' to continue......")


def get_variable_names(self, template: str) -> Set[str]:
    pattern = re.compile(r"\{\{([^{}]+)\}\}")
    return {match.group(1).strip() for match in pattern.finditer(template)}


def populate_template(template: str, variables: Dict[str, Any]) -> str:
    compiled_template = Template(template, undefined=StrictUndefined)
    try:
        return compiled_template.render(**variables)
    except Exception as e:
        raise Exception(f"Error during jinja template rendering: {type(e).__name__}: {e}")


class PlanningPromptTemplate(TypedDict):
    """
    Prompt templates for the planning step.

    Args:
        initial_facts (`str`): Initial facts prompt.
        initial_plan (`str`): Initial plan prompt.
        update_facts_pre_messages (`str`): Update facts pre-messages prompt.
        update_facts_post_messages (`str`): Update facts post-messages prompt.
        update_plan_pre_messages (`str`): Update plan pre-messages prompt.
        update_plan_post_messages (`str`): Update plan post-messages prompt.
    """

    initial_facts: str
    initial_plan: str
    update_facts_pre_messages: str
    update_facts_post_messages: str
    update_plan_pre_messages: str
    update_plan_post_messages: str


class ManagedAgentPromptTemplate(TypedDict):
    """
    Prompt templates for the managed agent.

    Args:
        task (`str`): Task prompt.
        report (`str`): Report prompt.
    """

    task: str
    report: str


class FinalAnswerPromptTemplate(TypedDict):
    """
    Prompt templates for the final answer.

    Args:
        pre_messages (`str`): Pre-messages prompt.
        post_messages (`str`): Post-messages prompt.
    """

    pre_messages: str
    post_messages: str


class PromptTemplates(TypedDict):
    """
    Prompt templates for the agent.

    Args:
        system_prompt (`str`): System prompt.
        planning ([`~agents.PlanningPromptTemplate`]): Planning prompt templates.
        managed_agent ([`~agents.ManagedAgentPromptTemplate`]): Managed agent prompt templates.
        final_answer ([`~agents.FinalAnswerPromptTemplate`]): Final answer prompt templates.
    """

    system_prompt: str
    planning: PlanningPromptTemplate
    managed_agent: ManagedAgentPromptTemplate
    final_answer: FinalAnswerPromptTemplate


EMPTY_PROMPT_TEMPLATES = PromptTemplates(
    system_prompt="",
    planning=PlanningPromptTemplate(
        initial_facts="",
        initial_plan="",
        update_facts_pre_messages="",
        update_facts_post_messages="",
        update_plan_pre_messages="",
        update_plan_post_messages="",
    ),
    managed_agent=ManagedAgentPromptTemplate(task="", report=""),
    final_answer=FinalAnswerPromptTemplate(pre_messages="", post_messages=""),
)


class MultiStepAgent:
    """
    Agent class that solves the given task step by step, using the ReAct framework:
    While the objective is not reached, the agent will perform a cycle of action (given by the LLM) and observation (obtained from the environment).

    Args:
        tools (`list[Tool]`): [`Tool`]s that the agent can use.
        model (`Callable[[list[dict[str, str]]], ChatMessage]`): Model that will generate the agent's actions.
        prompt_templates ([`~agents.PromptTemplates`], *optional*): Prompt templates.
        max_steps (`int`, default `6`): Maximum number of steps the agent can take to solve the task.
        tool_parser (`Callable`, *optional*): Function used to parse the tool calls from the LLM output.
        add_base_tools (`bool`, default `False`): Whether to add the base tools to the agent's tools.
        verbosity_level (`LogLevel`, default `LogLevel.INFO`): Level of verbosity of the agent's logs.
        grammar (`dict[str, str]`, *optional*): Grammar used to parse the LLM output.
        managed_agents (`list`, *optional*): Managed agents that the agent can call.
        step_callbacks (`list[Callable]`, *optional*): Callbacks that will be called at each step.
        reflection_interval (`int`, *optional*): Interval for reflection during the agent's process.
        name (`str`, *optional*): Necessary for a managed agent only - the name by which this agent can be called.
        description (`str`, *optional*): Necessary for a managed agent only - the description of this agent.
        knowledge_base_retrieval (`bool`, *optional*): Whether to provide knowledge retrieval from database.
        provide_run_summary (`bool`, *optional*): Whether to provide a run summary when called as a managed agent.
        final_answer_checks (`list`, *optional*): List of Callables to run before returning a final answer for checking validity.
    """

    def __init__(
        self,
        tools: List[Tool],
        model: Callable[[List[Dict[str, str]]], ChatMessage],
        prompt_templates: Optional[PromptTemplates] = None,
        max_steps: int = 6,
        tool_parser: Optional[Callable] = None,
        add_base_tools: bool = False,
        verbosity_level: LogLevel = LogLevel.INFO,
        grammar: Optional[Dict[str, str]] = None,
        managed_agents: Optional[List] = None,
        step_callbacks: Optional[List[Callable]] = None,
        planning_interval: Optional[int] = None,
        reflection_interval: Optional[int] = None,
        search_budget:Optional[int] = None,
        n_rollouts:Optional[int] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        knowledge_base_retrieval: bool = False,
        provide_run_summary: bool = False,
        final_answer_checks: Optional[List[Callable]] = None,
        debug: bool = False,
        prompts_type: Optional[str] = "default",
    ):
        if tool_parser is None:
            tool_parser = parse_json_tool_call
        self.agent_name = self.__class__.__name__
        self.model = model
        self.prompt_templates = prompt_templates or EMPTY_PROMPT_TEMPLATES
        self.max_steps = max_steps
        self.step_number: int = 0
        self.tool_parser = tool_parser
        self.grammar = grammar
        self.planning_interval = planning_interval
        self.reflection_interval = reflection_interval
        self.search_budget=search_budget
        self.n_rollouts=n_rollouts
        self.state = {}
        self.name = name
        self.description = description
        self.knowledge_base_retrieval = knowledge_base_retrieval
        self.provide_run_summary = provide_run_summary
        self.debug = debug
        self.action_trajectory=[]
        self.managed_agents = {}
        if managed_agents is not None:
            for managed_agent in managed_agents:
                assert managed_agent.name and managed_agent.description, (
                    "All managed agents need both a name and a description!"
                )
            self.managed_agents = {agent.name: agent for agent in managed_agents}

        tool_and_managed_agent_names = [tool.name for tool in tools]
        if managed_agents is not None:
            tool_and_managed_agent_names += [agent.name for agent in managed_agents]
        if len(tool_and_managed_agent_names) != len(set(tool_and_managed_agent_names)):
            raise ValueError(
                "Each tool or managed_agent should have a unique name! You passed these duplicate names: "
                f"{[name for name in tool_and_managed_agent_names if tool_and_managed_agent_names.count(name) > 1]}"
            )

        for tool in tools:
            assert isinstance(tool, Tool), f"This element is not of class Tool: {str(tool)}"
        self.tools = {tool.name: tool for tool in tools}

        if add_base_tools:
            for tool_name, tool_class in TOOL_MAPPING.items():
                if tool_name != "python_interpreter" or self.__class__.__name__ == "ToolCallingAgent":
                    self.tools[tool_name] = tool_class()
        self.tools["final_answer"] = FinalAnswerTool()

        self.system_prompt = self.initialize_system_prompt()
        self.input_messages = None
        self.task = None
        self.memory = AgentMemory(self.system_prompt)
        self.logger = AgentLogger(level=verbosity_level)
        self.monitor = Monitor(self.model, self.logger)
        self.step_callbacks = step_callbacks if step_callbacks is not None else []
        self.step_callbacks.append(self.monitor.update_metrics)
        self.final_answer_checks = final_answer_checks
        self.prompts_type = prompts_type
        try:
            orm_prompts=yaml.safe_load(
            importlib.resources.files(f"oagents.prompts.{self.prompts_type}").joinpath("ORM.yaml").read_text(encoding="utf-8"))
            self.ORM_prompt=orm_prompts['prompt']
            prm_prompts=yaml.safe_load(
            importlib.resources.files(f"oagents.prompts.{self.prompts_type}").joinpath("PRM.yaml").read_text(encoding="utf-8"))
            self.PRM_prompt=prm_prompts['prompt']
            list_wise_prompts=yaml.safe_load(
            importlib.resources.files(f"oagents.prompts.{self.prompts_type}").joinpath("list-wise.yaml").read_text(encoding="utf-8"))
            self.LIST_WISE_prompt=list_wise_prompts['prompt']
        except Exception as e:
            logger.error(f"Error loading ORM prompt: {e}")
            self.PRM_prompt=""
            self.ORM_prompt=""
            self.LIST_WISE_prompt=""

    @property
    def logs(self):
        logger.warning(
            "The 'logs' attribute is deprecated and will soon be removed. Please use 'self.memory.steps' instead."
        )
        return [self.memory.system_prompt] + self.memory.steps

    def initialize_system_prompt(self):
        """To be implemented in child classes"""
        pass

    def write_memory_to_messages(
        self,
        memory_steps:Optional[List[ActionStep]]=None,
        summary_mode: Optional[bool] = False,
    ) -> List[Dict[str, str]]:
        """
        Reads past llm_outputs, actions, and observations or errors from the memory into a series of messages
        that can be used as input to the LLM. Adds a number of keywords (such as PLAN, error, etc) to help
        the LLM.
        """
        messages = self.memory.system_prompt.to_messages(summary_mode=summary_mode)
        for memory_step in memory_steps if memory_steps else self.memory.steps:
            messages.extend(memory_step.to_messages(summary_mode=summary_mode))
        return messages

    def write_memory_to_string_messages(
            self,
            memory_steps: List[ActionStep | PlanningStep | TaskStep | ReflectionStep]
        ) -> str:
        """
        return 一串字符，cover每次branch的step信息，包含输入step_number，observations，action_output，model_output，error，score
        """


        def step_to_string(step: ActionStep | PlanningStep | TaskStep | ReflectionStep, max_string_per_field: int = 512) -> str:
            """
            将任意类型的 MemoryStep 转成可读字符串，不同 Step 类型可根据需要输出不同字段。
            """
            # 1) 如果是 ActionStep
            if isinstance(step, ActionStep):
                return (
                    f"[ActionStep]\n"
                    f"  step_number: {step.step_number}\n"
                    # f"  start_time: {step_info['start_time']}\n"
                    # f"  end_time: {step_info['end_time']}\n"
                    # f"  duration: {step_info['duration']}\n"
                    f"  observations: {truncate_content(str(step.observations), max_length=max_string_per_field)}\n"
                    f"  action_output: {truncate_content(str(step.action_output), max_length=max_string_per_field)}\n"
                    f"  model_output: {truncate_content(str(step.model_output), max_length=max_string_per_field)}\n"
                    f"  error: {truncate_content(str(step.error), max_length=max_string_per_field)}\n"
                    f"  score: {step.score}\n"
                    # f"  tool_calls_count: {len(step_info['tool_calls']) if step_info['tool_calls'] else 0}"
                )
            # 2) 如果是 TaskStep
            elif isinstance(step, TaskStep):
                return (
                    f"[TaskStep]\n"
                    f"  task: {step.task}\n"
                    f"  description: {step.description if hasattr(step, 'description') else None}"
                )
            # 3) 如果是 PlanningStep
            elif isinstance(step, PlanningStep):
                return (
                    f"[PlanningStep]\n"
                    f"  facts that we knows: {truncate_content(str(step.facts), max_length=max_string_per_field)}\n"
                    f"  current plan: {truncate_content(str(step.plan), max_length=max_string_per_field)}\n"
                )
            # 4) 如果是 ReflectionStep
            elif isinstance(step, ReflectionStep):
                return (
                    f"[ReflectionStep]\n"
                    f"  analysis of historic trajectory: {truncate_content(str(step.history_trajectory_analysis), max_length=max_string_per_field)}\n"
                    f"  score of historic trajectory: {truncate_content(str(step.history_trajectory_score), max_length=max_string_per_field)}\n"
                    f"  facts that we knows: {truncate_content(str(step.facts), max_length=max_string_per_field)}\n"
                    f"  current plan: {truncate_content(str(step.plan), max_length=max_string_per_field)}\n"
                )
            # 5) 否则，处理未知类型
            else:
                step_attrs = []
                for attr_name, attr_value in step.__dict__.items():
                    step_attrs.append(f"  {attr_name}: {attr_value}")
                step_attrs_str = "\n".join(step_attrs)
                return f"[Unknown Step Type: {type(step)}]\n{step_attrs_str}"

        # 将 current_step 也视为最后一个 step，需要包含在结果中
        lines = []
        for idx, step in enumerate(memory_steps, start=1):
            step_str = step_to_string(step)
            lines.append(f"===== Step {idx} =====\n{step_str}")

        # 拼成一个整块字符串
        final_merge_message = "\n".join(lines)
        return final_merge_message


    def visualize(self):
        """Creates a rich tree visualization of the agent's structure."""
        self.logger.visualize_agent_tree(self)

    def extract_think_and_answer(self, input_str):
        think_tags = ["<think>", "</think>"]
        answer_tags = ["<answer>", "</answer>"]
        think = self._extract_tagged_content(input_str, think_tags)
        answer = self._extract_tagged_content(input_str, answer_tags)
        return think, answer

    def _extract_tagged_content(self, input_str: str, tags: list) -> str:
        """
        Helper method to extract content between specified tags.

        Args:
            input_str (`str`): The input string to search within.
            tags (`list`): A list containing the opening and closing tags.

        Returns:
            `str`: The extracted content or an empty string if not found.
        """
        start_tag, end_tag = tags
        start_index = input_str.find(start_tag) + len(start_tag)
        end_index = input_str.find(end_tag, start_index)
        return input_str[start_index:end_index].strip() if end_index != -1 else ""

    # def extract_action(self, model_output: str, split_token: str) -> Tuple[str, str]:
    #     """
    #     Parse action from the LLM output

    #     Args:
    #         model_output (`str`): Output of the LLM
    #         split_token (`str`): Separator for the action. Should match the example in the system prompt.
    #     """
    #     try:
    #         split = model_output.split(split_token)
    #         rationale, action = (
    #             split[-2],
    #             split[-1],
    #         )  # NOTE: using indexes starting from the end solves for when you have more than one split_token in the output
    #     except Exception:
    #         raise AgentParsingError(
    #             f"No '{split_token}' token provided in your output.\nYour output:\n{model_output}\n. Be sure to include an action, prefaced with '{split_token}'!",
    #             self.logger,
    #         )
    #     return rationale.strip(), action.strip()

    def provide_final_answer(self, task: str, images: Optional[list[str]]) -> Tuple[str, str]:
        """
        Provide the final answer to the task, based on the logs of the agent's interactions.

        Args:
            task (`str`): Task to perform.
            images (`list[str]`, *optional*): Paths to image(s).

        Returns:
            `str`: Final answer to the task.
        """
        messages = [
            {
                "role": MessageRole.SYSTEM,
                "content": [
                    {
                        "type": "text",
                        "text": self.prompt_templates["final_answer"]["pre_messages"],
                    }
                ],
            }
        ]
        if images:
            messages[0]["content"].append({"type": "image"})
        messages += self.write_memory_to_messages()[1:]
        messages += [
            {
                "role": MessageRole.USER,
                "content": [
                    {
                        "type": "text",
                        "text": populate_template(
                            self.prompt_templates["final_answer"]["post_messages"], variables={"task": task}
                        ),
                    }
                ],
            }
        ]
        try:
            chat_message: ChatMessage = self.model(messages)
            final_answer = chat_message.content
            final_answer_think, final_answer_answer = self.extract_think_and_answer(final_answer)
            return final_answer_think, final_answer_answer
        except Exception as e:
            return None, f"Error in generating final LLM output:\n{e}"

    def execute_tool_call(self, tool_name: str, arguments: Union[Dict[str, str], str]) -> Any:
        """
        Execute tool with the provided input and returns the result.
        This method replaces arguments with the actual values from the state if they refer to state variables.

        Args:
            tool_name (`str`): Name of the Tool to execute (should be one from self.tools).
            arguments (Dict[str, str]): Arguments passed to the Tool.
        """
        available_tools = {**self.tools, **self.managed_agents}
        if tool_name not in available_tools:
            error_msg = f"Unknown tool {tool_name}, should be instead one of {list(available_tools.keys())}."
            raise AgentExecutionError(error_msg, self.logger)

        try:
            if isinstance(arguments, str):
                if tool_name in self.managed_agents:
                    observation = available_tools[tool_name].__call__(arguments)
                else:
                    observation = available_tools[tool_name].__call__(arguments, sanitize_inputs_outputs=True)
            elif isinstance(arguments, dict):
                for key, value in arguments.items():
                    if isinstance(value, str) and value in self.state:
                        arguments[key] = self.state[value]
                if tool_name in self.managed_agents:
                    observation = available_tools[tool_name].__call__(**arguments)
                else:
                    observation = available_tools[tool_name].__call__(**arguments, sanitize_inputs_outputs=True)
            else:
                error_msg = f"Arguments passed to tool should be a dict or string: got a {type(arguments)}."
                raise AgentExecutionError(error_msg, self.logger)
            return observation
        except Exception as e:
            if tool_name in self.tools:
                tool = self.tools[tool_name]
                error_msg = (
                    f"Error when executing tool {tool_name} with arguments {arguments}: {type(e).__name__}: {e}\nYou should only use this tool with a correct input.\n"
                    f"As a reminder, this tool's description is the following: '{tool.description}'.\nIt takes inputs: {tool.inputs} and returns output type {tool.output_type}"
                )
                raise AgentExecutionError(error_msg, self.logger)
            elif tool_name in self.managed_agents:
                error_msg = (
                    f"Error in calling team member: {e}\nYou should only ask this team member with a correct request.\n"
                    f"As a reminder, this team member's description is the following:\n{available_tools[tool_name]}"
                )
                raise AgentExecutionError(error_msg, self.logger)

    def step(self, memory_step: ActionStep) -> Union[None, Any]:
        """To be implemented in children classes. Should return either None if the step is not final."""
        pass

    def run(
        self,
        task: str,
        stream: bool = False,
        reset: bool = True,
        answer: Optional[str] = None,
        images: Optional[List[str]] = None,
        additional_args: Optional[Dict] = None,
    ):
        """
        Run the agent for the given task.

        Args:
            task (`str`): Task to perform.
            stream (`bool`): Whether to run in a streaming way.
            reset (`bool`): Whether to reset the conversation or keep it going from previous run.
            images (`list[str]`, *optional*): Paths to image(s).
            additional_args (`dict`): Any other variables that you want to pass to the agent run, for instance images or dataframes. Give them clear names!

        Example:
        ```py
        from oagents import CodeAgent
        agent = CodeAgent(tools=[])
        agent.run("What is the result of 2 power 3.7384?")
        ```
        """

        self.task = task
        self.answer = answer
        for tool_name, tool in self.tools.items():
            tool.task = self.task

        if additional_args is not None:
            self.state.update(additional_args)
            self.task += f"""
You have been provided with these additional arguments, that you can access using the keys as variables in your python code:
{str(additional_args)}."""

        self.system_prompt = self.initialize_system_prompt()
        self.memory.system_prompt = SystemPromptStep(system_prompt=self.system_prompt)
        if reset:
            self.memory.reset()
            self.monitor.reset()

        self.logger.log_task(
            content=self.task.strip(),
            subtitle=f"{type(self.model).__name__} - {(self.model.model_id if hasattr(self.model, 'model_id') else '')}",
            level=LogLevel.INFO,
            title=self.name if hasattr(self, "name") else None,
        )

        self.memory.steps.append(TaskStep(task=self.task, task_images=images))

        if stream:
            # The steps are returned as they are executed through a generator to iterate on.
            return self._run(task=self.task,images=images)
        # Outputs are returned only at the end as a string. We only look at the last step
        return deque(self._run(task=self.task, images=images), maxlen=1)[0]

    def _run(self, task: str, images: List[str] | None = None) -> Generator[ActionStep | AgentType, None, None]:
        """
        Run the agent in streaming mode and returns a generator of all the steps.

        Args:
            task (`str`): Task to perform.
            images (`list[str]`): Paths to image(s).
        """
        pass

    def retrieval_knowledge(self, task, knowledge_path):
        akb_manager = AKB_Manager(knowledge_path)
        workflow_result = akb_manager.search_by_text_similarity(task, top_k = 1)
        workflow_id, _ = workflow_result[0]
        workflow = akb_manager.get_workflow_details(workflow_id)
        return workflow

    def planning_step(self, task, is_first_step: bool, step: int) -> None:
        """
        Used periodically by the agent to plan the next steps to reach the objective.

        Args:
            task (`str`): Task to perform.
            is_first_step (`bool`): If this step is not the first one, the plan should be an update over a previous plan.
            step (`int`): The number of the current step, used as an indication for the LLM.
        """
        if is_first_step:
            input_messages = [
                {
                    "role": MessageRole.USER,
                    "content": [
                        {
                            "type": "text",
                            "text": populate_template(
                                self.prompt_templates["planning"]["initial_facts"], variables={"task": task}
                            ),
                        }
                    ],
                },
            ]

            chat_message_facts: ChatMessage = self.model(input_messages)
            facts = chat_message_facts.content
            facts_think, facts_answer = self.extract_think_and_answer(facts)

            if self.knowledge_base_retrieval:
                knowledge_data = self.retrieval_knowledge(task, KNOWLEDGE_DATABASE_PATH)
                if knowledge_data.resolved:
                    knowledge_bool = "resolved"
                else:
                    knowledge_bool = "failed to resolve"

                final_facts_knowledge = textwrap.dedent(
                    f"""Here are the similar task and plan that I retrieved:
                    ```
                    {knowledge_data.question}
                    {knowledge_data.plan}
                    ```""".strip()
                )

                initial_plan_template = populate_template(
                            self.prompt_templates["planning"]["initial_plan_with_knowledge"],
                            variables={
                                "task": task,
                                "tools": self.tools,
                                "managed_agents": self.managed_agents,
                                "answer_facts": facts_answer,
                                "knowledge_bool": knowledge_bool,
                                "knowledge_question": knowledge_data.question,
                                "knowledge_plan": knowledge_data.plan,
                            },
                        )
            else:
                initial_plan_template = populate_template(
                            self.prompt_templates["planning"]["initial_plan"],
                            variables={
                                "task": task,
                                "tools": self.tools,
                                "managed_agents": self.managed_agents,
                                "answer_facts": facts_answer,
                            },
                        )

            message_prompt_plan = {
                "role": MessageRole.USER,
                "content": [
                    {
                        "type": "text",
                        "text": initial_plan_template,
                    }
                ],
            }
            chat_message_plan: ChatMessage = self.model(
                [message_prompt_plan],
                # stop_sequences=["<end_plan>"],
            )
            plans = chat_message_plan.content
            plans_think, plans_answer = self.extract_think_and_answer(plans)

            final_plan_redaction = textwrap.dedent(
                f"""Here is the plan of action that I will follow to solve the task:
                ```
                {plans_answer}
                ```"""
            )
            final_plan_think_redaction = textwrap.dedent(
                f"""Here is the reasoning and thinking process to derive the plan:
                ```
                {plans_think}
                ```"""
            )
            final_facts_redaction = textwrap.dedent(
                f"""Here are the facts that I know so far:
                ```
                {facts_answer}
                ```""".strip()
            )
            final_facts_think_redaction = textwrap.dedent(
                f"""Here is the reasoning and thinking process to derive the facts:
                ```
                {facts_think}
                ```""".strip()
            )
            if self.knowledge_base_retrieval:
                self.logger.log(
                    Rule("[bold]retrieved task and plan", style="orange"),
                    Text(final_facts_knowledge),
                    level=LogLevel.INFO,
                )
            self.logger.log(
                Rule("[bold]Initial plan", style="orange"),
                Text(final_plan_redaction),
                level=LogLevel.INFO,
            )
            #########################################################
            self.memory.steps.append(
                PlanningStep(
                    model_input_messages=input_messages,
                    plan=final_plan_redaction,
                    plan_think=final_plan_think_redaction,
                    facts=final_facts_redaction,
                    facts_think=final_facts_think_redaction,
                )
            )
            #########################################################
            return PlanningStep(
                    model_input_messages=input_messages,
                    plan=final_plan_redaction,
                    plan_think=final_plan_think_redaction,
                    facts=final_facts_redaction,
                    facts_think=final_facts_think_redaction,
            )
            # self.logger.log(
            #     Rule("[bold]Initial plan", style="orange"),
            #     Text(final_plan_redaction),
            #     level=LogLevel.INFO,
            # )
        else:  # update plan
            # Do not take the system prompt message from the memory
            # summary_mode=False: Do not take previous plan steps to avoid influencing the new plan
            memory_messages = self.write_memory_to_messages()[1:]

            # Redact updated facts
            facts_update_pre_messages = {
                "role": MessageRole.SYSTEM,
                "content": [{"type": "text", "text": self.prompt_templates["planning"]["update_facts_pre_messages"]}],
            }
            facts_update_post_messages = {
                "role": MessageRole.USER,
                "content": [{"type": "text", "text": self.prompt_templates["planning"]["update_facts_post_messages"]}],
            }
            input_messages = [facts_update_pre_messages] + memory_messages + [facts_update_post_messages]
            chat_message_facts: ChatMessage = self.model(input_messages)
            facts_update = chat_message_facts.content
            facts_update_think, facts_update_answer = self.extract_think_and_answer(facts_update)

            # Redact updated plan
            update_plan_pre_messages = {
                "role": MessageRole.SYSTEM,
                "content": [
                    {
                        "type": "text",
                        "text": populate_template(
                            self.prompt_templates["planning"]["update_plan_pre_messages"], variables={"task": task}
                        ),
                    }
                ],
            }
            update_plan_post_messages = {
                "role": MessageRole.USER,
                "content": [
                    {
                        "type": "text",
                        "text": populate_template(
                            self.prompt_templates["planning"]["update_plan_post_messages"],
                            variables={
                                "task": task,
                                "tools": self.tools,
                                "managed_agents": self.managed_agents,
                                "facts_update": facts_update_answer,
                                "remaining_steps": (self.max_steps - step),
                            },
                        ),
                    }
                ],
            }
            chat_message_plan: ChatMessage = self.model(
                [update_plan_pre_messages] + memory_messages + [update_plan_post_messages],
                # stop_sequences=["<end_plan>"],
            )
            plans_updated = chat_message_plan.content
            plans_updated_think, plans_updated_answer = self.extract_think_and_answer(plans_updated)

            # Log final facts and plan
            final_plan_redaction = textwrap.dedent(
                f"""I still need to solve the task I was given:
                ```
                {task}
                ```

                Here is my new/updated plan of action to solve the task:
                ```
                {plans_updated_answer}
                ```"""
            )
            final_plan_think_redaction = textwrap.dedent(
                f"""
                Here is my reasoning and thinking process to derive the plan:
                ```
                {plans_updated_think}
                ```"""
            )

            final_facts_redaction = textwrap.dedent(
                f"""Here is the updated list of the facts that I know:
                ```
                {facts_update_answer}
                ```"""
            )
            final_facts_think_redaction = textwrap.dedent(
                f"""Here is my reasoning and thinking process to derive the facts:
                ```
                {facts_update_think}
                ```"""
            )
            self.memory.steps.append(
                PlanningStep(
                    model_input_messages=input_messages,
                    plan=final_plan_redaction,
                    plan_think=final_plan_think_redaction,
                    facts=final_facts_redaction,
                    facts_think=final_facts_think_redaction,
                )
            )
            self.logger.log(
                Rule("[bold]Updated plan", style="orange"),
                Text(final_plan_redaction),
                level=LogLevel.INFO,
            )
            return PlanningStep(
                    model_input_messages=input_messages,
                    plan=final_plan_redaction,
                    plan_think=final_plan_think_redaction,
                    facts=final_facts_redaction,
                    facts_think=final_facts_think_redaction,
                )

    def reflection_step(self, task: str, step: int) -> None:
        """
        Used periodically by the agent to plan the next steps to reach the objective.

        Args:
            task (`str`): Task to perform.
            step (`int`): The number of the current step, used as an indication for the LLM.
        """
        # 1. process reward model
        memory_steps = self.memory.steps
        answer = str(self.answer)
        if answer!=None:
            #训练模式已知answer
            answer_message = self.write_memory_to_string_messages(memory_steps)
            answer_message = f"standard_answer:{answer}\n\n\n"+answer_message
        else:
            answer_message = self.write_memory_to_string_messages(memory_steps)
        system_prompt = self.prompt_templates["reflection"]["process_rm"]
        process_reward_model_messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": answer_message  # Use the answer_message instead of text
            }
        ]
        chat_message_prm: ChatMessage = self.model(process_reward_model_messages)
        chat_message_prm_json = safe_json_loads(chat_message_prm.content)
        chat_message_prm_analysis, chat_message_prm_score = chat_message_prm_json["analysis"], \
            chat_message_prm_json["score"]


        # 2. reflect
        # Do not take the system prompt message from the memory
        # summary_mode=False: Do not take previous plan steps to avoid influencing the new plan
        memory_messages = self.write_memory_to_messages()[1:]
        # Redact updated facts
        facts_update_pre_messages = {
            "role": MessageRole.SYSTEM,
            "content": [{"type": "text", "text": self.prompt_templates["reflection"]["update_facts_pre_messages"]}],
        }
        facts_update_post_messages = {
            "role": MessageRole.USER,
            "content": [{"type": "text", "text": self.prompt_templates["reflection"]["update_facts_post_messages"]}],
        }
        input_messages = [facts_update_pre_messages] + memory_messages + [facts_update_post_messages]
        chat_message_facts: ChatMessage = self.model(input_messages)
        facts_update = chat_message_facts.content
        facts_update_think, facts_update_answer = self.extract_think_and_answer(facts_update)

        # Redact updated plan
        update_plan_pre_messages = {
            "role": MessageRole.SYSTEM,
            "content": [{"type": "text", "text": self.prompt_templates["reflection"]["update_plan_pre_messages"]}],
        }
        update_plan_post_messages = {
            "role": MessageRole.USER,
            "content": [
                {
                    "type": "text",
                    "text": populate_template(
                        self.prompt_templates["reflection"]["update_plan_post_messages"],
                        variables={
                            "task": task,
                            "tools": self.tools,
                            "managed_agents": self.managed_agents,
                            "trajectory_score": chat_message_prm_score,
                            "trajectory_analysis": chat_message_prm_analysis,
                            "facts_update": facts_update_answer,
                            "remaining_steps": (self.max_steps - step),
                            "standard_answer": self.answer,
                        },
                    ),
                }
            ],
        }
        chat_message_plan: ChatMessage = self.model(
            [update_plan_pre_messages] + memory_messages + [update_plan_post_messages],
            # stop_sequences=["<end_plan>"],
        )
        plans_updated = chat_message_plan.content
        plans_updated_think, plans_updated_answer = self.extract_think_and_answer(plans_updated)

        # Log final facts and plan
        final_plan_redaction = textwrap.dedent(
            f"""I still need to solve the task I was given:
            ```
            {task}
            ```

            Here is my new/updated plan of action to solve the task:
            ```
            {plans_updated_answer}
            ```"""
        )
        final_plan_think_redaction = textwrap.dedent(
            f"""
            Here is my reasoning and thinking process to derive the plan:
            ```
            {plans_updated_think}
            ```"""
        )

        final_facts_redaction = textwrap.dedent(
            f"""Here is the updated list of the facts that I know:
            ```
            {facts_update_answer}
            ```"""
        )
        final_facts_think_redaction = textwrap.dedent(
            f"""Here is my reasoning and thinking process to derive the facts:
            ```
            {facts_update_think}
            ```"""
        )
        self.memory.steps.append(
            ReflectionStep(
                model_input_messages=input_messages,
                history_trajectory_analysis=chat_message_prm_analysis,
                history_trajectory_score=chat_message_prm_score,
                plan=final_plan_redaction,
                plan_think=final_plan_think_redaction,
                facts=final_facts_redaction,
                facts_think=final_facts_think_redaction,
            )
        )
        self.logger.log(
            Rule("[bold]Updated plan", style="orange"),
            Text(final_plan_redaction),
            level=LogLevel.INFO,
        )
        return ReflectionStep(
                model_input_messages=input_messages,
                history_trajectory_analysis=chat_message_prm_analysis,
                history_trajectory_score=chat_message_prm_score,
                plan=final_plan_redaction,
                plan_think=final_plan_think_redaction,
                facts=final_facts_redaction,
                facts_think=final_facts_think_redaction,
            )

    def replay(self, detailed: bool = False):
        """Prints a pretty replay of the agent's steps.

        Args:
            detailed (bool, optional): If True, also displays the memory at each step. Defaults to False.
                Careful: will increase log length exponentially. Use only for debugging.
        """
        self.memory.replay(self.logger, detailed=detailed)

    # def __call__(self, task: str, **kwargs):
    #     """Adds additional prompting for the managed agent, runs it, and wraps the output.

    #     This method is called only by a managed agent.
    #     """
    #     full_task = populate_template(
    #         self.prompt_templates["managed_agent"]["task"],
    #         variables=dict(name=self.name, task=task),
    #     )
    #     report = self.run(full_task, **kwargs)
    #     answer = populate_template(
    #         self.prompt_templates["managed_agent"]["report"], variables=dict(name=self.name, final_answer=report)
    #     )
    #     if self.provide_run_summary:
    #         answer += "\n\nFor more detail, find below a summary of this agent's work:\n<summary_of_work>\n"
    #         for message in self.write_memory_to_messages(summary_mode=True):
    #             content = message["content"]
    #             answer += "\n" + truncate_content(str(content)) + "\n---"
    #         answer += "\n</summary_of_work>"
    #     return answer

    def save(self, output_dir: str, relative_path: Optional[str] = None):
        """
        Saves the relevant code files for your agent. This will copy the code of your agent in `output_dir` as well as autogenerate:

        - a `tools` folder containing the logic for each of the tools under `tools/{tool_name}.py`.
        - a `managed_agents` folder containing the logic for each of the managed agents.
        - an `agent.json` file containing a dictionary representing your agent.
        - a `prompt.yaml` file containing the prompt templates used by your agent.
        - an `app.py` file providing a UI for your agent when it is exported to a Space with `agent.push_to_hub()`
        - a `requirements.txt` containing the names of the modules used by your tool (as detected when inspecting its
          code)

        Args:
            output_dir (`str`): The folder in which you want to save your tool.
        """
        make_init_file(output_dir)

        # Recursively save managed agents
        if self.managed_agents:
            make_init_file(os.path.join(output_dir, "managed_agents"))
            for agent_name, agent in self.managed_agents.items():
                agent_suffix = f"managed_agents.{agent_name}"
                if relative_path:
                    agent_suffix = relative_path + "." + agent_suffix
                agent.save(os.path.join(output_dir, "managed_agents", agent_name), relative_path=agent_suffix)

        class_name = self.__class__.__name__

        # Save tools to different .py files
        for tool in self.tools.values():
            make_init_file(os.path.join(output_dir, "tools"))
            tool.save(os.path.join(output_dir, "tools"), tool_file_name=tool.name, make_gradio_app=False)

        # Save prompts to yaml
        yaml_prompts = yaml.safe_dump(
            self.prompt_templates,
            default_style="|",  # This forces block literals for all strings
            default_flow_style=False,
            width=float("inf"),
            sort_keys=False,
            allow_unicode=True,
            indent=2,
        )

        with open(os.path.join(output_dir, "prompts.yaml"), "w", encoding="utf-8") as f:
            f.write(yaml_prompts)

        # Save agent dictionary to json
        agent_dict = self.to_dict()
        agent_dict["tools"] = [tool.name for tool in self.tools.values()]
        with open(os.path.join(output_dir, "agent.json"), "w", encoding="utf-8") as f:
            json.dump(agent_dict, f, indent=4)

        # Save requirements
        with open(os.path.join(output_dir, "requirements.txt"), "w", encoding="utf-8") as f:
            f.writelines(f"{r}\n" for r in agent_dict["requirements"])

        # Make agent.py file with Gradio UI
        agent_name = f"agent_{self.name}" if getattr(self, "name", None) else "agent"
        managed_agent_relative_path = relative_path + "." if relative_path is not None else ""
        app_template = textwrap.dedent("""
            import yaml
            import os
            from oagents import GradioUI, {{ class_name }}, {{ agent_dict['model']['class'] }}

            # Get current directory path
            CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

            {% for tool in tools.values() -%}
            from {{managed_agent_relative_path}}tools.{{ tool.name }} import {{ tool.__class__.__name__ }} as {{ tool.name | camelcase }}
            {% endfor %}
            {% for managed_agent in managed_agents.values() -%}
            from {{managed_agent_relative_path}}managed_agents.{{ managed_agent.name }}.app import agent_{{ managed_agent.name }}
            {% endfor %}

            model = {{ agent_dict['model']['class'] }}(
            {% for key in agent_dict['model']['data'] if key not in ['class', 'last_input_token_count', 'last_output_token_count'] -%}
                {{ key }}={{ agent_dict['model']['data'][key]|repr }},
            {% endfor %})

            {% for tool in tools.values() -%}
            {{ tool.name }} = {{ tool.name | camelcase }}()
            {% endfor %}

            with open(os.path.join(CURRENT_DIR, "prompts.yaml"), 'r') as stream:
                prompt_templates = yaml.safe_load(stream)

            {{ agent_name }} = {{ class_name }}(
                model=model,
                tools=[{% for tool_name in tools.keys() if tool_name != "final_answer" %}{{ tool_name }}{% if not loop.last %}, {% endif %}{% endfor %}],
                managed_agents=[{% for subagent_name in managed_agents.keys() %}agent_{{ subagent_name }}{% if not loop.last %}, {% endif %}{% endfor %}],
                {% for attribute_name, value in agent_dict.items() if attribute_name not in ["model", "tools", "prompt_templates", "authorized_imports", "managed_agents", "requirements"] -%}
                {{ attribute_name }}={{ value|repr }},
                {% endfor %}prompt_templates=prompt_templates
            )
            if __name__ == "__main__":
                GradioUI({{ agent_name }}).launch()
            """).strip()
        template_env = jinja2.Environment(loader=jinja2.BaseLoader(), undefined=jinja2.StrictUndefined)
        template_env.filters["repr"] = repr
        template_env.filters["camelcase"] = lambda value: "".join(word.capitalize() for word in value.split("_"))
        template = template_env.from_string(app_template)

        # Render the app.py file from Jinja2 template
        app_text = template.render(
            {
                "agent_name": agent_name,
                "class_name": class_name,
                "agent_dict": agent_dict,
                "tools": self.tools,
                "managed_agents": self.managed_agents,
                "managed_agent_relative_path": managed_agent_relative_path,
            }
        )

        with open(os.path.join(output_dir, "app.py"), "w", encoding="utf-8") as f:
            f.write(app_text + "\n")  # Append newline at the end

    def to_dict(self) -> Dict[str, Any]:
        """Converts agent into a dictionary."""
        # TODO: handle serializing step_callbacks and final_answer_checks
        for attr in ["final_answer_checks", "step_callbacks"]:
            if getattr(self, attr, None):
                self.logger.log(f"This agent has {attr}: they will be ignored by this method.", LogLevel.INFO)

        tool_dicts = [tool.to_dict() for tool in self.tools.values()]
        tool_requirements = {req for tool in self.tools.values() for req in tool.to_dict()["requirements"]}
        managed_agents_requirements = {
            req for managed_agent in self.managed_agents.values() for req in managed_agent.to_dict()["requirements"]
        }
        requirements = tool_requirements | managed_agents_requirements
        if hasattr(self, "authorized_imports"):
            requirements.update(
                {package.split(".")[0] for package in self.authorized_imports if package not in BASE_BUILTIN_MODULES}
            )

        agent_dict = {
            "tools": tool_dicts,
            "model": {
                "class": self.model.__class__.__name__,
                "data": self.model.to_dict(),
            },
            "managed_agents": {
                managed_agent.name: managed_agent.__class__.__name__ for managed_agent in self.managed_agents.values()
            },
            "prompt_templates": self.prompt_templates,
            "max_steps": self.max_steps,
            "verbosity_level": int(self.logger.level),
            "grammar": self.grammar,
            "planning_interval": self.planning_interval,
            "reflection_interval": self.reflection_interval,
            "name": self.name,
            "description": self.description,
            "requirements": list(requirements),
        }
        if hasattr(self, "authorized_imports"):
            agent_dict["authorized_imports"] = self.authorized_imports
        if hasattr(self, "use_e2b_executor"):
            agent_dict["use_e2b_executor"] = self.use_e2b_executor
        if hasattr(self, "max_print_outputs_length"):
            agent_dict["max_print_outputs_length"] = self.max_print_outputs_length
        return agent_dict

    @classmethod
    def from_hub(
        cls,
        repo_id: str,
        token: Optional[str] = None,
        trust_remote_code: bool = False,
        **kwargs,
    ):
        """
        Loads an agent defined on the Hub.

        <Tip warning={true}>

        Loading a tool from the Hub means that you'll download the tool and execute it locally.
        ALWAYS inspect the tool you're downloading before loading it within your runtime, as you would do when
        installing a package using pip/npm/apt.

        </Tip>

        Args:
            repo_id (`str`):
                The name of the repo on the Hub where your tool is defined.
            token (`str`, *optional*):
                The token to identify you on hf.co. If unset, will use the token generated when running
                `huggingface-cli login` (stored in `~/.huggingface`).
            trust_remote_code(`bool`, *optional*, defaults to False):
                This flags marks that you understand the risk of running remote code and that you trust this tool.
                If not setting this to True, loading the tool from Hub will fail.
            kwargs (additional keyword arguments, *optional*):
                Additional keyword arguments that will be split in two: all arguments relevant to the Hub (such as
                `cache_dir`, `revision`, `subfolder`) will be used when downloading the files for your agent, and the
                others will be passed along to its init.
        """
        if not trust_remote_code:
            raise ValueError(
                "Loading an agent from Hub requires to acknowledge you trust its code: to do so, pass `trust_remote_code=True`."
            )

        # Get the agent's Hub folder.
        download_kwargs = {"token": token, "repo_type": "space"} | {
            key: kwargs.pop(key)
            for key in [
                "cache_dir",
                "force_download",
                "proxies",
                "revision",
                "local_files_only",
            ]
            if key in kwargs
        }

        download_folder = Path(snapshot_download(repo_id=repo_id, **download_kwargs))
        return cls.from_folder(download_folder, **kwargs)

    @classmethod
    def from_folder(cls, folder: Union[str, Path], **kwargs):
        """Loads an agent from a local folder.

        Args:
            folder (`str` or `Path`): The folder where the agent is saved.
            **kwargs: Additional keyword arguments that will be passed to the agent's init.
        """
        folder = Path(folder)
        agent_dict = json.loads((folder / "agent.json").read_text())

        # Recursively get managed agents
        managed_agents = []
        for managed_agent_name, managed_agent_class in agent_dict["managed_agents"].items():
            agent_cls = getattr(importlib.import_module("oagents.agents"), managed_agent_class)
            managed_agents.append(agent_cls.from_folder(folder / "managed_agents" / managed_agent_name))

        tools = []
        for tool_name in agent_dict["tools"]:
            tool_code = (folder / "tools" / f"{tool_name}.py").read_text()
            tools.append(Tool.from_code(tool_code))

        model_class: Model = getattr(importlib.import_module("oagents.models"), agent_dict["model"]["class"])
        model = model_class.from_dict(agent_dict["model"]["data"])

        args = dict(
            model=model,
            tools=tools,
            managed_agents=managed_agents,
            name=agent_dict["name"],
            description=agent_dict["description"],
            max_steps=agent_dict["max_steps"],
            planning_interval=agent_dict["planning_interval"],
            reflection_interval=agent_dict["reflection_interval"],
            grammar=agent_dict["grammar"],
            verbosity_level=agent_dict["verbosity_level"],
        )
        if cls.__name__ == "CodeAgent":
            args["additional_authorized_imports"] = agent_dict["authorized_imports"]
            args["use_e2b_executor"] = agent_dict["use_e2b_executor"]
            args["max_print_outputs_length"] = agent_dict["max_print_outputs_length"]
        args.update(kwargs)
        return cls(**args)

    def push_to_hub(
        self,
        repo_id: str,
        commit_message: str = "Upload agent",
        private: Optional[bool] = None,
        token: Optional[Union[bool, str]] = None,
        create_pr: bool = False,
    ) -> str:
        """
        Upload the agent to the Hub.

        Parameters:
            repo_id (`str`):
                The name of the repository you want to push to. It should contain your organization name when
                pushing to a given organization.
            commit_message (`str`, *optional*, defaults to `"Upload agent"`):
                Message to commit while pushing.
            private (`bool`, *optional*, defaults to `None`):
                Whether to make the repo private. If `None`, the repo will be public unless the organization's default is private. This value is ignored if the repo already exists.
            token (`bool` or `str`, *optional*):
                The token to use as HTTP bearer authorization for remote files. If unset, will use the token generated
                when running `huggingface-cli login` (stored in `~/.huggingface`).
            create_pr (`bool`, *optional*, defaults to `False`):
                Whether to create a PR with the uploaded files or directly commit.
        """
        repo_url = create_repo(
            repo_id=repo_id,
            token=token,
            private=private,
            exist_ok=True,
            repo_type="space",
            space_sdk="gradio",
        )
        repo_id = repo_url.repo_id
        metadata_update(
            repo_id,
            {"tags": ["oagents", "agent"]},
            repo_type="space",
            token=token,
            overwrite=True,
        )

        with tempfile.TemporaryDirectory() as work_dir:
            self.save(work_dir)
            logger.info(f"Uploading the following files to {repo_id}: {','.join(os.listdir(work_dir))}")
            return upload_folder(
                repo_id=repo_id,
                commit_message=commit_message,
                folder_path=work_dir,
                token=token,
                create_pr=create_pr,
                repo_type="space",
            )


class ToolCallingAgent(MultiStepAgent):
    """
    This agent uses JSON-like tool calls, using method `model.get_tool_call` to leverage the LLM engine's tool calling capabilities.

    Args:
        tools (`list[Tool]`): [`Tool`]s that the agent can use.
        model (`Callable[[list[dict[str, str]]], ChatMessage]`): Model that will generate the agent's actions.
        prompt_templates ([`~agents.PromptTemplates`], *optional*): Prompt templates.
        reflection_interval (`int`, *optional*): Interval at which the agent will run a reflection step.
        **kwargs: Additional keyword arguments.
    """

    def __init__(
        self,
        tools: List[Tool],
        model: Callable[[List[Dict[str, str]]], ChatMessage],
        prompt_templates: Optional[PromptTemplates] = None,
        reflection_interval: Optional[int] = None,
        knowledge_base_retrieval: bool = False,
        prompts_type: Optional[str] = "default",
        **kwargs,
    ):
        prompt_templates = prompt_templates or yaml.safe_load(
            importlib.resources.files(f"oagents.prompts.{prompts_type}").joinpath("toolcalling_agent.yaml").read_text(encoding="utf-8")
        )
        super().__init__(
            tools=tools,
            model=model,
            prompt_templates=prompt_templates,
            reflection_interval=reflection_interval,
            knowledge_base_retrieval=knowledge_base_retrieval,
            **kwargs,
        )

        self.task_records = {} # 记录self.task -> tools calling
        self.tool_call_records = []

    def initialize_system_prompt(self) -> str:
        system_prompt = populate_template(
            self.prompt_templates["system_prompt"],
            variables={"tools": self.tools, "managed_agents": self.managed_agents},
        )
        return system_prompt

    def _run(self, task: str,images: List[str] | None = None) -> Generator[ActionStep | AgentType, None, None]:
        """
        Run the agent in streaming mode and returns a generator of all the steps.

        Args:
            task (`str`): Task to perform.
            images (`list[str]`): Paths to image(s).
        """
        final_answer = None
        self.step_number = 1
        while final_answer is None and self.step_number <= self.max_steps:
            step_start_time = time.time()
            memory_step = ActionStep(
                step_number=self.step_number,
                start_time=step_start_time,
                observations_images=images,
            )
            try:
                # 第一次增加planning_step，planning_step不计入step_number，planning_step总是紧跟着一个action_step
                if self.step_number == 1 or (self.planning_interval is not None and self.step_number % self.planning_interval == 1):
                    self.planning_step(
                        task,
                        is_first_step=(self.step_number == 1),
                        step=self.step_number,
                    )
                # 除了第一次，后面每次，每隔reflection_interval个step，进行一次reflection_step，reflection_step不计入step_number，reflection_step总是紧跟着一个action_step
                elif self.reflection_interval is not None and self.step_number % self.reflection_interval == 1:
                    self.reflection_step(
                        task, step=self.step_number
                    )

                # Run one step!
                self.logger.log_rule(f"Step {self.step_number}", level=LogLevel.INFO)
                final_answer = self.step(memory_step)
                if final_answer is not None and self.final_answer_checks is not None:
                    for check_function in self.final_answer_checks:
                        try:
                            assert check_function(final_answer, self.memory)
                        except Exception as e:
                            final_answer = None
                            raise AgentError(f"Check {check_function.__name__} failed with error: {e}", self.logger)
            except AgentError as e:
                memory_step.error = e
                raise
            finally:
                memory_step.end_time = time.time()
                memory_step.duration = memory_step.end_time - step_start_time
                self.memory.steps.append(memory_step)
                for callback in self.step_callbacks:
                    # For compatibility with old callbacks that don't take the agent as an argument
                    if len(inspect.signature(callback).parameters) == 1:
                        callback(memory_step)
                    else:
                        callback(memory_step, agent=self)
                self.step_number += 1
                yield memory_step

        if final_answer is None and self.step_number == self.max_steps + 1:
            error_message = "Reached max steps."
            step_start_time = time.time()
            final_think, final_answer = self.provide_final_answer(task, images)
            final_memory_step = ActionStep(
                step_number=self.step_number, error=AgentMaxStepsError(error_message, self.logger)
            )
            final_memory_step.action_think = final_think
            final_memory_step.action_output = final_answer
            final_memory_step.end_time = time.time()
            final_memory_step.duration = memory_step.end_time - step_start_time
            self.memory.steps.append(final_memory_step)

            # 未得到final-answer, 将工具调用的log写入task字典
            _task_info = {
                'answer': final_answer,
                'tool_calls': self.tool_call_records
            }
            self.task_records[self.task] = _task_info

            for callback in self.step_callbacks:
                # For compatibility with old callbacks that don't take the agent as an argument
                if len(inspect.signature(callback).parameters) == 1:
                    callback(final_memory_step)
                else:
                    callback(final_memory_step, agent=self)
            yield final_memory_step

        yield handle_agent_output_types(final_answer)

    def step(self, memory_step: ActionStep, memory_messages=None) -> Union[None, Any]:
        """
        Perform one step in the ReAct framework: the agent thinks, acts, and observes the result.
        Returns None if the step is not final.
        """
        memory_messages = self.write_memory_to_messages() if memory_messages is None else memory_messages

        self.input_messages = memory_messages

        # Add new step in logs
        memory_step.model_input_messages = memory_messages.copy()

        try:
            model_message: ChatMessage = self.model(
                memory_messages,
                tools_to_call_from=list(self.tools.values()),
                stop_sequences=["Observation:"],
            )
            memory_step.model_output_messages = model_message
            if model_message.tool_calls is None or len(model_message.tool_calls) == 0:
                try:
                    content = model_message.content.strip()
                    if content.startswith("Action:") or content.startswith("{"):
                        model_message = json.loads(content.replace("Action:", "").strip())
                except:
                    # raise Exception("Model did not call any tools. Call `final_answer` tool to return a final answer.")
                    # 如果没有工具调用，直接返回模型生成的内容作为最终答案
                    final_answer = model_message.content
                    self.logger.log(
                        Text(f"Final answer (no tool call): {final_answer}", style=f"bold {YELLOW_HEX}"),
                        level=LogLevel.INFO,
                    )
                    memory_step.action_output = final_answer

                    _task_info = {
                        'answer': final_answer,
                        'tool_calls': self.tool_call_records
                    }
                    self.task_records[self.task] = _task_info
                    self.tool_call_records = []
                    return final_answer

            if isinstance(model_message, dict):
                tool_name, tool_call_id = model_message["name"], ""
                tool_arguments = model_message["arguments"]
            else:
                tool_call = model_message.tool_calls[0]
                tool_name, tool_call_id = tool_call.function.name, tool_call.id
                tool_arguments = tool_call.function.arguments
           
        except Exception as e:
            raise AgentGenerationError(f"Error in generating tool call with model:\n{e}", self.logger) from e

        memory_step.tool_calls = [ToolCall(name=tool_name, arguments=tool_arguments, id=tool_call_id)]
        memory_step.model_output = f"Calling tool: '{tool_name}' with arguments: {tool_arguments}"

        # Execute
        self.logger.log(
            Panel(Text(f"Calling tool: '{tool_name}' with arguments: {tool_arguments}")),
            level=LogLevel.INFO,
        )
        if tool_name == "final_answer":
            if isinstance(tool_arguments, dict):
                if "answer" in tool_arguments:
                    answer = tool_arguments["answer"]
                else:
                    answer = tool_arguments
            else:
                answer = tool_arguments
            if (
                isinstance(answer, str) and answer in self.state.keys()
            ):  # if the answer is a state variable, return the value
                final_answer = self.state[answer]
                self.logger.log(
                    f"[bold {YELLOW_HEX}]Final answer:[/bold {YELLOW_HEX}] Extracting key '{answer}' from state to return value '{final_answer}'.",
                    level=LogLevel.INFO,
                )
            else:
                final_answer = answer
                self.logger.log(
                    Text(f"Final answer: {final_answer}", style=f"bold {YELLOW_HEX}"),
                    level=LogLevel.INFO,
                )
            if self.debug:
                take_a_breath()
            memory_step.action_output = final_answer

            # 得到final-answer, task结束, 将工具调用的log写入task字典
            _task_info = {
                'answer': final_answer,
                'tool_calls': self.tool_call_records
            }
            self.task_records[self.task] = _task_info
            # tool call records 置空, 等待下一次task记录
            self.tool_call_records=[]

            return final_answer
            
        else:
            if tool_arguments is None:
                tool_arguments = {}
            try:
                observation = self.execute_tool_call(tool_name, tool_arguments)
            except Exception as e:
                observation= str(e)

            # record tool calling information
            _tool_info = {
                'name': tool_name,
                'args': tool_arguments,
                'observation': observation
            }
            self.tool_call_records.append(_tool_info)

            """
            [{'args': {'query': 'instruction backtranslation in Natural Language Processing (NLP)'}, 'name': 'web_search', 'observation': 'A Google search for \'instruction backtranslation in Natural Language Processing (NLP)\' found 10 results:\n\n## Web Results\...ed instruction backtranslation, starts with a language model finetuned on a small amount of seed data, and a given web corpus.'}]
            """

            observation_type = type(observation)
            if observation_type in [AgentImage, AgentAudio]:
                if observation_type == AgentImage:
                    observation_name = "image.png"
                elif observation_type == AgentAudio:
                    observation_name = "audio.mp3"
                # TODO: observation naming could allow for different names of same type

                self.state[observation_name] = observation
                updated_information = f"Stored '{observation_name}' in memory."
            else:
                updated_information = str(observation).strip()
            self.logger.log(
                f"Observations: {updated_information.replace('[', '|')}",  # escape potential rich-tag-like components
                level=LogLevel.INFO,
            )
            if self.debug:
                take_a_breath()
            memory_step.observations = updated_information
            return None


''' 暂时没有用上 '''
class CodeAgent(MultiStepAgent):
    """
    In this agent, the tool calls will be formulated by the LLM in code format, then parsed and executed.

    Args:
        tools (`list[Tool]`): [`Tool`]s that the agent can use.
        model (`Callable[[list[dict[str, str]]], ChatMessage]`): Model that will generate the agent's actions.
        prompt_templates ([`~agents.PromptTemplates`], *optional*): Prompt templates.
        grammar (`dict[str, str]`, *optional*): Grammar used to parse the LLM output.
        additional_authorized_imports (`list[str]`, *optional*): Additional authorized imports for the agent.
        planning_interval (`int`, *optional*): Interval at which the agent will run a planning step.
        use_e2b_executor (`bool`, default `False`): Whether to use the E2B executor for remote code execution.
        max_print_outputs_length (`int`, *optional*): Maximum length of the print outputs.
        **kwargs: Additional keyword arguments.

    """

    def __init__(
        self,
        tools: List[Tool],
        model: Callable[[List[Dict[str, str]]], ChatMessage],
        prompt_templates: Optional[PromptTemplates] = None,
        grammar: Optional[Dict[str, str]] = None,
        additional_authorized_imports: Optional[List[str]] = None,
        planning_interval: Optional[int] = None,
        search_budget: Optional[int] = None,
        n_rollouts: Optional[int] = None,
        search_type:str='A*',
        use_e2b_executor: bool = False,
        max_print_outputs_length: Optional[int] = None,
        prompts_type: Optional[str] = "default",
        **kwargs,
    ):
        self.additional_authorized_imports = additional_authorized_imports if additional_authorized_imports else []
        self.authorized_imports = list(set(BASE_BUILTIN_MODULES) | set(self.additional_authorized_imports))
        self.use_e2b_executor = use_e2b_executor
        self.search_type=search_type
        self.max_print_outputs_length = max_print_outputs_length
        prompt_templates = prompt_templates or yaml.safe_load(
            importlib.resources.files(f"oagents.prompts.{prompts_type}").joinpath("code_agent.yaml").read_text(encoding="utf-8")
        )
        super().__init__(
            tools=tools,
            model=model,
            prompt_templates=prompt_templates,
            grammar=grammar,
            planning_interval=planning_interval,
            search_budget=search_budget,
            n_rollouts=n_rollouts,
            **kwargs,
        )
        if "*" in self.additional_authorized_imports:
            self.logger.log(
                "Caution: you set an authorization for all imports, meaning your agent can decide to import any package it deems necessary. This might raise issues if the package is not installed in your environment.",
                0,
            )

        if use_e2b_executor and len(self.managed_agents) > 0:
            raise Exception(
                f"You passed both {use_e2b_executor} and some managed agents. Managed agents is not yet supported with remote code execution."
            )

        all_tools = {**self.tools, **self.managed_agents}
        if use_e2b_executor:
            self.python_executor = E2BExecutor(
                self.additional_authorized_imports,
                list(all_tools.values()),
                self.logger,
            )
        else:
            self.python_executor = LocalPythonInterpreter(
                self.additional_authorized_imports,
                all_tools,
                max_print_outputs_length=max_print_outputs_length,
            )

    def initialize_system_prompt(self) -> str:
        system_prompt = populate_template(
            self.prompt_templates["system_prompt"],
            variables={
                "tools": self.tools,
                "managed_agents": self.managed_agents,
                "authorized_imports": (
                    "You can import from any package you want."
                    if "*" in self.authorized_imports
                    else str(self.authorized_imports)
                ),
            },
        )
        return system_prompt
    def edit_code_by_user(self, failed_code: str):
        # 将code保存为.py
        import subprocess

        def y_n_prompt(prompt: str) -> bool:
            """
            提示用户输入 Y/n，并根据输入返回布尔值。

            Args:
                prompt (`str`): 提示信息。

            Returns:
                `bool`: 用户输入 'Y' 或 'y' 时返回 True，输入 'N' 或 'n' 时返回 False。
            """
            while True:
                user_input = input(prompt).strip().lower()
                if user_input in ['y', 'n']:
                    return user_input == 'y'
                else:
                    print("Please input 'Y' or 'n'.")

        new_code = ""
        os.makedirs('tmp', exist_ok=True)
        code_file = os.path.join('tmp', f"{hash(failed_code)}.py")
        with open(code_file, 'w') as f:
            f.write(failed_code)

        if y_n_prompt("Open Code Editor? (Y/n): "):
            result = subprocess.run(["vim", code_file], check=True)
            if result.returncode == 0:
                with open(code_file, 'r') as f:
                    new_code = f.read()
        return new_code

    def execute_code(self, memory_step: ActionStep, code_action: str):
        # Execute
        self.logger.log_code(title="Executing code:", content=code_action, level=LogLevel.INFO)
        try:
            output, execution_logs, is_final_answer = self.python_executor(
                code_action,
                self.state,
            )
            execution_outputs_console = []
            if len(execution_logs) > 0:
                execution_outputs_console += [
                    Text("Execution logs:", style="bold"),
                    Text(execution_logs),
                ]
            observation = "Execution logs:\n" + execution_logs
            return (
                observation,
                output,
                memory_step,
                is_final_answer,
                execution_outputs_console,
            )
        except Exception as e:
            if hasattr(self.python_executor, "state") and "_print_outputs" in self.python_executor.state:
                execution_logs = str(self.python_executor.state["_print_outputs"])
                if len(execution_logs) > 0:
                    execution_outputs_console = [
                        Text("Execution logs:", style="bold"),
                        Text(execution_logs),
                    ]
                    memory_step.observations = "Execution logs:\n" + execution_logs
                    self.logger.log(Group(*execution_outputs_console), level=LogLevel.INFO)
            error_msg = str(e)
            if "Import of " in error_msg and " is not allowed" in error_msg:
                self.logger.log(
                    "[bold red]Warning to user: Code execution failed due to an unauthorized import - Consider passing said import under `additional_authorized_imports` when initializing your CodeAgent.",
                    level=LogLevel.INFO,
                )
            if self.debug:
                new_code = self.edit_code_by_user(failed_code=code_action)
                if new_code:
                    self.execute_code(memory_step, new_code)
            raise AgentExecutionError(error_msg, self.logger)

    def step(self, memory_step: ActionStep, memory_messages=None) -> Union[None, Any]:
        """
        Perform one step in the ReAct framework: the agent thinks, acts, and observes the result.
        Returns None if the step is not final.
        """
        memory_messages = self.write_memory_to_messages() if memory_messages is None else memory_messages

        self.input_messages = memory_messages.copy()

        # Add new step in logs
        memory_step.model_input_messages = memory_messages.copy()
        try:
            additional_args = {"grammar": self.grammar} if self.grammar is not None else {}
            chat_message: ChatMessage = self.model(
                self.input_messages,
                stop_sequences=["<end_code>", "Observation:"],
                **additional_args,
            )
            memory_step.model_output_message = chat_message
            model_output = chat_message.content
            memory_step.model_output = model_output
        except Exception as e:
            raise AgentGenerationError(f"Error in generating model output:\n{e}", self.logger) from e

        self.logger.log_markdown(
            content=model_output,
            title="Output message of the LLM:",
            level=LogLevel.DEBUG,
        )

        # Parse
        try:
            code_action = fix_final_answer_code(parse_code_blobs(model_output))
        except Exception as e:
            error_msg = f"Error in code parsing:\n{e}\nMake sure to provide correct code blobs."
            if self.debug:
                print(error_msg)
                new_code = self.edit_code_by_user(failed_code=model_output)
                if new_code:
                    code_action = fix_final_answer_code(parse_code_blobs(new_code))
            else:
                raise AgentParsingError(error_msg, self.logger)

        memory_step.tool_calls = [
            ToolCall(
                name="python_interpreter",
                arguments=code_action,
                id=f"call_{len(self.memory.steps)}",
            )
        ]

        observation, output, memory_step, is_final_answer, execution_outputs_console = self.execute_code(
            memory_step, code_action
        )

        truncated_output = truncate_content(str(output))
        observation += "Last output from code snippet:\n" + truncated_output
        memory_step.observations = observation

        execution_outputs_console += [
            Text(
                f"{('Out - Final answer' if is_final_answer else 'Out')}: {truncated_output}",
                style=(f"bold {YELLOW_HEX}" if is_final_answer else ""),
            ),
        ]
        self.logger.log(Group(*execution_outputs_console), level=LogLevel.INFO)
        if self.debug:
            take_a_breath()
        memory_step.action_output = output
        return output if is_final_answer else None
    def track_action_state(self,current_step,search_count,new_search_id,answer_message):
        current_depth=current_step.step_number
        score=current_step.score
        evaluate_thought=current_step.evaluate_thought
        model_output=current_step.model_output
        observations=current_step.observations
        action_output=current_step.action_output
        error_message=current_step.error.message if current_step.error else None

        return {
            'search_id':new_search_id,
            'search_count':search_count,
            'current_depth':current_depth,
            'model_output':model_output,
            'action_output':action_output,
            'error_message':error_message,
            'observations':observations,
            "score":score,
            "evaluate_thought":evaluate_thought,
            "answer_message":answer_message
        }
    def track_planning_state(self,planning_step):
        return None

    def _run(self, task: str, images: List[str] | None = None) -> Generator[ActionStep | AgentType, None, None]:
        """
        Run the agent in streaming mode and returns a generator of all the steps.

        Args:
            task (`str`): Task to perform.
            images (`list[str]`): Paths to image(s).
        """
        Task_steps=self.memory.steps
        if self.search_type != 'BON' and self.search_type != 'Tree-Search':
            memory_steps=Task_steps.copy()
            final_answer = None
            self.step_number = 1
            # state=False
            search_count=0
            planning_step=self.planning_step(
                task,
                is_first_step=(self.step_number == 1),
                step=self.step_number,
            )
            self.logger.log_rule(f"Step {self.step_number}", level=LogLevel.INFO)
            # begin search
            memory_steps.append(planning_step)
            # self.memory.steps=memory_steps
        # if self.search_type == 'A*':
        #     open_set = []
        #     # 使用短UUID作为初始search_id
        #     heapq.heappush(open_set, (
        #         -0.5,
        #         self.step_number,
        #         str(uuid.uuid4())[:6],  # 使用UUID
        #         self.write_memory_to_messages(),
        #         memory_steps
        #     ))
        #     search_complete = False
        #     while open_set and search_count<self.search_budget and not search_complete and current_depth<self.max_steps + 1:
        #         current_score,current_depth,current_id,memory_messages,memory_steps  = heapq.heappop(open_set)

        #         for _ in range(self.n_rollouts): #rollout采样
        #             evaluate= False if self.n_rollouts==1 else True
        #             final_answer, evaluation_score, memory_steps_next,memory_messages_next = self.process_step(task, current_depth, images,memory_messages,memory_steps,evaluate=evaluate)

        #             search_count+=1
        #             new_search_id = str(uuid.uuid4())[:6]  # 为新状态生成UUID
        #             self.action_trajectory.append(self.track_action_state(memory_steps_next[-1],search_count,new_search_id))

        #             if final_answer: #search的终止条件
        #                 if evaluation_score==1.0:
        #                     search_complete = True  # Set flag when satisfactory answer found
        #                     break
        #             else:
        #                 if current_depth<self.max_steps + 1: #设置最深搜索深度
        #                     heapq.heappush(open_set, (-evaluation_score/10.0,current_depth+1,new_search_id,memory_messages_next,memory_steps_next))

        #         if search_complete:  # Break outer loop if search complete
        #             self.memory.steps = memory_steps_next  # 添加这行
        #             break

        #         self.step_number=current_depth

        #         if final_answer is not None and self.final_answer_checks is not None:
        #             for check_function in self.final_answer_checks:
        #                 try:
        #                     assert check_function(final_answer, self.memory)
        #                 except Exception as e:
        #                     final_answer = None
        #                     raise AgentError(f"Check {check_function.__name__} failed with error: {e}", self.logger)
        if self.search_type == 'BON':
            evaluate = False if self.n_rollouts == 1 else True
            final_answer_candidates = []  # 存储所有候选答案
            for i in range(self.n_rollouts):  # rollout采样
                memory_steps=Task_steps.copy()
                task_success = False
                self.step_number = 1
                # if self.planning_interval is not None and self.step_number % self.planning_interval == 1:
                planning_step=self.planning_step(
                    task,
                    is_first_step=(self.step_number == 1),
                    step=self.step_number,
                )
                self.logger.log_rule(f"Step {self.step_number}", level=LogLevel.INFO)
                #begin search
                memory_steps.append(planning_step) #不共用记忆池子，使用单独的存储记忆
                memory_messages=self.write_memory_to_messages(memory_steps=memory_steps)
                step_number=self.step_number
                while not task_success and step_number <= self.max_steps:
                    self.step_number=step_number #计数的时候使用step_number独立计数，避免使用同一个计数器
                    final_answer, evaluation_score, answer_message = self.process_step(
                            task, step_number, images, memory_messages, memory_steps, evaluate=evaluate)
                    # 记录当前节点的状态
                    new_search_id = str(uuid.uuid4())[:6]
                    self.action_trajectory.append(self.track_action_state(memory_steps[-1], step_number, new_search_id, answer_message))
                    if final_answer is None and self.step_number == self.max_steps: #如果搜索到最大深度，则提供最终答案
                        final_answer = self.provide_final_answer(task, images)
                        evaluation_score=0.0

                    # 存储每个采样的 final_answer
                    if final_answer:
                        final_answer_candidates.append((final_answer, evaluation_score))
                        task_success = True  # 标记任务成功，继续进行投票判断

                    step_number += 1
                    # 进行投票判断，选择出现次数最多的 final_answer

            if final_answer_candidates:
                final_answer=max(final_answer_candidates, key=lambda x: x[1])[0]
            else:
                final_answer = None
                print("No valid final answer found.")
        elif self.search_type == 'BON-wise':
            task_success=False
            evaluate= False if self.n_rollouts==1 else True
            evaluate=True
            memory_messages=self.write_memory_to_messages(memory_steps=memory_steps)
            # while final_answer is None and self.step_number <= self.max_steps:
            while not task_success and self.step_number <= self.max_steps:
                # if reflection:
                #     reflection_step=self.reflect_planing(task,answer_message,memory_steps_next[-1].evaluate_thought)
                #     memory_steps.append(reflection_step)
                #     memory_messages.extend(reflection_step.to_messages(summary_mode=False))
                #     reflection=False
                #begin search
                # memory_steps=self.memory.steps
                # current_depth=self.step_number
                best_evaluation_score = float('-inf')  # 假设更高的分数更好
                best_memory_steps = None
                best_memory_messages = None
                for i in range(self.n_rollouts): #rollout采样
                    final_answer, evaluation_score, answer_message = self.process_step(
                            task, self.step_number, images, memory_messages, memory_steps, evaluate=evaluate)
                # 比较当前 rollout 的结果与当前最优结果
                    new_search_id = str(uuid.uuid4())[:6]
                    if evaluation_score > best_evaluation_score:
                        best_evaluation_score = evaluation_score
                        best_memory_steps = memory_steps
                        best_memory_messages = memory_messages
                    self.action_trajectory.append(self.track_action_state(memory_steps[-1],self.step_number,new_search_id,answer_message))
                    # if final_answer is not None: #ORM判断任务成功
                    #     if memory_steps_next[-1].score:
                    #         task_success=True
                    #         break
                    #     else:
                    #         reflection=True
                    if final_answer is not None: #ORM判断任务成功
                        task_success=True
                        break
                # 使用最优的 memory_steps_next 和 memory_messages_next 作为下一次的输入
                memory_steps = best_memory_steps
                memory_messages = best_memory_messages
                self.step_number += 1
        elif self.search_type == 'BON-list-wise':
            system_prompt=self.LIST_WISE_prompt
            evaluate= False
            memory_messages=self.write_memory_to_messages(memory_steps=memory_steps)
            while final_answer is None and self.step_number <= self.max_steps:
                answer_message_str=''
                answer_message_list=[]
                memory_steps_list=[]
                memory_messages_list=[]
                for i in range(self.n_rollouts): #rollout采样
                    final_answer, evaluation_score, answer_message = self.process_step(
                            task, self.step_number, images, memory_messages, memory_steps, evaluate=evaluate)
                    # 比较当前 rollout 的结果与当前最优结果
                    new_search_id = str(uuid.uuid4())[:6]
                    self.action_trajectory.append(self.track_action_state(memory_steps[-1],self.step_number,new_search_id,answer_message))
                    answer_message_list.append(answer_message)
                    memory_steps_list.append(memory_steps)
                    memory_messages_list.append(memory_messages)
                    if final_answer is not None:
                        break
                if final_answer is None:
                    for i in range(len(answer_message_list)):
                        answer_message_str+=f'---Trajectroy{i}----\n'+answer_message_list[i]
                    # answer_message_str = '\n'.join(answer_message_list)
                    index,thought=evaluate_answer(answer_message_str,system_prompt,mode='list-wise')
                    if index < 0 or index >= len(answer_message_list):
                        # Re-evaluate if index is out of range
                        index, thought = evaluate_answer(answer_message_str, system_prompt, mode='list-wise')
                    # 使用最优的 memory_steps_next 和 memory_messages_next 作为下一次的输入
                    memory_steps = memory_steps_list[index]
                    memory_messages = memory_messages_list[index]
                    setattr(memory_steps[-1], 'score', index)
                    setattr(memory_steps[-1], 'evaluate_thought', thought)
                self.step_number += 1
        elif self.search_type == 'Beam-Search':
            beam_width = self.n_rollouts
            beam_size = 2
            task_success = False
            evaluate = True
            final_answer_candidates=[]
            memory_messages = self.write_memory_to_messages(memory_steps=memory_steps)
            current_nodes = [(self.step_number, memory_steps, memory_messages, 0.0)] * beam_size  # 初始节点

            while not task_success and self.step_number <= self.max_steps:
                next_nodes = []  # 存储下一层的节点
                successful_branches = 0  # 计数成功找到 final_answer 的分支数量

                for node in current_nodes:
                    step_number, memory_steps, memory_messages, _ = node

                    for i in range(beam_size):  # rollout采样
                        final_answer, evaluation_score, answer_message = self.process_step(
                            task, step_number, images, memory_messages, memory_steps, evaluate=evaluate
                        )
                        # 记录当前节点的状态
                        new_search_id = str(uuid.uuid4())[:6]
                        self.action_trajectory.append(self.track_action_state(memory_steps[-1], step_number, new_search_id, answer_message))

                        # 检查任务是否成功
                        if final_answer is not None:
                            final_answer_candidates.append((final_answer, evaluation_score))

                            successful_branches += 1  # 成功找到 final_answer 的分支数量加一

                        next_nodes.append((step_number + 1, memory_steps, memory_messages, evaluation_score))

                        # 检查是否所有分支都成功找到 final_answer
                        if successful_branches >= beam_size:
                            task_success = True
                            break

                # 选择 N 个最佳节点
                next_nodes.sort(key=lambda x: x[3], reverse=True)  # 根据分数排序
                current_nodes = next_nodes[:beam_width//beam_size]  # 保留前 N 个节点

                self.step_number += 1
            final_answer=max(final_answer_candidates, key=lambda x: x[1])[0]

        elif self.search_type == 'Tree-Search':
            evaluate = True
            final_answer_candidates = []  # 存储所有候选答案
            beam_size = 2  # 每个子树的beam size
            beam_width = self.n_rollouts  # 总的搜索宽度

            # 初始化两个独立的子树
            trees = []
            for tree_idx in range(beam_width//beam_size):  # 维护两棵独立的树
                tree_memory_steps=Task_steps.copy()
                self.step_number = 1
                # 初始化每棵树的规划
                planning_step=self.planning_step(
                    task,
                    is_first_step=(self.step_number == 1),
                    step=self.step_number,
                )
                self.logger.log_rule(f"Tree {tree_idx + 1} Step {self.step_number}", level=LogLevel.INFO)

                # 初始化树的起始状态
                # tree_memory_steps = self.memory.steps.copy()
                tree_memory_steps.append(planning_step)
                tree_memory_messages = self.write_memory_to_messages(memory_steps=tree_memory_steps)
                trees.append({
                    'memory_steps': tree_memory_steps,
                    'memory_messages': tree_memory_messages,
                    'step_number': self.step_number
                })
            for tree_idx, tree in enumerate(trees):
                memory_steps,memory_messages,step_number=tree['memory_steps'],tree['memory_messages'],tree['step_number']
                while step_number <= self.max_steps:
                    self.step_number=step_number
                    # for tree_idx, tree in enumerate(trees):
                    next_candidates = []  # 存储当前层的候选节点
                    # 在当前层进行beam_size次采样
                    for _ in range(beam_size):
                        best_evaluation_score = float('-inf')  # 假设更高的分数更好
                        final_answer, evaluation_score, answer_message = self.process_step(
                            task, step_number, images, memory_messages, memory_steps, evaluate=evaluate
                        )
                        # 记录搜索轨迹
                        new_search_id = str(uuid.uuid4())[:6]
                        self.action_trajectory.append(
                            self.track_action_state(memory_steps[-1], step_number, new_search_id, answer_message)
                        )
                        if evaluation_score > best_evaluation_score:
                            best_evaluation_score = evaluation_score
                            best_memory_steps = memory_steps
                            best_memory_messages = memory_messages

                        # 如果找到最终答案，添加到候选列表
                        if final_answer:
                            final_answer_candidates.append((final_answer, evaluation_score))
                            break
                    memory_steps = best_memory_steps
                    memory_messages = best_memory_messages
                    step_number += 1


            # 从所有候选答案中选择最佳答案
            if final_answer_candidates:
                final_answer = max(final_answer_candidates, key=lambda x: x[1])[0]
                # 使用得分最高的树的状态作为最终状态
                best_tree = max(trees, key=lambda x: max((step.score for step in x['memory_steps'] if hasattr(step, 'score')), default=0))
                self.memory.steps = best_tree['memory_steps']
            else:
                final_answer = self.provide_final_answer(task, images)
                print("No valid final answer found in tree search, using default answer.")
        elif self.search_type == 'none':
            while final_answer is None and self.step_number <= self.max_steps:
                step_start_time = time.time()
                memory_step = ActionStep(
                    step_number=self.step_number,
                    start_time=step_start_time,
                    observations_images=images,
                )
                try:
                    final_answer = self.step(memory_step)
                    if final_answer is not None and self.final_answer_checks is not None:
                        for check_function in self.final_answer_checks:
                            try:
                                assert check_function(final_answer, self.memory)
                            except Exception as e:
                                final_answer = None
                                raise AgentError(f"Check {check_function.__name__} failed with error: {e}", self.logger)
                except AgentError as e:
                    memory_step.error = e
                    raise
                finally:
                    memory_step.end_time = time.time()
                    memory_step.duration = memory_step.end_time - step_start_time
                    self.memory.steps.append(memory_step)
                    for callback in self.step_callbacks:
                        # For compatibility with old callbacks that don't take the agent as an argument
                        if len(inspect.signature(callback).parameters) == 1:
                            callback(memory_step)
                        else:
                            callback(memory_step, agent=self)
                    self.step_number += 1
                    memory_steps_next=self.memory.steps #just for Align data stream
                    yield memory_step
        else:
            raise ValueError

        assert memory_steps is not None, "memory_steps cannot be None"
        assert len(memory_steps) > 0, f"memory_steps cannot be empty, current length: {len(memory_steps)}"
        self.memory.steps=memory_steps
        if final_answer is None and self.step_number == self.max_steps + 1:
            error_message = "Reached max steps."
            final_answer = self.provide_final_answer(task, images)
            final_memory_step = ActionStep(
                step_number=self.step_number, error=AgentMaxStepsError(error_message, self.logger)
            )
            final_memory_step.action_output = final_answer
            final_memory_step.end_time = time.time()
            self.memory.steps.append(final_memory_step)
            for callback in self.step_callbacks:
                # For compatibility with old callbacks that don't take the agent as an argument
                if len(inspect.signature(callback).parameters) == 1:
                    callback(final_memory_step)
                else:
                    callback(final_memory_step, agent=self)
            yield final_memory_step

        yield handle_agent_output_types(final_answer)
    # def get_memory_message(self,memory_steps,memory_step):
    #     action_message=[]
    #     for step in memory_steps:
    #         if isinstance(step, ActionStep):
    #             # 提取 ActionStep 类型的字段信息
    #             action_message.append(step.actio_output nif step.action_output else step.observations)
    #         elif isinstance(step, TaskStep):
    #             # 提取 TaskStep 类型的字段信息
    #             task_message = step.task
    #         elif isinstance(step, PlanningStep):
    #             # 提取 PlanningStep 类型的字段信息
    #             plan = step.plan
    #             facts = step.facts
    #         else:
    #             # 处理未知类型或默认情况
    #             print("Unknown step type or default handling.")

    #     action_message.append(memory_step.action_output if memory_step.action_output else memory_step.observations)
    #     action_message_str = '\n'.join(f'Action{idx+1}:{message}' for idx, message in enumerate(action_message))
    #     final_merge_message = f"Question:{task_message}\n Plan:{plan}\n{action_message_str}"
    #     return final_merge_message
    def get_memory_step_message(self,
                                memory_steps: List[ActionStep | PlanningStep | TaskStep | ReflectionStep],
                                current_step: ActionStep | PlanningStep | TaskStep | ReflectionStep) -> str:
        #return 一串字符，cover每次branch的step信息，包含输入step_number，observations，action_output，model_output，error，score
        def step_to_string(step: ActionStep | PlanningStep | TaskStep | ReflectionStep) -> str:
            """
            将任意类型的 MemoryStep 转成可读字符串，不同 Step 类型可根据需要输出不同字段。
            """
            # 1) 如果是 ActionStep
            if isinstance(step, ActionStep):
                step_info = step.dict()
                return (
                    f"[ActionStep]\n"
                    f"  step_number: {step_info['step']}\n"
                    # f"  start_time: {step_info['start_time']}\n"
                    # f"  end_time: {step_info['end_time']}\n"
                    # f"  duration: {step_info['duration']}\n"
                    f"  observations: {step.observations}\n"
                    f"  action_output: {step_info['action_output']}\n"
                    f"  model_output: {step_info['model_output']}\n"
                    f"  error: {step_info['error']}\n"
                    f"  score: {step.score}\n"
                    # f"  tool_calls_count: {len(step_info['tool_calls']) if step_info['tool_calls'] else 0}"
                )
            # 2) 如果是 TaskStep
            elif isinstance(step, TaskStep):
                return (
                    f"[TaskStep]\n"
                    f"  task: {step.task}\n"
                    f"  description: {step.description if hasattr(step, 'description') else None}"
                )
            # 3) 如果是 PlanningStep
            elif isinstance(step, PlanningStep):
                return (
                    f"[PlanningStep]\n"
                    f"  facts that we knows: {step.facts}\n"
                    f"  current plan: {step.plan}\n"
                    f"  reason: {step.reason if hasattr(step, 'reason') else None}"
                )
            # 4) 否则，处理未知类型
            else:
                step_attrs = []
                for attr_name, attr_value in step.__dict__.items():
                    step_attrs.append(f"  {attr_name} = {attr_value}")
                step_attrs_str = "\n".join(step_attrs)
                return f"[Unknown Step Type: {type(step)}]\n{step_attrs_str}"

        # 将 current_step 也视为最后一个 step，需要包含在结果中
        all_steps = memory_steps + [current_step]
        lines = []
        for idx, step in enumerate(all_steps, start=1):
            step_str = step_to_string(step)
            lines.append(f"===== Step {idx} =====\n{step_str}")

        # 拼成一个整块字符串
        final_merge_message = "\n".join(lines)
        return final_merge_message
    def get_memory_message(self,memory_messages_next):
        final_message=''
        for item in memory_messages_next:
            final_message+=item['content'][0]['text']+'\n'
        return final_message



    def process_step(self, task, step_number, images,memory_messages,memory_steps,evaluate=True):
        """
        Process a single step in the agent's execution.

        Args:
            task (`str`): The task to perform.
            step_number (`int`): The current step number.
            images (`list[str]`): Paths to image(s).

        Returns:
            Tuple[Optional[str], float, ActionStep]: Final answer, evaluation score, and memory step.
        """
        self.step_number=step_number
        step_start_time = time.time()
        memory_step = ActionStep(
            step_number=self.step_number,
            start_time=step_start_time,
            observations_images=images,
            score=0.0  # 初始化score
        )
        final_answer = None
        evaluation_score = float('-inf')  # Initialize to a very low score
        # if True:
        try:
            evaluation_content,answer_message="",""
            final_answer = self.step(memory_step,memory_messages)# Evaluate the quality of the answer
            if final_answer is not None and self.final_answer_checks is not None:
                for check_function in self.final_answer_checks:
                    try:
                        assert check_function(final_answer, self.memory)
                    except Exception as e:
                        final_answer = None
                        raise AgentError(f"Check {check_function.__name__} failed with error: {e}", self.logger)
            if final_answer is not None:
                if not isinstance(final_answer, str):
                    final_answer = str(final_answer)  # 转换为字符串
                mode='ORM'
                # memory_message=self.get_memory_message(memory_messages_next)
                memory_message = self.get_memory_step_message(memory_steps, memory_step)
                answer_message = memory_message + '\n' + 'Final_Answer: ' + final_answer
                system_prompt = self.ORM_prompt
            else:
                mode='PRM'
                #TODO 修改成基于全部process的函数
                # answer_message=self.get_memory_message(memory_messages_next)
                answer_message = self.get_memory_step_message(memory_steps, memory_step)
                system_prompt = self.PRM_prompt
            if evaluate:
                evaluation_score,evaluation_content = evaluate_answer(answer_message, system_prompt, mode)
            else:
                evaluation_score,evaluation_content=0.0,""

        except AgentError as e:
            memory_step.error = e
            evaluation_score=0.0
            evaluation_content=""
            answer_message=""
        finally:
            setattr(memory_step, 'score', evaluation_score)
            setattr(memory_step, 'evaluate_thought', evaluation_content)
            memory_step.end_time = time.time()
            memory_step.duration = memory_step.end_time - step_start_time
            # self.memory.steps.append(memory_step)
            memory_messages.extend(memory_step.to_messages())
            memory_steps.append(memory_step)
            for callback in self.step_callbacks:
                # For compatibility with old callbacks that don't take the agent as an argument
                if len(inspect.signature(callback).parameters) == 1:
                    callback(memory_step)
                else:
                    callback(memory_step, agent=self)
        return final_answer, evaluation_score,answer_message