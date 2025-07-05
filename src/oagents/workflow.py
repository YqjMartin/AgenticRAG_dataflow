import re
from typing import Union, List
import os
import json

save_dir = 'workflow'
os.makedirs(save_dir, exist_ok=True)


class Step:
    """步骤对象，包含原始编号和描述"""
    __slots__ = ('index', 'description')

    def __init__(self, index: int, description: str):
        self.index = index
        self.description = description.strip()

    def __repr__(self):
        return f"Step({self.index}, '{self.description}')"

    def __str__(self):
        return f"{self.index}. {self.description}"


class Workflow:
    def __init__(self, steps: Union[str, List[Step]] = None, wf_name='gaia_validation'):
        self._steps = []
        self.load(steps)
        self.wf_name = wf_name

    def load(self, steps: Union[str, List[Step]] = None):
        if isinstance(steps, str):
            self._steps = self.load_from_str(steps)
        elif steps:
            self._steps = list(steps)

    def load_from_str(self, s: str):
        """完全替换步骤列表"""
        return self._parse_initial_str(s)

    def apply_update(self, s: str):
        """增量更新步骤（支持中间插入和尾部追加）"""
        new_steps = self._parse_update_str(s)
        if not new_steps:
            return

        start_num = new_steps[0].index
        max_allowed_start = len(self._steps) + 1

        # 校验起始位置有效性（允许在末尾追加）
        if not (1 <= start_num <= max_allowed_start):
            raise ValueError(
                f"更新起始步骤 {start_num} 超出允许范围（1-{max_allowed_start}）"
            )

        # 检查步骤覆盖范围
        overlap_end = start_num + len(new_steps) - 1
        original_end = len(self._steps)

        # 动态调整保留区间
        if start_num <= original_end:
            # 覆盖现有步骤的情况
            self._steps = self._steps[:start_num-1] + new_steps
        else:
            # 纯追加新步骤的情况
            self._steps += new_steps

        # 自动补全编号连续性检查
        if start_num <= original_end and (overlap_end < original_end):
            print(f"警告：步骤 {overlap_end+1}-{original_end} 已被丢弃")

    @staticmethod
    def _parse_update_str(s: str) -> List[Step]:
        """解析更新字符串（允许独立编号序列）"""
        steps = []
        current_base = None
        for line in s.splitlines():
            if step := Workflow._parse_line(line):
                if not steps:
                    current_base = step.index
                adjusted_index = current_base + len(steps)
                if step.index != adjusted_index:
                    raise ValueError(
                        f"更新步骤不连续，期望 {adjusted_index}，实际 {step.index}"
                    "\n提示：更新块需要自成连续序列"
                    )
                steps.append(Step(adjusted_index, step.description))
        return steps

    @staticmethod
    def _parse_initial_str(s: str) -> List[Step]:
        """解析初始化字符串（强制从1开始连续）"""
        steps = []
        expected = 1
        for line in s.splitlines():
            if step := Workflow._parse_line(line):
                if step.index != expected:
                    raise ValueError(f"步骤不连续，期望 {expected}，实际 {step.index}")
                expected += 1
                steps.append(step)
        return steps

    @staticmethod
    def _parse_line(line: str) -> Union[Step, None]:
        """解析单行步骤"""
        pattern = re.compile(
            r'^\s*[([{]?(\d+)[.)\]、}]\s*(.*)$',
            flags=re.UNICODE
        )
        line = line.strip()
        if match := pattern.match(line):
            return Step(int(match.group(1)), match.group(2))
        return None
    def load_from_file(self):
        wf_path = os.path.join(save_dir, f"{self.wf_name}.json")
        if os.path.exists(wf_path):
            with open(wf_path, 'r') as f:
                data = json.load(f)
            if self.task_id in data:
                self._steps = self.load_from_str(data[self.task_id]['workflow'])
            else:
                raise ValueError(f"task_id {self.task_id} not found in {wf_path}")
        else:
           raise FileNotFoundError

    def save_to_file(self, data_dict):
        # todo: 加锁提高线程安全性
        wf_path = os.path.join(save_dir, f"{self.wf_name}.jsonl")
        with open(wf_path, 'a+', encoding='utf-8') as f:
            f.write(json.dumps(data_dict) + "\n")

    def __getitem__(self, index: int):
        return self._steps[index - 1]  # 1-based 访问

    def __len__(self):
        return len(self._steps)

    def __repr__(self):
        return f"Workflow({self._steps})"

    def __str__(self):
        return "\n".join(
            f"{str(step)}"
            for step in self._steps
        )

# 使用示例
if __name__ == "__main__":
    doc = """
    <问题分析>
    这里是对问题的分析说明...

    1. 收集用户需求
    2) 创建技术方案
    (3) 编写核心代码
    4、进行单元测试
    5. 部署生产环境

    <总结>
    这里是总结内容...
    """

    try:
        workflow = Workflow(doc)
        print("解析结果:", workflow)

        print("\n访问单个步骤:")
        print(workflow[1])  # 访问第一个步骤（基于1的索引）

    except ValueError as e:
        print(f"解析错误: {e}")

    new_doc = """
    <问题分析>
    这里是对问题的分析说明...

    4、代码走查
    5. 代码评审
    (6) 做PPT

    <总结>
    这里是总结内容...
    """
    try:
        workflow.apply_update(new_doc)
        print("\n更新后的流程:")
        print(workflow)
    except ValueError as e:
        print(f"更新错误: {e}")