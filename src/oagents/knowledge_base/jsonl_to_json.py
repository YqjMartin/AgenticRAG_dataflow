import json
from json import JSONDecodeError
from typing import Any, Dict, List, Union

def deep_parse_jsonlike_strings(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: deep_parse_jsonlike_strings(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [deep_parse_jsonlike_strings(item) for item in obj]
    elif isinstance(obj, str):
        try:
            parsed = json.loads(obj.replace("'", "\""))
            return deep_parse_jsonlike_strings(parsed)
        except (JSONDecodeError, AttributeError):
            if r'\"' in obj or r"\'" in obj:
                try:
                    parsed = json.loads(obj.replace(r'\"', '"').replace(r"\'", "'"))
                    return deep_parse_jsonlike_strings(parsed)
                except JSONDecodeError:
                    return obj
            return obj
    return obj

def jsonl_to_nested_json(input_path: str, output_path: str) -> None:
    processed_data = []
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                entry: Dict[str, Union[dict, list, str]] = json.loads(line)
            except JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
                continue
            
            entry = deep_parse_jsonlike_strings(entry)
            
            if "intermediate_steps" in entry:
                for step in entry["intermediate_steps"]:
                    if "start_time" in step and isinstance(step["start_time"], float):
                        step["start_time"] = f"{step['start_time']:.6f}"
                    if "end_time" in step and isinstance(step["end_time"], float):
                        step["end_time"] = f"{step['end_time']:.6f}"
            
            processed_data.append(entry)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(
            processed_data,
            f,
            indent=2,
            ensure_ascii=False,
            separators=(',', ': '),
            default=str
        )

if __name__ == "__main__":
    jsonl_to_nested_json('knowledge_database.jsonl', 'knowledge_database.json')