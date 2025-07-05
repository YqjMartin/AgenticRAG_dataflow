'''
    @ Guan
    用于search agent的reflection/augmentation module
    功能包括: search query reflection, search result reflection以及search query rollout
'''
import yaml
import json
import os
import importlib

from oagents.models import OpenAIServerModel, ChatMessage

class SearchReflector:
    def __init__(self,
                 model:OpenAIServerModel=None):
        
        self.model = model if model is not None else \
                    OpenAIServerModel(
                        model_id="gpt-4.1",
                        api_base=os.getenv("OPENAI_BASE_URL"),
                        api_key=os.getenv("OPENAI_API_KEY")
                    )

        try:
            prompts=yaml.safe_load(
            importlib.resources.files("reflectors.prompts").joinpath("search_prompts.yaml").read_text())

            self.query_rollout_prompt = prompts['query_rollout']
            self.query_reflect_prompt = prompts['query_reflection']
            self.result_reflect_prompt = prompts['result_reflection']
        except:
            self.query_rollout_prompt = ""
            self.query_reflect_prompt = ""
            self.result_reflect_prompt = ""


    def _pack_message(self, role: str, content: str) -> list[dict]:
        packed_message = [
                {
                    "role": role,
                    "content": content,
                }
            ]
        return packed_message
    

    def query_rollout(self, query:str, n_rollout:int=1) -> list[str]:
        # query改写
        prompted_query = self.query_rollout_prompt.format(query=query, roll_out=n_rollout)
        input_messages = self._pack_message(role="user", content=prompted_query)

        chat_message :ChatMessage = self.model(
            messages = input_messages,
            stop_sequences=["<end>"],
        )
        model_output = chat_message.content

        # extract querys
        try:
            queries = model_output.split('<begin>')[1].strip()
            queries = queries.split("\n")[:n_rollout] #避免返回过多的query
        except:
            queries = []

        queries.append(query) # 添加原本的query
        return queries
    

    def query_reflect(self, origin_query: str):
        messages = []
        # Add System Prompt
        if self.query_reflect_prompt != "":
            messages += self._pack_message(role='system', content=self.query_reflect_prompt)
        # Prepare Query message
        query_message = "Now you will receive a search query, please help me revise it and output strictly with Output Format." \
        f"The original search query is {origin_query}"

        messages += self._pack_message(role="user", content=query_message)
        # Response
        chat_message :ChatMessage = self.model(
                messages = messages
            )
        model_output = chat_message.content

        # 提取答案
        try:
            result = json.loads(model_output)
            analysis_info = result['Analysis']
            augmented_query = result['Augmented Query']
        except Exception as e:
            print(e)
            return "", origin_query

        return analysis_info, augmented_query


    def result_reflect(self, origin_result) -> bool:
        '''
            @ function: determine whether the result is enough for current search turn
            @ return: True if need further search else False
        '''

        return False


# test case
if __name__=="__main__":
    from tqdm import tqdm

    KEY = ""
    URL = ""

    model = OpenAIServerModel(model_id="gpt-4.1",
                              api_base=URL,
                              api_key=KEY
                              )
    reflector = SearchReflector(model=model)

    search_querys = [
        "Pie Menus or Linear Menus, Which Is Better? 2015 full list of authors",
        "Black to move forced win puzzle image data/gaia/validation/cca530fc-4052-43b2-b130-b30968d8aa44.png",
        "Recommend something nice to watch",
        "Find a hotel suitable for a family trip this weekend with children's play facilities, moderate prices, and good reviews",
        "Find the price of a luxury king room at a five-star hotel in Chaoyang District, Beijing, on October 15, 2024",
        "Explain the meaning of the idiom 'after a spring rain, bamboo shoots come up in great numbers"
                    ]

    with open("./test_query_reflect.txt", 'w') as fw:
        idx = 0
        for query in tqdm(search_querys):
            info, new_query = reflector.query_reflect(query)
            message = f"===== Case [{idx+1}] =====\n" + \
                        f"original query: {query}\n" + \
                        f"reflection: {info}\n" + \
                        f"augmented query: {new_query}\n\n"
            fw.write(message)
            idx += 1
