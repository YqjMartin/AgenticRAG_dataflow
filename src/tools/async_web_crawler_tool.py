'''
@ TOOL for Web Crawler
'''

# Shamelessly stolen from Microsoft Autogen team: thanks to them for this great resource!
# https://github.com/microsoft/autogen/blob/gaia_multiagent_v01_march_1st/autogen/browser_utils.py
import os
import time
from typing import Dict, List, Optional
import requests
from serpapi import GoogleSearch
import asyncio
from crawl4ai import AsyncWebCrawler

from oagents import Tool, OpenAIServerModel
from camel.embeddings import OpenAIEmbedding
from oagents.simple_vector_retriever import SimpleVectorRetriever
from camel.storages.vectordb_storages import QdrantStorage
from taskcraft.src.similarity import MinHash  # similarity scorer
from taskcraft.src.reflectors import SearchReflector


class SimpleCrawler:
    '''
        @overview: a simple crawler for agent to search contents related to query, crawl pages through url.etc
        @func:

    '''

    def __init__(self,
                 serpapi_key: Optional[str] = None,
                 model: OpenAIServerModel=None,
                 reflection: bool=True,
                 roll_out: int=0,
                 search_limit: int=10,
                 serp_num: int=10,
                 topk: int=1,
                 rerank: bool=False,
                 chunk_size: int=500,
                 chunk_overlap: int=50,
                 use_db: Optional[bool] = False,
                 path: Optional[str] = None,
                 ):
        self.serpapi_key = os.getenv("SERP_API_KEY") if serpapi_key is None else serpapi_key
        self.model = model

        if model is not None and reflection:
            self.reflector = SearchReflector(model=model)
            self.reflection = True
        else:
            self.reflection = False

        # 需要加入拆分逻辑
        self.prompt_template = """
Your task is to receive a search query, analyze the user's intent, break down the search task, and generate N new search queries to improve search quality. The original query is: {query}. Generate {roll_out} alternative versions of this query. Format your response as follows:
<begin>
query_1
query_2
...
query_N
<end>
Where N represents the total number of queries you've generated. Each query should be distinct and maintain the core intent of the original while broadening its scope for more comprehensive search results.
"""
        self.evaluate_prompt_template = """
Prompt:
You are an expert in evaluating search result relevance. Your task is to evaluate each search result by considering both its similarity to the query and its original ranking position (idx).
Inputs:
A search query
Multiple search results, each containing:
idx (original ranking position, with smaller idx indicating more prominent original ranking)
title
snippet
Output:
For each search result, provide two scores between 0-10:
Similarity score (0-10): Based on how well the title and snippet match the query's intent and keywords
Overall score (0-10): Combination of similarity score and idx position
Scoring Guidelines:
Similarity Score Evaluation:
9-10: Directly answers the query with precise keywords and relevant context
7-8: Strongly relevant but may lack some specific details
5-6: Partially relevant with some related information
3-4: Marginally related with minimal relevant content
0-2: Virtually unrelated or completely off-topic
Overall Score Formula:
overall_score = (similarity_score * 0.7) + (idx_weight * 0.3)
Where idx_weight is calculated as:
idx_weight = (max_idx - current_idx + 1) / max_idx * 10
Example:
Query: "climate change effects"
Search Result 1:
idx: 1
title: "Global Warming Impacts"
snippet: "Detailed analysis of how rising temperatures affect ecosystems..."
Similarity score: 9/10 (directly addresses climate change effects)
idx_weight: 9.5/10
Overall score: (9 * 0.7) + (9.5 * 0.3) = 9.15
Search Result 2:
idx: 5
title: "Weather Patterns Change"
snippet: "Study on shifting precipitation patterns..."
Similarity score: 7/10 (related but less direct)
idx_weight: 6/10
Overall score: (7 * 0.7) + (6 * 0.3) = 7.3
Instructions:
Analyze the query's core intent
Evaluate each result's relevance to the query for similarity score
Calculate idx_weight using the formula
Compute overall score using the weighted formula
------
Now you get the inputs:
query: {query}
idx: {idx}
title: {title}
snippet: {snippet}
Please return ONLY the overall scores as format: score:[final score]
<end>
"""

        self.history = []
        self.roll_out = roll_out
        self.serp_num = serp_num
        self.search_limit = search_limit
        self.topk = topk
        self.rerank = rerank

        # Retriever for extract related information in the crawled contents
        self.retriever = SimpleVectorRetriever(embedding_model=OpenAIEmbedding(),
                                               chunk_size=chunk_size, 
                                               chunk_overlap=chunk_overlap,
                                               path=None if not use_db else path)
    

    # serpapi return snippets and concat them to content
    def _search(self, query: str, filter_year: Optional[int] = None) -> List[str]:
        if self.serpapi_key is None:
            raise ValueError("Missing SerpAPI key.")

        self.history.append((query, time.time()))

        params = {
            "engine": "google",  # google
            "q": query,
            "api_key": self.serpapi_key,
            "num": self.serp_num
        }
        if filter_year is not None:
            params["tbs"] = f"cdr:1,cd_min:01/01/{filter_year},cd_max:12/31/{filter_year}"

        search = GoogleSearch(params)

        results = search.get_dict()
        '''
        @ serp result format -> json dict
        dict_keys(['search_metadata', 
                    'search_parameters', 
                    'search_information', 
                    'knowledge_graph', 
                    'inline_images', 
                    'related_questions', 
                    'organic_results', 
                    'top_stories_link', 
                    'top_stories_serpapi_link', 
                    'related_searches', 
                    'pagination', 
                    'serpapi_pagination']
                    )
        '''

        self.page_title = f"{query} - Search"
        if "organic_results" not in results.keys():
            raise Exception(f"No results found for query: '{query}'. Use a less specific query.")
        if len(results["organic_results"]) == 0:
            year_filter_message = f" with filter year={filter_year}" if filter_year is not None else ""
            return f"No results found for '{query}'{year_filter_message}. Try with a more general query, or remove the year filter."

        web_snippets: List[str] = list()
        idx = 0
        if "organic_results" in results:
            for page in results["organic_results"]:
                idx += 1
                date_published = ""
                if "date" in page:
                    date_published = "\nDate published: " + page["date"]

                source = ""
                if "source" in page:
                    source = "\nSource: " + page["source"]

                snippet = ""
                if "snippet" in page:
                    snippet = "\n" + page["snippet"]

                _search_result = {
                    "idx": idx,
                    "title": page["title"],
                    "date": date_published,
                    "snippet": snippet,
                    "source": source,
                    "link": page['link']
                }

                web_snippets.append(_search_result)

        return web_snippets

    def _pre_visit(self, url):
        for i in range(len(self.history) - 1, -1, -1):
            if self.history[i][0] == url:
                return f"You previously visited this page {round(time.time() - self.history[i][1])} seconds ago.\n"
        return ""

    def _to_contents(self, query: str, snippets: List):
        web_snippets = []
        idx = 1
        for search_info in snippets:
            redacted_version = f"{idx}. [{search_info['title']}]({search_info['link']})" + \
                               f"{search_info['date']}{search_info['source']}\n{self._pre_visit(search_info['link'])}{search_info['snippet']}"

            redacted_version = redacted_version.replace("Your browser can't play this video.", "")
            web_snippets.append(redacted_version)
            idx += 1

        content = (
                f"A Google search for '{query}' found {len(web_snippets)} results:\n\n## Web Results\n"
                + "\n\n".join(web_snippets)
        )
        return content

    def _expand_query(self, query: str) -> List[str]:
        # query改写
        prompted_query = self.query_rollout_prompt_template.format(query=query, roll_out=self.roll_out)
        input_messages = [
            {
                "role": "user",
                "content": prompted_query,
            }
        ]
        chat_message = self.model(
            messages=input_messages,
            stop_sequences=["<end>"],
        )
        model_output = chat_message.content
        # extract querys
        try:
            queries = model_output.split('<begin>')[1].strip()
            queries = queries.split("\n")[:self.roll_out]  # 避免返回过多的query
        except:
            queries = []

        queries.append(query)  # 添加原本的query
        return queries

    # asynchronicly crawl page base on url
    async def _crawl_page(self, url):
        async with AsyncWebCrawler(verbose=True) as crawler:
            # Crawl the specified URL
            result = await crawler.arun(url=url)
            # Return the extracted content in Markdown format
            return result.markdown
    # static method for query related contents
    def _query(self, query, contents):
        return self.retriever.retrieve(query=query, contents=contents)

    # check if the url or query has been searched before
    def _check_history(self, url_or_query):
        header = ''
        for i in range(len(self.history) - 2, -1, -1):  # Start from the second last
            if self.history[i][0] == url_or_query:
                header += f"You previously visited this page {round(time.time() - self.history[i][1])} seconds ago.\n"
                return header
        # 未出现过则加入history
        self.history.append((url_or_query, time.time()))
        return header

    # evaluate the similarity between snippet and reference snippet
    def evaluate_similarity_score(self, ref_snippets, info, style="minhash"):
        '''
        @ evaluate similarity between snippet and snippets/title
        @ support compute style
            - minhash (default)
            - bm25 (需要将info换成完整的snippets列表)
            - llm_score: use prompted llm to output score response
        '''
        snippet = info['snippet']
        title = info['title']
        idx = info['idx']
        score = 0

        assert style in ["minhash", 'bm25', 'llm_score']

        # ToDo: 增加其他打分逻辑, 目前只有minhash能用, llm太慢, 这个函数的命名暂时还有些混淆
        if style == "minhash":
            minhash = MinHash(num_perm=128)
            for ref_snippet in ref_snippets:
                ref = ref_snippet['title'] + ref_snippet['snippet']
                score += minhash.similarity(ref, title + snippet)
            return score / len(ref_snippets)

        elif style == "llm_score":
            # 需要重构
            prompt = self.evaluate_prompt_template.format(query="query", idx=idx, title=title, snippet=snippet)
            input_messages = [
                {
                    "role": "user",
                    "content": prompt,
                }
            ]
            chat_message = self.model(
                messages=input_messages,
                stop_sequences=["<end>"],
            )
            model_output = chat_message.content
            # extract score
            score = model_output.split("score:")[1].strip()
            score = float(score)

        return score

    def aggregate(self, ref_query, search_results: Dict, intersect: bool = False, rerank: bool = False) -> List[str]:
        def _dedup(raw_list: List) -> List:
            # dedup
            seen_set = set()
            unique_results = []
            intersect_results = []
            for result in raw_list:
                if result['link'] in seen_set:
                    intersect_results.append(result)
                else:
                    seen_set.add(result['link'])
                    unique_results.append(result)
            return unique_results, intersect_results

        # 每个expanded_query对应的搜索结果重排序
        new_search_results = []
        ref_results = []
        tail_results = []
        for q, results in search_results.items():
            # 保证最开始的query的结果在最前面, 取top-k, 这里k=1
            if q == ref_query:
                ref_results = [results[0]] + ref_results
            else:
                ref_results.append(results[0])
            tail_results += results[1:]
        # 加入尾部的搜索结果
        new_search_results = ref_results + tail_results
        # 去重
        new_search_results, intersect_results = _dedup(new_search_results)

        # 取交集
        if intersect:
            if len(intersect_results) < 2:
                print("Not enough intersect results\n")
                return search_results[ref_query]
            intersect_results, _ = _dedup(intersect_results)
            for i in range(len(intersect_results)):
                intersect_results[i]['idx'] = i + 1
            return intersect_results[:self.search_limit]

        # 重新分配索引
        for i in range(len(new_search_results)):
            new_search_results[i]['idx'] = i + 1

        # 无需重排, 直接返回聚合结果
        if not rerank:
            return new_search_results[:self.search_limit]

        # 根据参考的query进行重排序, 保留前k个作为参考, default k=3
        rerank_results = new_search_results[self.topk:]
        # ref_snippet = new_search_results[0]['title'] + new_search_results[0]['snippet']
        ref_results = new_search_results[:self.topk]
        # 保留前k个作为参考

        for item in rerank_results:
            score = self.evaluate_similarity_score(ref_results, item)
            item["score"] = score
        rerank_results = sorted(rerank_results, key=lambda x: x["score"], reverse=True)
        new_search_results = ref_results + rerank_results

        # 重新分配索引
        for i in range(len(new_search_results)):
            new_search_results[i]['idx'] = i + 1

        return new_search_results[:self.search_limit]

    # ===================== 类接口 ========================
    # search snippets by using google search through serpapi
    def search(self, query, filter_year=None):
        use_rollout = self.roll_out > 0 and self.model is not None

        header = self._check_history(query)

        if self.reflection:
            analysis, query = self.reflector.query_reflect(query)

        if use_rollout:
            queries = self._expand_query(query)
            search_results = {}
            for q in queries:
                try:
                    snippets = self._search(q, filter_year)
                    assert len(snippets) > 0
                    search_results[q] = snippets
                except:
                    pass

            if len(search_results)==0:
                error_messages = f"Search for query '{query}' failed! Search query should be less specific\n"
                return error_messages

            # aggregate and rerank
            web_snippets = self.aggregate(query, search_results, intersect=False, rerank=self.rerank)

        else:
            # no rollout, perform general search
            web_snippets = self._search(query, filter_year)
            if type(web_snippets) == str:
                return web_snippets

        # to final contents
        content = self._to_contents(query, web_snippets)

        return header + content

    # crawl whole pages on the website, return completed contents
    def crawl_page(self, url):
        header = self._check_history(url)
        pages = asyncio.run(self._crawl_page(url=url))
        return header + pages

    # crawl page and retrieve, return contents filtered by query-similarity
    def crawl_page_with_rag(self, url, query):
        header = self._check_history(url)
        contents = self.crawl_page(url=url)
        return header + self._query(query, contents)
    # read page through jina api
    def read_page(self, url):
        def jina_read(url):
            jina_url = f'https://r.jina.ai/{url}'
            headers = {
                'Authorization': f'Bearer {os.getenv("JINA_API_KEY")}',
                'X-Engine': 'browser',
                'X-Return-Format': 'text',
                'X-Timeout': '10',
                'X-Token-Budget': '80000'
            }

            response = requests.get(jina_url, headers=headers)
            return response.text
        return jina_read(url)


class CrawlerSearchTool(Tool):
    name = "web_search"
    description = "Perform a web search query (think a google search) and returns the search results."
    inputs = {"query": {"type": "string", "description": "The web search query to perform."}}
    inputs["filter_year"] = {
        "type": "string",
        "description": "[Optional parameter]: filter the search results to only include pages from a specific year. For example, '2020' will only include pages from 2020. Make sure to use this parameter if you're trying to search for articles from a specific date!",
        "nullable": True,
    }
    output_type = "string"

    def __init__(self,
                 crawler: SimpleCrawler,
                 rollout: int = 0,
                 search_limit: int = 10,
                 serp_num: int = 10,
                 rerank: bool = False,
                 topk: int = 1):
        super().__init__()
        # define a crawler
        self.crawler = crawler
        # reinitialize crawler's configs
        self.crawler.serp_num = serp_num
        self.crawler.roll_out = rollout
        self.crawler.search_limit = search_limit
        self.crawler.rerank = rerank
        self.crawler.topk = topk

    def forward(self, query: str, filter_year: Optional[int] = None) -> str:
        '''
            @serpapi -> 根据query返回top-k个相关搜索结果
        '''
        return self.crawler.search(query, filter_year)


class CrawlerArchiveSearchTool(Tool):
    name = "find_archived_url"
    description = "Given a url, searches the Wayback Machine and returns the archived version of the url that's closest in time to the desired date."
    inputs = {
        "url": {"type": "string", "description": "The url you need the archive for."},
        "date": {
            "type": "string",
            "description": "The date that you want to find the archive for. Give this date in the format 'YYYYMMDD', for instance '27 June 2008' is written as '20080627'.",
        },
    }
    output_type = "string"

    def __init__(self, crawler: SimpleCrawler):
        super().__init__()
        self.crawler = crawler

    def forward(self, url, date) -> str:
        no_timestamp_url = f"https://archive.org/wayback/available?url={url}"
        archive_url = no_timestamp_url + f"&timestamp={date}"
        response = requests.get(archive_url).json()
        response_notimestamp = requests.get(no_timestamp_url).json()
        if "archived_snapshots" in response and "closest" in response["archived_snapshots"]:
            closest = response["archived_snapshots"]["closest"]
            print("Archive found!", closest)

        elif "archived_snapshots" in response_notimestamp and "closest" in response_notimestamp["archived_snapshots"]:
            closest = response_notimestamp["archived_snapshots"]["closest"]
            print("Archive found!", closest)
        else:
            # raise Exception(f"Your {url} was not archived on Wayback Machine, try a different url.")
            return "Your {url} was not archived on Wayback Machine, try a different url."
        target_url = closest["url"]

        content = self.crawler.crawl_page(target_url)
        return (
                f"Web archive for url {url}, snapshot taken at date {closest['timestamp'][:8]}:\n"
                + content
        )


class CrawlerReadTool(Tool):
    name = "crawl_pages"
    description = "Access a webpage using the provided URL and return completed contents of the webpage. In the case of a YouTube video URL, extract and return the video transcript."
    inputs = {
        "url": {
            "type": "string",
            "description": "The relative or absolute url of the webpage to visit."
        },
    }
    output_type = "string"

    def __init__(self, crawler: SimpleCrawler, read_type: str = "jina_read"):
        super().__init__()
        self.crawler = crawler
        self.read_type = read_type

    def forward(self, url) -> str:
        # 考虑arxiv pdf的路径情况, 或许增加一个判断条件, 如果url=https://xxx.pdf, 则建议调用inspect工具, 增加文件解析确定性
        if self.read_type == "crawl":
            result = self.crawler.crawl_page(url)
        else:
            result = self.crawler.read_page(url)
        if result == '\n':
            return f"Crawling for url: {url} return None, maybe it is a url for .pdf file which is unable to crawl. " \
                   "Please try to use tool: inspect_file_as_text() to get the contents."

        if  'BalanceError' in result:
            raise ValueError(f"Failed to read content from {url}: {result}")

        return result


# ToDo: 集成到一个readertool, 用参数控制
class CrawlerRAGTool(Tool):
    name = "crawl_pages_with_retrieve"
    description = "Access a webpage using the provided URL and retrieve its text content. In the case of a YouTube video URL, extract and return the video transcript."
    inputs = {
        "url": {
            "type": "string",
            "description": "The relative or absolute url of the webpage to visit."
        },
        "query": {
            "type": "string",
            "description": "the search query for contents relative to your task",
        },
    }
    output_type = "string"

    def __init__(self, crawler: SimpleCrawler):
        super().__init__()
        self.crawler = crawler

    def forward(self, url, query) -> str:
        # --------------- 默认为无db版rag，直接返回结果
        return self.crawler.crawl_page_with_rag(url, query)
    
