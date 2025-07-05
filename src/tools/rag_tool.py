import requests
from oagents import Tool

class RAGTool(Tool):
    name = "RAGTool"
    description = """
    A retrieval-augmented tool that searches for documents relevant to a natural language query.
    It supports dynamic control over how many documents to retrieve via the 'topk' parameter.
    It will return the 'topk' most relevant retrieved documents and their similarity scores.
    """
    inputs = {
        "query": {
            "type": "string",
            "description": "The natural language query to search the knowledge base for.",
        },
        "topk": {
            "type": "integer",
            "description": "The number of top relevant documents to retrieve (e.g., 5). Default is 5 if not specified.",
            "nullable": True
        },
    }
    output_type = "string"

    def forward(self, query: str, topk: int = 5) -> str:
        try:
            print("forward topk:",topk)
            response = requests.post(
                "http://localhost:8000/retrieve",
                json={"query": query, "topk": topk},
                timeout=30
            )
            data = response.json()
            if not data.get('results'):
                return f"No results found. Server says: {data.get('message')}"

            formatted = [
                f"[Doc ID: {doc['doc_id']}]\nContent: \"{doc['contents']}\"\nSimilarity Score: {doc['score']}"
                for doc in data["results"]
            ]
            
            return (
                f"[RAGTool Results | query: \"{query}\" | topk={topk}]\n\n" +
                "\n\n".join(formatted)
            )
        except Exception as e:
            return (
                "[RAGTool Error] An exception occurred during retrieval.\n"
                f"Error Message: {str(e)}\n"
            )

if __name__ == "__main__":
    # Example usage
    rag_tool = RAGTool()
    query = "What is the capital of France?"
    topk = 5
    result = rag_tool.forward(query=query, topk=topk)
    print(result)

# from oagents import Tool
# from flashrag.config import Config
# from flashrag.utils import get_retriever

# class RAGTool(Tool):
#     name = "RAGTool"
#     description = """
#     A retrieval-augmented tool that searches for documents relevant to a natural language query using an FAISS-based vector index.
#     It supports dynamic control over how many documents to retrieve via the 'topk' parameter.
#     """

#     inputs = {
#         "query": {
#             "type": "string",
#             "description": "The natural language query to search the knowledge base for.",
#         },
#         "topk": {
#             "type": "integer",
#             "description": "The number of top relevant documents to retrieve (e.g., 5). Default is 5 if not specified.",
#             "nullable": True
#         },
#     }

#     output_type = "string"

#     def forward(self, query: str, topk: int = 5) -> str:
#         try:
#             config = Config("/mnt/public/data/lh/yqj/TaskCraft/retriever_config.yaml")
#             config["retrieval_topk"] = topk
#             retriever = get_retriever(config)

#             retrieval_results, doc_scores = retriever.batch_search(
#                 [query], return_score=True
#             )

#             if not retrieval_results or not retrieval_results[0]:
#                 return (
#                     f"No relevant documents found for query: '{query}'. "
#                     "Consider rephrasing or using a broader question."
#                 )

#             results = []
#             for doc, score in zip(retrieval_results[0], doc_scores[0]):
#                 doc_id = doc.get("id", "N/A")
#                 content = doc.get("contents", "").strip()
#                 results.append(
#                     f"[Doc ID: {doc_id}]\nContent: \"{content}\"\nSimilarity Score: {score:.4f}"
#                 )

#             return (
#                 f"[RAGTool Results | query: \"{query}\" | topk={topk}]\n\n"
#                 + "\n\n".join(results)
#             )

#         except Exception as e:
#             return (
#                 "[RAGTool Error] An exception occurred during retrieval.\n"
#                 f"Error Message: {str(e)}\n"
#             )
