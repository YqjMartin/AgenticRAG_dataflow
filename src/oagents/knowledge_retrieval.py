import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# import torch
# from sentence_transformers import SentenceTransformer

@dataclass
class WorkflowStep:
    step_id: int
    query: str 
    tool_call: Optional[Dict[str, Any]] = None 
    output: Optional[str] = None  


@dataclass
class WorkflowInstance:
    """A complete workflow instance"""
    workflow_id: str
    question: str
    plan: str
    steps: List[WorkflowStep] = field(default_factory=list)
    true_answer: Optional[str] = None
    predicted_answer: Optional[str] = None
    resolved: bool = False
    additional_metadata: Dict[str, Any] = field(default_factory=dict)
    # For embedding-based search
    # question_embedding: Optional[np.ndarray] = None


class AgenticKnowledgeBase:
    """
    Agentic Knowledge Base (AKB) class for managing agentic workflow knowledge.
    """

    def __init__(self, json_file_paths=None):
        # Core storage
        self.workflows: Dict[str, WorkflowInstance] = {}  # ID to instance
        
        # Initialize embedding model
        # self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = None
        self.workflow_ids = []
        
        self.step_tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        self.step_tfidf_matrix = None
        self.steps_info = []

        if json_file_paths:
            for json_path in json_file_paths:
                if not os.path.exists(json_path):
                    raise FileNotFoundError(f'[ERROR] JSON file: {json_path} does not exist.')
                print(f'[INFO] Parsing workflow instances from {json_path}')
                self.parse_json_file(json_path)
            
            self.build_search_indices()

    def parse_json_file(self, json_file_path):
        """Parse JSON file containing workflow instances"""
        try:
            with open(json_file_path, 'r') as f:
                data = json.load(f)
                
            if isinstance(data, list):
                samples = data
            else:
                samples = [data]
                
            for sample in samples:
                workflow_instance = self.parse_workflow_sample(sample)
                if workflow_instance:
                    self.add_workflow_instance(workflow_instance)
                    
        except json.JSONDecodeError:
            with open(json_file_path, 'r') as f:
                for line in f:
                    if line.strip():
                        try:
                            sample = json.loads(line)
                            workflow_instance = self.parse_workflow_sample(sample)
                            if workflow_instance:
                                self.add_workflow_instance(workflow_instance)
                        except json.JSONDecodeError as e:
                            print(f"[ERROR] Failed to parse line: {e}")
                            continue
        
        print(f'[INFO] Successfully parsed {len(self.workflows)} workflow instances')

    def parse_workflow_sample(self, sample):
        try:
            task_id = sample.get('task_id', str(len(self.workflows)))
            question = sample.get('question', '')
            augmented_question = sample.get('augmented_question', '')
            prediction = sample.get('prediction', '')
            true_answer = sample.get('true_answer', '')
            
            intermediate_steps = sample.get('intermediate_steps', [])
            
            plan = ""
            for step in intermediate_steps:
                if step.get('step_type') == 'planning' and 'plan' in step:
                    plan = step.get('plan', '')
                    break
            
            workflow_steps = []
            for step in intermediate_steps:
                if step.get('step_type') == 'action':
                    query = step.get('model_output', '')
                    
                    tool_call = None
                    if 'tool_calls' in step:
                        tool_call = step.get('tool_calls')
                    
                    output = step.get('observations', '')
                    if not output and 'action_output' in step:
                        output = step.get('action_output', '')
                    
                    workflow_step = WorkflowStep(
                        step_id=len(workflow_steps),
                        query=query,
                        tool_call=tool_call,
                        output=output
                    )
                    workflow_steps.append(workflow_step)
            
            workflow = WorkflowInstance(
                workflow_id=task_id,
                question=question or augmented_question,
                plan=plan,
                steps=workflow_steps,
                true_answer=true_answer,
                predicted_answer=prediction,
                resolved=prediction == true_answer,
                additional_metadata={
                    "augmented_question": augmented_question,
                    "parsing_error": sample.get('parsing_error', False),
                    "iteration_limit_exceeded": sample.get('iteration_limit_exceeded', False),
                    "agent_error": sample.get('agent_error', None)
                }
            )
            
            return workflow
            
        except Exception as e:
            print(f'[ERROR] Failed to parse workflow sample: {e}')
            return None

    def add_workflow_instance(self, workflow: WorkflowInstance):
        self.workflows[workflow.workflow_id] = workflow
        return workflow
    
    # def build_search_indices(self):
    #     questions = []
    #     self.workflow_ids = []
    #     for wf_id, workflow in self.workflows.items():
    #         questions.append(workflow.question)
    #         self.workflow_ids.append(wf_id)
        
    #     if questions:
    #         self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(questions)
        

    def build_search_indices(self):
        questions = []
        self.workflow_ids = []
        for wf_id, workflow in self.workflows.items():
            questions.append(workflow.question)
            self.workflow_ids.append(wf_id)
        if questions:
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(questions)
        # Generate embeddings for semantic search
        # with torch.no_grad():
        #     for wf_id, workflow in self.workflows.items():
        #         workflow.question_embedding = self.embedding_model.encode(workflow.question)

        step_queries = []
        self.steps_info = []
        for wf_id, workflow in self.workflows.items():
            for step_idx, step in enumerate(workflow.steps):
                step_queries.append(step.query)
                self.steps_info.append((wf_id, step_idx))
        if step_queries:
            self.step_tfidf_matrix = self.step_tfidf_vectorizer.fit_transform(step_queries)

    def get_workflow_step(self, workflow_id: str, step_id: int) -> Optional[WorkflowStep]:
        workflow = self.workflows.get(workflow_id)
        if workflow and 0 <= step_id < len(workflow.steps):
            return workflow.steps[step_id]
        return None

    def search_steps_by_query(self, query: str, top_k: int = 1) -> List[Tuple[str, int, float]]:
        if self.step_tfidf_matrix is None:
            print("[WARN] Step TF-IDF matrix is not built yet. No search results.")
            return []
        
        query_vec = self.step_tfidf_vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.step_tfidf_matrix).flatten()
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        return [
            (self.steps_info[idx][0], self.steps_info[idx][1], float(similarities[idx]))
            for idx in top_indices
        ]

    
    def search_text_similarity(self, query: str, top_k: int = 1) -> List[Tuple[str, float]]:
        if not self.tfidf_matrix is not None:
            print("[WARN] Workflow TF-IDF matrix is not built yet. No search results.")
            return []
        query_vec = self.tfidf_vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            wf_id = self.workflow_ids[idx]
            results.append((wf_id, float(similarities[idx])))
        
        return results
    
    # def search_semantic(self, query: str, top_k: int = 1) -> List[Tuple[str, float]]:
    #     if not self.workflows:
    #         print("[WARN] No workflows in the knowledge base. No search results.")
    #         return []
    #     with torch.no_grad():
    #         query_embedding = self.embedding_model.encode(query)
        
    #     similarities = []
    #     for wf_id, workflow in self.workflows.items():
    #         if workflow.question_embedding is not None:
    #             sim_score = cosine_similarity(
    #                 [query_embedding], 
    #                 [workflow.question_embedding]
    #             )[0][0]
    #             similarities.append((wf_id, float(sim_score)))
        
    #     similarities.sort(key=lambda x: x[1], reverse=True)
    #     return similarities[:top_k]

    def get_workflow(self, workflow_id: str) -> Optional[WorkflowInstance]:
        return self.workflows.get(workflow_id)
    
    def get_all_workflows(self) -> Dict[str, WorkflowInstance]:
        return self.workflows


class AKB_Manager:

    def __init__(self, json_file_paths=None):
        self.knowledge_base = AgenticKnowledgeBase(json_file_paths=json_file_paths)

    def search_by_text_similarity(self, query: str, top_k: int = 1):
        return self.knowledge_base.search_text_similarity(query, top_k)
    
    def search_by_semantic_similarity(self, query: str, top_k: int = 1):
        return self.knowledge_base.search_semantic(query, top_k)
    
    def get_workflow_details(self, workflow_id: str):
        return self.knowledge_base.get_workflow(workflow_id)
    
    def get_all_workflows(self):
        return self.knowledge_base.get_all_workflows()
    
    def search_steps_by_query(self, query: str, top_k: int = 1):
        return self.knowledge_base.search_steps_by_query(query, top_k)

    def get_workflow_step_details(self, workflow_id: str, step_id: int):
        return self.knowledge_base.get_workflow_step(workflow_id, step_id)

# Run the tests
if __name__ == "__main__":
    akb_manager = AKB_Manager(json_file_paths=['./knowledge_base/memory_base.json'])
    query = "What is the oldest movie in a rental store?"
    
    print(f"Testing query: '{query}'")
    print("\nText similarity results:")

    results = akb_manager.search_by_text_similarity(query)
    for wf_id, score in results:
        workflow = akb_manager.get_workflow_details(wf_id)
        print(f"- ID: {wf_id}")
        print(f"- Score: {score:.4f}")
        print(f"- Question: {workflow.question}")
        print(f"- Plan: {workflow.plan}")

    results_step = akb_manager.search_steps_by_query(query, top_k=1)
    for wf_id, step_idx, score in results_step:
        step = akb_manager.get_workflow_step_details(wf_id, step_idx)
        workflow = akb_manager.knowledge_base.get_workflow(wf_id)
        
        print(f"- Match Score: {score:.4f}")
        print(f"- Workflow ID: {wf_id}")
        print(f"- Original Question: {workflow.question}")
        print(f"- Step: {step_idx}:")
        print(f"- Query: {step.query}")
        if step.tool_call:
            print(f"- Tool Call: {json.dumps(step.tool_call, indent=2)}")
        if step.output:
            print(f"- Output: {step.output[:200]}")