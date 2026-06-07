

import json
from typing import List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import time
from langchain_core.callbacks import BaseCallbackHandler
from rag_core.llm import get_llm
from rag_core.qdrant import get_vectorstore_retriever
from rag_core.reranker import get_reranker
from langchain_core.outputs import LLMResult


query_rewrite_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an expert query planner and query rewriter for a RAG system.

The user may ask questions whose answer is not explicitly stated in one passage. Your task is to generate diverse search queries that retrieve the evidence needed to answer the user's question accurately.

Use the user's original question and the optional context below:

Additional context:
{pre_context}

Generate several clearly different search queries that cover the following retrieval strategies:

1. Direct query  
   Closely rephrase the original question while preserving its exact intent.

2. Keyword query  
   Create a short keyword-based query using the most important terms.

3. Concept queries  
   Search for key concepts, related ideas, synonyms, or implied themes from the question.

4. Mechanism queries  
   Search for causes, reasons, effects, consequences, mechanisms, or relationships relevant to the question.

5. Technical or domain-specific queries  
   Use terminology from the relevant domain, especially when applicable: electricity market simulation, agent-based modeling, reinforcement learning, market design, policy support, validation, interpretability, transparency, or explainability.

6. Alternative-perspective queries  
   Reformulate the question from a different but still relevant perspective, such as evaluation, policy relevance, model behavior, stakeholder use, or decision support.

7. Negative or risk queries  
   If useful, generate queries that retrieve risks, limitations, or failure modes related to the question, such as market power, price manipulation, black-box behavior, invalid assumptions, lack of validation, biased outcomes, or poor interpretability.

8. Abbreviation/entity queries  
   If useful, include abbreviations, key entities, technical terms, or domain-specific shorthand that may appear in documents.

Rules:
- Preserve the original intent exactly.
- Do not invent an answer.
- Do not invent facts, names, numbers, entities, or assumptions.
- Do not create irrelevant or overly broad queries.
- Use the additional context only to improve retrieval, not to change the question.
- Queries may use related concepts, synonyms, and domain terminology.
- Make sure the queries are clearly different from each other.
- Return only a valid JSON list of strings.
- Do not include explanations, Markdown, comments, or text outside the JSON list.
"""
    ),
    (
        "human",
        """Question:
{question}

Number of variants:
{num_queries}"""
    )
])

def query_rewriting(question: str, max_queries = 5) -> List[str]:

    llm_milliseconds_taken = 0.0
    llm_model_input_tokens_used = 0
    llm_model_output_tokens_used = 0
    
    class MetricsCallbackHandler(BaseCallbackHandler):
        def on_llm_start(self, serialized, prompts, **kwargs):
            self._llm_start_time = time.time()

        def on_llm_end(self, response: LLMResult, **kwargs):
            nonlocal llm_milliseconds_taken, llm_model_input_tokens_used, llm_model_output_tokens_used
            llm_milliseconds_taken += (time.time() - self._llm_start_time) * 1000
            usage = (response.llm_output or {}).get("token_usage", {})
            llm_model_input_tokens_used += usage.get("prompt_tokens", 0)
            llm_model_output_tokens_used += usage.get("completion_tokens", 0)


    llm = get_llm(MetricsCallbackHandler())

    query_rewriter = (
        query_rewrite_prompt
        | llm
        | StrOutputParser()
    )
    
    reranker = get_reranker()
    
    
    retriever = get_vectorstore_retriever()
    basic_raw = retriever.invoke(question)
    basic_info = reranker.compress_documents(basic_raw, question)
    
    raw_queries = query_rewriter.invoke({"question": question, "num_queries": max_queries, "pre_context": basic_info})
    queries = parse_queries(raw_queries, question, max_queries)
    return queries



def parse_queries(raw: str, original_question: str, max_queries: int) -> list[str]:
    """
    Parsed die LLM-Ausgabe robust und fügt die Originalfrage immer hinzu.
    """
    try:
        queries = json.loads(raw)
        if not isinstance(queries, list):
            queries = []
    except Exception:
        queries = []

    queries = [q.strip() for q in queries if isinstance(q, str) and q.strip()]

    # Originalfrage immer mit aufnehmen
    queries = [original_question] + queries

    # Deduplizieren, Reihenfolge erhalten
    seen = set()
    unique_queries = []
    for q in queries:
        key = q.lower()
        if key not in seen:
            seen.add(key)
            unique_queries.append(q)

    return unique_queries[:max_queries]
