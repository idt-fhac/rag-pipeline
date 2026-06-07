from typing import List

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from rag_core.llm import get_llm
from rag_core.mongodb import get_mongo_collection
from rag_core.qdrant import get_vectorstore_retriever
from langchain_core.prompts import ChatPromptTemplate
import time
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_core.documents import Document
from rag_core.queryRewrite import query_rewriting
from rag_core.reranker import get_reranker, reciprocal_rank_fusion
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def retrieve_multi_query_with_parents(question: str):
    multi_questions = query_rewriting(question) 
    print(multi_questions)
    scored_chunks = retrieve_multi_query(multi_questions)  # list of (Document, rrf_score)
    return fetch_top_parent_from_mongo(scored_chunks)

def build_pipeline():
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
    multi_query_parent_retriever = RunnableLambda(retrieve_multi_query_with_parents)

    prompt = ChatPromptTemplate.from_template("""You are an expert assistant for the ASSUME energy simulation framework.
        Answer the question using ONLY the information provided in the context below.
        If the context does not contain enough information to answer, say "I don't have enough information to answer this question."
        Do not make up or infer facts beyond what is explicitly stated in the context.

        Context:
        {context}

        Question: {question}

        Answer concisely and precisely:""")
    
    
    _answer_chain = (
        prompt
        | llm
        | StrOutputParser()
    )
    
    chain_with_sources = (
        RunnableParallel({"context": multi_query_parent_retriever, "question": RunnablePassthrough()})
        .assign(answer=_answer_chain)
    )
    
    return chain_with_sources

def retrieve_multi_query(queries: List[str]):
    """
    Query-Rewriting + Multi-Query Retrieval mit RRF-Fusion.

    Für jede Query-Variante werden Dokumente über den Hybrid-Retriever
    (dense + sparse) + Cross-Encoder abgerufen.  Die pro-Query-Ergebnislisten
    werden anschließend via Reciprocal Rank Fusion (RRF) zusammengeführt:
    Dokumente, die in mehreren Query-Varianten hoch gerankt werden,
    steigen im kombinierten Ranking auf.
    """

    # Collect one result list per query variant
    all_doc_lists: list[list] = []
    
    retriever = get_vectorstore_retriever()
    reranker = get_reranker(top_n=10)

    for q in queries:
        docs_raw = retriever.invoke(q)  # holt 25 Dokumente pro Query-Variante
        docs = reranker.compress_documents(docs_raw, q)  # reranked, best first
        all_doc_lists.append(docs)

    # RRF across query variants: documents appearing in multiple lists
    # and at high ranks get a boosted combined score
    # Returns (doc, rrf_score) tuples - scores are preserved for downstream parent selection
    return reciprocal_rank_fusion(all_doc_lists, k=40)

def fetch_top_parent_from_mongo(scored_docs):
    """
    scored_docs: list of (Document, rrf_score) from reciprocal_rank_fusion.
    Every chunk carries parent_id in its metadata (set at index time).
    Aggregates RRF scores per parent and returns the top-10 parent documents.
    """

    parent_id_scores: dict[str, float] = {}

    for doc, rrf_score in scored_docs:
        parent_id = doc.metadata.get("parent_id")
        if parent_id:
            parent_id_scores[parent_id] = parent_id_scores.get(parent_id, 0.0) + rrf_score

    top_parent_ids = sorted(parent_id_scores, key=parent_id_scores.__getitem__, reverse=True)[:10]
    print(f"Top parent IDs: {top_parent_ids}")
    
    mongo_collection = get_mongo_collection()

    parent_docs = []
    for pid in top_parent_ids:
        t0 = time.time()
        parent = mongo_collection.find_one(
            {"element_id": pid},
            {"page_content": 1, "element_id": 1, "source": 1, "_id": 0},
        )

        if parent:
            parent_docs.append(
                Document(
                    page_content=parent["page_content"],
                    metadata={
                        "element_id": parent["element_id"],
                        "retrieval": "parent",
                        "source": parent.get("source", "unknown"),
                    },
                )
            )

    return parent_docs

