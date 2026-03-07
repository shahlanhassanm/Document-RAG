"""
Query Transformation: HyDE + Query Rewriter.
Pre-retrieval strategies to bridge the semantic gap between user queries and document chunks.
"""

from core.llm import get_llm
from core.embeddings import get_embeddings
from core.logger import get_logger

log = get_logger("query_transform")

# HyDE prompt — generates a hypothetical answer to use as the search vector
HYDE_PROMPT = """You are a document expert. Given the following question, write a short, 
factual paragraph that would likely appear in a document answering this question. 
Write ONLY the answer paragraph, nothing else. Do not include any preamble or explanation.

Question: {question}

Hypothetical answer:"""

# Query rewriter prompt — extracts core search terms
REWRITE_PROMPT = """You are a search query optimizer. Given the following user question, 
extract and rewrite it as a concise, keyword-rich search query that would match 
relevant document passages. Output ONLY the rewritten query, nothing else.

User question: {question}

Optimized search query:"""


def hyde_transform(question: str, model_name: str = "qwen3:8b") -> str:
    """
    Hypothetical Document Embeddings (HyDE).
    
    Generates a hypothetical answer using the LLM, which is then used
    as the search query instead of the original question. The hypothetical
    answer uses vocabulary and patterns closer to the actual document chunks,
    improving vector similarity matching.
    
    Args:
        question: Original user question
    
    Returns:
        Hypothetical answer text to be used for embedding search
    """
    log.info(f"HyDE transform for: '{question[:80]}...'")
    
    try:
        llm = get_llm(temperature=0.3, model_name=model_name)
        prompt = HYDE_PROMPT.format(question=question)
        hypothetical_answer = llm.invoke(prompt)
        
        # Clean up — remove thinking tags if qwen3 includes them
        if "</think>" in hypothetical_answer:
            hypothetical_answer = hypothetical_answer.split("</think>")[-1].strip()
        
        log.info(f"HyDE generated: '{hypothetical_answer[:100]}...'")
        return hypothetical_answer
    except Exception as e:
        log.warning(f"HyDE transform failed: {e}. Using original query.")
        return question


def rewrite_query(question: str, model_name: str = "qwen3:8b") -> str:
    """
    Rewrite a verbose or vague query into a keyword-rich search query.
    
    Args:
        question: Original user question
    
    Returns:
        Optimized search query
    """
    log.info(f"Rewriting query: '{question[:80]}...'")
    
    try:
        llm = get_llm(temperature=0.1, model_name=model_name)
        prompt = REWRITE_PROMPT.format(question=question)
        rewritten = llm.invoke(prompt)
        
        # Clean up thinking tags
        if "</think>" in rewritten:
            rewritten = rewritten.split("</think>")[-1].strip()
        
        log.info(f"Query rewritten to: '{rewritten[:100]}...'")
        return rewritten
    except Exception as e:
        log.warning(f"Query rewrite failed: {e}. Using original query.")
        return question
