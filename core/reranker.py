"""
FlashRank Cross-Encoder Reranker + LongContextReorder.
Post-retrieval refinement for precision and positional bias mitigation.
"""

from langchain_core.documents import Document
from core.logger import get_logger

log = get_logger("reranker")


def rerank(query: str, documents: list[Document], top_k: int = 6) -> list[Document]:
    """
    Rerank documents using FlashRank cross-encoder.
    
    Cross-encoders process (query, doc) pairs together, capturing interaction
    between query terms and document content — much more precise than bi-encoders.
    
    Args:
        query: User's search query
        documents: Candidate documents from hybrid search
        top_k: Number of top results to return
    
    Returns:
        Top-k documents reranked by cross-attention score
    """
    if not documents:
        return []
    
    log.info(f"Reranking {len(documents)} candidates with FlashRank")
    
    try:
        from flashrank import Ranker, RerankRequest
        
        ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="/tmp/flashrank_cache")
        
        # Prepare passages for FlashRank
        passages = [
            {"id": i, "text": doc.page_content, "meta": doc.metadata}
            for i, doc in enumerate(documents)
        ]
        
        rerank_request = RerankRequest(query=query, passages=passages)
        results = ranker.rerank(rerank_request)
        
        # Rebuild Document objects in reranked order
        reranked_docs = []
        for result in results[:top_k]:
            idx = result["id"]
            reranked_docs.append(documents[idx])
        
        log.info(f"Reranking complete: {len(documents)} -> top {len(reranked_docs)}")
        return reranked_docs
        
    except Exception as e:
        log.warning(f"FlashRank reranking failed: {e}. Falling back to original order.")
        return documents[:top_k]


def long_context_reorder(documents: list[Document]) -> list[Document]:
    """
    Reorder documents to mitigate 'lost-in-the-middle' positional bias.
    
    Places the most relevant documents at the BEGINNING and END of the list,
    with less relevant ones in the middle. LLMs pay more attention to content
    at the edges of their context window.
    
    Input should already be ranked by relevance (best first).
    """
    if len(documents) <= 2:
        return documents
    
    log.debug(f"Applying LongContextReorder to {len(documents)} documents")
    
    # Split into two groups: odd-indexed go to middle, even stay at edges
    reordered = []
    
    # Even indices (0,2,4...) — higher relevance, placed at edges
    edge_docs = [documents[i] for i in range(0, len(documents), 2)]
    # Odd indices (1,3,5...) — lower relevance, placed in middle
    middle_docs = [documents[i] for i in range(1, len(documents), 2)]
    
    # Build: best at start, worst in middle, second-best at end
    reordered = edge_docs[:len(edge_docs)//2] + middle_docs + edge_docs[len(edge_docs)//2:]
    
    log.debug("LongContextReorder applied")
    return reordered
