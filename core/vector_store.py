"""
Vector Store with Hybrid Retrieval.
ChromaDB HNSW (dense) + BM25 (sparse) fused via Reciprocal Rank Fusion.
"""

import os
import shutil
import gc
import pickle
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from core.embeddings import get_embeddings
from core.logger import get_logger

log = get_logger("vector_store")

DB_DIR = os.path.join(os.path.dirname(__file__), '..', 'chroma_db')
BM25_PATH = os.path.join(os.path.dirname(__file__), '..', 'bm25_index.pkl')

# Chunking config
CHUNK_SIZE = 800
CHUNK_OVERLAP = 200
SEPARATORS = ["\n\n", "\n", ". ", " "]


def _get_splitter() -> RecursiveCharacterTextSplitter:
    """Returns the configured text splitter."""
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=SEPARATORS,
        length_function=len,
    )


def build_vector_store(documents: list[Document]) -> tuple[Chroma, int]:
    """
    Chunks documents, builds ChromaDB vector store and BM25 index.
    
    Args:
        documents: List of Document objects from document_processor
    
    Returns:
        (Chroma vector store, number of chunks created)
    """
    log.info(f"Building vector store from {len(documents)} document segments")
    
    # Clear existing stores
    clear_store()
    
    # Chunk the documents
    splitter = _get_splitter()
    chunks = splitter.split_documents(documents)
    log.info(f"Created {len(chunks)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    
    if not chunks:
        log.warning("No chunks created — documents may be empty")
        return None, 0
    
    # Build ChromaDB dense index
    embeddings = get_embeddings()
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_DIR,
    )
    log.info(f"ChromaDB HNSW index built with {len(chunks)} vectors at {DB_DIR}")
    
    # Build BM25 sparse index
    tokenized_chunks = [chunk.page_content.lower().split() for chunk in chunks]
    bm25_index = BM25Okapi(tokenized_chunks)
    
    # Save BM25 index + chunk data for retrieval
    bm25_data = {
        "index": bm25_index,
        "chunks": chunks,
        "tokenized": tokenized_chunks,
    }
    with open(BM25_PATH, 'wb') as f:
        pickle.dump(bm25_data, f)
    log.info(f"BM25 sparse index built and saved to {BM25_PATH}")
    
    return vector_store, len(chunks)


def _load_chroma() -> Chroma:
    """Load existing ChromaDB store."""
    if not os.path.exists(DB_DIR):
        raise ValueError("Vector store not found. Upload documents first.")
    embeddings = get_embeddings()
    return Chroma(persist_directory=DB_DIR, embedding_function=embeddings)


def _load_bm25() -> tuple[BM25Okapi, list[Document]]:
    """Load existing BM25 index."""
    if not os.path.exists(BM25_PATH):
        raise ValueError("BM25 index not found. Upload documents first.")
    with open(BM25_PATH, 'rb') as f:
        data = pickle.load(f)
    return data["index"], data["chunks"]


def dense_mmr_search(query: str, k: int = 6, fetch_k: int = 20) -> list[Document]:
    """Dense MMR search via ChromaDB."""
    log.debug(f"Dense MMR search: k={k}, fetch_k={fetch_k}")
    store = _load_chroma()
    results = store.max_marginal_relevance_search(query, k=k, fetch_k=fetch_k)
    log.debug(f"Dense search returned {len(results)} results")
    return results


def bm25_search(query: str, k: int = 6) -> list[Document]:
    """Sparse BM25 keyword search."""
    log.debug(f"BM25 search: k={k}")
    bm25_index, chunks = _load_bm25()
    tokenized_query = query.lower().split()
    scores = bm25_index.get_scores(tokenized_query)
    
    # Get top-k indices
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    results = [chunks[i] for i in top_indices if scores[i] > 0]
    log.debug(f"BM25 search returned {len(results)} results")
    return results


def hybrid_search_rrf(query: str, k: int = 6, fetch_k: int = 20, rrf_k: int = 60) -> list[Document]:
    """
    Hybrid search: Dense MMR + BM25 fused via Reciprocal Rank Fusion (RRF).
    
    RRF score = sum(1 / (rrf_k + rank)) across all result lists.
    """
    log.info(f"Hybrid RRF search: query='{query[:80]}...'")
    
    # Run both searches
    dense_results = dense_mmr_search(query, k=k, fetch_k=fetch_k)
    sparse_results = bm25_search(query, k=k)
    
    # RRF fusion
    doc_scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}
    
    for rank, doc in enumerate(dense_results):
        doc_id = doc.page_content[:200]  # Use content prefix as ID
        doc_scores[doc_id] = doc_scores.get(doc_id, 0) + 1.0 / (rrf_k + rank + 1)
        doc_map[doc_id] = doc
    
    for rank, doc in enumerate(sparse_results):
        doc_id = doc.page_content[:200]
        doc_scores[doc_id] = doc_scores.get(doc_id, 0) + 1.0 / (rrf_k + rank + 1)
        doc_map[doc_id] = doc
    
    # Sort by fused score
    sorted_ids = sorted(doc_scores.keys(), key=lambda x: doc_scores[x], reverse=True)
    fused_results = [doc_map[doc_id] for doc_id in sorted_ids[:k * 2]]  # Return more for reranking
    
    log.info(f"RRF fusion: {len(dense_results)} dense + {len(sparse_results)} sparse -> {len(fused_results)} fused")
    return fused_results


def clear_store():
    """Remove both ChromaDB and BM25 index."""
    gc.collect()
    
    if os.path.exists(DB_DIR):
        def on_rm_error(func, path, exc_info):
            import stat
            os.chmod(path, stat.S_IWRITE)
            try:
                func(path)
            except Exception:
                pass
        shutil.rmtree(DB_DIR, onerror=on_rm_error)
        log.info("ChromaDB store cleared")
    
    if os.path.exists(BM25_PATH):
        os.remove(BM25_PATH)
        log.info("BM25 index cleared")


def store_exists() -> bool:
    """Check if the vector store has been built."""
    return os.path.exists(DB_DIR) and os.path.exists(BM25_PATH)
