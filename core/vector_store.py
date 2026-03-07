"""
Vector Store with Hybrid Retrieval — Milvus Standalone Backend.
Dense HNSW (Milvus) + BM25 (sparse) fused via Reciprocal Rank Fusion.
"""

import os
import gc
import pickle
from langchain_milvus import Milvus
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from pymilvus import connections, utility
from pymilvus.exceptions import MilvusException
from core.embeddings import get_embeddings
from core.logger import get_logger

log = get_logger("vector_store")

# Milvus connection config
MILVUS_HOST = "127.0.0.1"
MILVUS_PORT = "19530"
MILVUS_URI = f"http://{MILVUS_HOST}:{MILVUS_PORT}"
COLLECTION_NAME = "rag_documents"

# BM25 sparse index (still local file — Milvus handles dense only)
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


def _connect_milvus():
    """Ensure a connection to the Milvus server exists."""
    try:
        connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
    except Exception as e:
        log.debug(f"Milvus connection note: {e}")


def build_vector_store(documents: list[Document], embedding_model_name: str = "qwen3-embedding:0.6b") -> tuple:
    """
    Chunks documents, builds Milvus vector store and BM25 index.

    Args:
        documents: List of Document objects from document_processor
        embedding_model_name: The embedding model to use

    Returns:
        (Milvus vector store, number of chunks created)
    """
    log.info(f"Building vector store from {len(documents)} document segments using {embedding_model_name}")

    # Clear existing stores
    clear_store()

    # Chunk the documents
    splitter = _get_splitter()
    chunks = splitter.split_documents(documents)
    log.info(f"Created {len(chunks)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")

    if not chunks:
        log.warning("No chunks created — documents may be empty")
        return None, 0

    # Build Milvus dense index
    embeddings = get_embeddings(embedding_model_name)

    # Normalize metadata — Milvus requires consistent fields across all documents
    # (some versions of the Milvus client will error if the batch insert
    # contains rows with differing columns, so we guarantee every chunk has
    # the same set of fields before we call `from_documents`).
    required_fields = {
        "source": "",
        "type": "",
        "page": 0,
        "slide": 0,
        "sheet": "",
    }

    for i, chunk in enumerate(chunks):
        # metadata should always be a dict; guard against corruption just in
        # case some upstream code accidentally replaced it with None.
        if not isinstance(chunk.metadata, dict):
            log.warning(f"chunk[{i}] metadata was not a dict (got {type(chunk.metadata)}); resetting")
            chunk.metadata = {}

        for field, default in required_fields.items():
            if field not in chunk.metadata:
                log.debug(f"adding missing metadata field '{field}' to chunk[{i}]")
                # Use copy of default in case it is mutable
                chunk.metadata[field] = default

    # sanity check: no chunk should still be missing any required field
    missing_source = [i for i,c in enumerate(chunks) if "source" not in c.metadata]
    if missing_source:
        log.error(f"chunks {missing_source} still missing 'source' metadata after normalization")
        # this should never happen, but we raise a clear error so the caller
        # can know that something went wrong with the document processing
        raise ValueError("Internal error: some document chunks lack 'source' metadata")

    from pymilvus import exceptions as milvus_exceptions

    try:
        vector_store = Milvus.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name=COLLECTION_NAME,
            connection_args={"host": MILVUS_HOST, "port": MILVUS_PORT},
            drop_old=True,  # Drop existing collection if any
        )
    except milvus_exceptions.DataNotMatchException as dne:
        # This error generally means a document in the batch lacked one of
        # the metadata fields used to define the collection schema. Our
        # earlier normalization step should prevent this, so raise a more
        # descriptive message to help debugging.
        log.error("Milvus insertion failed due to metadata mismatch", exc_info=True)
        raise ValueError(
            "Failed to upload documents: at least one chunk is missing a required "
            "metadata field (e.g. 'source'). Please ensure all documents have "
            "consistent metadata and try again.") from dne
    except Exception as e:
        # rethrow anything else unchanged
        raise

    log.info(f"Milvus HNSW index built with {len(chunks)} vectors (collection: {COLLECTION_NAME})")

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


def _load_milvus(embedding_model_name: str = "qwen3-embedding:0.6b") -> Milvus:
    """Load existing Milvus vector store."""
    _connect_milvus()
    if not utility.has_collection(COLLECTION_NAME):
        raise ValueError("Vector store not found. Upload documents first.")
    embeddings = get_embeddings(embedding_model_name)
    return Milvus(
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
        connection_args={"host": MILVUS_HOST, "port": MILVUS_PORT},
    )


def _load_bm25() -> tuple[BM25Okapi, list[Document]]:
    """Load existing BM25 index."""
    if not os.path.exists(BM25_PATH):
        raise ValueError("BM25 index not found. Upload documents first.")
    with open(BM25_PATH, 'rb') as f:
        data = pickle.load(f)
    return data["index"], data["chunks"]


def dense_mmr_search(query: str, k: int = 6, fetch_k: int = 20, embedding_model_name: str = "qwen3-embedding:0.6b") -> list[Document]:
    """Dense MMR search via Milvus."""
    log.debug(f"Dense MMR search: k={k}, fetch_k={fetch_k}")
    store = _load_milvus(embedding_model_name=embedding_model_name)
    try:
        results = store.max_marginal_relevance_search(query, k=k, fetch_k=fetch_k)
        log.debug(f"Dense search returned {len(results)} results")
        return results
    except MilvusException as e:
        if "dimension mismatch" in str(e).lower() or "vector size" in str(e).lower():
            log.error("Milvus dimension mismatch detected (likely due to hot-reload). Auto-clearing database.")
            clear_store()
            raise ValueError("The database dimensions did not match the current model! I have automatically cleared the database for you. **Please re-upload your files.**")
        raise e


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


def hybrid_search_rrf(query: str, original_query: str, k: int = 6, fetch_k: int = 20, rrf_k: int = 60, embedding_model_name: str = "qwen3-embedding:0.6b") -> list[Document]:
    """
    Hybrid search: Dense MMR (Milvus) + BM25 fused via Reciprocal Rank Fusion (RRF).

    RRF score = sum(1 / (rrf_k + rank)) across all result lists.
    """
    log.info(f"Hybrid RRF search: query='{query[:80]}...'")

    # Run both searches (Dense uses HyDE query, Sparse uses original keyword query)
    dense_results = dense_mmr_search(query, k=k, fetch_k=fetch_k, embedding_model_name=embedding_model_name)
    sparse_results = bm25_search(original_query, k=k)

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
    """Remove both Milvus collection and BM25 index.

    Milvus occasionally returns transient errors when dropping a collection
    (see the log messages about `InvalidateCollectionMetaCache`). in that case
    we retry once after a short pause so the UI can continue working instead
    of quietly leaving an old collection around with an incompatible schema.
    """
    gc.collect()

    _connect_milvus()
    if utility.has_collection(COLLECTION_NAME):
        try:
            utility.drop_collection(COLLECTION_NAME)
            log.info(f"Milvus collection '{COLLECTION_NAME}' dropped")
        except Exception as e:
            log.warning(f"Failed to drop Milvus collection on first attempt: {e}")
            # retry once
            try:
                log.info("Retrying Milvus drop after a brief pause...")
                import time
                time.sleep(1)
                utility.drop_collection(COLLECTION_NAME)
                log.info(f"Milvus collection '{COLLECTION_NAME}' dropped on retry")
            except Exception as e2:
                log.warning(f"Second drop attempt also failed: {e2}")

    if os.path.exists(BM25_PATH):
        os.remove(BM25_PATH)
        log.info("BM25 index cleared")


def store_exists() -> bool:
    """Check if the vector store has been built."""
    try:
        _connect_milvus()
        return utility.has_collection(COLLECTION_NAME) and os.path.exists(BM25_PATH)
    except Exception:
        return False
