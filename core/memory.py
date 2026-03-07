"""
Persistent Vector-Based Conversation Memory — Milvus Backend.
Stores past Q&A pairs in a separate Milvus collection for semantic recall.
"""

import os
import uuid
from datetime import datetime
from langchain_milvus import Milvus
from langchain_core.documents import Document
from pymilvus import connections, utility
from core.embeddings import get_embeddings
from core.logger import get_logger

log = get_logger("memory")

# Milvus connection config (same server as vector_store)
MILVUS_HOST = "127.0.0.1"
MILVUS_PORT = "19530"
MEMORY_COLLECTION = "conversation_memory"


def _connect_milvus():
    """Ensure a connection to the Milvus server exists."""
    try:
        connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
    except Exception as e:
        log.debug(f"Milvus connection note: {e}")


def _get_memory_store() -> Milvus:
    """Get or create the memory Milvus collection."""
    embeddings = get_embeddings()
    return Milvus(
        embedding_function=embeddings,
        collection_name=MEMORY_COLLECTION,
        connection_args={"host": MILVUS_HOST, "port": MILVUS_PORT},
    )


def store_exchange(question: str, answer: str, session_id: str = "default"):
    """
    Store a Q&A exchange in persistent vector memory.

    Args:
        question: User's question
        answer: Assistant's answer
        session_id: Session identifier for grouping conversations
    """
    log.info(f"Storing exchange in memory (session: {session_id})")

    store = _get_memory_store()

    # Combine Q&A into a single document for semantic search
    content = f"Question: {question}\nAnswer: {answer}"
    doc = Document(
        page_content=content,
        metadata={
            "question": question[:500],  # Truncate for metadata limits
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "type": "conversation",
        }
    )

    store.add_documents([doc], ids=[str(uuid.uuid4())])
    log.debug("Exchange stored in memory")


def recall_relevant(query: str, top_k: int = 3) -> list[dict]:
    """
    Retrieve the top-k most semantically relevant past exchanges.

    Args:
        query: Current user question
        top_k: Number of past exchanges to retrieve

    Returns:
        List of dicts with 'question', 'answer', 'timestamp' keys
    """
    if not memory_exists():
        return []

    log.debug(f"Recalling top-{top_k} relevant past exchanges")

    try:
        store = _get_memory_store()
        results = store.similarity_search(query, k=top_k)

        exchanges = []
        for doc in results:
            content = doc.page_content
            # Parse back into Q&A
            if "Question:" in content and "Answer:" in content:
                parts = content.split("Answer:", 1)
                q = parts[0].replace("Question:", "").strip()
                a = parts[1].strip()
                exchanges.append({
                    "question": q,
                    "answer": a,
                    "timestamp": doc.metadata.get("timestamp", ""),
                })

        log.info(f"Recalled {len(exchanges)} relevant past exchanges")
        return exchanges
    except Exception as e:
        log.warning(f"Memory recall failed: {e}")
        return []


def get_recent_messages(messages: list[dict], n: int = 5) -> str:
    """
    Format the last n messages as conversation context string.

    Args:
        messages: List of {"role": ..., "content": ...} dicts
        n: Number of recent messages to include
    """
    if not messages:
        return ""

    recent = messages[-n:]
    lines = []
    for m in recent:
        prefix = "User" if m["role"] == "user" else "Assistant"
        lines.append(f"{prefix}: {m['content']}")
    return "\n".join(lines)


def clear_memory():
    """Wipe all conversation memory."""
    import gc
    gc.collect()

    try:
        _connect_milvus()
        if utility.has_collection(MEMORY_COLLECTION):
            utility.drop_collection(MEMORY_COLLECTION)
            log.info(f"Milvus memory collection '{MEMORY_COLLECTION}' dropped")
    except Exception as e:
        log.warning(f"Failed to clear memory: {e}")


def memory_exists() -> bool:
    """Check if memory store exists."""
    try:
        _connect_milvus()
        return utility.has_collection(MEMORY_COLLECTION)
    except Exception:
        return False
