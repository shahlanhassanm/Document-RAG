"""
Persistent Vector-Based Conversation Memory.
Stores past Q&A pairs in a separate ChromaDB collection for semantic recall.
"""

import os
import uuid
from datetime import datetime
from langchain_chroma import Chroma
from langchain_core.documents import Document
from core.embeddings import get_embeddings
from core.logger import get_logger

log = get_logger("memory")

MEMORY_DIR = os.path.join(os.path.dirname(__file__), '..', 'memory_db')
COLLECTION_NAME = "conversation_memory"


def _get_memory_store() -> Chroma:
    """Get or create the memory ChromaDB collection."""
    embeddings = get_embeddings()
    return Chroma(
        persist_directory=MEMORY_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
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
    if not os.path.exists(MEMORY_DIR):
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
    import shutil
    import gc
    gc.collect()
    
    if os.path.exists(MEMORY_DIR):
        def on_rm_error(func, path, exc_info):
            import stat
            os.chmod(path, stat.S_IWRITE)
            try:
                func(path)
            except Exception:
                pass
        shutil.rmtree(MEMORY_DIR, onerror=on_rm_error)
        log.info("Conversation memory cleared")


def memory_exists() -> bool:
    """Check if memory store exists."""
    return os.path.exists(MEMORY_DIR)
