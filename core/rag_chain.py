"""
RAG Chain — Full Pipeline Orchestrator.
HyDE → Hybrid Search (MMR + BM25 + RRF) → FlashRank Rerank → LongContextReorder → Generate
"""

from langchain_core.documents import Document
from core.query_transform import hyde_transform
from core.vector_store import hybrid_search_rrf, store_exists
from core.reranker import rerank, long_context_reorder
from core.memory import recall_relevant, get_recent_messages
from core.llm import get_llm
from core.logger import get_logger

log = get_logger("rag_chain")

# Strict grounded-answer prompt
RAG_PROMPT = """You are a precise document analyst. Your ONLY job is to answer questions 
based strictly on the provided document excerpts. Follow these rules absolutely:

1. Answer ONLY using information from the "Document Context" below
2. CITE the source (filename, page/slide/sheet number) for every factual claim
3. If the answer is NOT in the provided context, say: "I could not find this information in the uploaded documents."
4. NEVER make up information or use outside knowledge
5. Be detailed and comprehensive in your answers
6. Format your response with clear structure (use bullet points, headers if needed)

{memory_context}

---
Document Context:
{context}
---

{history}

Question: {question}

Answer (with citations):"""

# General chat prompt (no documents uploaded)
GENERAL_PROMPT = """You are a helpful AI assistant running locally. Answer the following 
question to the best of your ability. Be clear, detailed, and honest about what you know.

{history}

Question: {question}

Answer:"""


def format_context(documents: list[Document]) -> str:
    """Format retrieved documents into a context string with source citations."""
    if not documents:
        return "No relevant documents found."
    
    parts = []
    for i, doc in enumerate(documents, 1):
        meta = doc.metadata
        source = meta.get("source", "Unknown")
        
        # Build location string
        location_parts = []
        if "page" in meta:
            location_parts.append(f"Page {meta['page']}")
        if "slide" in meta:
            location_parts.append(f"Slide {meta['slide']}")
        if "sheet" in meta:
            location_parts.append(f"Sheet '{meta['sheet']}'")
        
        location = ", ".join(location_parts) if location_parts else ""
        header = f"[Source: {source}" + (f", {location}]" if location else "]")
        
        parts.append(f"--- Excerpt {i} {header} ---\n{doc.page_content}")
    
    return "\n\n".join(parts)


def format_memory(exchanges: list[dict]) -> str:
    """Format recalled past exchanges into context."""
    if not exchanges:
        return ""
    
    lines = ["Relevant past conversations:"]
    for ex in exchanges:
        lines.append(f"  Q: {ex['question'][:200]}")
        lines.append(f"  A: {ex['answer'][:300]}")
        lines.append("")
    
    return "\n".join(lines)


def ask(question: str, chat_history: list[dict] = None, num_ctx: int = 8192, model_name: str = "qwen3:8b", embedding_model_name: str = "qwen3-embedding:0.6b") -> dict:
    """
    Full RAG pipeline: HyDE → Hybrid Search → Rerank → Reorder → Generate.
    
    Args:
        question: User's question
        chat_history: List of {"role": ..., "content": ...} message dicts
        num_ctx: Context window size for the LLM
        model_name: Ollama model to use for generation
    
    Returns:
        dict with keys:
            - "answer": Generated answer text
            - "sources": List of source documents used
            - "pipeline": Dict of pipeline stats (for UI transparency)
    """
    chat_history = chat_history or []
    pipeline_stats = {}
    
    if not store_exists():
        # No documents uploaded
        log.info("No documents in store")
        yield {
            "answer": "No documents uploaded. Please upload a document from the sidebar to ask questions.",
            "sources": [],
            "pipeline": {"mode": "no_documents"},
        }
        return
    
    # === RAG MODE ===
    log.info(f"RAG pipeline started for: '{question[:80]}...'")
    
    # Stage 1: HyDE Query Transform (always on)
    search_query = hyde_transform(question, model_name=model_name)
    pipeline_stats["hyde"] = search_query[:100]
    
    # Stage 2: Hybrid Search (Dense MMR + BM25 + RRF)
    candidates = hybrid_search_rrf(search_query, original_query=question, k=6, fetch_k=20, embedding_model_name=embedding_model_name)
    pipeline_stats["candidates_found"] = len(candidates)
    
    if not candidates:
        log.warning("No candidates found from hybrid search")
        yield {
            "answer": "I could not find any relevant information in the uploaded documents.",
            "sources": [],
            "pipeline": pipeline_stats,
        }
        return
    
    # Stage 3: FlashRank Reranking
    reranked = rerank(question, candidates, top_k=6)  # Use original question for reranking
    pipeline_stats["after_rerank"] = len(reranked)
    
    # Stage 4: LongContextReorder
    reordered = long_context_reorder(reranked)
    pipeline_stats["context_chunks"] = len(reordered)
    
    # Format context with source citations
    context_str = format_context(reordered)
    
    # Recall relevant past exchanges from memory
    past_exchanges = recall_relevant(question, top_k=3)
    memory_str = format_memory(past_exchanges)
    pipeline_stats["memory_recalls"] = len(past_exchanges)
    
    # Recent chat history
    history_str = get_recent_messages(chat_history, n=5)
    history_section = f"Recent conversation:\n{history_str}" if history_str else ""
    
    # Build final prompt
    prompt = RAG_PROMPT.format(
        memory_context=memory_str,
        context=context_str,
        history=history_section,
        question=question,
    )
    
    # Stage 5: Generate with LLM (Streaming)
    log.info(f"Starting streaming generation with {model_name}")
    llm = get_llm(model_name=model_name)
    
    # Collect source info
    sources = []
    for doc in reordered:
        meta = doc.metadata
        source_info = {"file": meta.get("source", "Unknown")}
        if "page" in meta:
            source_info["page"] = meta["page"]
        if "slide" in meta:
            source_info["slide"] = meta["slide"]
        if "sheet" in meta:
            source_info["sheet"] = meta["sheet"]
        sources.append(source_info)
        
    pipeline_stats["mode"] = "rag"
    
    # We yield tokens first, then yield a dict with metadata at the very end
    full_answer = ""
    in_think_block = False
    
    for chunk in llm.stream(prompt):
        text = chunk
        full_answer += text
        
        # Simple heuristic to strip <think> tags from streaming output
        if "<think>" in full_answer and "</think>" not in full_answer:
            continue
        elif "</think>" in text:
            text = text.split("</think>")[-1]
            if not text:
                continue
                
        yield text

    log.info(f"RAG streaming complete. Answer length: {len(full_answer)} chars")
    
    # Final yield is the metadata block
    yield {
        "is_meta": True,
        "full_answer": full_answer,
        "sources": sources,
        "pipeline": pipeline_stats,
    }
