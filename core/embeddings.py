"""
Embedding model wrapper.
Uses qwen3-embedding:0.6b via Ollama for fully local embeddings.
"""

from langchain_ollama import OllamaEmbeddings
from core.logger import get_logger

log = get_logger("embeddings")

_embeddings_instance = None


def get_embeddings() -> OllamaEmbeddings:
    """Returns the Ollama embedding model instance (singleton)."""
    global _embeddings_instance
    if _embeddings_instance is None:
        log.info("Initializing embedding model: qwen3-embedding:0.6b")
        _embeddings_instance = OllamaEmbeddings(model="qwen3-embedding:0.6b")
        log.info("Embedding model ready")
    return _embeddings_instance
