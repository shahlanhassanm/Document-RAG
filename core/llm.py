"""
LLM wrapper for qwen3:8b via Ollama.
Fully local, no cloud dependencies.
"""

import ollama
from langchain_ollama import OllamaLLM
from core.logger import get_logger

log = get_logger("llm")


def get_llm(temperature: float = 0.3, num_ctx: int = 32768) -> OllamaLLM:
    """
    Returns the qwen3:8b LLM instance.
    
    Args:
        temperature: Lower = more factual. Default 0.3 for grounded answers.
        num_ctx: Context window size. Default 32K, upgradeable to 128K.
    """
    log.info(f"Initializing LLM: qwen3:8b (temp={temperature}, ctx={num_ctx})")
    llm = OllamaLLM(
        model="qwen3:8b",
        temperature=temperature,
        num_ctx=num_ctx,
    )
    log.info("LLM ready")
    return llm


def check_model(model_name: str) -> bool:
    """Check if an Ollama model is available locally."""
    try:
        list_resp = ollama.list()
        if hasattr(list_resp, 'models'):
            models = [m.model for m in list_resp.models]
        else:
            models = [m.get('name', '') for m in list_resp.get('models', [])]
        
        for existing in models:
            if existing.startswith(model_name):
                log.debug(f"Model '{model_name}' found as '{existing}'")
                return True
        
        log.warning(f"Model '{model_name}' not found in Ollama")
        return False
    except Exception as e:
        log.error(f"Error checking Ollama models: {e}")
        return False


def get_available_models() -> list[str]:
    """Returns list of all available Ollama models."""
    try:
        list_resp = ollama.list()
        if hasattr(list_resp, 'models'):
            return [m.model for m in list_resp.models]
        return [m.get('name', '') for m in list_resp.get('models', [])]
    except Exception:
        return []
