"""
Optimized and Robust LLM Wrapper for Qwen3:8B.
Developed for production-grade local AI orchestration.
"""

import ollama
from langchain_ollama import OllamaLLM
from core.logger import get_logger
from typing import List, Optional, Union

# Initialize high-fidelity logger
log = get_logger("llm_engine")

class OllamaRegistryManager:
    """Handles verification and lifecycle for local Ollama models."""
    
    def __init__(self, host: str = "http://localhost:11434"):
        self.host = host

    def is_server_online(self) -> bool:
        """Verifies if the Ollama server is responsive."""
        try:
            # A simple call to list models serves as a connectivity test
            ollama.list()
            return True
        except Exception as e:
            log.critical(f"Ollama server unreachable at {self.host}. Error: {e}")
            return False

    def ensure_model_presence(self, model_name: str, auto_pull: bool = True) -> bool:
        """
        Confirms model existence or initiates pull if missing.
        """
        try:
            list_resp = ollama.list()
            # Modern SDK uses ListResponse object with a 'models' sequence
            local_models = [m.model for m in list_resp.models] if hasattr(list_resp, 'models') else
            
            if any(m.startswith(model_name) for m in local_models):
                log.info(f"Verified model presence: {model_name}")
                return True
            
            if auto_pull:
                log.warning(f"Model {model_name} not found locally. Initiating pull...")
                ollama.pull(model_name)
                log.info(f"Model {model_name} successfully pulled to local registry.")
                return True
            
            return False
        except ollama.ResponseError as re:
            log.error(f"Ollama API error during model check: {re.error}")
            return False
        except Exception as e:
            log.error(f"Unexpected error during registry audit: {e}")
            return False

def get_llm(
    temperature: float = 0.3,
    num_ctx: int = 32768,
    model_name: str = "qwen3:8b",
    reasoning_mode: Optional[bool] = None,
    keep_alive: str = "5m",
    auto_pull: bool = True
) -> OllamaLLM:
    """
    Returns a validated and optimized LangChain Ollama LLM instance.
    
    Args:
        temperature: Controls output variance. Default 0.3 for grounded tasks. 
        num_ctx: Size of the context window. Qwen3 native is 32,768. 
        model_name: Qualified name of the local model. [5]
        reasoning_mode: Explicit control for <think> tags. 
        keep_alive: Memory persistence duration. Use -1 for infinite. 
        auto_pull: Whether to automatically download a missing model. 
    """
    manager = OllamaRegistryManager()
    
    # Pre-flight check: Server connectivity
    if not manager.is_server_online():
        raise ConnectionError("Ollama service must be running locally to proceed.")

    # Pre-flight check: Model registry audit
    if not manager.ensure_model_presence(model_name, auto_pull=auto_pull):
        raise FileNotFoundError(f"Model {model_name} is unavailable and auto_pull is disabled.")

    log.info(f"Configuring LLM: {model_name} | Temp: {temperature} | Ctx: {num_ctx}")
    
    # Instantiate with fail-fast validation and advanced parameters
    llm = OllamaLLM(
        model=model_name,
        temperature=temperature,
        num_ctx=num_ctx,
        keep_alive=keep_alive,
        validate_model_on_init=True,
        reasoning=reasoning_mode
    )
    
    log.info(f"LLM Engine {model_name} initialized and ready for inference.")
    return llm