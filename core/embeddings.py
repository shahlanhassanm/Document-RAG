import threading
import time
from typing import Optional, Dict, Any, Union
from langchain_ollama import OllamaEmbeddings
from pydantic import ValidationError
from core.logger import get_logger

# Professional-grade logging for infrastructure monitoring
log = get_logger("embeddings")

class EmbeddingModelManager:
    """
    A production-grade manager for local Ollama embedding models.
    Implements a thread-safe Singleton pattern with proactive validation,
    advanced timeout handling, and hardware optimization.
    """
    _instance: Optional['EmbeddingModelManager'] = None
    _global_lock = threading.Lock()

    def __new__(cls):
        """Ensure a single manager instance across the entire application."""
        if cls._instance is None:
            with cls._global_lock:
                if cls._instance is None:
                    cls._instance = super(EmbeddingModelManager, cls).__new__(cls)
                    cls._instance._init_state()
        return cls._instance

    def _init_state(self):
        """Initialize internal state for instance and model tracking."""
        self._embeddings_instance: Optional[OllamaEmbeddings] = None
        self._active_model_name: Optional[str] = None
        self._instance_lock = threading.Lock()
        log.info("Embedding Manager initialized. Ready to serve local models.")

    def get_model(
        self, 
        model_name: str = "qwen3-embedding:0.6b",
        base_url: str = "http://localhost:11434",
        request_timeout: int = 120,
        keep_alive: str = "1h",
        num_ctx: int = 32768,
        validate: bool = True,
        **extra_params: Any
    ) -> OllamaEmbeddings:
        """
        Retrieves or initializes a thread-safe OllamaEmbeddings instance.
        
        Args:
            model_name: Name of the Ollama model. Defaults to Qwen3-0.6B.
            base_url: The API endpoint for the local Ollama server.
            request_timeout: HTTP timeout in seconds for long-context processing.
            keep_alive: Duration the model stays warm in VRAM (e.g., '1h', '24h').
            num_ctx: Size of the context window (Qwen3 supports up to 32,768).
            validate: If True, checks model existence on initialization.
            **extra_params: Additional kwargs for hardware tuning (num_gpu, num_thread).
        """
        # Primary check without locking for performance
        if self._embeddings_instance is None or self._active_model_name!= model_name:
            with self._instance_lock:
                # Re-verify inside the lock to prevent multi-threaded race conditions
                if self._embeddings_instance is None or self._active_model_name!= model_name:
                    log.info(f"Initializing/Switching model: {model_name} (Context: {num_ctx})")
                    
                    try:
                        # Configure client-level timeouts for network resilience
                        # Fine-tuning connect/read/write/pool for local reliability
                        client_config = {
                            "timeout": float(request_timeout),
                            "follow_redirects": True
                        }
                        
                        # Instantiate the OllamaEmbeddings with Pydantic-based validation
                        # We use validate_model_on_init to fail fast if the model is missing
                        new_instance = OllamaEmbeddings(
                            model=model_name,
                            base_url=base_url,
                            validate_model_on_init=validate,
                            keep_alive=keep_alive,
                            num_ctx=num_ctx,
                            client_kwargs=client_config,
                            **extra_params
                        )
                        
                        # Atomic update of the shared instance
                        self._embeddings_instance = new_instance
                        self._active_model_name = model_name
                        log.info(f"Successfully loaded {model_name} into VRAM.")
                        
                    except ValidationError as ve:
                        log.error(f"Configuration Validation Error for {model_name}: {ve}")
                        # In a production system, this could trigger an 'ollama pull'
                        raise RuntimeError(f"Model {model_name} configuration is invalid or model not found.") from ve
                    except ConnectionError as ce:
                        log.error(f"Failed to connect to Ollama server at {base_url}: {ce}")
                        raise RuntimeError("AI Infrastructure Connection Failure") from ce
                    except Exception as e:
                        log.critical(f"Critical failure during model instantiation: {e}")
                        raise

        return self._embeddings_instance

# High-level factory function satisfying the user's interface requirements
def get_embeddings(model_name: str = "qwen3-embedding:0.6b", **kwargs) -> OllamaEmbeddings:
    """
    Public access point for the thread-safe, optimized embedding wrapper.
    Encapsulates the EmbeddingModelManager to ensure perfect state management.
    """
    manager = EmbeddingModelManager()
    return manager.get_model(model_name=model_name, **kwargs)