from sentence_transformers import SentenceTransformer
from typing import Optional
import numpy as np
import logging
from pathlib import Path

log = logging.getLogger(__name__)

_MODEL_CACHE = {}

class EmbeddingModel:
    """Multilingual E5-base for semantic similarity."""
    
    MODEL_NAME = "intfloat/multilingual-e5-base"
    
    def __new__(cls, model_name: Optional[str] = None, device: str = "cpu"):
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_name = model_name or cls.MODEL_NAME
        cache_key = (model_name, device)
        if cache_key not in _MODEL_CACHE:
            instance = super(EmbeddingModel, cls).__new__(cls)
            _MODEL_CACHE[cache_key] = instance
        return _MODEL_CACHE[cache_key]
    
    def __init__(self, model_name: Optional[str] = None, device: str = "cpu"):
        if hasattr(self, 'model'):
            return
        model_name = model_name or self.MODEL_NAME
        log.info(f"Loading {model_name} on {device}")
        self.model = SentenceTransformer(model_name)
        self.device = device
        
    _MAX_TEXT_LEN = 512

    def encode(self, texts: list[str], batch_size: int = 64, normalize: bool = True) -> np.ndarray:
        """Encode texts to embeddings with input validation."""
        if not texts:
            raise ValueError("texts must be a non-empty list")
        sanitized = []
        for i, t in enumerate(texts):
            if not isinstance(t, str):
                raise TypeError(f"texts[{i}] must be str, got {type(t).__name__}")
            sanitized.append(t[:self._MAX_TEXT_LEN])
        embeddings = self.model.encode(
            sanitized,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=normalize,
        )
        return embeddings
    
    @classmethod
    def from_pretrained(cls, model_path: Path):
        """Load from local path."""
        return cls(model_path.as_posix())
    
    def save(self, path: Path):
        """Save model."""
        self.model.save(path.as_posix())
        log.info(f"Model saved to {path}")

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

