"""Stage 1 Neural Retrieval - Top 100 semantic candidates."""
from typing import List, Tuple
import numpy as np
from src.embeddings.embedding_model import EmbeddingModel
from ..embeddings.vector_store import VectorStore

class SemanticRetriever:
    def __init__(self, embedding_model: EmbeddingModel, vector_store: VectorStore):
        self.embedding_model = embedding_model
        self.vector_store = vector_store
    
    def retrieve(self, query: str, top_k: int = 100) -> List[Tuple[int, float]]:
        """Retrieve top-K products by semantic similarity."""
        query_emb = self.embedding_model.encode([query])
        scores, indices = self.vector_store.search(query_emb, top_k)
        return list(zip(indices[0], scores[0]))

