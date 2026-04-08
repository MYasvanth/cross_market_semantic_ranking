import faiss
import numpy as np
from typing import Tuple, List
from pathlib import Path
import logging

log = logging.getLogger(__name__)

class VectorStore:
    """FAISS index for Stage 1 retrieval."""
    
    def __init__(self, dimension: int = 768, index_type: str = "IndexFlatIP"):
        self.dimension = dimension
        self.index = getattr(faiss, index_type)(dimension)
        self.ntotal = 0
        
    def add(self, embeddings: np.ndarray, ids: List[int]):
        """Add product embeddings to index."""
        assert embeddings.shape[1] == self.dimension
        self.index.add(embeddings.astype('float32'))
        self.ntotal = self.index.ntotal
        log.info(f"Added {len(ids)} vectors. Total: {self.ntotal}")
    
    def search(self, query_embedding: np.ndarray, k: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """Search top-k similar products."""
        scores, indices = self.index.search(query_embedding.astype('float32'), k)
        return scores, indices
    
    def save(self, path: Path):
        """Save FAISS index."""
        path.parent.mkdir(exist_ok=True)
        faiss.write_index(self.index, str(path))
        log.info(f"Index saved to {path}")
    
    @classmethod
    def load(cls, path: Path) -> 'VectorStore':
        """Load FAISS index."""
        index = faiss.read_index(str(path))
        store = cls(index.d)
        store.index = index
        store.ntotal = index.ntotal
        log.info(f"Index loaded from {path}, ntotal={store.ntotal}")
        return store
    
    def normalize_index(self):
        """Normalize for cosine similarity."""
        faiss.normalize_L2(self.index)

