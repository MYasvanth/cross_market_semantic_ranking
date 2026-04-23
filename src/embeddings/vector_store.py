import faiss
import numpy as np
from typing import Tuple, List
from pathlib import Path
import logging

log = logging.getLogger(__name__)

# HNSW_M: number of graph connections per node — 32 is optimal recall/speed tradeoff
_HNSW_M = 32


class VectorStore:
    """FAISS index for Stage 1 retrieval — IndexHNSWFlat for ANN search."""

    def __init__(self, dimension: int = 768, use_hnsw: bool = True):
        self.dimension = dimension
        if use_hnsw:
            self.index = faiss.IndexHNSWFlat(dimension, _HNSW_M)
            self.index.hnsw.efSearch = 64   # search-time accuracy vs speed
            log.info(f"FAISS IndexHNSWFlat(d={dimension}, M={_HNSW_M}, efSearch=64)")
        else:
            self.index = faiss.IndexFlatIP(dimension)
            log.info(f"FAISS IndexFlatIP(d={dimension})")
        self.ntotal = 0
        
    _MAX_VECTORS = 10_000_000
    _MAX_K = 1_000

    def add(self, embeddings: np.ndarray, ids: List[int]):
        """Add product embeddings to index with validation."""
        if not isinstance(embeddings, np.ndarray) or embeddings.ndim != 2:
            raise ValueError("embeddings must be a 2-D numpy array")
        if embeddings.shape[1] != self.dimension:
            raise ValueError(f"Expected dim {self.dimension}, got {embeddings.shape[1]}")
        if len(ids) != len(embeddings):
            raise ValueError("ids length must match number of embeddings")
        if self.ntotal + len(embeddings) > self._MAX_VECTORS:
            raise ValueError(f"Index would exceed max capacity of {self._MAX_VECTORS}")
        if not np.isfinite(embeddings).all():
            raise ValueError("embeddings contain NaN or Inf values")
        self.index.add(embeddings.astype('float32'))
        self.ntotal = self.index.ntotal
        log.info(f"Added {len(ids)} vectors. Total: {self.ntotal}")

    def search(self, query_embedding: np.ndarray, k: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """Search top-k similar products with validation."""
        if not isinstance(query_embedding, np.ndarray) or query_embedding.ndim != 2:
            raise ValueError("query_embedding must be a 2-D numpy array")
        if query_embedding.shape[1] != self.dimension:
            raise ValueError(f"Expected dim {self.dimension}, got {query_embedding.shape[1]}")
        if not (1 <= k <= self._MAX_K):
            raise ValueError(f"k must be between 1 and {self._MAX_K}")
        if not np.isfinite(query_embedding).all():
            raise ValueError("query_embedding contains NaN or Inf values")
        k = min(k, self.ntotal)
        scores, indices = self.index.search(query_embedding.astype('float32'), k)
        return scores, indices
    
    def save(self, path: Path):
        """Save FAISS index."""
        path.parent.mkdir(exist_ok=True)
        faiss.write_index(self.index, str(path))
        log.info(f"Index saved to {path}")
    
    @classmethod
    def load(cls, path: Path) -> 'VectorStore':
        """Load FAISS index with path validation."""
        path = Path(path).resolve()
        allowed_base = Path("artifacts").resolve()
        if not str(path).startswith(str(allowed_base)):
            raise ValueError(f"Index path must be inside artifacts/: {path}")
        if not path.exists():
            raise FileNotFoundError(f"Index file not found: {path}")
        index = faiss.read_index(str(path))
        store = cls(index.d)
        store.index = index
        store.ntotal = index.ntotal
        log.info(f"Index loaded from {path}, ntotal={store.ntotal}")
        return store
    
    def normalize_index(self):
        """Normalize for cosine similarity."""
        faiss.normalize_L2(self.index)

