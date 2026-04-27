"""Stage 1 Retrieval — Semantic + Hybrid candidates."""
from typing import List, Tuple, Dict
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


class HybridRetriever:
    """
    Hybrid Stage 1 retrieval: BM25 + FAISS with Reciprocal Rank Fusion (RRF).

    When semantic retrieval alone yields low scores (e.g., 0.075 on realistic
    data), keyword-based BM25 can rescue relevant candidates that embeddings miss.
    """

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_store: VectorStore,
        bm25_tokenized_docs: List[List[str]],
        rrf_k: int = 60,
        semantic_weight: float = 0.5,
        bm25_weight: float = 0.5,
    ):
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.rrf_k = rrf_k
        self.semantic_weight = semantic_weight
        self.bm25_weight = bm25_weight

        # Build BM25 index
        from rank_bm25 import BM25Okapi
        self.bm25 = BM25Okapi(bm25_tokenized_docs)
        self.doc_count = len(bm25_tokenized_docs)

    def retrieve(self, query: str, top_k: int = 200) -> List[Tuple[int, float]]:
        """
        Retrieve top-K products using hybrid BM25 + FAISS fusion.

        Args:
            query: raw user query
            top_k: number of candidates to return (default 200 for training)

        Returns:
            List of (doc_index, fused_score) tuples, sorted by score descending.
        """
        # ── Semantic retrieval ─────────────────────────────────────
        query_emb = self.embedding_model.encode([query])
        sem_scores, sem_indices = self.vector_store.search(query_emb, min(top_k * 2, self.doc_count))
        sem_indices = sem_indices[0]
        sem_scores = sem_scores[0]

        # Convert L2 distance to similarity score (lower distance = higher score)
        # For normalized vectors: L2^2 = 2 - 2*cos_sim → cos_sim = 1 - L2^2/2
        sem_sims = 1.0 - (sem_scores ** 2) / 2.0
        sem_sims = np.clip(sem_sims, 0.0, 1.0)

        # ── BM25 retrieval ─────────────────────────────────────────
        query_tokens = query.lower().split()
        bm25_scores = self.bm25.get_scores(query_tokens)

        # Normalize BM25 to [0, 1] using percentile-based scaling
        if bm25_scores.max() > 0:
            bm25_norm = bm25_scores / bm25_scores.max()
        else:
            bm25_norm = np.zeros_like(bm25_scores)

        # ── Reciprocal Rank Fusion ─────────────────────────────────
        # RRF: score = Σ 1 / (k + rank) for each ranking
        fused_scores = {}

        # Semantic ranking contribution
        for rank, idx in enumerate(sem_indices):
            if idx < self.doc_count:
                fused_scores[idx] = fused_scores.get(idx, 0.0) + \
                    self.semantic_weight / (self.rrf_k + rank + 1)

        # BM25 ranking contribution
        bm25_ranks = np.argsort(-bm25_scores)[:top_k * 2]
        for rank, idx in enumerate(bm25_ranks):
            if idx < self.doc_count:
                fused_scores[idx] = fused_scores.get(idx, 0.0) + \
                    self.bm25_weight / (self.rrf_k + rank + 1)

        # ── Score-based fusion fallback ────────────────────────────
        # For indices that only appear in one ranking, add their raw score
        for rank, idx in enumerate(sem_indices[:top_k]):
            if idx not in fused_scores:
                fused_scores[idx] = self.semantic_weight * sem_sims[rank]

        top_bm25 = np.argsort(-bm25_scores)[:top_k]
        for idx in top_bm25:
            if idx not in fused_scores:
                fused_scores[idx] = self.bm25_weight * bm25_norm[idx]

        # Sort by fused score descending
        sorted_results = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]


