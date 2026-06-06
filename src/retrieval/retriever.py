"""Stage 1 Retrieval — Semantic + Hybrid candidates."""
from typing import List, Tuple
import re
import numpy as np
from src.embeddings.embedding_model import EmbeddingModel
from ..embeddings.vector_store import VectorStore

_STOPWORDS = {"a", "an", "the", "for", "of", "in", "on", "at", "to", "and", "or", "with"}


def _normalize_tokens(text: str) -> List[str]:
    """Canonical tokenizer used by BOTH retrieval and feature BM25.
    Single definition eliminates the train/serve IDF skew that arises
    when retrieval uses stopword-stripped tokens but features use raw split.
    """
    tokens = re.sub(r"[^\w\s]", " ", text.lower()).split()
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 1]


class SemanticRetriever:
    def __init__(self, embedding_model: EmbeddingModel, vector_store: VectorStore):
        self.embedding_model = embedding_model
        self.vector_store = vector_store

    def retrieve(self, query: str, top_k: int = 100, query_emb: np.ndarray = None) -> List[Tuple[int, float]]:
        if query_emb is None:
            query_emb = self.embedding_model.encode([query])
        scores, indices = self.vector_store.search(query_emb, top_k)
        return list(zip(indices[0], scores[0]))


class HybridRetriever:
    """
    Hybrid Stage 1 retrieval: BM25 + FAISS with Reciprocal Rank Fusion (RRF).
    BM25 uses normalized tokens (no stopwords, no punctuation) for better recall.
    Semantic weight is higher (0.7) since multilingual-e5-base is the stronger signal.
    """

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_store: VectorStore,
        bm25_tokenized_docs: List[List[str]],
        rrf_k: int = 30,
        semantic_weight: float = 0.7,
        bm25_weight: float = 0.3,
    ):
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.rrf_k = rrf_k
        self.semantic_weight = semantic_weight
        self.bm25_weight = bm25_weight

        # Re-tokenize docs with normalized tokens for better recall
        from rank_bm25 import BM25Okapi
        normalized_docs = [_normalize_tokens(" ".join(doc)) for doc in bm25_tokenized_docs]
        self.bm25 = BM25Okapi(normalized_docs)
        self.doc_count = len(bm25_tokenized_docs)

    def retrieve(
        self,
        query: str,
        top_k: int = 400,
        query_emb: np.ndarray = None,
    ) -> List[Tuple[int, float]]:
        if query_emb is None:
            query_emb = self.embedding_model.encode([query])
        sem_scores, sem_indices = self.vector_store.search(query_emb, min(top_k * 2, self.doc_count))
        sem_indices = sem_indices[0]

        query_tokens = _normalize_tokens(query)
        bm25_scores  = self.bm25.get_scores(query_tokens)

        fused_scores = {}
        for rank, idx in enumerate(sem_indices):
            fused_scores[int(idx)] = self.semantic_weight / (self.rrf_k + rank + 1)

        bm25_ranks = np.argsort(-bm25_scores)[:top_k * 2]
        for rank, idx in enumerate(bm25_ranks):
            idx = int(idx)
            fused_scores[idx] = fused_scores.get(idx, 0.0) + \
                self.bm25_weight / (self.rrf_k + rank + 1)

        sorted_results = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]
