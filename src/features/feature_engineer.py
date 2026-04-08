"""Universal Feature Engineering — Semantic, Lexical, Entity, Cross-Lingual."""
import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity
from langdetect import detect, LangDetectException
from src.embeddings.embedding_model import EmbeddingModel

# Cross-lingual query translation map (local -> English keywords)
# Covers the 4 target markets: India (hi), Egypt (ar), Poland (pl)
_TRANSLATION_MAP = {
    # Hindi
    "जूते":      "shoes",
    "इलेक्ट्रॉनिक्स": "electronics",
    "कपड़े":     "clothing",
    "सोनी":      "sony",
    "नाइकी":     "nike",
    # Arabic
    "أحذية":     "shoes",
    "إلكترونيات": "electronics",
    "ملابس":     "clothing",
    "سوني":      "sony",
    # Polish
    "buty":      "shoes",
    "elektronika": "electronics",
    "odzież":    "clothing",
    "tanie":     "cheap",
    "najlepszy": "best",
}


def _translate_query(query: str) -> str:
    """Map local language tokens to English equivalents."""
    tokens = query.split()
    return " ".join(_TRANSLATION_MAP.get(t, t) for t in tokens)


def _detect_lang(query: str) -> str:
    try:
        return detect(query)
    except LangDetectException:
        return "en"


class FeatureEngineer:
    """
    Extracts 8 universal signals per query-product pair:

    Semantic  : [0] cosine similarity (E5 embeddings)
                [1] cross-lingual cosine (translated query vs product)
    Lexical   : [2] BM25 score
                [3] Jaccard overlap
    Entity    : [4] brand match (boolean)
                [5] category match (boolean)
                [6] exact title match (boolean)
    Query     : [7] query length (normalisation signal)
    """

    FEATURE_NAMES = [
        "semantic_sim",
        "cross_lingual_sim",
        "bm25_score",
        "jaccard",
        "brand_match",
        "category_match",
        "exact_title_match",
        "query_len",
    ]

    def __init__(self, embedding_model: EmbeddingModel, bm25_tokenized_docs: list):
        self.embedding_model = embedding_model
        self.bm25 = BM25Okapi(bm25_tokenized_docs)

    def extract_features(
        self,
        query: str,
        products: list,
        prod_embs: np.ndarray = None,
        query_emb: np.ndarray = None,
        translated_emb: np.ndarray = None,
        candidate_indices: list = None,
    ) -> np.ndarray:
        """
        Extract 8 features for every (query, product) pair.

        Args:
            query:             raw user query in any language
            products:          list of dicts with keys: title, brand, category
            prod_embs:         pre-computed product embeddings (n, 768)
            query_emb:         pre-computed query embedding (1, 768)
            translated_emb:    pre-computed translated query embedding (1, 768)
            candidate_indices: original indices of candidates in the BM25 corpus
        """
        lang           = _detect_lang(query)
        is_non_english = lang != "en"
        query_lower    = query.lower()
        query_tokens   = set(query_lower.split())
        query_len      = len(query_tokens)

        if query_emb is None:
            query_emb = self.embedding_model.encode([query])

        if translated_emb is None:
            translated_query = _translate_query(query) if is_non_english else query
            translated_emb   = (
                self.embedding_model.encode([translated_query])
                if is_non_english else query_emb
            )

        # ── Product embeddings ────────────────────────────────────────
        titles = [p.get("title", "") for p in products]
        if prod_embs is None:
            prod_embs = self.embedding_model.encode(titles)

        # ── BM25: score query once against full corpus, slice by candidate positions ──
        full_bm25_scores = self.bm25.get_scores(query_lower.split())  # (corpus_size,)
        if candidate_indices is not None:
            bm25_per_title = full_bm25_scores[np.array(candidate_indices)]
        else:
            # fallback: re-rank by title position in corpus
            bm25_per_title = np.array([
                full_bm25_scores[i] for i in range(len(titles))
            ])

        # ── Cosine similarities ─────────────────────────────────────────
        semantic_sims      = cosine_similarity(query_emb, prod_embs)[0]
        cross_lingual_sims = (
            cosine_similarity(translated_emb, prod_embs)[0]
            if is_non_english else semantic_sims
        )

        features = []
        for i, prod in enumerate(products):
            title        = titles[i]
            brand        = prod.get("brand", "").lower()
            category     = prod.get("category", "").lower()
            title_tokens = set(title.lower().split())
            union        = query_tokens | title_tokens
            jaccard      = len(query_tokens & title_tokens) / len(union) if union else 0.0

            features.append([
                float(semantic_sims[i]),
                float(cross_lingual_sims[i]),
                float(bm25_per_title[i]),
                jaccard,
                1.0 if brand    and brand    in query_lower else 0.0,
                1.0 if category and category in query_lower else 0.0,
                1.0 if query_lower in title.lower() else 0.0,
                query_len,
            ])

        return np.array(features, dtype=np.float32)
