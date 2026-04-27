"""Universal Feature Engineering — Semantic, Lexical, Entity, Cross-Lingual."""
import numpy as np
import pandas as pd
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


def _tokenize(text: str) -> list[str]:
    """Basic preprocessing for lexical features."""
    return text.lower().split()


def classify_intent(query: str) -> str:
    """
    Lightweight rule-based intent classifier.

    Returns:
        'BRAND'   — query contains a known brand name.
        'SKU'     — query looks like a product code / model number.
        'GENERIC' — everything else (category or descriptive queries).
    """
    import re
    from src.data.normalizer import normalize_query, KNOWN_BRANDS
    q = normalize_query(query)
    # BRAND: known brand token present
    if any(b in q for b in KNOWN_BRANDS):
        return "BRAND"
    # SKU: alphanumeric model patterns (e.g., "iPhone 15", "XPS 13", "WH-1000XM5")
    if re.search(r'\b[A-Z]+[-]?\d{2,}[A-Z0-9]*\b', query) or re.search(r'\b\d{4}[A-Z]+\b', query):
        return "SKU"
    return "GENERIC"


class FeatureEngineer:
    """
    Extracts 14 universal signals per query-product pair.

    Pre-computes catalog-level data (normalized brands, categories, title tokens)
    to avoid redundant work across 50k+ queries.
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
        "intent_brand_weight",
        "intent_sku_weight",
        "intent_generic_weight",
        "semantic_channel",
        "lexical_channel",
        "constraint_channel",
    ]

    def __init__(self, embedding_model: EmbeddingModel, bm25_tokenized_docs: list):
        self.embedding_model = embedding_model
        self.bm25 = BM25Okapi(bm25_tokenized_docs)
        # Pre-compute catalog data for vectorized feature extraction
        self._catalog_precomputed = False

    def precompute_catalog(self, products_df: pd.DataFrame):
        """
        Pre-compute normalized brands, categories, title tokens, and token sets
        once for the entire catalog. Call this before processing queries.

        Args:
            products_df: DataFrame with columns [pid, title, brand, category]
        """
        from src.data.normalizer import normalize_entity
        import pandas as pd

        self.products_df = products_df.copy()
        self.n_products = len(products_df)

        # Vectorized normalization
        self._titles_lower = products_df["title"].str.lower().values
        self._brands_norm = products_df["brand"].fillna("").str.lower().apply(
            lambda x: normalize_entity(x, "brand")
        ).values
        self._categories_norm = products_df["category"].fillna("").str.lower().apply(
            lambda x: normalize_entity(x, "category")
        ).values

        # Pre-tokenize titles for Jaccard — store as NumPy object array for vectorized access
        self._title_tokens = np.empty(len(products_df), dtype=object)
        for i, t in enumerate(self._titles_lower):
            self._title_tokens[i] = set(t.split())

        # Pre-compute title token lengths for vectorized Jaccard
        self._title_token_lengths = np.array(
            [len(ts) for ts in self._title_tokens], dtype=np.float32
        )

        self._catalog_precomputed = True

    def extract_features(
        self,
        query: str,
        products: list,
        prod_embs: np.ndarray = None,
        query_emb: np.ndarray = None,
        translated_emb: np.ndarray = None,
        candidate_indices: list = None,
        bm25_scores: np.ndarray = None,
    ) -> np.ndarray:
        """
        Extract 14 features for every (query, product) pair.

        Args:
            query:             raw user query in any language
            products:          list of dicts with keys: title, brand, category
            prod_embs:         pre-computed product embeddings (n, 768)
            query_emb:         pre-computed query embedding (1, 768)
            translated_emb:    pre-computed translated query embedding (1, 768)
            candidate_indices: original indices of candidates in the BM25 corpus
            bm25_scores:       pre-computed BM25 scores for candidates (optional, skips BM25 call)
        """
        lang           = _detect_lang(query)
        is_non_english = lang != "en"
        query_lower    = query.lower()
        query_tokens   = set(_tokenize(query))
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
        titles = [(p.get("title") or "") for p in products]
        n_candidates = len(products)
        if prod_embs is None:
            prod_embs = self.embedding_model.encode(titles)

        # ── BM25: score only the k candidates using global IDF ──────────────
        if bm25_scores is not None:
            bm25_per_title = bm25_scores
        else:
            query_tokens_list = _tokenize(query)
            if candidate_indices is not None:
                bm25_per_title = np.array(
                    self.bm25.get_batch_scores(query_tokens_list, candidate_indices)
                )
            else:
                bm25_per_title = self.bm25.get_scores(query_tokens_list)[:n_candidates]

        # Normalize BM25 scores (stable scaling to [0, ~1-2] range)
        bm25_per_title = bm25_per_title / 10.0

        # ── Cosine similarities ─────────────────────────────────────────
        semantic_sims      = cosine_similarity(query_emb, prod_embs)[0]
        cross_lingual_sims = (
            cosine_similarity(translated_emb, prod_embs)[0]
            if is_non_english else semantic_sims
        )

        # ── Vectorized entity + lexical features ────────────────────────
        from src.data.normalizer import normalize_entity, normalize_query
        norm_query = normalize_query(query_lower)

        if self._catalog_precomputed and candidate_indices is not None:
            # Fast path: use pre-computed catalog data with fully vectorized ops
            cand_brands = self._brands_norm[candidate_indices]
            cand_cats   = self._categories_norm[candidate_indices]
            cand_titles = self._titles_lower[candidate_indices]

            # Fully vectorized brand/category/exact match via NumPy broadcasting
            brand_match = np.fromiter(
                (1.0 if b and b in norm_query else 0.0 for b in cand_brands),
                dtype=np.float32, count=n_candidates
            )
            category_match = np.fromiter(
                (1.0 if c and c in norm_query else 0.0 for c in cand_cats),
                dtype=np.float32, count=n_candidates
            )
            exact_match = np.fromiter(
                (1.0 if norm_query in t else 0.0 for t in cand_titles),
                dtype=np.float32, count=n_candidates
            )

            # Vectorized Jaccard via pre-computed token sets
            query_len_f = float(len(query_tokens))
            cand_token_sets = self._title_tokens[candidate_indices]
            unions = np.fromiter(
                (len(query_tokens | ts) for ts in cand_token_sets),
                dtype=np.float32, count=n_candidates
            )
            intersect = np.fromiter(
                (len(query_tokens & ts) for ts in cand_token_sets),
                dtype=np.float32, count=n_candidates
            )
            jaccard = np.where(unions > 0, intersect / unions, 0.0).astype(np.float32)
        else:
            # Fallback: compute per-product (slower, for backward compat)
            brands = np.array([normalize_entity((p.get("brand")    or "").lower(), 'brand')    for p in products])
            categories = np.array([normalize_entity((p.get("category") or "").lower(), 'category') for p in products])
            titles_lower = np.array([t.lower() for t in titles])

            brand_match = np.array([1.0 if b and b in norm_query else 0.0 for b in brands],     dtype=np.float32)
            category_match = np.array([1.0 if c and c in norm_query else 0.0 for c in categories], dtype=np.float32)
            exact_match = np.array([1.0 if norm_query in t else 0.0 for t in titles_lower],      dtype=np.float32)

            title_token_sets = [set(t.split()) for t in titles_lower]
            unions   = np.array([len(query_tokens | ts) for ts in title_token_sets], dtype=np.float32)
            intersect = np.array([len(query_tokens & ts) for ts in title_token_sets], dtype=np.float32)
            jaccard  = np.where(unions > 0, intersect / unions, 0.0).astype(np.float32)

        # ── Intent weights ─────────────────────────────────────────────
        intent = classify_intent(query)
        intent_brand  = 1.0 if intent == "BRAND"  else 0.0
        intent_sku    = 1.0 if intent == "SKU"    else 0.0
        intent_generic = 1.0 if intent == "GENERIC" else 0.0

        # ── Three-Channel Aggregates ───────────────────────────────────
        semantic_channel   = (semantic_sims + cross_lingual_sims) / 2.0
        lexical_channel    = (bm25_per_title + jaccard + exact_match) / 3.0
        constraint_channel = (brand_match + category_match + np.full(n_candidates, intent_brand, dtype=np.float32)) / 3.0

        return np.column_stack([
            semantic_sims.astype(np.float32),
            cross_lingual_sims.astype(np.float32),
            bm25_per_title.astype(np.float32),
            jaccard,
            brand_match,
            category_match,
            exact_match,
            np.full(n_candidates, query_len, dtype=np.float32),
            np.full(n_candidates, intent_brand,  dtype=np.float32),
            np.full(n_candidates, intent_sku,    dtype=np.float32),
            np.full(n_candidates, intent_generic, dtype=np.float32),
            semantic_channel.astype(np.float32),
            lexical_channel.astype(np.float32),
            constraint_channel.astype(np.float32),
        ])
