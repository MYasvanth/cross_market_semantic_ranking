"""Universal Feature Engineering — Semantic, Lexical, Entity, Cross-Lingual."""
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity
from langdetect import detect, LangDetectException
from src.embeddings.embedding_model import EmbeddingModel
# Single canonical tokenizer shared with retriever — eliminates IDF skew
from src.retrieval.retriever import _normalize_tokens as _bm25_tokenize

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
    """Basic whitespace tokenizer for token-set features (coverage, prefix, Jaccard).
    Intentionally different from _bm25_tokenize: these features measure surface
    overlap between raw query tokens and raw title tokens, not BM25 relevance.
    """
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
    Extracts 14 universal signals per query-product pair (or 15 with cross-encoder).

    Pre-computes catalog-level data (normalized brands, categories, title tokens)
    to avoid redundant work across 50k+ queries.
    """

    FEATURE_NAMES = [
        "semantic_sim",         # col 0  — cosine sim query vs product
        "cross_lingual_sim",    # col 1  — cosine sim translated query vs product
        "bm25_score",           # col 2  — BM25 normalized score
        "brand_match",          # col 3  — query contains product brand
        "category_match",       # col 4  — query contains product category
        "query_len",            # col 5  — number of query tokens
        "lexical_channel",      # col 6  — avg(bm25, token_coverage)
        "title_len_ratio",      # col 7  — title tokens / query tokens
        "prefix_match",         # col 8  — query tokens in title prefix
        "token_coverage",       # col 9  — query tokens found in title
        "semantic_rank",        # col 10 — log-rank: 1/(log1p(rank)+1), flatter than 1/(rank+1)
        "bm25_rank",            # col 11 — log-rank: 1/(log1p(rank)+1), flatter than 1/(rank+1)
        "title_brand_in_query", # col 12 — title word found in query
        # cols 13-14 only present when use_cross_encoder=True
        "ce_score",             # col 13 — cross-encoder sigmoid score
        "ce_evaluated",         # col 14 — 1 if CE was run on this candidate
    ]

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        bm25_tokenized_docs: list,
        use_cross_encoder: bool = False,
        cross_encoder_model_name: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
        ce_top_k: int = 50,
        bm25_norm_factor: float = 10.0,
    ):
        self.embedding_model = embedding_model
        normalized_docs = [_bm25_tokenize(" ".join(doc)) for doc in bm25_tokenized_docs]
        self.bm25 = BM25Okapi(normalized_docs)
        self.use_cross_encoder = use_cross_encoder
        self._ce_model_name = cross_encoder_model_name
        self._ce_top_k = ce_top_k
        self._bm25_norm_factor = bm25_norm_factor  # explicit param, not a hidden side-effect
        self._ce_model = None
        self._catalog_precomputed = False

    def _get_catalog_slice(self, candidate_indices, titles):
        """Return (brands, categories, titles_lower, title_token_sets) for candidates.
        Uses pre-computed catalog arrays when available, falls back to per-call compute.
        Centralises the fast-path/fallback branching that was duplicated 5x in extract_features.
        """
        from src.data.normalizer import normalize_entity
        if self._catalog_precomputed and candidate_indices is not None:
            return (
                self._brands_norm[candidate_indices],
                self._categories_norm[candidate_indices],
                self._titles_lower[candidate_indices],
                self._title_tokens[candidate_indices],
            )
        titles_lower = np.array([t.lower() for t in titles])
        brands       = np.array([normalize_entity((t or "").lower(), 'brand')    for t in titles_lower])
        categories   = np.array([normalize_entity((t or "").lower(), 'category') for t in titles_lower])
        token_sets   = np.empty(len(titles), dtype=object)
        for i, t in enumerate(titles_lower):
            token_sets[i] = set(t.split())
        return brands, categories, titles_lower, token_sets

    def _load_cross_encoder(self):
        """Lazily load cross-encoder model on first use."""
        if self._ce_model is None:
            from sentence_transformers import CrossEncoder
            self._ce_model = CrossEncoder(self._ce_model_name)
        return self._ce_model

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
        retrieval_ranks: np.ndarray = None,  # rank position from semantic retrieval
        bm25_ranks: np.ndarray = None,       # rank position from BM25
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

        if n_candidates == 0:
            if prod_embs is not None:
                n_candidates = prod_embs.shape[0]
            elif candidate_indices is not None:
                n_candidates = len(candidate_indices)
            elif bm25_scores is not None:
                n_candidates = len(bm25_scores)

        if prod_embs is None:
            prod_embs = self.embedding_model.encode(titles)

        # ── BM25 ──────────────────────────────────────────────────────────────
        if bm25_scores is not None:
            bm25_per_title = np.asarray(bm25_scores, dtype=np.float32)
        else:
            if self.bm25 is None:
                raise RuntimeError(
                    "bm25_scores must be provided when FeatureEngineer.bm25 is None."
                )
            q_tokens = _bm25_tokenize(query)
            if candidate_indices is not None:
                bm25_per_title = np.array(
                    self.bm25.get_batch_scores(q_tokens, candidate_indices), dtype=np.float32
                )
            else:
                bm25_per_title = np.array(
                    self.bm25.get_scores(q_tokens)[:n_candidates], dtype=np.float32
                )

        bm25_per_title = bm25_per_title / self._bm25_norm_factor

        # ── Cosine similarities ───────────────────────────────────────────────
        semantic_sims = cosine_similarity(
            query_emb.astype(np.float32), prod_embs.astype(np.float32)
        )[0].astype(np.float32)
        cross_lingual_sims = (
            cosine_similarity(
                translated_emb.astype(np.float32), prod_embs.astype(np.float32)
            )[0].astype(np.float32)
            if is_non_english else semantic_sims
        )

        # ── Entity + lexical features (single catalog slice call) ─────────────
        from src.data.normalizer import normalize_query
        norm_query = normalize_query(query_lower)

        cand_brands, cand_cats, cand_titles_lower, cand_token_sets = \
            self._get_catalog_slice(candidate_indices, titles)

        brand_match = np.fromiter(
            (1.0 if b and b in norm_query else 0.0 for b in cand_brands),
            dtype=np.float32, count=n_candidates
        )
        category_match = np.fromiter(
            (1.0 if c and c in norm_query else 0.0 for c in cand_cats),
            dtype=np.float32, count=n_candidates
        )
        title_brand_in_query = np.fromiter(
            (1.0 if any(w in norm_query for w in t.split() if len(w) > 3) else 0.0
             for t in cand_titles_lower),
            dtype=np.float32, count=n_candidates
        )
        # ── Cross-encoder
        # ce_score=0 is ambiguous: it could mean "irrelevant" or "not evaluated".
        # ce_evaluated=1 breaks that ambiguity so LambdaMART can learn separate
        # decision boundaries for scored vs unscored candidates.
        if self.use_cross_encoder:
            ce_model = self._load_cross_encoder()
            if candidate_indices is not None and self._catalog_precomputed:
                all_titles_ce = self._titles_lower[candidate_indices].tolist()
            else:
                all_titles_ce = titles

            top_k = min(getattr(self, '_ce_top_k', 50), n_candidates)
            top_k_idx = np.argpartition(-semantic_sims, top_k - 1)[:top_k]

            ce_pairs = [(query, all_titles_ce[i]) for i in top_k_idx]
            ce_scores_raw = ce_model.predict(ce_pairs, batch_size=64).astype(np.float32)
            ce_scores_sigmoid = (1.0 / (1.0 + np.exp(-ce_scores_raw))).astype(np.float32)

            ce_scores     = np.zeros(n_candidates, dtype=np.float32)
            ce_evaluated  = np.zeros(n_candidates, dtype=np.float32)
            ce_scores[top_k_idx]    = ce_scores_sigmoid
            ce_evaluated[top_k_idx] = 1.0
        else:
            ce_scores    = np.zeros(n_candidates, dtype=np.float32)
            ce_evaluated = np.zeros(n_candidates, dtype=np.float32)

        features = [
            semantic_sims.astype(np.float32),
            cross_lingual_sims.astype(np.float32),
            bm25_per_title.astype(np.float32),
            brand_match,
            category_match,
            np.full(n_candidates, query_len, dtype=np.float32),
        ]

        # ── Title-level features ──────────────────────────────────────────────
        t_lens = np.array([len(ts) for ts in cand_token_sets], dtype=np.float32)

        query_len_f     = float(max(query_len, 1))
        title_len_ratio = np.clip(t_lens / query_len_f, 0.0, 10.0).astype(np.float32)

        prefix_sets  = [set(t.split()[:5]) for t in cand_titles_lower]
        prefix_match = np.array(
            [len(query_tokens & ps) / query_len_f for ps in prefix_sets], dtype=np.float32
        )

        token_coverage = np.array(
            [len(query_tokens & ts) / query_len_f for ts in cand_token_sets], dtype=np.float32
        )

        # lexical_channel: bm25 + token_coverage (computed here after token_coverage is ready)
        lexical_channel = (bm25_per_title + token_coverage) / 2.0

        features += [lexical_channel.astype(np.float32), title_len_ratio, prefix_match, token_coverage]

        # ── Rank position features — log-rank transform ──────────────────
        # log1p compresses the top-rank advantage: rank=0->1.0, rank=9->0.41,
        # rank=99->0.20, rank=399->0.16. Much flatter than 1/(rank+1), forcing
        # the model to rely on semantic/lexical features to break ties instead
        # of memorizing retrieval order from training queries.
        if retrieval_ranks is not None:
            sem_rank_feat = 1.0 / (np.log1p(retrieval_ranks.astype(np.float32)) + 1.0)
        else:
            order = np.argsort(-semantic_sims)
            ranks = np.empty_like(order)
            ranks[order] = np.arange(len(order))
            sem_rank_feat = 1.0 / (np.log1p(ranks.astype(np.float32)) + 1.0)

        if bm25_ranks is not None:
            bm25_rank_feat = 1.0 / (np.log1p(bm25_ranks.astype(np.float32)) + 1.0)
        else:
            order = np.argsort(-bm25_per_title)
            ranks = np.empty_like(order)
            ranks[order] = np.arange(len(order))
            bm25_rank_feat = 1.0 / (np.log1p(ranks.astype(np.float32)) + 1.0)

        features += [sem_rank_feat, bm25_rank_feat, title_brand_in_query]
        # Only append CE features when cross-encoder is actually enabled (Root Cause 4 fix).
        # Dead zero columns add noise and consume feature slots without contributing signal.
        if self.use_cross_encoder:
            features.append(ce_scores)
            features.append(ce_evaluated)
        return np.column_stack(features)
