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
    ):
        self.embedding_model = embedding_model
        # Use canonical _bm25_tokenize (stopword-stripped) so IDF weights match
        # the retriever's BM25 index — eliminates train/serve tokenizer skew.
        normalized_docs = [_bm25_tokenize(" ".join(doc)) for doc in bm25_tokenized_docs]
        self.bm25 = BM25Okapi(normalized_docs)
        self.use_cross_encoder = use_cross_encoder
        self._ce_model_name = cross_encoder_model_name
        self._ce_top_k = ce_top_k
        self._ce_model = None
        self._catalog_precomputed = False

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
        
        # If products list is empty, infer count from pre-computed embeddings or indices
        if n_candidates == 0:
            if prod_embs is not None:
                n_candidates = prod_embs.shape[0]
            elif candidate_indices is not None:
                n_candidates = len(candidate_indices)
            elif bm25_scores is not None:
                n_candidates = len(bm25_scores)

        if prod_embs is None:
            prod_embs = self.embedding_model.encode(titles)

        # ── BM25: score candidates using canonical tokenizer ──────────────────────
        # bm25_scores takes priority (pre-computed in main process for training).
        # Falls back to live scoring only at inference time (full_catalog_eval).
        # Both paths use _bm25_tokenize for consistent IDF weights.
        if bm25_scores is not None:
            bm25_per_title = np.asarray(bm25_scores, dtype=np.float32)
        else:
            if self.bm25 is None:
                raise RuntimeError(
                    "bm25_scores must be provided when FeatureEngineer.bm25 is None. "
                    "Pass pre-computed scores from the main process."
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

        # Normalize BM25 scores — factor configurable via RankerConfig.bm25_norm_factor
        bm25_per_title = bm25_per_title / getattr(self, '_bm25_norm_factor', 10.0)

        # ── Cosine similarities ─────────────────────────────────────────
        # Keep float32 throughout — sklearn cosine_similarity upcasts to float64
        # internally, doubling per-worker memory. Explicit float32 cast prevents this.
        semantic_sims      = cosine_similarity(
            query_emb.astype(np.float32), prod_embs.astype(np.float32)
        )[0].astype(np.float32)
        cross_lingual_sims = (
            cosine_similarity(
                translated_emb.astype(np.float32), prod_embs.astype(np.float32)
            )[0].astype(np.float32)
            if is_non_english else semantic_sims
        )

        # ── Vectorized entity + lexical features ────────────────────────────────
        from src.data.normalizer import normalize_entity, normalize_query
        norm_query = normalize_query(query_lower)

        if self._catalog_precomputed and candidate_indices is not None:
            # Fast path: use pre-computed catalog data with fully vectorized ops.
            # brand_norm and category_norm come from product titles/metadata only —
            # never from training labels — so these features are leakage-free at
            # both train and inference time.
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
        else:
            # Fallback: compute per-product (slower, for backward compat)
            brands = np.array([normalize_entity((p.get("brand")    or "").lower(), 'brand')    for p in products])
            categories = np.array([normalize_entity((p.get("category") or "").lower(), 'category') for p in products])
            titles_lower = np.array([t.lower() for t in titles])

            brand_match    = np.array([1.0 if b and b in norm_query else 0.0 for b in brands],      dtype=np.float32)
            category_match = np.array([1.0 if c and c in norm_query else 0.0 for c in categories],  dtype=np.float32)

        if self._catalog_precomputed and candidate_indices is not None:
            title_brand_in_query = np.fromiter(
                (1.0 if any(w in norm_query for w in t.split() if len(w) > 3) else 0.0
                 for t in self._titles_lower[candidate_indices]),
                dtype=np.float32, count=n_candidates
            )
        else:
            title_brand_in_query = np.fromiter(
                (1.0 if any(w in norm_query for w in t.lower().split() if len(w) > 3) else 0.0
                 for t in titles),
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

        # ── Title-level features (3 new signals) ──────────────────────────
        if self._catalog_precomputed and candidate_indices is not None:
            t_lens = np.array([len(ts) for ts in self._title_tokens[candidate_indices]], dtype=np.float32)
        else:
            t_lens = np.array([len(set(t.split())) for t in titles], dtype=np.float32)

        query_len_f = float(max(query_len, 1))
        # title_len_ratio: how much longer is the title vs query (penalizes very long titles)
        title_len_ratio = np.clip(t_lens / query_len_f, 0.0, 10.0).astype(np.float32)

        # prefix_match: fraction of query tokens that appear in title prefix (first 5 tokens)
        if self._catalog_precomputed and candidate_indices is not None:
            prefix_sets = [set(t.split()[:5]) for t in self._titles_lower[candidate_indices]]
        else:
            prefix_sets = [set(t.lower().split()[:5]) for t in titles]
        prefix_match = np.array(
            [len(query_tokens & ps) / query_len_f for ps in prefix_sets], dtype=np.float32
        )

        # token_coverage: fraction of query tokens covered by title tokens
        if self._catalog_precomputed and candidate_indices is not None:
            cand_sets = self._title_tokens[candidate_indices]
        else:
            cand_sets = [set(t.lower().split()) for t in titles]
        token_coverage = np.array(
            [len(query_tokens & ts) / query_len_f for ts in cand_sets], dtype=np.float32
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
