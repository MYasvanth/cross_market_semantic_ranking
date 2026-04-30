#!/usr/bin/env python3
"""Cross-Market Semantic Ranking — Inference Script (ONNX Runtime).

Loads artifacts produced by the training pipeline and serves a 3-stage
ranking pipeline via CLI or Python API:

    Stage 1: Hybrid retrieval (BM25 + FAISS) → top-K candidates
    Stage 2: Feature engineering (14 features)
    Stage 3: ONNX reranking + post-process guardrails
"""
import argparse
import logging
import pickle
import sys
from pathlib import Path
from typing import List, Optional

# Ensure project root is on path when running from inference/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import onnxruntime as ort
import pandas as pd

from src.embeddings.embedding_model import EmbeddingModel
from src.embeddings.vector_store import VectorStore
from src.features.feature_engineer import FeatureEngineer, _detect_lang, _translate_query

log = logging.getLogger(__name__)

ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"
CATALOG_PATH = ARTIFACTS_DIR / "catalog.pkl"
EMBEDDINGS_PATH = ARTIFACTS_DIR / "embeddings.npy"
ONNX_PATH = ARTIFACTS_DIR / "ranker.onnx"


class RankerPredictor:
    """End-to-end predictor: retrieval → features → ONNX reranking."""

    def __init__(
        self,
        catalog_path: Path = CATALOG_PATH,
        embeddings_path: Path = EMBEDDINGS_PATH,
        onnx_path: Path = ONNX_PATH,
        embedding_model_name: str = "intfloat/multilingual-e5-base",
        retrieval_k: int = 100,
    ):
        self.catalog_path = catalog_path
        self.embeddings_path = embeddings_path
        self.onnx_path = onnx_path
        self.embedding_model_name = embedding_model_name
        self.retrieval_k = retrieval_k

        self.session: Optional[ort.InferenceSession] = None
        self.products: Optional[pd.DataFrame] = None
        self.product_embs: Optional[np.ndarray] = None
        self.vector_store: Optional[VectorStore] = None
        self.feat_eng: Optional[FeatureEngineer] = None
        self.input_name: Optional[str] = None

    # ------------------------------------------------------------------ #
    # Artifact loading
    # ------------------------------------------------------------------ #

    def load_artifacts(self) -> "RankerPredictor":
        """Load ONNX session, catalog, embeddings, and build FAISS index."""
        # ONNX model — handle both .onnx and .onnx.zip
        onnx_path = self.onnx_path
        if not onnx_path.exists() and Path(str(onnx_path) + ".zip").exists():
            import zipfile, tempfile
            zip_path = Path(str(onnx_path) + ".zip")
            tmpdir = Path(tempfile.mkdtemp())
            zipfile.ZipFile(zip_path).extractall(tmpdir)
            extracted = list(tmpdir.glob("*.onnx"))
            if not extracted:
                raise FileNotFoundError(f"No .onnx file found inside {zip_path}")
            onnx_path = extracted[0]
            log.info(f"Extracted ONNX from zip: {onnx_path}")
        elif not onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

        self.session = ort.InferenceSession(str(onnx_path))
        self.input_name = self.session.get_inputs()[0].name
        log.info(f"ONNX model loaded: {self.onnx_path}")

        # 2. Product catalog
        with open(self.catalog_path, "rb") as f:
            self.products = pickle.load(f)
        log.info(f"Catalog loaded: {len(self.products)} products")

        # 3. Product embeddings
        self.product_embs = np.load(self.embeddings_path)
        log.info(f"Embeddings loaded: {self.product_embs.shape}")

        # 4. FAISS index (rebuild from saved embeddings)
        self.vector_store = VectorStore(dimension=self.product_embs.shape[1])
        self.vector_store.add(self.product_embs, ids=self.products.index.tolist())
        log.info(f"FAISS index rebuilt: {self.vector_store.ntotal} vectors")

        # 5. Embedding model + feature engineer
        embed_model = EmbeddingModel(self.embedding_model_name)
        bm25_docs = self.products["title"].str.split().tolist()
        self.feat_eng = FeatureEngineer(embed_model, bm25_docs)

        return self

    # ------------------------------------------------------------------ #
    # Core ranking API
    # ------------------------------------------------------------------ #

    def rank(self, query: str, top_k: int = 10) -> pd.DataFrame:
        """Run full two-stage ranking pipeline for a single query.

        Args:
            query: raw user query (any language)
            top_k: number of final results to return

        Returns:
            DataFrame with columns: [pid, title, brand, category, score]
            sorted by score descending.
        """
        if self.session is None:
            raise RuntimeError("Artifacts not loaded. Call load_artifacts() first.")

        # ── Stage 1: Hybrid Retrieval ────────────────────────────────────
        # Use direct FAISS search at inference — avoids BM25 fusion score inversion
        # (HNSW returns L2 distances; RRF fusion in HybridRetriever treats them as
        # higher=better, which inverts ranking. Direct FAISS search is correct.)
        query_emb = self.feat_eng.embedding_model.encode([query])
        faiss_scores, faiss_indices = self.vector_store.search(query_emb, self.retrieval_k)
        l2 = faiss_scores[0]
        cos_sims = np.clip(1.0 - (l2 ** 2) / 2.0, 0.0, 1.0)
        order = np.argsort(-cos_sims)
        candidate_indices = faiss_indices[0][order].tolist()
        if not candidate_indices:
            log.warning(f"No candidates retrieved for query: {query}")
            return pd.DataFrame(columns=["pid", "title", "brand", "category", "score"])

        # Ensure indices are valid
        candidate_indices = [
            i for i in candidate_indices if i < len(self.products)
        ]
        if not candidate_indices:
            log.warning(f"All retrieved indices out of range for query: {query}")
            return pd.DataFrame(columns=["pid", "title", "brand", "category", "score"])

        candidate_products = self.products.iloc[candidate_indices][
            ["title", "brand", "category"]
        ].to_dict("records")

        # ── Stage 2: Feature Engineering ─────────────────────────────────
        lang = _detect_lang(query)
        if lang != "en":
            translated = _translate_query(query)
            translated_emb = self.feat_eng.embedding_model.encode([translated])
        else:
            translated_emb = query_emb

        candidate_embs = self.product_embs[candidate_indices]

        X = self.feat_eng.extract_features(
            query=query,
            products=candidate_products,
            prod_embs=candidate_embs,
            query_emb=query_emb,
            translated_emb=translated_emb,
            candidate_indices=candidate_indices,
        )

        # ── Stage 3: ONNX Reranking ──────────────────────────────────────
        scores = self.session.run(None, {self.input_name: X.astype(np.float32)})[0].ravel()

        # ── Post-Process Guardrails ──────────────────────────────────────
        scores = self._apply_guardrails(
            scores, X, query, candidate_products, candidate_indices
        )

        # ── Assemble results ─────────────────────────────────────────────
        results = self.products.iloc[candidate_indices][
            ["pid", "title", "brand", "category"]
        ].copy()
        results["score"] = scores
        results = results.sort_values("score", ascending=False).reset_index(drop=True)

        # Drop candidates whose score is below the top-score gap threshold
        # (filters BM25 noise that the ONNX model already scored very low)
        top_score = results["score"].iloc[0]
        score_range = top_score - results["score"].iloc[-1]
        threshold = top_score - max(3.0, score_range * 0.4)
        results = results[results["score"] > threshold]

        return results.head(top_k)

    # ------------------------------------------------------------------ #
    # Guardrails (mirrors LambdaRanker.post_process)
    # ------------------------------------------------------------------ #

    def _apply_guardrails(
        self,
        scores: np.ndarray,
        X: np.ndarray,
        query: str,
        products: List[dict],
        candidate_indices: List[int],
    ) -> np.ndarray:
        """Apply deterministic guardrails after model scoring."""
        from src.data.normalizer import normalize_entity, normalize_query, KNOWN_BRANDS

        scores = scores.copy()
        norm_query = normalize_query(query)
        # Also check translated query for non-English brand detection
        from src.features.feature_engineer import _detect_lang, _translate_query
        if _detect_lang(query) != "en":
            norm_query_translated = normalize_query(_translate_query(query))
        else:
            norm_query_translated = norm_query

        # Rule 1: Brand Guardrail
        query_brand = None
        for b in KNOWN_BRANDS:
            if b in norm_query or b in norm_query_translated:
                query_brand = b
                break

        if query_brand is not None:
            non_brand_mask = X[:, 4] == 0.0  # brand_match column
            # Use median as threshold — works for any score scale
            threshold = np.median(scores)
            demote_mask = non_brand_mask & (scores < threshold)
            scores[demote_mask] *= 0.1

        # Rule 2: Category Hierarchy Guardrail
        primary_categories = {"laptops", "phones", "electronics", "shoes", "footwear", "clothing", "smartphone", "mobile", "laptop"}
        accessory_categories = {"accessories", "chargers", "cases", "cables", "covers"}

        # Detect query's target category from translated query
        # shoes/footwear are treated as the same category
        _CAT_ALIASES = {"footwear": "shoes", "sneakers": "shoes", "smartphone": "phones", "mobile": "phones", "laptop": "laptops", "notebook": "laptops"}
        query_category = None
        for cat in primary_categories:
            if cat in norm_query or cat in norm_query_translated:
                query_category = cat
                break

        if query_category is not None:
            def _norm_cat(raw):
                c = normalize_entity((raw or "").lower(), "category")
                return _CAT_ALIASES.get(c, c)

            target = _CAT_ALIASES.get(query_category, query_category)
            unknown_cat = {"us", "es", "jp", ""}

            # Demote off-category products — exempt on-category and unknown (ESCI locale)
            off_category_mask = np.array([
                _norm_cat(p.get("category")) not in (target, *unknown_cat)
                for p in products
            ])
            demote_cat_threshold = np.median(scores)
            scores[off_category_mask & (scores < demote_cat_threshold)] *= 0.05

            # Boost on-category products above the median
            on_category_mask = np.array([
                _norm_cat(p.get("category")) == target
                for p in products
            ])
            scores[on_category_mask] += abs(demote_cat_threshold) * 0.5

            # Also demote accessories below the weakest on-category product
            top10_idx = np.argsort(scores)[-10:]
            top10_products = [products[i] for i in top10_idx]
            top10_categories = [_norm_cat(p.get("category")) for p in top10_products]
            primary_in_top10 = any(c == target for c in top10_categories)
            if primary_in_top10:
                primary_scores = [
                    scores[top10_idx[i]]
                    for i, c in enumerate(top10_categories)
                    if c == target
                ]
                min_primary_score = min(primary_scores)
                score_gap = abs(scores.max() - scores.min()) * 0.01 or 0.01
                for i, idx in enumerate(top10_idx):
                    if top10_categories[i] in accessory_categories:
                        scores[idx] = min(scores[idx], min_primary_score - score_gap)

        return scores

    # ------------------------------------------------------------------ #
    # Batch / convenience API
    # ------------------------------------------------------------------ #

    def predict(self, query: str, top_k: int = 10) -> pd.DataFrame:
        """Alias for rank()."""
        return self.rank(query, top_k=top_k)


# ---------------------------------------------------------------------- #
# CLI
# ---------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(
        description="Cross-Market Semantic Ranking — ONNX Inference"
    )
    parser.add_argument("query", type=str, help="User query string")
    parser.add_argument(
        "-k", "--top-k", type=int, default=10, help="Number of results (default: 10)"
    )
    parser.add_argument(
        "--catalog", type=Path, default=CATALOG_PATH, help="Path to catalog.pkl"
    )
    parser.add_argument(
        "--embeddings", type=Path, default=EMBEDDINGS_PATH, help="Path to embeddings.npy"
    )
    parser.add_argument(
        "--onnx", type=Path, default=ONNX_PATH, help="Path to ranker.onnx"
    )
    parser.add_argument(
        "--retrieval-k", type=int, default=100, help="Stage-1 retrieval candidates"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug logging"
    )
    parser.add_argument(
        "--full-eval", type=Path, default=None,
        metavar="TRAIN_DF_PKL",
        help="Run full-catalog NDCG eval. Pass path to a pickled val_df DataFrame."
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
    )

    predictor = RankerPredictor(
        catalog_path=args.catalog,
        embeddings_path=args.embeddings,
        onnx_path=args.onnx,
        retrieval_k=args.retrieval_k,
    )
    predictor.load_artifacts()

    # ── Full-catalog eval (unbiased NDCG) ────────────────────────────────
    if args.full_eval:
        import pickle
        from src.ranking.evaluator import full_catalog_eval
        from src.ranking.ranker import LambdaRanker
        from src.config import RankerConfig

        with open(args.full_eval, "rb") as f:
            val_df = pickle.load(f)

        # Wrap ONNX session as a ranker-compatible object
        class _OnnxRanker:
            def __init__(self, session, input_name):
                self.session = session
                self.input_name = input_name
            def predict(self, X):
                return self.session.run(None, {self.input_name: X.astype(np.float32)})[0].ravel()

        onnx_ranker = _OnnxRanker(predictor.session, predictor.input_name)
        result = full_catalog_eval(
            ranker=onnx_ranker,
            val_df=val_df,
            products=predictor.products,
            product_embs=predictor.product_embs,
            feat_eng=predictor.feat_eng,
            vector_store=predictor.vector_store,
        )
        print("\nFull-Catalog Evaluation (unbiased):")
        for k, v in result.items():
            print(f"  {k}: {v}")
        return

    results = predictor.rank(args.query, top_k=args.top_k)

    print(f"\nQuery: '{args.query}'")
    print("=" * 80)
    print(results.to_string(index=False).encode("utf-8", errors="replace").decode("utf-8"))
    print("=" * 80)


if __name__ == "__main__":
    main()
