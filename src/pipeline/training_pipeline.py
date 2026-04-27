"""ZenML Pipeline — Cross-Market Semantic Ranking."""
import numpy as np
import pandas as pd
import logging
from typing import List, Tuple, Annotated

from zenml import step, pipeline

from src.config import PipelineConfig
from src.data.data_generator import DataGenerator
from src.embeddings.embedding_model import EmbeddingModel
from src.embeddings.vector_store import VectorStore
from src.features.feature_engineer import FeatureEngineer, _translate_query, _detect_lang
from src.ranking.ranker import LambdaRanker
from src.ranking.evaluator import ablation_study
from src.ranking.materializer import LambdaRankerMaterializer

log = logging.getLogger(__name__)


# ── Step 1: Data Ingestion ─────────────────────────────────────────────────

@step(enable_cache=True)
def ingest_data(
    cfg: PipelineConfig,
) -> Annotated[pd.DataFrame, "train_df"]:
    config = {
        "num_products":            cfg.data.num_products,
        "queries_per_product":     cfg.data.queries_per,
        "use_esci":                cfg.data.use_esci,
        "esci_max_rows":           cfg.data.esci_max_rows,
        "categories":              cfg.data.categories,
        "brands":                  cfg.data.brands,
        # Augmentation config
        "use_augmentation":        cfg.data.use_augmentation,
        "use_llm":                 cfg.data.use_llm,
        "grok_api_key":            cfg.data.grok_api_key,
        "grok_api_endpoint":       cfg.data.grok_api_endpoint,
        "llm_model_name":          cfg.data.llm_model_name,
        "augmentation_cache_path": cfg.data.augmentation_cache_path,
        "hard_negative_ratio":     cfg.data.hard_negative_ratio,
        "attribute_noise_ratio":   cfg.data.attribute_noise_ratio,
        "synonym_injection_ratio": cfg.data.synonym_injection_ratio,
        "seed":                    cfg.data.seed,
    }
    generator = DataGenerator(config)
    chunks = []
    for chunk in generator.generate(use_esci=cfg.data.use_esci):
        chunks.append(chunk)
        log.info(f"Chunk loaded: {len(chunk)} rows")
    df = pd.concat(chunks, ignore_index=True)
    # ── Input validation ───────────────────────────────────────────────────
    required_cols = {"qid", "pid", "query", "product_title", "brand", "category", "relevance"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")
    if df.empty:
        raise ValueError("Dataset is empty after ingestion")
    if not df["relevance"].between(0, 3).all():
        raise ValueError("relevance scores must be in range [0, 3]")
    str_cols = ["qid", "pid", "query", "product_title", "brand", "category"]
    for col in str_cols:
        df[col] = df[col].astype(str).str.slice(0, 512)
    log.info(f"Total rows: {len(df)}")
    rel_dist = df['relevance'].value_counts().sort_index()
    log.info(f"Relevance distribution: {dict(rel_dist)}")
    log.info(f"Unique queries: {df['qid'].nunique()}, Unique products: {df['pid'].nunique()}")
    return df


# ── Step 2: Build Embeddings + Vector Store ────────────────────────────────

@step(enable_cache=True)
def build_embeddings(
    train_df: pd.DataFrame,
    cfg: PipelineConfig,
) -> Tuple[
    Annotated[np.ndarray, "product_embs"],
    Annotated[pd.DataFrame, "products"],
]:
    embed_model = EmbeddingModel(cfg.embedding_model_name)
    products = (
        train_df[["pid", "product_title", "brand", "category"]]
        .drop_duplicates("pid")
        .rename(columns={"product_title": "title"})
        .reset_index(drop=True)
    )
    log.info(f"Catalog size: {len(products)} products")
    product_embs = embed_model.encode(products["title"].tolist())
    log.info(f"Encoded {len(products)} products -> shape {product_embs.shape}")

    # Save for predict.py
    import pickle
    from pathlib import Path
    artifacts = Path("artifacts")
    artifacts.mkdir(exist_ok=True)
    with open(artifacts / "catalog.pkl", "wb") as f:
        pickle.dump(products, f)
    np.save(artifacts / "embeddings.npy", product_embs)
    log.info(f"Saved catalog.pkl and embeddings.npy to artifacts/")

    return product_embs, products


# ── Step 3: Feature Engineering ────────────────────────────────────────────

@step(enable_cache=True)
def build_features(
    train_df: pd.DataFrame,
    products: pd.DataFrame,
    product_embs: np.ndarray,
    cfg: PipelineConfig,
) -> Tuple[
    Annotated[np.ndarray, "X"],
    Annotated[np.ndarray, "y"],
    Annotated[List[int], "groups"],
]:
    embed_model = EmbeddingModel(cfg.embedding_model_name)
    vector_store = VectorStore()
    vector_store.add(product_embs, ids=products.index.tolist())

    bm25_docs = products["title"].str.split().tolist()
    feat_eng  = FeatureEngineer(embed_model, bm25_docs)
    feat_eng.precompute_catalog(products)  # pre-normalize once for 50k+ queries

    # Batch encode all unique queries
    unique_queries   = train_df[["qid", "query"]].drop_duplicates("qid")
    q_texts          = unique_queries["query"].tolist()
    all_query_embs   = embed_model.encode(q_texts)

    # Only re-encode non-English queries for cross-lingual sim — reuse query embs for English
    non_en_mask      = [_detect_lang(q) != "en" for q in q_texts]
    non_en_indices   = [i for i, m in enumerate(non_en_mask) if m]
    log.info(f"Non-English queries: {len(non_en_indices)} / {len(q_texts)} — skipping re-encode for the rest")

    all_translated_embs = all_query_embs.copy()
    if non_en_indices:
        non_en_translated = [_translate_query(q_texts[i]) for i in non_en_indices]
        all_translated_embs[non_en_indices] = embed_model.encode(non_en_translated)

    # ── Stage 1 Retrieval: Hybrid or Pure Semantic ───────────────────────
    retrieval_k = cfg.data.retrieval_k
    use_hybrid = cfg.data.use_hybrid_retrieval

    if use_hybrid:
        from src.retrieval.retriever import HybridRetriever
        retriever = HybridRetriever(
            embed_model, vector_store, bm25_docs,
            rrf_k=cfg.data.rrf_k,
            semantic_weight=cfg.data.semantic_weight,
            bm25_weight=cfg.data.bm25_weight,
        )
        log.info(f"Using HybridRetriever (k={retrieval_k}, rrf_k={cfg.data.rrf_k})")
    else:
        from src.retrieval.retriever import SemanticRetriever
        retriever = SemanticRetriever(embed_model, vector_store)
        log.info(f"Using SemanticRetriever (k={retrieval_k})")

    # Pre-compute retrieval results for all unique queries
    query_lookup = {}
    for i, row in enumerate(unique_queries.to_dict("records")):
        qid = row["qid"]
        query_text = row["query"]

        if use_hybrid:
            # HybridRetriever returns (doc_idx, fused_score) sorted by score desc
            hybrid_results = retriever.retrieve(query_text, top_k=retrieval_k)
            pids = [r[0] for r in hybrid_results]
            scores = [r[1] for r in hybrid_results]
        else:
            # Pure semantic: FAISS returns L2 distances
            scores_arr, pids_arr = vector_store.search(all_query_embs[i:i+1], k=retrieval_k)
            pids = pids_arr[0].tolist()
            scores = scores_arr[0].tolist()

        query_lookup[qid] = {
            "emb":            all_query_embs[i].reshape(1, -1),
            "translated_emb": all_translated_embs[i].reshape(1, -1),
            "pids":           pids,
            "scores":         scores,
        }

    pid_to_idx = {pid: idx for idx, pid in enumerate(products["pid"])}

    # Debug: check pid overlap between train_df and products catalog
    train_pids    = set(train_df["pid"].unique())
    catalog_pids  = set(pid_to_idx.keys())
    overlap       = train_pids & catalog_pids
    log.info(f"PID overlap: {len(overlap)} / {len(train_pids)} train pids found in catalog")
    rel_dist = dict(train_df['relevance'].value_counts().sort_index())
    log.info(f"Relevance distribution: {rel_dist}")
    if set(rel_dist.keys()) == {0}:
        raise ValueError(
            "All relevance scores are 0 — ESCI label mapping failed. "
            "Check that 'esci_label' or 'label' field exists in the dataset."
        )

    X_all, y_all, groups = [], [], []
    skipped = 0

    from tqdm import tqdm
    for qid, qgroup in tqdm(train_df.groupby("qid"), desc="Extracting features", total=train_df["qid"].nunique()):
        query          = qgroup["query"].iloc[0]
        lookup         = query_lookup[qid]
        q_emb          = lookup["emb"]
        t_emb          = lookup["translated_emb"]
        retrieved_pids_int = lookup["pids"]
        retrieved_scores   = lookup["scores"]

        # Labeled pids that exist in catalog
        labeled_pids   = [p for p in qgroup["pid"].unique() if p in pid_to_idx]
        labeled_set    = set(labeled_pids)

        # Filter retrieved candidates
        if use_hybrid:
            # Hybrid: fused scores — higher is better. Use percentile threshold.
            if len(retrieved_scores) > 0:
                score_threshold = np.percentile(retrieved_scores, 25)  # Keep top 75%
            else:
                score_threshold = 0.0
            retrieved_candidates = [
                (products.iloc[i]["pid"], s)
                for i, s in zip(retrieved_pids_int, retrieved_scores)
                if s >= score_threshold and i < len(products)
            ]
        else:
            # Pure semantic: FAISS returns L2 distances
            retrieved_candidates = [
                (products.iloc[i]["pid"], s)
                for i, s in zip(retrieved_pids_int, retrieved_scores) if s < 1.0
            ]

        # ── Hard-Negative Mining ──────────────────────────────────────
        # Select "near misses": high retrieval score but brand does NOT match
        from src.data.normalizer import normalize_entity, normalize_query
        norm_query = normalize_query(query)
        hard_neg_pids = []
        random_extra  = []

        if use_hybrid:
            # Hybrid: top-scoring non-brand matches are hard negatives
            for pid, score in retrieved_candidates:
                if pid in labeled_set:
                    continue
                prod = products.iloc[pid_to_idx[pid]]
                prod_brand = normalize_entity((prod.get("brand") or "").lower(), 'brand')
                is_brand_match = prod_brand and prod_brand in norm_query
                # Top 30% scores that don't brand-match = hard negatives
                score_percentile = score / max(retrieved_scores[0], 1e-6) if retrieved_scores else 0
                if score_percentile > 0.7 and not is_brand_match:
                    hard_neg_pids.append(pid)
                else:
                    random_extra.append(pid)
        else:
            # Pure semantic: L2 < 0.8 ≈ cos > 0.68 but no brand match
            for pid, dist in retrieved_candidates:
                if pid in labeled_set:
                    continue
                prod = products.iloc[pid_to_idx[pid]]
                prod_brand = normalize_entity((prod.get("brand") or "").lower(), 'brand')
                is_brand_match = prod_brand and prod_brand in norm_query
                if dist < 0.8 and not is_brand_match:
                    hard_neg_pids.append(pid)
                else:
                    random_extra.append(pid)

        # Compose candidate list: labeled + hard-negs (priority) + random filler
        candidate_pids = (labeled_pids + hard_neg_pids[:10] + random_extra)[:retrieval_k]
        pids_int       = [pid_to_idx[p] for p in candidate_pids if p in pid_to_idx]
        if hard_neg_pids:
            log.debug(f"Query '{query[:40]}': {len(hard_neg_pids)} hard negatives selected")

        if len(pids_int) < 2:
            skipped += 1
            continue

        candidate_products = products.iloc[pids_int][["title", "brand", "category"]].to_dict("records")
        candidate_embs     = product_embs[pids_int]

        X_group = feat_eng.extract_features(
            query, candidate_products,
            prod_embs=candidate_embs, query_emb=q_emb,
            translated_emb=t_emb, candidate_indices=pids_int
        )

        labeled_relevance = qgroup.groupby("pid")["relevance"].max()
        y_group = labeled_relevance.reindex(candidate_pids, fill_value=0).values[:len(pids_int)]

        if labeled_relevance.max() == 0:
            skipped += 1
            continue

        X_all.append(X_group)
        y_all.append(y_group)
        groups.append(len(y_group))

    log.info(f"Valid groups: {len(groups)} | Skipped: {skipped} / {train_df['qid'].nunique()}")
    if not X_all:
        raise ValueError("No valid query groups — check ESCI pids exist in product catalog.")
    return np.vstack(X_all), np.hstack(y_all), groups


# ── Step 4: Train Ranker ───────────────────────────────────────────────────

@step(output_materializers={"ranker": LambdaRankerMaterializer}, enable_cache=True)
def train_ranker(
    X: np.ndarray,
    y: np.ndarray,
    groups: List[int],
    cfg: PipelineConfig,
) -> Annotated[LambdaRanker, "ranker"]:
    ranker = LambdaRanker(cfg.ranker)
    X_val, y_val, group_val = ranker.fit(X, y, groups)

    ranker.save_model("artifacts")
    ranker.export_onnx("artifacts", num_features=X.shape[1])

    from src.ranking.evaluator import ablation_study
    import pandas as pd

    raw_scores = ranker.predict(X_val)
    feature_columns = [
        'semantic_sim', 'cross_lingual_sim', 'bm25_score', 'jaccard',
        'brand_match', 'category_match', 'exact_title_match', 'query_len',
        'intent_brand_weight', 'intent_sku_weight', 'intent_generic_weight',
        'semantic_channel', 'lexical_channel', 'constraint_channel',
    ]
    eval_df = pd.DataFrame(X_val, columns=feature_columns)
    eval_df['relevance'] = y_val
    # Note: post_process needs per-query products; for val eval we use raw scores
    # Guardrails are applied at inference time in the serving layer.
    eval_df['ranker_score'] = raw_scores

    # Log feature statistics for debugging
    log.info(f"Feature means: {X_val.mean(axis=0)}")
    log.info(f"Feature stds: {X_val.std(axis=0)}")
    log.info(f"Relevance distribution in val: {np.bincount(y_val.astype(int))}")

    metrics_df = ablation_study(eval_df, group_val)
    log.info(f"Validation Metrics:\n{metrics_df.to_string()}")

    try:
        import mlflow
        import mlflow.lightgbm
        from pathlib import Path
        mlflow.set_tracking_uri(f"sqlite:///{Path(__file__).resolve().parents[2] / 'mlflow.db'}")
        mlflow.set_experiment("cross_market_semantic_ranking")
        with mlflow.start_run():
            mlflow.log_params({
                'num_boost_round':   cfg.ranker.num_boost_round,
                'learning_rate':     cfg.ranker.learning_rate,
                'num_leaves':        cfg.ranker.num_leaves,
                'objective':         cfg.ranker.objective,
                'best_iteration':    ranker.model.best_iteration,
            })
            # Ablation metrics — NDCG + MRR for BM25 / Semantic / Full
            flat_metrics = {}
            for row in metrics_df.itertuples():
                flat_metrics[f"{row.Index}_ndcg"] = row.ndcg
                flat_metrics[f"{row.Index}_mrr"]  = row.mrr
            mlflow.log_metrics(flat_metrics)
            # Model artifact
            mlflow.lightgbm.log_model(ranker.model, "lambda_ranker")
        log.info("MLflow run logged successfully")
    except Exception as e:
        log.warning(f"MLflow logging skipped: {e}")

    return ranker




# ── Step 5: Evaluate ───────────────────────────────────────────────────────

@step(enable_cache=False)
def evaluate(
    ranker: LambdaRanker,
    X_val: np.ndarray,
    y_val: np.ndarray,
    group_val: List[int],
) -> Annotated[pd.DataFrame, "eval_results"]:
    raw_scores = ranker.predict(X_val)
    eval_df = pd.DataFrame(X_val, columns=[
        "semantic_sim", "cross_lingual_sim", "bm25_score", "jaccard",
        "brand_match", "category_match", "exact_title_match", "query_len",
        "intent_brand_weight", "intent_sku_weight", "intent_generic_weight",
        "semantic_channel", "lexical_channel", "constraint_channel",
    ])
    eval_df["relevance"]    = y_val
    eval_df["ranker_score"] = raw_scores
    results = ablation_study(eval_df, group_val)
    print(results)
    return results


# ── ZenML Pipeline ─────────────────────────────────────────────────────────

@pipeline(name="cross_market_semantic_ranking")
def ranking_pipeline(cfg: PipelineConfig = PipelineConfig()):
    train_df               = ingest_data(cfg)
    product_embs, products = build_embeddings(train_df, cfg)
    X, y, groups           = build_features(train_df, products, product_embs, cfg)
    ranker                 = train_ranker(X, y, groups, cfg)
