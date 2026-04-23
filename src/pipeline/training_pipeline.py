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

@step(enable_cache=False)
def ingest_data(
    cfg: PipelineConfig,
) -> Annotated[pd.DataFrame, "train_df"]:
    config = {
        "num_products":        cfg.data.num_products,
        "queries_per_product": cfg.data.queries_per,
        "use_esci":            cfg.data.use_esci,
        "esci_max_rows":       cfg.data.esci_max_rows,
        "categories":          cfg.data.categories,
        "brands":              cfg.data.brands,
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

@step(enable_cache=False)
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
    product_embs = embed_model.encode(products["title"].tolist())
    log.info(f"Encoded {len(products)} products -> shape {product_embs.shape}")
    return product_embs, products


# ── Step 3: Feature Engineering ────────────────────────────────────────────

@step(enable_cache=False)
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

    _, all_pids = vector_store.search(all_query_embs, k=100)
    all_scores, all_pids = vector_store.search(all_query_embs, k=100)

    query_lookup = {
        row["qid"]: {
            "emb":            all_query_embs[i].reshape(1, -1),
            "translated_emb": all_translated_embs[i].reshape(1, -1),
            "pids":           all_pids[i].tolist(),
            "scores":         all_scores[i].tolist(),
        }
        for i, row in enumerate(unique_queries.to_dict("records"))
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
        faiss_pids_int = lookup["pids"]
        faiss_scores   = lookup["scores"]

        # Labeled pids that exist in catalog
        labeled_pids   = [p for p in qgroup["pid"].unique() if p in pid_to_idx]
        # FAISS augmentation — HNSW returns L2 distances (lower=better);
        # distance < 1.0 on unit vectors ≈ cosine_sim > 0.5 — filters noisy irrelevant candidates
        faiss_pids_str = [
            products.iloc[i]["pid"]
            for i, s in zip(faiss_pids_int, faiss_scores) if s < 1.0
        ]
        extra_pids     = [p for p in faiss_pids_str if p not in set(labeled_pids)]
        candidate_pids = (labeled_pids + extra_pids)[:100]
        pids_int       = [pid_to_idx[p] for p in candidate_pids]

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
        y_group = labeled_relevance.reindex(candidate_pids, fill_value=0).values

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

@step(output_materializers={"ranker": LambdaRankerMaterializer}, enable_cache=False)
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

    scores = ranker.predict(X_val)
    feature_columns = [
        'semantic_sim', 'cross_lingual_sim', 'bm25_score', 'jaccard',
        'brand_match', 'category_match', 'exact_title_match', 'query_len'
    ]
    eval_df = pd.DataFrame(X_val, columns=feature_columns)
    eval_df['relevance'] = y_val
    eval_df['ranker_score'] = scores

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
    scores = ranker.predict(X_val)
    eval_df = pd.DataFrame(X_val, columns=[
        "semantic_sim", "cross_lingual_sim", "bm25_score", "jaccard",
        "brand_match", "category_match", "exact_title_match", "query_len"
    ])
    eval_df["relevance"]    = y_val
    eval_df["ranker_score"] = scores
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

