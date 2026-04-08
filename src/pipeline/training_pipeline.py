"""ZenML Pipeline — Cross-Market Semantic Ranking."""
import numpy as np
import pandas as pd
import logging
from typing import List, Tuple, Annotated

from zenml import step, pipeline
from zenml.config import DockerSettings

from src.data.data_generator import DataGenerator
from src.embeddings.embedding_model import EmbeddingModel
from src.embeddings.vector_store import VectorStore
from src.features.feature_engineer import FeatureEngineer, _translate_query, _detect_lang
from src.ranking.ranker import LambdaRanker
from src.ranking.evaluator import ablation_study
from src.ranking.materializer import LambdaRankerMaterializer

log = logging.getLogger(__name__)


# ── Step 1: Data Ingestion ─────────────────────────────────────────────────

@step
def ingest_data(
    num_products: int = 10_000,
    queries_per: int = 5,
    use_esci: bool = False,
    esci_max_rows: int = 50_000,
    categories: List[str] = None,
    brands: List[str] = None,
) -> Annotated[pd.DataFrame, "train_df"]:
    config = {
        "num_products":        num_products,
        "queries_per_product": queries_per,
        "use_esci":            use_esci,
        "esci_max_rows":       esci_max_rows,
        "categories":          categories or ["Footwear", "Electronics", "Clothing", "Home"],
        "brands":              brands or ["Nike", "Sony", "Samsung", "Apple"],
    }
    generator = DataGenerator(config)
    chunks = []
    for chunk in generator.generate(use_esci=use_esci):
        chunks.append(chunk)
        log.info(f"Chunk loaded: {len(chunk)} rows")
    df = pd.concat(chunks, ignore_index=True)
    log.info(f"Total rows: {len(df)}")
    return df


# ── Step 2: Build Embeddings + Vector Store ────────────────────────────────

@step
def build_embeddings(
    train_df: pd.DataFrame,
    embedding_model_name: str = "intfloat/multilingual-e5-base",
) -> Tuple[
    Annotated[np.ndarray, "product_embs"],
    Annotated[pd.DataFrame, "products"],
]:
    embed_model = EmbeddingModel(embedding_model_name)
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

@step
def build_features(
    train_df: pd.DataFrame,
    products: pd.DataFrame,
    product_embs: np.ndarray,
    embedding_model_name: str = "intfloat/multilingual-e5-base",
) -> Tuple[
    Annotated[np.ndarray, "X"],
    Annotated[np.ndarray, "y"],
    Annotated[List[int], "groups"],
]:
    embed_model = EmbeddingModel(embedding_model_name)
    vector_store = VectorStore()
    vector_store.add(product_embs, ids=products.index.tolist())

    bm25_docs = products["title"].str.split().tolist()
    feat_eng  = FeatureEngineer(embed_model, bm25_docs)

    all_prod_embs = product_embs

    # Batch encode all unique queries + translated queries in two forward passes
    unique_queries      = train_df[["qid", "query"]].drop_duplicates("qid")
    q_texts             = unique_queries["query"].tolist()
    translated_texts    = [_translate_query(q) if _detect_lang(q) != "en" else q for q in q_texts]
    all_query_embs      = embed_model.encode(q_texts)
    all_translated_embs = embed_model.encode(translated_texts)

    _, all_pids = vector_store.search(all_query_embs, k=100)

    query_lookup = {
        row["qid"]: {
            "emb":            all_query_embs[i].reshape(1, -1),
            "translated_emb": all_translated_embs[i].reshape(1, -1),
            "pids":           all_pids[i].tolist()
        }
        for i, row in enumerate(unique_queries.to_dict("records"))
    }

    X_all, y_all, groups = [], [], []

    for qid, qgroup in train_df.groupby("qid"):
        query = qgroup["query"].iloc[0]
        lookup = query_lookup[qid]
        pids_int = lookup["pids"]
        q_emb    = lookup["emb"]
        t_emb    = lookup["translated_emb"]

        # Map FAISS integer indices to string product IDs (pids)
        pids_str = products.iloc[pids_int]["pid"].tolist()

        # Stage 2: features using pre-computed product embeddings
        candidate_products = products.iloc[pids_int][["title", "brand", "category"]].to_dict("records")
        candidate_embs     = all_prod_embs[pids_int]
        
        X_group = feat_eng.extract_features(
            query,
            candidate_products,
            prod_embs=candidate_embs,
            query_emb=q_emb,
            translated_emb=t_emb,
            candidate_indices=pids_int
        )

        y_group = (
            qgroup.groupby("pid")["relevance"].max()
            .reindex(pids_str, fill_value=0)
            .values
        )

        X_all.append(X_group)
        y_all.append(y_group)
        groups.append(len(y_group))

    return np.vstack(X_all), np.hstack(y_all), groups


# ── Step 4: Train Ranker ───────────────────────────────────────────────────

@step(output_materializers={"ranker": LambdaRankerMaterializer}, enable_cache=False)
def train_ranker(
    X: np.ndarray,
    y: np.ndarray,
    groups: List[int],
    objective: str = "lambdarank",
    metric: str = "ndcg",
    num_leaves: int = 31,
    learning_rate: float = 0.05,
    num_boost_round: int = 300,
    test_size: float = 0.2,
    seed: int = 42,
) -> Tuple[
    Annotated[LambdaRanker, "ranker"],
    Annotated[np.ndarray, "X_val"],
    Annotated[np.ndarray, "y_val"],
    Annotated[List[int], "group_val"],
]:
    from omegaconf import OmegaConf
    cfg = OmegaConf.create({
        "ranker":        {"objective": objective, "metric": metric, "num_leaves": num_leaves},
        "learning_rate": learning_rate,
        "num_boost_round": num_boost_round,
        "test_size":     test_size,
        "seed":          seed,
    })
    ranker = LambdaRanker(cfg)
    X_val, y_val, group_val = ranker.fit(X, y, groups)

    ranker.save_model("artifacts")
    ranker.export_onnx("artifacts", num_features=X.shape[1])

    log.info("Ranker trained and exported to ONNX")
    return ranker, X_val, y_val, group_val


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
def ranking_pipeline(
    num_products: int = 10_000,
    queries_per: int = 5,
    use_esci: bool = False,
    esci_max_rows: int = 50_000,
    embedding_model_name: str = "intfloat/multilingual-e5-base",
    num_boost_round: int = 300,
    num_leaves: int = 31,
    learning_rate: float = 0.05,
    categories: List[str] = None,
    brands: List[str] = None,
):
    train_df                        = ingest_data(num_products, queries_per, use_esci, esci_max_rows, categories, brands)
    product_embs, products          = build_embeddings(train_df, embedding_model_name)
    X, y, groups                    = build_features(train_df, products, product_embs, embedding_model_name)
    ranker, X_val, y_val, group_val = train_ranker(X, y, groups, num_boost_round=num_boost_round, num_leaves=num_leaves, learning_rate=learning_rate)
    evaluate(ranker, X_val, y_val, group_val)
