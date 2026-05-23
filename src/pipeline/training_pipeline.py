"""ZenML Pipeline — Cross-Market Semantic Ranking."""
import numpy as np
import pandas as pd
import logging
import os
from typing import List, Tuple, Annotated
from multiprocessing import Pool, cpu_count, shared_memory

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
        "use_augmentation":        cfg.data.use_augmentation,
        "use_llm":                 cfg.data.use_llm,
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

    # ── Cross-encoder distillation for synthetic labels ─────────────────────
    if cfg.data.use_cross_encoder_distillation:
        from src.data.label_distiller import LabelDistiller
        log.info("Running cross-encoder distillation for synthetic labels")
        df = LabelDistiller(model_name=cfg.data.cross_encoder_model_name).distill(df)
        log.info(f"After distillation: {df['relevance'].value_counts().sort_index().to_dict()}")

    return df


# ── Step 2: Build Embeddings + Vector Store ────────────────────────────────

@step(enable_cache=True)
def build_embeddings(
    train_df: pd.DataFrame,
    cfg: PipelineConfig,
) -> Annotated[pd.DataFrame, "products"]:
    import pickle
    from pathlib import Path

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

    artifacts = Path("artifacts")
    artifacts.mkdir(exist_ok=True)
    with open(artifacts / "catalog.pkl", "wb") as f:
        pickle.dump(products, f)
    np.save(str(artifacts / "embeddings.npy"), product_embs)
    log.info(f"Saved catalog.pkl and embeddings.npy to artifacts/")

    return products


# ── Parallel worker for feature extraction ─────────────────────────────────

_worker_feat_eng     = None
_worker_products     = None
_worker_pid_to_idx   = None
_worker_product_embs = None  # shared_memory view — zero copy
_worker_cfg          = {}

def _init_worker(products_df, pid_to_idx, bm25_tokenized_docs, worker_cfg,
                 shm_name=None, emb_shape=None):
    """Initialize process-local globals once per worker.
    BM25 index is built here only for the bm25_norm_factor access in
    FeatureEngineer — actual BM25 scoring is done in main process (OPT 4/5).
    Workers only use catalog metadata: titles, brands, categories, token sets.
    """
    global _worker_feat_eng, _worker_products, _worker_pid_to_idx, _worker_product_embs, _worker_cfg
    from src.features.feature_engineer import FeatureEngineer

    _worker_products   = products_df
    _worker_pid_to_idx = pid_to_idx
    _worker_cfg        = worker_cfg

    # Attach to the shared memory block by name — zero copy, same physical RAM
    # as the main process. Workers read directly from the OS page cache.
    if shm_name is not None:
        shm = shared_memory.SharedMemory(name=shm_name)
        _worker_product_embs = np.ndarray(emb_shape, dtype=np.float32, buffer=shm.buf)
    else:
        _worker_product_embs = None

    _worker_feat_eng = FeatureEngineer.__new__(FeatureEngineer)
    # Workers no longer do BM25 scoring — set bm25=None, scores come pre-computed
    _worker_feat_eng.bm25 = None
    _worker_feat_eng.use_cross_encoder    = False
    _worker_feat_eng._ce_model            = None
    _worker_feat_eng._ce_model_name       = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    _worker_feat_eng._ce_top_k            = worker_cfg.get("ce_top_k", 50)
    _worker_feat_eng._bm25_norm_factor    = worker_cfg["bm25_norm_factor"]
    _worker_feat_eng._catalog_precomputed = True
    _worker_feat_eng._titles_lower        = products_df["title"].str.lower().values
    _worker_feat_eng._brands_norm         = products_df["brand_norm"].values
    _worker_feat_eng._categories_norm     = products_df["category_norm"].values
    _worker_feat_eng._title_tokens        = np.empty(len(products_df), dtype=object)
    for i, t in enumerate(_worker_feat_eng._titles_lower):
        _worker_feat_eng._title_tokens[i] = set(t.split())


def _process_query_group(args):
    """Worker reads candidate embeddings directly from shared memory — zero IPC copy."""
    (qid, query, qgroup_data, lookup, use_hybrid, retrieval_k) = args

    global _worker_feat_eng, _worker_products, _worker_pid_to_idx, _worker_product_embs

    import numpy as np

    q_emb             = lookup["emb"].astype(np.float32)
    t_emb             = lookup["translated_emb"].astype(np.float32)
    candidate_indices = lookup["candidate_indices"]
    ret_ranks         = np.array(lookup["ret_ranks"],  dtype=np.float32)
    bm25_rank_arr     = np.array(lookup["bm25_ranks"], dtype=np.float32)

    # Read directly from shared memory — no deserialization, no copy
    candidate_embs = np.array(_worker_product_embs[candidate_indices], dtype=np.float32)

    labeled_pids = qgroup_data["pids"]
    labeled_rels = qgroup_data["relevance"]
    pid_to_rel   = dict(zip(labeled_pids, labeled_rels))

    total_positives_in_group   = sum(1 for r in labeled_rels if r > 0)
    retrieved_with_label_count = sum(
        1 for p_idx in candidate_indices
        if _worker_products["pid"].values[p_idx] in pid_to_rel
        and pid_to_rel[_worker_products["pid"].values[p_idx]] > 0
    )

    if len(candidate_indices) < 2:
        return None

    X_group = _worker_feat_eng.extract_features(
        query, [],
        prod_embs=candidate_embs, query_emb=q_emb,
        translated_emb=t_emb, candidate_indices=candidate_indices,
        retrieval_ranks=ret_ranks, bm25_ranks=bm25_rank_arr,
    )

    pids    = _worker_products["pid"].values[candidate_indices]
    y_group = np.array([pid_to_rel.get(pid, 0) for pid in pids])

    assert y_group.max() <= 3 and y_group.min() >= 0, \
        f"Label out of [0,3] range: min={y_group.min()}, max={y_group.max()}"

    if np.max(y_group) == 0:
        return None

    return (X_group, y_group, len(y_group), retrieved_with_label_count, total_positives_in_group)


# ── Step 3: Feature Engineering ────────────────────────────────────────────

@step(enable_cache=True)
def build_features(
    train_df: pd.DataFrame,
    products: pd.DataFrame,
    cfg: PipelineConfig,
) -> Tuple[
    Annotated[np.ndarray, "X_train"],
    Annotated[np.ndarray, "y_train"],
    Annotated[List[int], "groups_train"],
    Annotated[np.ndarray, "X_val"],
    Annotated[np.ndarray, "y_val"],
    Annotated[List[int], "groups_val"],
    Annotated[np.ndarray, "X_test"],
    Annotated[np.ndarray, "y_test"],
    Annotated[List[int], "groups_test"],
]:
    # ── FIX 1: Held-out test split BEFORE train/val ─────────────────────
    rng      = np.random.default_rng(cfg.ranker.seed)
    all_qids = train_df["qid"].unique()
    rng.shuffle(all_qids)
    n_total = len(all_qids)
    n_test  = int(n_total * 0.05)
    n_train = int(n_total * 0.80)
    test_qids  = set(all_qids[:n_test])
    train_qids = set(all_qids[n_test:n_test + n_train])
    val_qids   = set(all_qids[n_test + n_train:])
    log.info(f"Held-out split: {len(test_qids)} test / {len(train_qids)} train / {len(val_qids)} val queries")

    from pathlib import Path
    # Load embeddings into shared memory — one allocation in the OS page cache,
    # all workers attach by name with zero copies. Standard production pattern
    # used by PyTorch DataLoader, Ray, and Dask for large read-only arrays.
    _raw          = np.load(str(Path("artifacts") / "embeddings.npy"))
    product_embs  = np.ascontiguousarray(_raw, dtype=np.float32)
    del _raw
    shm           = shared_memory.SharedMemory(create=True, size=product_embs.nbytes)
    shm_arr       = np.ndarray(product_embs.shape, dtype=np.float32, buffer=shm.buf)
    np.copyto(shm_arr, product_embs)
    del product_embs  # main process no longer needs its own copy
    product_embs  = shm_arr  # alias for _build_tasks slicing below
    log.info(f"Loaded product_embs into shared_memory: shape {shm_arr.shape}, shm={shm.name}")

    embed_model  = EmbeddingModel(cfg.embedding_model_name)
    vector_store = VectorStore(
        hnsw_m=cfg.vector_store.hnsw_m,
        ef_search=cfg.vector_store.ef_search,
        max_vectors=cfg.vector_store.max_vectors,
        max_k=cfg.vector_store.max_k,
    )
    vector_store.add(product_embs, ids=products.index.tolist())

    bm25_docs = products["title"].str.split().tolist()
    feat_eng  = FeatureEngineer(
        embed_model, bm25_docs,
        use_cross_encoder=cfg.data.use_cross_encoder,
        cross_encoder_model_name=cfg.data.cross_encoder_model_name,
        ce_top_k=cfg.data.ce_top_k,
    )
    feat_eng.precompute_catalog(products)

    unique_queries = train_df[["qid", "query"]].drop_duplicates("qid")
    q_texts        = unique_queries["query"].tolist()
    all_query_embs = embed_model.encode(q_texts)

    non_en_mask    = [_detect_lang(q) != "en" for q in q_texts]
    non_en_indices = [i for i, m in enumerate(non_en_mask) if m]
    log.info(f"Non-English queries: {len(non_en_indices)} / {len(q_texts)}")

    all_translated_embs = all_query_embs.copy()
    if non_en_indices:
        non_en_translated = [_translate_query(q_texts[i]) for i in non_en_indices]
        all_translated_embs[non_en_indices] = embed_model.encode(non_en_translated)

    retrieval_k = cfg.data.retrieval_k
    use_hybrid  = cfg.data.use_hybrid_retrieval

    if use_hybrid:
        from src.retrieval.retriever import HybridRetriever, _normalize_tokens
        retriever = HybridRetriever(
            embed_model, vector_store, bm25_docs,
            rrf_k=cfg.data.rrf_k,
            semantic_weight=cfg.data.semantic_weight,
            bm25_weight=cfg.data.bm25_weight,
        )
        log.info(f"Using HybridRetriever (k={retrieval_k}, rrf_k={cfg.data.rrf_k})")
    else:
        from src.retrieval.retriever import SemanticRetriever, _normalize_tokens
        retriever = SemanticRetriever(embed_model, vector_store)
        log.info(f"Using SemanticRetriever (k={retrieval_k})")

    # OPT 1: Store query embeddings as numpy arrays keyed by index, not .tolist()
    # .tolist() converts float32 -> Python float (8 bytes each) inflating query_lookup
    # RAM by ~3x. Keep as float32 numpy and only convert in worker via np.array().
    query_lookup = {}
    qids         = unique_queries["qid"].tolist()
    batch_scores, batch_indices = vector_store.search(all_query_embs, k=min(retrieval_k * 2, len(products)))

    if use_hybrid:
        for i, qid in enumerate(qids):
            sem_indices   = batch_indices[i]
            bm25_scores_i = retriever.bm25.get_scores(_normalize_tokens(q_texts[i]))
            fused = {}
            for rank, idx in enumerate(sem_indices):
                fused[int(idx)] = retriever.semantic_weight / (retriever.rrf_k + rank + 1)
            bm25_ranks_i = np.argsort(-bm25_scores_i)[:retrieval_k * 2]
            for rank, idx in enumerate(bm25_ranks_i):
                idx = int(idx)
                fused[idx] = fused.get(idx, 0.0) + retriever.bm25_weight / (retriever.rrf_k + rank + 1)
            sorted_results = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:retrieval_k]
            query_lookup[qid] = {
                "emb":            all_query_embs[i].reshape(1, -1),
                "translated_emb": all_translated_embs[i].reshape(1, -1),
                "pids":           [r[0] for r in sorted_results],
            }
    else:
        for i, qid in enumerate(qids):
            query_lookup[qid] = {
                "emb":            all_query_embs[i].reshape(1, -1),
                "translated_emb": all_translated_embs[i].reshape(1, -1),
                "pids":           batch_indices[i].tolist(),
            }

    pid_to_idx = {pid: idx for idx, pid in enumerate(products["pid"])}

    # OPT 4: Pre-compute BM25 scores for all queries in main process in one pass.
    # Workers currently call bm25.get_batch_scores per query, each scoring the
    # full 84k corpus. Doing it once here in the main process reuses the BM25
    # IDF weights without re-initializing per worker and avoids redundant scoring.
    from src.retrieval.retriever import _normalize_tokens
    log.info("Pre-computing BM25 scores for all queries...")
    all_q_tokens = [_normalize_tokens(q) for q in q_texts]
    qid_to_bm25_ranks: dict = {}
    for i, qid in enumerate(qids):
        bm25_scores_q = feat_eng.bm25.get_scores(all_q_tokens[i]).astype(np.float32)
        # inverse permutation: rank_of[catalog_idx] = BM25 rank
        argsort_desc  = np.argsort(-bm25_scores_q).astype(np.int32)
        rank_of       = np.empty_like(argsort_desc)
        rank_of[argsort_desc] = np.arange(len(argsort_desc), dtype=np.int32)
        qid_to_bm25_ranks[qid] = rank_of  # O(1) rank lookup per candidate
    log.info("BM25 pre-computation complete.")

    def _evaluate_retrieval(query_lookup, df, pid_to_idx, products_df):
        """Compute Retrieval Recall@k for k in [10, 50, 100]."""
        idx_to_pid = {idx: pid for pid, idx in pid_to_idx.items()}
        recall_results = {}
        for k in [10, 50, 100]:
            hits = 0
            total = 0
            for qid, qgroup in df.groupby("qid"):
                if qid not in query_lookup:
                    continue
                retrieved_indices = query_lookup[qid]["pids"][:k]
                retrieved_pids = {idx_to_pid.get(i) for i in retrieved_indices}
                positives = set(qgroup[qgroup["relevance"] > 0]["pid"])
                if positives.intersection(retrieved_pids):
                    hits += 1
                total += 1
            recall = hits / max(total, 1)
            recall_results[f"retrieval_recall_{k}"] = recall
            log.info(f"Retrieval Recall@{k}: {recall:.3f} ({hits}/{total})")
        return recall_results

    retrieval_recalls = _evaluate_retrieval(query_lookup, train_df[train_df["qid"].isin(train_qids)], pid_to_idx, products)
    try:
        import mlflow
        mlflow.log_metrics(retrieval_recalls)
    except Exception as e:
        log.warning(f"MLflow retrieval recall logging skipped: {e}")

    train_pids = set(train_df["pid"].unique())
    overlap    = train_pids & set(pid_to_idx.keys())
    log.info(f"PID overlap: {len(overlap)} / {len(train_pids)} train pids found in catalog")
    rel_dist = dict(train_df['relevance'].value_counts().sort_index())
    log.info(f"Relevance distribution: {rel_dist}")
    if set(rel_dist.keys()) == {0}:
        raise ValueError(
"All relevance scores are 0 — ESCI label mapping failed. "
            "Check that 'esci_label' or 'label' field exists in the dataset."
        )

    n_workers = min(cfg.data.num_workers, cpu_count())
    log.info(f"Feature extraction workers: {n_workers}")

    from src.data.normalizer import normalize_entity
    products_norm = products.copy()
    products_norm["brand_norm"] = products_norm["brand"].fillna("").str.lower().apply(lambda x: normalize_entity(x, "brand"))
    products_norm["category_norm"] = products_norm["category"].fillna("").str.lower().apply(lambda x: normalize_entity(x, "category"))

    worker_cfg = {
        "max_hard_negatives":       cfg.data.max_hard_negatives,
        "hard_neg_score_threshold": cfg.data.hard_neg_score_threshold,
        "semantic_score_threshold": cfg.data.semantic_score_threshold,
        "bm25_norm_factor":         cfg.ranker.bm25_norm_factor,
        "ce_top_k":                 cfg.data.ce_top_k,
    }

    def _build_tasks(df_subset):
        """Pre-slice candidate embeddings in main process — workers receive only
        their 400x768 slice (~1.2MB) instead of the full 245MB catalog copy.
        OPT 5: BM25 ranks pre-computed here from qid_to_bm25_ranks — workers
        do zero BM25 scoring."""
        tasks = []
        for qid, qgroup in df_subset.groupby("qid"):
            if qid not in query_lookup:
                continue
            qlookup = query_lookup[qid]
            retrieved_pids_int = qlookup["pids"]

            labeled_pids = qgroup["pid"].tolist()
            labeled_rels = qgroup["relevance"].tolist()
            pid_to_rel   = dict(zip(labeled_pids, labeled_rels))

            retrieved_candidates = [p for p in retrieved_pids_int if p < len(products_norm)]
            candidate_indices, retrieval_rank_map = [], {}
            for rank, p_idx in enumerate(retrieved_candidates):
                if len(candidate_indices) >= retrieval_k:
                    break
                candidate_indices.append(p_idx)
                retrieval_rank_map[p_idx] = rank

            candidate_set = set(candidate_indices)
            for pid, rel in zip(labeled_pids, labeled_rels):
                if rel > 0 and pid in pid_to_idx:
                    p_idx = pid_to_idx[pid]
                    if p_idx not in candidate_set:
                        candidate_indices.append(p_idx)
                        candidate_set.add(p_idx)
                        retrieval_rank_map[p_idx] = retrieval_k

            if len(candidate_indices) < 2:
                continue

            candidate_indices = candidate_indices[:retrieval_k]
            ret_ranks = [retrieval_rank_map.get(i, retrieval_k) for i in candidate_indices]

            # OPT 5: O(k) rank lookup using pre-computed inverse permutation.
            # rank_of[catalog_idx] = BM25 rank, so indexing is direct array access.
            rank_of = qid_to_bm25_ranks[qid]
            bm25_ranks_cand = rank_of[candidate_indices].tolist()

            task_lookup = {
                "emb":               qlookup["emb"],
                "translated_emb":    qlookup["translated_emb"],
                "candidate_indices": candidate_indices,
                # No embedding bytes in task — workers read directly from shm
                "ret_ranks":         ret_ranks,
                "bm25_ranks":        bm25_ranks_cand,
            }
            tasks.append((
                qid, qgroup["query"].iloc[0],
                {"pids": labeled_pids, "relevance": labeled_rels},
                task_lookup, use_hybrid, retrieval_k
            ))
        return tasks

    def _run_tasks(task_list, desc):
        from tqdm import tqdm
        X_out, y_out, g_out, n_skip = [], [], [], 0
        total_retrieved_pos = 0
        total_positives = 0

        if n_workers > 1:
            with Pool(
                processes=n_workers,
                initializer=_init_worker,
                initargs=(products_norm, pid_to_idx, None, worker_cfg,
                          shm.name, shm_arr.shape)  # workers attach by name
            ) as pool:
                chunksize = max(1, len(task_list) // (n_workers * 8))
                for result in tqdm(pool.imap(_process_query_group, task_list, chunksize=chunksize), desc=desc, total=len(task_list)):
                    if result is None: n_skip += 1; continue
                    X_out.append(result[0]); y_out.append(result[1]); g_out.append(result[2])
                    total_retrieved_pos += result[3]
                    total_positives += result[4]
        else:
            _init_worker(products_norm, pid_to_idx, None, worker_cfg)
            for result in tqdm(map(_process_query_group, task_list), desc=desc, total=len(task_list)):
                if result is None: n_skip += 1; continue
                X_out.append(result[0]); y_out.append(result[1]); g_out.append(result[2])
                total_retrieved_pos += result[3]
                total_positives += result[4]

        recall_at_k = total_retrieved_pos / max(total_positives, 1)
        log.info(f"{desc}: valid={len(g_out)}, skipped={n_skip}, recall@k={recall_at_k:.3f} ({total_retrieved_pos}/{total_positives})")
        return X_out, y_out, g_out

# ── FIX 2: Extract features separately for train, val, and test query sets ──
    train_df_split = train_df[train_df["qid"].isin(train_qids)]
    val_df_split   = train_df[train_df["qid"].isin(val_qids)]
    test_df_split  = train_df[train_df["qid"].isin(test_qids)]

    try:
        X_tr, y_tr, g_tr = _run_tasks(_build_tasks(train_df_split), "Train features")
        X_vl, y_vl, g_vl = _run_tasks(_build_tasks(val_df_split),   "Val features")
        X_ts, y_ts, g_ts = _run_tasks(_build_tasks(test_df_split),  "Test features")
    finally:
        # Always release shared memory — even if a task raises an exception.
        # On Windows, leaked shm blocks persist until reboot.
        shm.close()
        shm.unlink()

    if not X_tr:
        raise ValueError("No valid train query groups — check ESCI pids exist in product catalog.")
    if not X_vl:
        raise ValueError("No valid val query groups.")
    if not X_ts:
        raise ValueError("No valid test query groups.")

    import pickle
    from pathlib import Path
    with open(Path("artifacts") / "val_df.pkl", "wb") as f:
        pickle.dump(val_df_split, f)
    log.info(f"Saved val_df.pkl ({len(val_df_split)} rows) to artifacts/")

    return (
        np.vstack(X_tr), np.hstack(y_tr), g_tr,
        np.vstack(X_vl), np.hstack(y_vl), g_vl,
        np.vstack(X_ts), np.hstack(y_ts), g_ts,
    )


# ── Step 4: Train Ranker ───────────────────────────────────────────────────

@step(output_materializers={"ranker": LambdaRankerMaterializer}, enable_cache=True)
def train_ranker(
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups_train: List[int],
    X_val: np.ndarray,
    y_val: np.ndarray,
    groups_val: List[int],
    X_test: np.ndarray,
    y_test: np.ndarray,
    groups_test: List[int],
    cfg: PipelineConfig,
) -> Annotated[LambdaRanker, "ranker"]:
    ranker = LambdaRanker(cfg.ranker)
    # ── FIX 3: Pass pre-split query-disjoint val set directly ────────────
    ranker.fit(X_train, y_train, groups_train, X_val=X_val, y_val=y_val, group_val=groups_val)

    ranker.save_model("artifacts")
    ranker.export_onnx("artifacts", num_features=X_train.shape[1])

    from src.features.feature_engineer import FeatureEngineer
    # Use actual column count — CE features absent when use_cross_encoder=False
    feature_columns = FeatureEngineer.FEATURE_NAMES[:X_val.shape[1]]

    raw_scores = ranker.predict(X_val)
    eval_df = pd.DataFrame(X_val, columns=feature_columns)
    eval_df['relevance']    = y_val
    eval_df['ranker_score'] = raw_scores

    log.info(f"Feature means: {X_val.mean(axis=0)}")
    log.info(f"Feature stds:  {X_val.std(axis=0)}")
    log.info(f"Relevance distribution in val: {np.bincount(y_val.astype(int))}")

    metrics_df = ablation_study(eval_df, groups_val)
    log.info(f"Validation Metrics (pre-filtered candidates):\n{metrics_df.to_string()}")

    # ── Full-catalog eval — unbiased headline metric ─────────────────────
    import pickle
    from pathlib import Path
    from src.ranking.evaluator import full_catalog_eval
    from src.features.feature_engineer import FeatureEngineer as FE

    try:
        with open(Path("artifacts") / "val_df.pkl", "rb") as f:
            val_df_full = pickle.load(f)
        with open(Path("artifacts") / "catalog.pkl", "rb") as f:
            products_df = pickle.load(f)
        product_embs_full = np.load(str(Path("artifacts") / "embeddings.npy"))

        from src.embeddings.embedding_model import EmbeddingModel
        from src.embeddings.vector_store import VectorStore
        embed_model_eval  = EmbeddingModel(cfg.embedding_model_name)
        vs_eval = VectorStore(
            hnsw_m=cfg.vector_store.hnsw_m,
            ef_search=cfg.vector_store.ef_search,
            max_vectors=cfg.vector_store.max_vectors,
            max_k=cfg.vector_store.max_k,
        )
        vs_eval.add(product_embs_full, ids=products_df.index.tolist())
        bm25_docs_eval = products_df["title"].str.split().tolist()
        feat_eng_eval  = FE(
            embed_model_eval, bm25_docs_eval,
            use_cross_encoder=cfg.data.use_cross_encoder,
            cross_encoder_model_name=cfg.data.cross_encoder_model_name,
            ce_top_k=cfg.data.ce_top_k,
        )
        feat_eng_eval.precompute_catalog(products_df)

        full_eval = full_catalog_eval(
            ranker, val_df_full, products_df, product_embs_full,
            feat_eng_eval, vs_eval, top_k=cfg.data.retrieval_k, sample_queries=500
        )
        log.info(f"Full-catalog NDCG@10: {full_eval['full_catalog_ndcg_10']:.4f} | MRR: {full_eval['full_catalog_mrr']:.4f}")
    except Exception as e:
        log.warning(f"Full-catalog eval skipped: {e}")
        full_eval = {}

    # ── Held-out test set — honest headline metric ──────────────────────
    test_scores  = ranker.predict(X_test)
    test_eval_df = pd.DataFrame(X_test, columns=feature_columns)
    test_eval_df['relevance']    = y_test
    test_eval_df['ranker_score'] = test_scores
    test_metrics_df = ablation_study(test_eval_df, groups_test)
    log.info(f"Test Metrics (held-out, honest headline):\n{test_metrics_df.to_string()}")

    try:
        import mlflow
        import mlflow.lightgbm
        from pathlib import Path
        mlflow.set_tracking_uri(f"sqlite:///{Path(__file__).resolve().parents[2] / 'mlflow.db'}")
        mlflow.set_experiment("cross_market_semantic_ranking")
        with mlflow.start_run():
            mlflow.log_params({
                'num_boost_round': cfg.ranker.num_boost_round,
                'learning_rate':   cfg.ranker.learning_rate,
                'num_leaves':      cfg.ranker.num_leaves,
                'objective':       cfg.ranker.objective,
                'best_iteration':  ranker.model.best_iteration,
            })
            flat_metrics = {}
            for row in metrics_df.itertuples():
                flat_metrics[f"val_{row.Index}_ndcg"] = row.ndcg
                flat_metrics[f"val_{row.Index}_mrr"]  = row.mrr
            for row in test_metrics_df.itertuples():
                flat_metrics[f"test_{row.Index}_ndcg"] = row.ndcg
                flat_metrics[f"test_{row.Index}_mrr"]  = row.mrr
            for k, v in full_eval.items():
                if isinstance(v, float):
                    flat_metrics[k] = v  # keys already prefixed full_catalog_ in evaluator
            # Sanitize: replace any chars MLflow rejects
            import re
            flat_metrics = {
                re.sub(r'[^a-zA-Z0-9_\-\. /]', '_', k): v
                for k, v in flat_metrics.items()
            }
            mlflow.log_metrics(flat_metrics)
            mlflow.lightgbm.log_model(ranker.model, "lambda_ranker")
        log.info("MLflow run logged successfully")
    except Exception as e:
        log.warning(f"MLflow logging skipped: {e}")

    return ranker


# ── ZenML Pipeline ─────────────────────────────────────────────────────────

@pipeline(name="cross_market_semantic_ranking")
def ranking_pipeline(cfg: PipelineConfig = PipelineConfig()):
    train_df                                                                   = ingest_data(cfg)
    products                                                                   = build_embeddings(train_df, cfg)
    X_train, y_train, g_train, X_val, y_val, g_val, X_test, y_test, g_test   = build_features(train_df, products, cfg)
    train_ranker(X_train, y_train, g_train, X_val, y_val, g_val, X_test, y_test, g_test, cfg)
