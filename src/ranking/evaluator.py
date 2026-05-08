"""Evaluation Metrics: NDCG@10, MRR, Ablation Study."""
from typing import List
from sklearn.metrics import ndcg_score
import pandas as pd
import numpy as np


import logging

log = logging.getLogger(__name__)

def compute_ndcg(y_true: np.ndarray, y_score: np.ndarray, k=10) -> float:
    return ndcg_score([y_true], [y_score], k=k)


def compute_mrr(y_true: np.ndarray, y_score: np.ndarray) -> float:
    order = np.argsort(-y_score)
    ranks = np.where(y_true[order] > 0)[0]
    return 1.0 / (ranks[0] + 1) if len(ranks) > 0 else 0.0


def _mean_metrics(df: pd.DataFrame, score_col: str, groups: List[int]):
    """Compute mean NDCG@10 and MRR across query groups."""
    ndcgs, mrrs = [], []
    y_true_all  = df["relevance"].values
    y_score_all = df[score_col].values

    if sum(groups) != len(df):
        log.warning(f"Group mismatch in evaluator: sum(groups)={sum(groups)}, len(df)={len(df)}")
        return {"ndcg": 0.0, "mrr": 0.0}

    start = 0
    for gsize in groups:
        end     = start + gsize
        y_true  = y_true_all[start:end]
        y_score = y_score_all[start:end]
        start   = end
        if y_true.sum() == 0:
            continue
        ndcgs.append(compute_ndcg(y_true, y_score))
        mrrs.append(compute_mrr(y_true, y_score))

    return {"ndcg": float(np.mean(ndcgs)) if ndcgs else 0.0, "mrr": float(np.mean(mrrs)) if mrrs else 0.0}


def ablation_study(df: pd.DataFrame, groups: List[int]) -> pd.DataFrame:
    """Per-query ablation: BM25 vs Semantic vs Cross-Encoder vs Full ranker."""
    if len(df) == 0:
        raise ValueError(
            "ablation_study called with empty DataFrame — no metrics to compute. "
            "Check that feature extraction produced valid query groups."
        )

    results = {
        "BM25":     _mean_metrics(df, "bm25_score",   groups),
        "Semantic": _mean_metrics(df, "semantic_sim", groups),
        "Full":     _mean_metrics(df, "ranker_score", groups),
    }
    if "ce_score" in df.columns and df["ce_score"].any():
        results["CrossEncoder"] = _mean_metrics(df, "ce_score", groups)
    return pd.DataFrame(results).T


def full_catalog_eval(
    ranker,
    val_df: pd.DataFrame,
    products: pd.DataFrame,
    product_embs: np.ndarray,
    feat_eng,
    vector_store,
    top_k: int = 400,
    sample_queries: int = 500,
) -> dict:
    """
    Evaluate on the FULL catalog (not pre-filtered candidates).
    top_k must match retrieval_k used during training so the eval candidate
    pool is identical to the serving pool — otherwise NDCG is underestimated
    because positives at ranks 201-400 are missed.
    """
    from src.features.feature_engineer import _detect_lang, _translate_query

    rng        = np.random.default_rng(42)
    val_qids   = val_df["qid"].unique()
    if len(val_qids) > sample_queries:
        val_qids = rng.choice(val_qids, size=sample_queries, replace=False)

    pid_to_idx = {pid: idx for idx, pid in enumerate(products["pid"])}
    ndcgs, mrrs = [], []

    for qid in val_qids:
        qgroup = val_df[val_df["qid"] == qid]
        query  = qgroup["query"].iloc[0]

        # Retrieve from full catalog
        q_emb          = feat_eng.embedding_model.encode([query])
        scores_arr, idx_arr = vector_store.search(q_emb, k=top_k)
        cand_indices   = idx_arr[0].tolist()

        if len(cand_indices) < 2:
            continue

        cand_embs = product_embs[cand_indices]
        lang      = _detect_lang(query)
        t_emb     = feat_eng.embedding_model.encode([_translate_query(query)]) if lang != "en" else q_emb

        X_cand = feat_eng.extract_features(
            query, [],
            prod_embs=cand_embs, query_emb=q_emb,
            translated_emb=t_emb, candidate_indices=cand_indices
        )

        ranker_scores = ranker.predict(X_cand)

        labeled = qgroup.groupby("pid")["relevance"].max()
        cand_pids = products["pid"].values[cand_indices]
        y_true    = np.array([labeled.get(pid, 0) for pid in cand_pids])

        if y_true.sum() == 0:
            continue

        ndcgs.append(compute_ndcg(y_true, ranker_scores))
        mrrs.append(compute_mrr(y_true, ranker_scores))

    result = {
        "full_catalog_ndcg_10": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "full_catalog_mrr":     float(np.mean(mrrs))  if mrrs  else 0.0,
        "n_queries_evaluated":  len(ndcgs),
    }
    log.info(f"Full-catalog eval ({len(ndcgs)} queries): NDCG@10={result['full_catalog_ndcg_10']:.4f}, MRR={result['full_catalog_mrr']:.4f}")
    return result
