"""LightGBM LambdaMART Ranker for Precision Re-ranking."""
import lightgbm as lgb
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import List, Tuple

from src.config import RankerConfig

log = logging.getLogger(__name__)


def _downsample_negatives(
    X: np.ndarray, y: np.ndarray, group: List[int],
    neg_ratio: float, rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """Keep neg_ratio * n_positives label=0 rows per query group.
    Groups with zero positives are dropped — they produce zero LambdaMART
    gradient and inflate the negative pool without signal.
    The previous max(len(pos_idx), ...) floor was dead code: groups with
    zero positives are already filtered by _process_query_group before
    reaching the ranker, so the floor never fired.
    """
    sizes  = np.array(group)
    starts = np.concatenate([[0], np.cumsum(sizes[:-1])])
    keep_rows, new_group = [], []
    for s, sz in zip(starts, sizes):
        idx      = np.arange(s, s + sz)
        pos_mask = y[idx] > 0
        pos_idx  = idx[pos_mask]
        neg_idx  = idx[~pos_mask]
        if len(pos_idx) == 0:
            continue  # drop all-negative groups — no gradient signal
        n_keep = int(len(pos_idx) * neg_ratio)
        if len(neg_idx) > n_keep:
            neg_idx = rng.choice(neg_idx, size=n_keep, replace=False)
        kept = np.concatenate([pos_idx, neg_idx])
        keep_rows.append(kept)
        new_group.append(len(kept))
    keep_rows = np.concatenate(keep_rows)
    return X[keep_rows], y[keep_rows], new_group


def _upsample_rare_labels(
    X: np.ndarray, y: np.ndarray, group: List[int],
    rare_label: int, target_ratio: float, rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """Upsample rare label rows globally until they reach target_ratio of positives."""
    pos_mask  = y > 0
    rare_mask = y == rare_label
    n_pos     = pos_mask.sum()
    n_rare    = rare_mask.sum()
    target_n  = int(n_pos * target_ratio)
    if n_rare >= target_n or n_rare == 0:
        return X, y, group
    n_extra = target_n - n_rare
    rare_idx = np.where(rare_mask)[0]
    extra    = rng.choice(rare_idx, size=n_extra, replace=True)

    # Distribute extra rows back into their source groups to preserve valid
    # group structure. A synthetic single-label group produces zero gradient.
    sizes  = np.array(group)
    starts = np.concatenate([[0], np.cumsum(sizes[:-1])])
    extra_set = set(extra.tolist())
    extra_by_group: dict = {}
    for g_idx, (s, sz) in enumerate(zip(starts, sizes)):
        for row_idx in range(s, s + sz):
            if row_idx in extra_set:
                extra_by_group.setdefault(g_idx, []).append(row_idx)

    result_X, result_y, result_group = [], [], []
    for g_idx, (s, sz) in enumerate(zip(starts, sizes)):
        rows     = list(range(s, s + sz))
        injected = extra_by_group.get(g_idx, [])
        all_rows = rows + injected
        result_X.append(X[all_rows])
        result_y.append(y[all_rows])
        result_group.append(len(all_rows))

    X     = np.vstack(result_X)
    y     = np.concatenate(result_y)
    group = result_group
    log.info(f"Upsampled label={rare_label}: {n_rare} -> {n_rare + n_extra} rows")
    return X, y, group


class LambdaRanker:
    def __init__(self, cfg: RankerConfig):
        self.cfg   = cfg
        self.model = None

    def fit(
        self,
        X: np.ndarray, y: np.ndarray, group: List[int],
        X_val: np.ndarray = None, y_val: np.ndarray = None,
        group_val: List[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        """Train LambdaMART with query-disjoint train/val split."""
        rng = np.random.default_rng(self.cfg.seed)

        if X_val is None:
            # Fallback: random group-level split (not query-disjoint)
            n_queries  = len(group)
            sizes      = np.array(group)
            starts     = np.concatenate([[0], np.cumsum(sizes[:-1])])
            perm       = rng.permutation(n_queries)
            row_idx    = np.concatenate([np.arange(starts[i], starts[i] + sizes[i]) for i in perm])
            X, y       = X[row_idx], y[row_idx]
            group      = [group[i] for i in perm]
            split_idx  = int(n_queries * (1 - self.cfg.test_size))
            train_rows = sum(group[:split_idx])
            X_train, X_val       = X[:train_rows], X[train_rows:]
            y_train, y_val       = y[:train_rows], y[train_rows:]
            group_train, group_val = group[:split_idx], group[split_idx:]
        else:
            X_train, y_train, group_train = X, y, group

        # -- Downsample negatives on TRAIN only --------------------------------
        # Val is intentionally NOT downsampled: it must reflect the real serving
        # distribution (~39 negatives per positive at retrieval_k=400) so that
        # early stopping fires at the correct operating point.
        X_train, y_train, group_train = _downsample_negatives(
            X_train, y_train, group_train,
            neg_ratio=self.cfg.neg_downsample_ratio, rng=rng
        )
        log.info(f"After downsampling — train rows: {len(X_train)}, "
                 f"label dist: {np.unique(y_train, return_counts=True)}")
        log.info(f"Val rows (unmodified): {len(X_val)}, "
                 f"label dist: {np.unique(y_val, return_counts=True)}")

        # -- Upsample rare label=1 (Complement) on TRAIN -----------------------
        # label=1 is ~3% of positives in ESCI. Without upsampling LambdaMART
        # sees too few label=1 pairs to learn meaningful Complement ranking.
        # target_ratio=0.15 brings label=1 to 15% of all positives.
        X_train, y_train, group_train = _upsample_rare_labels(
            X_train, y_train, group_train,
            rare_label=1, target_ratio=0.15, rng=rng
        )
        log.info(f"After upsampling label=1 — train rows: {len(X_train)}, "
                 f"label dist: {np.unique(y_train, return_counts=True)}")

        # -- Cast labels to int ------------------------------------------------
        # np.hstack produces float64; LightGBM silently truncates floats
        # (e.g. 2.9 -> 2), corrupting the label_gain mapping.
        y_train = y_train.astype(np.int32)
        y_val   = y_val.astype(np.int32)

        # -- Group integrity check ---------------------------------------------
        assert sum(group_train) == len(X_train), (
            f"Train group sum {sum(group_train)} != X_train rows {len(X_train)}"
        )
        assert sum(group_val) == len(X_val), (
            f"Val group sum {sum(group_val)} != X_val rows {len(X_val)}"
        )

        train_set = lgb.Dataset(X_train, y_train, group=group_train)
        val_set   = lgb.Dataset(X_val,   y_val,   group=group_val,
                                reference=train_set)

        max_label  = int(y_train.max()) if y_train.max() > 0 else 1
        counts     = np.bincount(y_train, minlength=max_label + 1)
        # Exponential gains: 2^rel - 1 = [0, 1, 3, 7]
        label_gain = [int(2 ** i - 1) for i in range(max_label + 1)]
        if max_label >= 1 and counts[1] > 0 and counts[2] > 0:
            label_gain[1] = max(2, label_gain[1])
        log.info(f"label_gain={label_gain}, label counts={counts.tolist()}")
        log.info(f"Label distribution: {np.unique(y_train, return_counts=True)}")

        min_rows_needed = self.cfg.num_leaves * self.cfg.min_data_in_leaf
        if len(X_train) < min_rows_needed:
            raise ValueError(
                f"Training set too small: {len(X_train)} rows < "
                f"num_leaves({self.cfg.num_leaves}) x min_data_in_leaf"
                f"({self.cfg.min_data_in_leaf}) = {min_rows_needed}."
            )
        if not np.any(y_val > 0):
            raise ValueError(
                "Val set has no positive labels — NDCG@5 will be 0 every round."
            )

        evals_result: dict = {}
        self.model = lgb.train(
            {
                "objective":               self.cfg.objective,
                "metric":                  self.cfg.metric,
                "eval_at":                 [5, 10],
                "num_leaves":              self.cfg.num_leaves,
                "max_depth":               self.cfg.max_depth,
                "learning_rate":           self.cfg.learning_rate,
                "min_data_in_leaf":        self.cfg.min_data_in_leaf,
                "min_sum_hessian_in_leaf": 1,
                "min_gain_to_split":       self.cfg.min_gain_to_split,
                "label_gain":              label_gain,
                "feature_fraction":        self.cfg.feature_fraction,
                "bagging_fraction":        self.cfg.bagging_fraction,
                "bagging_freq":            self.cfg.bagging_freq,
                "lambda_l1":               self.cfg.lambda_l1,
                "lambda_l2":               self.cfg.lambda_l2,
                "pos_bagging_fraction":    1.0,
                "neg_bagging_fraction":    self.cfg.neg_bagging_fraction,
                "verbose":                 -1,
            },
            train_set,
            valid_sets=[train_set, val_set],
            num_boost_round=self.cfg.num_boost_round,
            callbacks=[
                lgb.early_stopping(self.cfg.early_stopping_rounds),
                lgb.log_evaluation(self.cfg.early_stopping_rounds),
                lgb.record_evaluation(evals_result),
            ],
        )

        # -- Feature importance ------------------------------------------------
        from src.features.feature_engineer import FeatureEngineer
        feat_names    = FeatureEngineer.FEATURE_NAMES[:X_train.shape[1]]
        importance    = self.model.feature_importance(importance_type="gain")
        importance_df = pd.DataFrame(
            {"feature": feat_names[:len(importance)], "gain": importance}
        ).sort_values("gain", ascending=False)
        log.info(f"Feature importances (gain):\n{importance_df.to_string(index=False)}")
        zero_imp = importance_df[importance_df["gain"] == 0]["feature"].tolist()
        if zero_imp:
            log.warning(f"Zero-importance features (candidates for removal): {zero_imp}")

        # -- Training metrics --------------------------------------------------
        train_scores  = self.model.predict(X_train)
        from src.ranking.evaluator import ablation_study
        train_eval_df = pd.DataFrame(X_train, columns=feat_names)
        train_eval_df["relevance"]    = y_train
        train_eval_df["ranker_score"] = train_scores
        train_metrics = ablation_study(train_eval_df, group_train)
        log.info(f"Training Metrics:\n{train_metrics.to_string()}")

        # -- Overfitting diagnostics -------------------------------------------
        self._log_overfitting_diagnostics(evals_result, self.model.best_iteration)

        return X_val, y_val, group_val

    def _log_overfitting_diagnostics(
        self, evals_result: dict, best_iter: int
    ) -> None:
        """Log train/val NDCG gap and per-round table. Extracted from fit() for readability."""
        train_ndcg = (evals_result.get("training", {}).get("ndcg@5", []) or
                      evals_result.get("training", {}).get("ndcg@1", []))
        val_ndcg   = (evals_result.get("valid_1",  {}).get("ndcg@5", []) or
                      evals_result.get("valid_1",  {}).get("ndcg@1", []))

        if not (train_ndcg and val_ndcg):
            return

        best_idx    = min(best_iter - 1, len(train_ndcg) - 1)
        final_train = train_ndcg[best_idx]
        final_val   = val_ndcg[best_idx]
        gap         = final_train - final_val
        peak_val    = max(val_ndcg)
        peak_iter   = val_ndcg.index(peak_val) + 1

        log.info("-- Overfitting Diagnostics --")
        log.info(f"  Best iteration : {best_iter}")
        log.info(f"  Train NDCG@5   : {final_train:.4f}")
        log.info(f"  Val   NDCG@5   : {final_val:.4f}")
        log.info(f"  Train/Val gap  : {gap:.4f}")
        log.info(f"  Peak val NDCG@5: {peak_val:.4f} @ iter {peak_iter}")

        if gap > 0.05:
            log.warning(f"  OVERFIT DETECTED: gap={gap:.4f} > 0.05 threshold.")
        elif gap > 0.02:
            log.warning(f"  MILD OVERFIT: gap={gap:.4f}. Monitor on next run.")
        else:
            log.info(f"  Generalisation OK: gap={gap:.4f} <= 0.02")

        log.info("  Round | Train NDCG@5 | Val NDCG@5 | Gap")
        for i in range(0, len(train_ndcg), 50):
            log.info(f"  {i+1:5d} | {train_ndcg[i]:.4f}       | "
                     f"{val_ndcg[i]:.4f}     | {train_ndcg[i]-val_ndcg[i]:.4f}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict ranking scores."""
        return self.model.predict(X)

    def post_process(
        self,
        scores: np.ndarray,
        X: np.ndarray,
        query: str,
        products: list,
    ) -> np.ndarray:
        """Deterministic guardrails applied after model scoring.

        Column indices are derived from FeatureEngineer.FEATURE_NAMES so they
        stay correct when the feature set changes — never hardcoded.

        Rules:
        1. Brand Guardrail: demote non-brand products when query contains a
           known brand, unless their score already exceeds brand_demote_threshold.
        2. Category Hierarchy Guardrail: prevent accessories outranking primary
           devices when both appear in the top-10.
        """
        from src.data.normalizer import normalize_entity, normalize_query, KNOWN_BRANDS
        from src.features.feature_engineer import FeatureEngineer

        # Derive column indices from FEATURE_NAMES — never hardcode
        brand_col    = FeatureEngineer.FEATURE_NAMES.index("brand_match")

        scores     = scores.copy()
        norm_query = normalize_query(query)

        # -- Rule 1: Brand Guardrail -------------------------------------------
        query_brand = None
        for b in KNOWN_BRANDS:
            if b in norm_query:
                query_brand = b
                break

        if query_brand is not None:
            non_brand_mask = X[:, brand_col] == 0.0
            demote_mask    = non_brand_mask & (scores < self.cfg.brand_demote_threshold)
            scores[demote_mask] *= self.cfg.brand_demote_factor

        # -- Rule 2: Category Hierarchy Guardrail ------------------------------
        primary_categories   = {"laptops", "phones", "electronics", "shoes", "clothing"}
        accessory_categories = {"accessories", "chargers", "cases", "cables", "covers"}
        query_targets_primary = any(cat in norm_query for cat in primary_categories)

        if query_targets_primary:
            top10_idx        = np.argsort(scores)[-10:]
            top10_products   = [products[i] for i in top10_idx]
            top10_categories = [
                normalize_entity((p.get("category") or "").lower(), "category")
                for p in top10_products
            ]
            primary_in_top10 = any(c in primary_categories for c in top10_categories)
            if primary_in_top10:
                primary_scores = [
                    scores[top10_idx[i]]
                    for i, c in enumerate(top10_categories)
                    if c in primary_categories
                ]
                min_primary_score = min(primary_scores)
                for i, idx in enumerate(top10_idx):
                    if top10_categories[i] in accessory_categories:
                        scores[idx] = min(
                            scores[idx],
                            min_primary_score - self.cfg.brand_demote_factor
                        )

        return scores

    def _safe_path(self, base: str, filename: str) -> Path:
        """Resolve path and guard against traversal outside base dir."""
        base_resolved = Path(base).resolve()
        target        = (base_resolved / filename).resolve()
        if not str(target).startswith(str(base_resolved)):
            raise ValueError(f"Path traversal detected: {target}")
        return target

    def save_model(self, path: str):
        model_path = self._safe_path(path, "ranker.txt")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(model_path))
        log.info(f"Model saved: {model_path}")

    def export_onnx(self, path: str, num_features: int):
        from hummingbird.ml import convert
        onnx_path = self._safe_path(path, "ranker.onnx")
        onnx_path.parent.mkdir(parents=True, exist_ok=True)
        model = convert(self.model, "onnx",
                        np.zeros((1, num_features), dtype=np.float32))
        model.save(str(onnx_path))
        log.info(f"ONNX model exported: {onnx_path}")
