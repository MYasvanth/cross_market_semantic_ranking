"""LightGBM LambdaMART Ranker for Precision Re-ranking."""
import lightgbm as lgb
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import List, Tuple

from src.config import RankerConfig

log = logging.getLogger(__name__)

class LambdaRanker:
    def __init__(self, cfg: RankerConfig):
        self.cfg = cfg
        self.model = None
    
    def fit(self, X: np.ndarray, y: np.ndarray, group: List[int]) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        """Train LambdaMART with shuffled group-aware train/val split."""
        rng       = np.random.default_rng(self.cfg.seed)
        n_queries = len(group)
        # Build per-query row slices then shuffle
        sizes  = np.array(group)
        starts = np.concatenate([[0], np.cumsum(sizes[:-1])])
        perm   = rng.permutation(n_queries)
        
        # Build list of row indices, shuffling within each group to avoid position bias
        shuffled_row_indices = []
        for i in perm:
            group_rows = np.arange(starts[i], starts[i] + sizes[i])
            rng.shuffle(group_rows)
            shuffled_row_indices.append(group_rows)
        
        row_idx = np.concatenate(shuffled_row_indices)
        X, y    = X[row_idx], y[row_idx]
        group   = [group[i] for i in perm]
        split_idx  = int(n_queries * (1 - self.cfg.test_size))
        train_rows = sum(group[:split_idx])
        X_train, X_val = X[:train_rows], X[train_rows:]
        y_train, y_val = y[:train_rows], y[train_rows:]
        group_train, group_val = group[:split_idx], group[split_idx:]

        train_set = lgb.Dataset(X_train, y_train, group=group_train)
        val_set   = lgb.Dataset(X_val,   y_val,   group=group_val, reference=train_set)

        max_label = int(y.max()) if y.max() > 0 else 1
        label_gain = list(range(max_label + 1))
        log.info(f"Training on {len(X)} rows, {n_queries} queries. Max label: {max_label}")
        log.info(f"Label distribution: {np.unique(y, return_counts=True)}")

        self.model = lgb.train(
            {
                "objective":               self.cfg.objective,
                "metric":                  self.cfg.metric,
                "num_leaves":              self.cfg.num_leaves,
                "learning_rate":           self.cfg.learning_rate,
                "min_data_in_leaf":        10,
                "min_sum_hessian_in_leaf": 0,
                "label_gain":              label_gain,
                "feature_fraction":        0.8,
                "bagging_fraction":        0.8,
                "bagging_freq":            1,
                "lambda_l2":               0.1,
                "verbose":                 -1,
            },
            train_set,
            valid_sets=[train_set, val_set],
            num_boost_round=self.cfg.num_boost_round,
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)],
        )
        # Log training metrics
        train_scores = self.model.predict(X_train)
        from src.ranking.evaluator import ablation_study
        train_eval_df = pd.DataFrame(X_train, columns=[
            'semantic_sim', 'cross_lingual_sim', 'bm25_score', 'jaccard',
            'brand_match', 'category_match', 'exact_title_match', 'query_len'
        ])
        train_eval_df['relevance'] = y_train
        train_eval_df['ranker_score'] = train_scores
        train_metrics = ablation_study(train_eval_df, group_train)
        log.info(f"Training Metrics:\n{train_metrics.to_string()}")

        return X_val, y_val, group_val
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict ranking scores."""
        return self.model.predict(X)
    
    def _safe_path(self, base: str, filename: str) -> Path:
        """Resolve path and guard against traversal outside base dir."""
        base_resolved = Path(base).resolve()
        target = (base_resolved / filename).resolve()
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
        model = convert(self.model, "onnx", np.zeros((1, num_features), dtype=np.float32))
        model.save(str(onnx_path))
        log.info(f"ONNX model exported: {onnx_path}")

