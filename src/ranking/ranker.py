"""LightGBM LambdaMART Ranker for Precision Re-ranking."""
import lightgbm as lgb
import numpy as np
import logging
from pathlib import Path
from typing import List, Tuple
from omegaconf import DictConfig

log = logging.getLogger(__name__)

class LambdaRanker:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.model = None
    
    def fit(self, X: np.ndarray, y: np.ndarray, group: List[int]) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        """Train LambdaMART with group-aware train/val split. Returns (X_val, y_val, group_val)."""
        n_queries = len(group)
        split_idx = int(n_queries * (1 - self.cfg.test_size))

        train_rows = sum(group[:split_idx])
        X_train, X_val = X[:train_rows], X[train_rows:]
        y_train, y_val = y[:train_rows], y[train_rows:]
        group_train, group_val = group[:split_idx], group[split_idx:]

        train_set = lgb.Dataset(X_train, y_train, group=group_train)
        val_set   = lgb.Dataset(X_val,   y_val,   group=group_val, reference=train_set)

        self.model = lgb.train(
            {
                "objective":    self.cfg.ranker.objective,
                "metric":       self.cfg.ranker.metric,
                "num_leaves":   self.cfg.ranker.num_leaves,
                "learning_rate": self.cfg.learning_rate,
                "verbose":      -1,
            },
            train_set,
            valid_sets=[train_set, val_set],
            num_boost_round=self.cfg.num_boost_round,
            callbacks=[lgb.early_stopping(50)],
        )
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

