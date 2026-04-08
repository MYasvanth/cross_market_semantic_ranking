"""Export LightGBM ranker to ONNX for C++ inference (<30ms p99)."""
import lightgbm as lgb
import numpy as np
from pathlib import Path
import logging

log = logging.getLogger(__name__)


def export_ranker(model_path: str, output_path: str, num_features: int = 8):
    """
    Convert trained LightGBM booster to ONNX.
    Uses hummingbird-ml — no pkg_resources dependency.

    Args:
        model_path:   path to ranker.txt
        output_path:  path to save ranker.onnx
        num_features: number of input features (default 8)
    """
    from hummingbird.ml import convert

    booster  = lgb.Booster(model_file=model_path)
    model    = convert(booster, "onnx", X=np.zeros((1, num_features), dtype=np.float32))
    model.save(output_path)
    log.info(f"ONNX model exported: {output_path}")


if __name__ == "__main__":
    export_ranker(
        model_path   = "artifacts/ranker.txt",
        output_path  = "artifacts/ranker.onnx",
        num_features = 8
    )
