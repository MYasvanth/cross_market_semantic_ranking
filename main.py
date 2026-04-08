#!/usr/bin/env python3
"""
Cross-Market Semantic Ranking — ZenML Pipeline Entry Point
Usage:
  python main.py                          # synthetic data, default config
  python main.py data.use_esci=true       # stream from HuggingFace ESCI
  python main.py data.num_products=50000  # larger synthetic dataset
"""
import os
import sys
import hydra
from omegaconf import DictConfig

os.environ["HYDRA_FULL_ERROR"] = "1"
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr.encoding != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8")

from src.pipeline.training_pipeline import ranking_pipeline


@hydra.main(version_base=None, config_path="configs", config_name="pipeline")
def main(cfg: DictConfig):
    ranking_pipeline(
        num_products         = cfg.data.num_products,
        queries_per          = cfg.data.queries_per_product,
        use_esci             = cfg.data.use_esci,
        esci_max_rows        = cfg.data.esci_max_rows,
        embedding_model_name = cfg.model.embedding,
        num_boost_round      = cfg.model.ranker.num_boost_round,
        num_leaves           = cfg.model.ranker.num_leaves,
        learning_rate        = cfg.model.ranker.learning_rate,
        categories           = list(cfg.data.categories),
        brands               = list(cfg.data.brands),
    )


if __name__ == "__main__":
    main()
