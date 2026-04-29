"""ZenML materializer for LambdaRanker — uses LightGBM's native txt format."""
import os
from pathlib import Path
from typing import Type

import lightgbm as lgb
from omegaconf import OmegaConf
from zenml.materializers.base_materializer import BaseMaterializer
from zenml.enums import ArtifactType

from src.ranking.ranker import LambdaRanker
from src.config import RankerConfig

MODEL_FILE  = "ranker.txt"
CONFIG_FILE = "ranker_cfg.yaml"


class LambdaRankerMaterializer(BaseMaterializer):
    ASSOCIATED_TYPES = (LambdaRanker,)
    ASSOCIATED_ARTIFACT_TYPE = ArtifactType.MODEL

    def load(self, data_type: Type[LambdaRanker]) -> LambdaRanker:
        loaded_cfg = OmegaConf.load(os.path.join(self.uri, CONFIG_FILE))
        cfg_dict = OmegaConf.to_container(loaded_cfg, resolve=True)
        ranker = LambdaRanker(RankerConfig(**cfg_dict))
        ranker.model = lgb.Booster(model_file=os.path.join(self.uri, MODEL_FILE))
        return ranker

    def save(self, ranker: LambdaRanker) -> None:
        Path(self.uri).mkdir(parents=True, exist_ok=True)
        ranker.model.save_model(os.path.join(self.uri, MODEL_FILE))
        # Handle Pydantic model vs OmegaConf container
        if hasattr(ranker.cfg, "dict"):
            cfg_dict = ranker.cfg.dict()
        else:
            cfg_dict = OmegaConf.to_container(ranker.cfg, resolve=True)
        OmegaConf.save(cfg_dict, os.path.join(self.uri, CONFIG_FILE))
