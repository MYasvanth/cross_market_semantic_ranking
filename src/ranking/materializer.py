"""ZenML materializer for LambdaRanker — uses LightGBM's native txt format."""
import os
from pathlib import Path
from typing import Type

import lightgbm as lgb
from omegaconf import OmegaConf
from zenml.materializers.base_materializer import BaseMaterializer
from zenml.enums import ArtifactType

from src.ranking.ranker import LambdaRanker

MODEL_FILE  = "ranker.txt"
CONFIG_FILE = "ranker_cfg.yaml"


class LambdaRankerMaterializer(BaseMaterializer):
    ASSOCIATED_TYPES = (LambdaRanker,)
    ASSOCIATED_ARTIFACT_TYPE = ArtifactType.MODEL

    def load(self, data_type: Type[LambdaRanker]) -> LambdaRanker:
        cfg = OmegaConf.load(os.path.join(self.uri, CONFIG_FILE))
        ranker = LambdaRanker(cfg)
        ranker.model = lgb.Booster(model_file=os.path.join(self.uri, MODEL_FILE))
        return ranker

    def save(self, ranker: LambdaRanker) -> None:
        Path(self.uri).mkdir(parents=True, exist_ok=True)
        ranker.model.save_model(os.path.join(self.uri, MODEL_FILE))
        OmegaConf.save(ranker.cfg, os.path.join(self.uri, CONFIG_FILE))
