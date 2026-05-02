#!/usr/bin/env python3
"""Cross-Market Semantic Ranking — ZenML Pipeline Entry Point"""
import os
import sys
import tempfile

# Redirect temp dir to D: before any imports — prevents OSError on C: (low space)
# numpy.save and ZenML materializers write large temp files before renaming
_tmp = "D:\\tmp"
os.makedirs(_tmp, exist_ok=True)
os.environ["TEMP"] = os.environ["TMP"] = os.environ["TMPDIR"] = _tmp
tempfile.tempdir = _tmp

from dotenv import load_dotenv
load_dotenv()
from omegaconf import OmegaConf
from src.config import PipelineConfig, DataConfig, RankerConfig
from src.pipeline.training_pipeline import ranking_pipeline

if __name__ == "__main__":
    # Create default config
    base_cfg = PipelineConfig()
    
    # Merge with command line arguments (e.g. data.use_esci=true)
    cli_cfg = OmegaConf.from_cli(sys.argv[1:])
    merged_cfg = OmegaConf.merge(base_cfg.model_dump(), cli_cfg)
    
    # Convert back to PipelineConfig pydantic model
    cfg = PipelineConfig(**merged_cfg)
    
    ranking_pipeline(cfg)
