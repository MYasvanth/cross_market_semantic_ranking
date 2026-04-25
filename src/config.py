"""Typed config — single source of truth for the entire pipeline."""
import os
from typing import Optional

from pydantic import BaseModel, Field


class DataConfig(BaseModel):
    num_products:  int   = Field(1000,    ge=0)
    queries_per:   int   = Field(5,       gt=0)
    use_esci:      bool  = False
    esci_max_rows: int   = Field(50_000,  gt=0)
    categories:    list  = ["Footwear", "Electronics", "Clothing", "Home"]
    brands:        list  = ["Nike", "Sony", "Samsung", "Apple"]

    # ── Augmentation settings ────────────────────────────────────────────
    use_augmentation:        bool    = True
    use_llm:                 bool    = True
    grok_api_key:            Optional[str] = Field(default_factory=lambda: os.getenv("GROQ_API_KEY") or os.getenv("GROK_API_KEY"))
    grok_api_endpoint:       Optional[str] = "https://api.groq.com/openai/v1/chat/completions"
    llm_model_name:          str     = "llama-3.1-8b-instant"
    augmentation_cache_path: str     = "artifacts/synthetic_cache.pkl"
    hard_negative_ratio:     float   = Field(0.10, ge=0.0, le=1.0)
    attribute_noise_ratio:   float   = Field(0.05, ge=0.0, le=1.0)
    synonym_injection_ratio: float   = Field(0.30, ge=0.0, le=1.0)

    # ── Retrieval settings ───────────────────────────────────────────────
    use_hybrid_retrieval:    bool    = True
    retrieval_k:             int     = Field(50, gt=0)
    rrf_k:                   int     = Field(60, gt=0)
    semantic_weight:         float   = Field(0.5, ge=0.0, le=1.0)
    bm25_weight:             float   = Field(0.5, ge=0.0, le=1.0)
    seed:                    int     = 42


class RankerConfig(BaseModel):
    objective:       str   = "lambdarank"
    metric:          str   = "ndcg"
    num_leaves:      int   = Field(31,   gt=0)
    learning_rate:   float = Field(0.05, gt=0.0, lt=1.0)
    num_boost_round: int   = Field(300,  gt=0)
    test_size:       float = Field(0.2,  gt=0.0, lt=1.0)
    seed:            int   = 42


class PipelineConfig(BaseModel):
    embedding_model_name: str          = "intfloat/multilingual-e5-base"
    data:                 DataConfig   = DataConfig()
    ranker:               RankerConfig = RankerConfig()
