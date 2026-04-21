"""Typed config — single source of truth for the entire pipeline."""
from pydantic import BaseModel, Field


class DataConfig(BaseModel):
    num_products:  int   = Field(1000,    ge=0)
    queries_per:   int   = Field(5,       gt=0)
    use_esci:      bool  = False
    esci_max_rows: int   = Field(50_000,  gt=0)
    categories:    list  = ["Footwear", "Electronics", "Clothing", "Home"]
    brands:        list  = ["Nike", "Sony", "Samsung", "Apple"]


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
