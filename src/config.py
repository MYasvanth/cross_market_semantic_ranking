"""Typed config — single source of truth for the entire pipeline."""
from pydantic import BaseModel, Field


class VectorStoreConfig(BaseModel):
    hnsw_m:      int = Field(32,          gt=0)
    ef_search:   int = Field(256,         gt=0)  # increased from 64 for better recall
    max_vectors: int = Field(10_000_000,  gt=0)
    max_k:       int = Field(1_000,       gt=0)


class DataConfig(BaseModel):
    num_products:  int   = Field(2000,    ge=0)   # balanced synthetic coverage
    queries_per:   int   = Field(7,       gt=0)   # enough diversity, not excessive
    use_esci:      bool  = True
    esci_max_rows: int   = Field(200_000, gt=0)   # 2x diversity, ~6h total runtime
    categories:    list  = [
        "Running Shoes", "Basketball Shoes", "Casual Sneakers",
        "Smartphones", "Laptops", "Headphones", "Tablets",
        "T-Shirts", "Jackets", "Jeans",
        "Kitchen", "Bedroom", "Living Room",
    ]
    brands:        list  = ["Nike", "Sony", "Samsung", "Apple", "Adidas", "LG", "Dell", "Puma"]

    # ── Augmentation settings ────────────────────────────────────────────
    use_augmentation:        bool    = True
    use_llm:                 bool    = False
    use_llm_titles:          bool    = True   # 1 call/product → 5000 calls total, fits Groq free tier
    use_llm_queries:         bool    = False  # 1 call/product → same 5000 calls, enable when keys available
    # LLM keys are read from env: GROQ_API_KEY, GEMINI_API_KEY, TOGETHER_API_KEY
    # Provider chain: Groq (14400/day) → Gemini (1500/day) → Together AI (no daily limit)
    # Both titles+queries together = 2 calls/product = 10,000 calls — still fits Groq free tier
    augmentation_cache_path: str     = "artifacts/synthetic_cache.pkl"
    hard_negative_ratio:     float   = Field(0.15, ge=0.0, le=1.0)  # safer than 0.25 — avoids cannibalizing Substitute class
    attribute_noise_ratio:   float   = Field(0.05, ge=0.0, le=1.0)
    synonym_injection_ratio: float   = Field(0.30, ge=0.0, le=1.0)

    # ── Retrieval settings ───────────────────────────────────────────────
    use_hybrid_retrieval:      bool  = True
    retrieval_k:               int   = Field(400, gt=0)   # was 200 — more candidates raises recall ceiling
    rrf_k:                     int   = Field(30,  gt=0)   # was 60 — lower rrf_k sharpens top-rank signal
    semantic_weight:           float = Field(0.7, ge=0.0, le=1.0)  # e5-base is strong; trust it more
    bm25_weight:               float = Field(0.3, ge=0.0, le=1.0)  # BM25 as recall safety net
    seed:                      int   = 42
    # ── Worker / candidate settings ─────────────────────────────────────
    num_workers:               int   = Field(6,   gt=0)
    max_hard_negatives:        int   = Field(10,  gt=0)
    hard_neg_score_threshold:  float = Field(0.7, ge=0.0, le=1.0)
    semantic_score_threshold:  float = Field(0.8, ge=0.0, le=1.0)
    # ── Cross-encoder settings ───────────────────────────────────────────
    use_cross_encoder_distillation: bool = False
    use_cross_encoder:             bool = False
    cross_encoder_model_name:      str  = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    ce_top_k:                      int  = Field(50, gt=0)


class RankerConfig(BaseModel):
    objective:              str   = "lambdarank"
    metric:                 str   = "ndcg"
    num_leaves:             int   = Field(15,   gt=0)   # halved to reduce tree complexity
    max_depth:              int   = Field(6,    gt=0)   # cap tree depth — was unbounded
    learning_rate:          float = Field(0.03, gt=0.0, lt=1.0)
    num_boost_round:        int   = Field(2000, gt=0)
    test_size:              float = Field(0.2,  gt=0.0, lt=1.0)
    seed:                   int   = 42
    # ── LightGBM regularization ──────────────────────────────────────────
    min_data_in_leaf:       int   = Field(300,  gt=0)   # increased to force broader splits
    min_gain_to_split:      float = Field(0.01, ge=0.0) # prune splits with insufficient gain
    feature_fraction:       float = Field(0.8,  gt=0.0, le=1.0)
    bagging_fraction:       float = Field(0.8,  gt=0.0, le=1.0)
    bagging_freq:           int   = Field(1,    ge=0)
    lambda_l2:              float = Field(20.0, ge=0.0)  # stronger regularization
    lambda_l1:              float = Field(2.0,  ge=0.0)
    neg_bagging_fraction:   float = Field(0.5,  gt=0.0, le=1.0)
    early_stopping_rounds:  int   = Field(100,  gt=0)
    neg_downsample_ratio:   float = Field(39.0, gt=0.0)  # match real serving distribution (~39:1 at retrieval_k=400)
    # ── Post-processing guardrails ───────────────────────────────────────
    brand_demote_threshold: float = Field(0.9,  ge=0.0, le=1.0)
    brand_demote_factor:    float = Field(0.1,  ge=0.0, le=1.0)
    # ── Feature normalization ────────────────────────────────────────────
    bm25_norm_factor:       float = Field(10.0, gt=0.0)


class PipelineConfig(BaseModel):
    embedding_model_name: str               = "intfloat/multilingual-e5-base"
    data:                 DataConfig        = DataConfig()
    ranker:               RankerConfig      = RankerConfig()
    vector_store:         VectorStoreConfig = VectorStoreConfig()
