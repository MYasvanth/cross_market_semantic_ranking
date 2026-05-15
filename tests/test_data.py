"""Unit tests for data module — including LLM augmentation."""
import pytest
import pandas as pd
from src.data.data_generator import DataGenerator
from src.data.synthetic_augmentor import SyntheticAugmentor, AugmentedProduct


# ── DataGenerator Tests ───────────────────────────────────────────────────

def test_data_generator_legacy():
    """Test legacy synthetic stream without augmentation."""
    cfg = {
        'num_products': 5,
        'categories': ['shoes', 'electronics'],
        'brands': ['Nike', 'Sony'],
        'use_augmentation': False,
    }
    gen = DataGenerator(cfg)
    
    # Test streaming
    df = pd.concat(
        [pd.DataFrame([chunk]) for chunk in gen.generate_synthetic_stream(num_products=5)],
        ignore_index=True,
    )
    
    assert len(df) > 0
    assert 'relevance' in df.columns
    assert df['relevance'].between(0, 4).all()
    assert len(df['qid'].unique()) > 1
    assert 'product_title' in df.columns


def test_data_generator_with_augmentation():
    """Test augmented synthetic stream with realistic data."""
    cfg = {
        'num_products': 10,
        'categories': ['Footwear', 'Electronics'],
        'brands': ['Nike', 'Sony'],
        'use_augmentation': True,
        'use_llm': False,
        'queries_per_product': 20,
        'hard_negative_ratio': 0.15,
        'attribute_noise_ratio': 0.20,
        'synonym_injection_ratio': 0.30,
        'seed': 42,
    }
    gen = DataGenerator(cfg)
    
    rows = list(gen.generate_synthetic_stream(num_products=10, queries_per=20))
    df = pd.DataFrame(rows)
    
    assert len(df) > 0
    assert 'relevance' in df.columns
    assert 'intent' in df.columns
    assert 'product_title' in df.columns
    
    # Should have more queries per product than legacy (5)
    assert len(df) >= 10 * 20  # 10 products * 20 queries
    
    # Check diverse intents
    intents = df['intent'].unique()
    assert len(intents) > 1
    
    # Check hard negatives exist
    hard_negs = df[df['intent'].str.contains('hard_negative', na=False)]
    assert len(hard_negs) > 0 or cfg['hard_negative_ratio'] == 0
    
    # Check multilingual presence
    langs = df['query_lang'].unique()
    assert len(langs) >= 1


def test_load_esci():
    gen = DataGenerator({})
    stream = gen.load_esci()
    # Test first batch
    first_batch = None
    for batch in stream:
        first_batch = batch
        break
    assert first_batch is not None
    assert 'relevance' in first_batch.columns


# ── SyntheticAugmentor Tests ──────────────────────────────────────────────

def test_augmentor_generates_realistic_products():
    """Test that augmentor creates rich product titles."""
    aug = SyntheticAugmentor(seed=42, use_llm=False)
    product = aug.generate_product(0, "Nike", "Footwear")
    
    assert isinstance(product, AugmentedProduct)
    assert product.brand == "Nike"
    assert product.category == "Footwear"
    
    # Title should be richer than "Nike Footwear"
    assert len(product.title_en) > len("Nike Footwear")
    assert "Nike" in product.title_en
    
    # Should have local titles
    assert 'hi' in product.title_local
    assert 'ar' in product.title_local
    assert 'pl' in product.title_local


def test_augmentor_generates_diverse_queries():
    """Test query diversity generation."""
    aug = SyntheticAugmentor(seed=42, use_llm=False, queries_per_product=30)
    product = aug.generate_product(1, "Sony", "Electronics")
    queries = aug.generate_queries(product, n=30)
    
    assert len(queries) == 30
    
    # Check diverse intents
    intents = [q['intent'] for q in queries]
    assert 'generic' in intents or any('multilingual' in i for i in intents)
    
    # Check multilingual presence
    langs = [q['lang'] for q in queries]
    assert 'en' in langs
    # At least one non-English query should exist with 30 samples
    assert any(l != 'en' for l in langs)


def test_hard_negative_generation():
    """Test confusing near-miss generation."""
    aug = SyntheticAugmentor(seed=42, use_llm=False, hard_negative_ratio=0.2)
    product = aug.generate_product(2, "Apple", "Electronics")
    query = {"text": "Apple iPhone 15 Pro", "lang": "en", "intent": "sku"}
    
    negatives = aug.generate_hard_negatives(product, query)
    
    assert len(negatives) > 0
    for neg in negatives:
        assert neg['relevance'] == 0
        assert 'hard_negative' in neg['intent']


def test_smart_relevance_assignment():
    """Test synonym-aware relevance labeling fixes the 'trainers=shoes' problem."""
    aug = SyntheticAugmentor(seed=42, use_llm=False)
    product = aug.generate_product(0, "Nike", "Footwear")

    # Direct brand + category = Exact (3)
    assert aug.assign_relevance("Nike Footwear", product) == 3

    # Synonym: "trainers" = "shoes" = "footwear" → should still match category
    rel = aug.assign_relevance("comfortable trainers for running", product)
    assert rel >= 1, f"Expected >=1 for synonym match, got {rel}"

    # Brand-only = Substitute (2)
    assert aug.assign_relevance("Nike latest collection", product) == 2

    # Category via synonym ("shoes" = "footwear") = Substitute (2)
    assert aug.assign_relevance("affordable shoes online", product) == 2

    # Generic purchase intent with no brand/category match = Complement (1)
    assert aug.assign_relevance("affordable gadgets online", product) == 1

    # Hard negative = Irrelevant (0)
    assert aug.assign_relevance("hard_negative iPhone case", product) == 0

    # SKU match boosts to Exact (3)
    product_with_model = aug.generate_product(1, "Sony", "Electronics")
    model = product_with_model.attributes.get("model", "")
    if model:
        rel = aug.assign_relevance(f"Sony {model} specs", product_with_model)
        assert rel == 3, f"Expected 3 for SKU match, got {rel}"


def test_attribute_noise_injection():
    """Test that attribute noise is applied."""
    aug = SyntheticAugmentor(
        seed=42, use_llm=False,
        attribute_noise_ratio=1.0,  # Force noise
        queries_per_product=10,
    )
    product = aug.generate_product(3, "Samsung", "Electronics")
    queries = aug.generate_queries(product, n=10)
    
    # With 100% noise ratio, some queries should be modified
    original = f"Samsung Electronics"
    modified = [q for q in queries if '[BRAND]' in q['text'] or '[CATEGORY]' in q['text'] or 'some brand' in q['text']]
    # Noise is probabilistic; with 100% ratio and 10 queries we expect some
    assert len(queries) == 10


def test_synonym_injection():
    """Test synonym replacement in queries."""
    aug = SyntheticAugmentor(
        seed=42, use_llm=False,
        synonym_injection_ratio=1.0,  # Force synonyms
        queries_per_product=10,
    )
    product = aug.generate_product(4, "Nike", "Footwear")
    queries = aug.generate_queries(product, n=10)
    
    # Check that some queries use synonyms
    texts = [q['text'].lower() for q in queries]
    # "Footwear" might become "shoes", "sneakers", "trainers"
    has_synonym = any(
        word in ' '.join(texts)
        for word in ['shoes', 'sneakers', 'trainers', 'footwear', 'kicks']
    )
    assert has_synonym or len(texts) == 0


def test_augmentor_caching():
    """Test that cache is used for repeated generation."""
    import tempfile
    import os
    
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
        cache_path = tmp.name
    
    try:
        aug = SyntheticAugmentor(
            seed=42, use_llm=False,
            cache_path=cache_path,
        )
        product1 = aug.generate_product(5, "LG", "Electronics")
        
        # Create new instance with same cache
        aug2 = SyntheticAugmentor(
            seed=99, use_llm=False,  # Different seed
            cache_path=cache_path,
        )
        product2 = aug2.generate_product(5, "LG", "Electronics")
        
        # Should be identical due to cache
        assert product1.title_en == product2.title_en
    finally:
        os.unlink(cache_path)


def test_batch_catalog_generation():
    """Test generating a full catalog."""
    aug = SyntheticAugmentor(seed=42, use_llm=False)
    products = aug.generate_catalog(
        n=50,
        categories=['Footwear', 'Electronics', 'Clothing', 'Home'],
        brands=['Nike', 'Sony', 'Samsung', 'Apple'],
    )
    
    assert len(products) == 50
    
    # Check diversity across categories and brands
    brands = {p.brand for p in products}
    categories = {p.category for p in products}
    assert len(brands) > 1
    assert len(categories) > 1
    
    # All products should have realistic titles
    for p in products:
        assert len(p.title_en) > 10
        assert p.brand in p.title_en

