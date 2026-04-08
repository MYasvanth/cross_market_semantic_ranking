"""Unit tests for data module."""
import pytest
import pandas as pd
from src.data.data_generator import DataGenerator

def test_data_generator():
    cfg = {
        'num_products': 5,
        'categories': ['shoes', 'electronics'],
        'brands': ['Nike', 'Sony']
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

def test_load_esci():
    gen = DataGenerator({})
    stream = gen.load_esci()
    # Test first batch
    first_batch = None
    for batch in stream():
        first_batch = batch
        break
    assert first_batch is not None
    assert 'relevance' in first_batch.columns

