import pandas as pd
import numpy as np
from typing import List, Dict, Generator
from datasets import load_dataset
from src.embeddings.embedding_model import EmbeddingModel
from sklearn.metrics.pairwise import cosine_similarity
import logging

log = logging.getLogger(__name__)

# Amazon ESCI dataset — correct HuggingFace ID
ESCI_DATASET_ID = "tasksource/esci"

# ESCI label -> relevance score (0-3)
ESCI_LABEL_MAP = {'E': 3, 'S': 2, 'C': 1, 'I': 0}


class DataGenerator:
    """
    Data loading strategy:
      Primary   -> Amazon ESCI (real queries + products, streamed)
      Secondary -> Synthetic cold-start data (fills new market gaps)
    """

    def __init__(self, config: Dict):
        self.categories    = config.get('categories', ['Footwear', 'Electronics', 'Clothing', 'Home'])
        self.brands        = config.get('brands', ['Nike', 'Sony', 'Samsung', 'Apple'])
        self.num_products  = config.get('num_products', 1_000)
        self.queries_per   = config.get('queries_per_product', 5)
        self.esci_max_rows = config.get('esci_max_rows', 50_000)
        self.embedding_model = EmbeddingModel(config.get('model_name', 'intfloat/multilingual-e5-base'))

    # ── Primary: Amazon ESCI ───────────────────────────────────────────────

    def load_esci(self, max_rows: int = None) -> Generator[pd.DataFrame, None, None]:
        """
        Stream ESCI in 10k chunks — never loads full dataset into RAM.
        max_rows: cap total rows (None = full dataset ~1.8M rows)
        """
        log.info(f"Streaming ESCI from HuggingFace: {ESCI_DATASET_ID}")
        dataset = load_dataset(ESCI_DATASET_ID, streaming=True, split="train")

        batch = []
        total = 0
        for row in dataset:
            if max_rows and total >= max_rows:
                break
            batch.append({
                'qid':           row['query_id'],
                'pid':           row['product_id'],
                'query':         row['query'],
                'query_lang':    row.get('query_locale', 'en'),
                'product_title': row['product_title'],
                'brand':         row.get('brand', ''),
                'category':      row.get('product_type', row.get('product_locale', '')),
                'relevance':     ESCI_LABEL_MAP.get(row['esci_label'], 0)
            })
            total += 1
            if len(batch) >= 10_000:
                df = pd.DataFrame(batch)
                log.info(f"ESCI chunk: {len(df)} rows (total: {total})")
                yield df
                batch = []
        if batch:
            yield pd.DataFrame(batch)
            log.info(f"ESCI stream complete. Total rows: {total}")

    # ── Secondary: Synthetic Cold-Start ───────────────────────────────────

    def generate_synthetic_stream(
        self, num_products: int = 1_000, queries_per: int = 5
    ) -> Generator[Dict, None, None]:
        """
        Synthetic data for cold-start markets not covered by ESCI.
        Batch encodes all titles + queries — no per-row encode calls.
        """
        products = self._generate_products(num_products)

        # Batch encode all product titles once
        all_prod_embs = self.embedding_model.encode(
            products["title_en"].tolist()
        )

        # Pre-generate all queries then batch encode
        all_queries, all_meta = [], []
        for i, (_, prod) in enumerate(products.iterrows()):
            for q in self._generate_queries(prod, queries_per):
                all_queries.append(q['text'])
                all_meta.append((i, prod, q))

        all_query_embs = self.embedding_model.encode(
            all_queries
        )

        for idx, (prod_idx, prod, q) in enumerate(all_meta):
            sim = cosine_similarity([all_query_embs[idx]], [all_prod_embs[prod_idx]])[0][0]
            if q['lang'] != 'en':
                sim *= 0.9
            relevance = 0 if 'irrelevant' in q['text'] else min(4, max(0, int(sim * 4 + np.random.normal(0, 0.2))))
            yield {
                'qid':                 f"synth_q{prod['product_id']}_{idx}",
                'pid':                 f"synth_p{prod['product_id']}",
                'query':               q['text'],
                'query_lang':          q['lang'],
                'product_title':       prod['title_en'],
                'product_title_local': prod['title_local'],
                'brand':               prod['brand'],
                'category':            prod['category'],
                'relevance':           relevance
            }

    # ── Entry Point: ESCI primary + Synthetic secondary ───────────────────

    def generate(self, use_esci: bool = True) -> Generator[pd.DataFrame, None, None]:
        """
        Always yields synthetic cold-start data first (fast, guaranteed).
        Then streams ESCI if use_esci=True.

        Flow:
            synthetic (1000 products) -> always included
            ESCI stream (10k chunks)  -> if use_esci=True
        """
        # Secondary: synthetic cold-start always runs first
        log.info(f"Generating {self.num_products} synthetic cold-start products")
        
        # Stream synthetic data in 10k chunks
        batch = []
        for row in self.generate_synthetic_stream(self.num_products, self.queries_per):
            batch.append(row)
            if len(batch) >= 10_000:
                yield pd.DataFrame(batch)
                batch = []
        if batch:
            yield pd.DataFrame(batch)
        
        # Primary: ESCI streamed on top
        if use_esci:
            try:
                yield from self.load_esci(max_rows=self.esci_max_rows)
            except Exception as e:
                log.warning(f"ESCI stream failed ({e}) — using synthetic only")

    # ── Helpers ───────────────────────────────────────────────────────────

    def _generate_products(self, n: int) -> pd.DataFrame:
        np.random.seed(42)
        cats   = np.random.choice(self.categories, n)
        brands = np.random.choice(self.brands, n)
        return pd.DataFrame({
            'product_id':  range(n),
            'title_en':    [f"{brands[i]} {cats[i].title()}" for i in range(n)],
            'title_local': [f"Local title {i}" for i in range(n)],
            'brand':       brands,
            'category':    cats
        })

    def _generate_queries(self, prod: pd.Series, n: int) -> List[Dict]:
        templates = [
            {'text': f"{prod['brand']} {prod['category']}",      'lang': 'en'},
            {'text': f"best {prod['category']}",                  'lang': 'en'},
            {'text': f"{prod['brand']} {prod['category']}",       'lang': 'hi'},
            {'text': "irrelevant query",                           'lang': 'en'},
            {'text': f"cheap {prod['brand']} {prod['category']}", 'lang': 'ar'},
        ]
        return templates[:n]
