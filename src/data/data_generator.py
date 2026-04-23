import pandas as pd
import numpy as np
from typing import List, Dict, Generator
from datasets import load_dataset
import logging

log = logging.getLogger(__name__)

# Amazon ESCI dataset — correct HuggingFace ID
ESCI_DATASET_ID = "tasksource/esci"

# ESCI label -> relevance score (0-3)
# tasksource/esci uses full strings: 'Exact', 'Substitute', 'Complement', 'Irrelevant'
ESCI_LABEL_MAP = {
    'Exact': 3, 'Substitute': 2, 'Complement': 1, 'Irrelevant': 0,
    'E': 3,     'S': 2,         'C': 1,           'I': 0,          # legacy short-form
}


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
        first_row_logged = False
        for row in dataset:
            if not first_row_logged:
                log.info(f"ESCI sample row keys: {list(row.keys())}")
                log.info(f"ESCI sample esci_label: {row.get('esci_label')} | label: {row.get('label')}")
                first_row_logged = True
            if max_rows and total >= max_rows:
                break
            raw_label = row.get('esci_label')
            relevance = ESCI_LABEL_MAP.get(raw_label, 0)
            batch.append({
                'qid':           str(row['query_id']),
                'pid':           str(row['product_id']),
                'query':         row.get('query') or '',
                'query_lang':    row.get('product_locale') or 'en',
                'product_title': row.get('product_title') or '',
                'brand':         row.get('product_brand') or '',
                'category':      row.get('product_locale') or '',
                'relevance':     relevance
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
        Synthetic data for cold-start markets.
        Relevance is rule-based (text overlap) — no embedding used to avoid leakage.
        """
        products = self._generate_products(num_products)

        for _, prod in products.iterrows():
            for q in self._generate_queries(prod, queries_per):
                relevance = self._rule_based_relevance(q['text'], prod)
                yield {
                    'qid':                 f"synth_q{prod['product_id']}_{q['text'][:20]}",
                    'pid':                 f"synth_p{prod['product_id']}",
                    'query':               q['text'],
                    'query_lang':          q['lang'],
                    'product_title':       prod['title_en'],
                    'product_title_local': prod['title_local'],
                    'brand':               prod['brand'],
                    'category':            prod['category'],
                    'relevance':           relevance
                }

    @staticmethod
    def _rule_based_relevance(query: str, prod: pd.Series) -> int:
        """Assign relevance purely from text rules — no embeddings, no leakage."""
        if 'irrelevant' in query.lower():
            return 0
        q = query.lower()
        brand    = prod['brand'].lower()
        category = prod['category'].lower()
        brand_match    = brand in q
        category_match = category in q
        if brand_match and category_match:
            return 3   # Exact
        if brand_match or category_match:
            return 2   # Substitute
        if any(w in q for w in ['best', 'cheap', 'buy']):
            return 1   # Complement
        return 0       # Irrelevant

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
        if self.num_products > 0:
            log.info(f"Generating {self.num_products} synthetic cold-start products")
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
