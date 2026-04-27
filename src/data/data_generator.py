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

    Augmentation support (optional):
      - LLM-generated realistic titles & 50+ diverse queries per product
      - Hard negative injection for precision@1 training
      - Attribute noise to force semantic channel reliance
      - Synonym injection for cross-lingual robustness
    """

    def __init__(self, config: Dict):
        self.categories    = config.get('categories', ['Footwear', 'Electronics', 'Clothing', 'Home'])
        self.brands        = config.get('brands', ['Nike', 'Sony', 'Samsung', 'Apple'])
        self.num_products  = config.get('num_products', 1_000)
        self.queries_per   = config.get('queries_per_product', 5)
        self.esci_max_rows = config.get('esci_max_rows', 50_000)

        # Augmentation config
        self.use_augmentation = config.get('use_augmentation', False)
        self.augmentor = None
        if self.use_augmentation:
            from src.data.synthetic_augmentor import SyntheticAugmentor
            self.augmentor = SyntheticAugmentor(
                api_key=config.get('grok_api_key'),
                api_endpoint=config.get('grok_api_endpoint'),
                cache_path=config.get('augmentation_cache_path', 'artifacts/synthetic_cache.pkl'),
                use_llm=config.get('use_llm', False),
                queries_per_product=config.get('queries_per_product', 50),
                hard_negative_ratio=config.get('hard_negative_ratio', 0.15),
                attribute_noise_ratio=config.get('attribute_noise_ratio', 0.20),
                synonym_injection_ratio=config.get('synonym_injection_ratio', 0.30),
                llm_model_name=config.get('llm_model_name', 'grok-2-latest'),
                seed=config.get('seed', 42),
            )
            log.info("Synthetic augmentation enabled via SyntheticAugmentor")

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

        If augmentation is enabled, uses SyntheticAugmentor for:
          - Realistic product titles with attributes
          - 50+ diverse queries per product (generic, brand, SKU, descriptive, multilingual)
          - Hard negative injection (confusing near-misses)
          - Attribute noise and synonym injection
        """
        if self.augmentor:
            yield from self._generate_augmented_stream(num_products, queries_per)
        else:
            yield from self._generate_legacy_stream(num_products, queries_per)

    def _generate_augmented_stream(self, num_products: int, queries_per: int) -> Generator[Dict, None, None]:
        """Augmented synthetic stream with LLM/fallback-generated realistic data."""
        products = self.augmentor.generate_catalog(
            n=num_products,
            categories=self.categories,
            brands=self.brands,
        )

        for prod in products:
            # Positive queries
            queries = self.augmentor.generate_queries(prod, n=queries_per)
            for q in queries:
                # Use synonym-aware relevance assignment from augmentor
                relevance = self.augmentor.assign_relevance(q['text'], prod)
                yield {
                    'qid':                 f"synth_q{prod.product_id}_{hash(q['text']) & 0xFFFFFF:06x}",
                    'pid':                 f"synth_p{prod.product_id}",
                    'query':               q['text'],
                    'query_lang':          q.get('lang', 'en'),
                    'product_title':       prod.title_en,
                    'product_title_local': prod.title_local.get(q.get('lang', 'en'), prod.title_local.get('en', '')),
                    'brand':               prod.brand,
                    'category':            prod.category,
                    'relevance':           relevance,
                    'intent':              q.get('intent', 'generic'),
                }

            # Hard negatives (confusing near-misses)
            for q in queries[:max(1, int(queries_per * 0.3))]:
                hard_negs = self.augmentor.generate_hard_negatives(prod, q)
                for neg in hard_negs:
                    yield {
                        'qid':                 f"synth_q{prod.product_id}_hn_{hash(neg['text']) & 0xFFFFFF:06x}",
                        'pid':                 f"synth_p{prod.product_id}",
                        'query':               neg['text'],
                        'query_lang':          neg.get('lang', 'en'),
                        'product_title':       prod.title_en,
                        'product_title_local': prod.title_local.get(neg.get('lang', 'en'), ''),
                        'brand':               prod.brand,
                        'category':            prod.category,
                        'relevance':           0,  # Hard negatives are explicitly irrelevant
                        'intent':              neg.get('intent', 'hard_negative'),
                    }

    def _generate_legacy_stream(self, num_products: int, queries_per: int) -> Generator[Dict, None, None]:
        """Original static template-based synthetic stream (fallback)."""
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
    def _rule_based_relevance(query: str, prod) -> int:
        """Assign relevance purely from text rules — no embeddings, no leakage."""
        from src.data.normalizer import normalize_entity, normalize_query
        if 'irrelevant' in query.lower() or 'hard_negative' in query.lower():
            return 0
        q = normalize_query(query)

        # Handle both AugmentedProduct (dataclass) and pd.Series (dict-like)
        if hasattr(prod, 'brand'):
            # AugmentedProduct or object with attributes
            brand = normalize_entity(getattr(prod, 'brand', ''), 'brand')
            category = normalize_entity(getattr(prod, 'category', ''), 'category')
        else:
            # pd.Series or dict
            brand = normalize_entity(prod.get('brand', ''), 'brand')
            category = normalize_entity(prod.get('category', ''), 'category')

        brand_match    = brand in q
        category_match = category in q
        if brand_match and category_match:
            return 3   # Exact
        if brand_match or category_match:
            return 2   # Substitute
        if any(w in q for w in ['best', 'cheap', 'buy', 'top', 'affordable']):
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
            if self.augmentor:
                log.info(f"Augmentation active: {self.augmentor.queries_per_product} queries/product, "
                         f"hard_neg_ratio={self.augmentor.hard_negative_ratio}, "
                         f"noise_ratio={self.augmentor.attribute_noise_ratio}")
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

    # ── Legacy Helpers ────────────────────────────────────────────────────

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
