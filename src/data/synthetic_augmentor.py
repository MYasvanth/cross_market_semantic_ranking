"""LLM-Augmented Synthetic Data Generator with rich local fallback.

Provides:
  1. Reality Injection — realistic product titles instead of "Brand Category"
  2. Query Diversity — 50+ diverse queries per product (generic, brand, SKU, descriptive, multilingual)
  3. Hard Negative Generation — confusing near-misses for precision@1 training
  4. Attribute Noise — random masking of brands/categories to force semantic reliance
  5. Synonym Injection — thesaurus-based term replacement for cross-lingual robustness
  6. Realistic Local Titles — translated product descriptions in target languages
"""

import json
import logging
import os
import pickle
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.data.normalizer import (
    BRAND_SYNONYMS,
    CATEGORY_SYNONYMS,
    normalize_entity,
)

log = logging.getLogger(__name__)

# ── Rich Local Fallback Data ──────────────────────────────────────────────

PRODUCT_TEMPLATES = {
    "Footwear": [
        "{brand} Air {model} — {adjective} {gender} {category}, {color}, Size {size}",
        "{brand} {model} Pro — Lightweight {gender} {category} for {activity}, {color}",
        "{brand} {model} Ultra — Cushioned {category} with {tech}, {color}, {size}",
        "{brand} {category} {model} — Breathable Mesh Upper, {color}, {width} Fit",
        "{brand} Performance {category} — {activity}-Ready, {tech} Sole, {color}",
    ],
    "Electronics": [
        "{brand} {model} — {adjective} {category} with {tech}, {color}, {storage}",
        "{brand} {model} Pro — Flagship {category}, {display} Display, {storage}",
        "{brand} {category} {model} — {tech} Processor, {color}, {storage} Storage",
        "{brand} Wireless {category} — Noise Cancelling, {battery} Battery, {color}",
        "{brand} Smart {category} — {tech} Sensor, {display} Screen, {color}",
    ],
    "Clothing": [
        "{brand} {model} — {adjective} {gender} {category}, {color}, Size {size}",
        "{brand} {category} {model} — {tech} Fabric, {color}, {fit} Fit",
        "{brand} Performance {category} — Moisture-Wicking, {color}, {size}",
        "{brand} {adjective} {category} — {tech} Material, {color}, {gender}",
        "{brand} Everyday {category} — Comfortable {fit} Fit, {color}, Size {size}",
    ],
    "Home": [
        "{brand} {model} — {adjective} {category}, {color}, {size} Dimensions",
        "{brand} Smart {category} — {tech} Enabled, {color}, {storage} Capacity",
        "{brand} Premium {category} — {tech} Material, {color}, {size}",
        "{brand} {category} {model} — Energy Efficient, {color}, {display} Controls",
        "{brand} Modern {category} — {adjective} Design, {color}, {size}",
    ],
}

# Slot fillers for templates
_SLOT_FILLERS = {
    "model": [
        "Max 270", "Ultra Boost", "Phantom", "Zoom Fly", "React",
        "X1000", "Pro 5", "Elite", "Studio", "Vision",
        "Aero", "Nova", "Prime", "Edge", "Core",
    ],
    "adjective": [
        "Premium", "Breathable", "Durable", "Lightweight", "Ergonomic",
        "Stylish", "Comfortable", "High-Performance", "Sleek", "Versatile",
    ],
    "gender": ["Men's", "Women's", "Unisex", "Kids'", "Youth"],
    "color": [
        "Triple Black", "White/Red", "Navy Blue", "Olive Green", "Charcoal Grey",
        "Midnight Blue", "Pearl White", "Crimson Red", "Stealth Grey", "Electric Blue",
    ],
    "size": ["7", "8", "9", "10", "11", "12", "S", "M", "L", "XL", "XXL"],
    "width": ["Standard", "Wide", "Narrow", "Extra Wide"],
    "activity": ["Running", "Training", "Marathons", "Gym", "Casual Wear", "Sports"],
    "tech": [
        "Air Cushion", "Gel", "React Foam", "Memory Foam", "Carbon Fiber",
        "AI-Powered", "Quantum", "Neural", "Bluetooth 5.3", "Wi-Fi 6E",
    ],
    "storage": ["128GB", "256GB", "512GB", "1TB", "64GB"],
    "display": ["OLED", "AMOLED", "Retina", "4K", "QHD", "FHD+"],
    "battery": ["24h", "36h", "48h", "72h", "All-Day"],
    "fit": ["Slim", "Regular", "Relaxed", "Athletic", "Tailored"],
}

# Query diversity templates by intent type
QUERY_TEMPLATES = {
    "generic": [
        "affordably priced {syn_category} for {activity}",
        "best {syn_category} under {price}",
        "top rated {syn_category} {year}",
        "where to buy {syn_category} online",
        "{syn_category} recommendations for beginners",
        "cheap {syn_category} with free shipping",
        "high quality {syn_category} reviews",
        "most comfortable {syn_category}",
        "{syn_category} for {activity} in {location}",
        "what are the best {syn_category} this year",
    ],
    "brand": [
        "{syn_brand} {syn_category} latest collection",
        "official {syn_brand} store {syn_category}",
        "{syn_brand} {syn_category} price comparison",
        "authentic {syn_brand} {syn_category} online",
        "{syn_brand} new releases {year}",
        "{syn_brand} {syn_category} warranty",
        "why choose {syn_brand} {syn_category}",
        "{syn_brand} vs competitors {syn_category}",
    ],
    "sku": [
        "{syn_brand} {model} {syn_category} specs",
        "{syn_brand} {model} price in {location}",
        "{syn_brand} {model} review and unboxing",
        "buy {syn_brand} {model} {syn_category} online",
        "{syn_brand} {model} {color} availability",
        "{syn_brand} {model} {storage} best deal",
        "latest {year} {syn_brand} {model} release date",
    ],
    "descriptive": [
        "{color} {syn_category} with {tech}",
        "{adjective} {syn_category} for {activity}",
        "{syn_category} with {tech} and {color}",
        "{gender} {syn_category} {color} size {size}",
        "{syn_category} featuring {tech} technology",
        "lightweight {syn_category} for {activity}",
        "{color} {syn_category} {adjective} design",
    ],
}

# Multilingual query templates
MULTILINGUAL_QUERIES = {
    "hi": [
        "{syn_brand} {syn_category} खरीदें",  # Buy {brand} {category}
        "सस्ते {syn_category} ऑनलाइन",         # Cheap {category} online
        "{syn_category} रनिंग के लिए",         # {category} for running
        "{syn_brand} {model} कीमत",            # {brand} {model} price
        "बेहतरीन {syn_category} {year}",       # Best {category} {year}
    ],
    "ar": [
        "شراء {syn_brand} {syn_category}",     # Buy {brand} {category}
        "أفضل {syn_category} بسعر رخيص",        # Best {category} cheap price
        "{syn_category} للجري",                # {category} for running
        "سعر {syn_brand} {model}",             # Price of {brand} {model}
        "{syn_category} عالي الجودة",          # High quality {category}
    ],
    "pl": [
        "kup {syn_brand} {syn_category}",      # Buy {brand} {category}
        "tanie {syn_category} online",         # Cheap {category} online
        "{syn_category} do biegania",          # {category} for running
        "cena {syn_brand} {model}",            # Price of {brand} {model}
        "najlepsze {syn_category} {year}",     # Best {category} {year}
    ],
    "en": [
        "buy {syn_brand} {syn_category}",
        "{syn_category} for {activity}",
        "{syn_brand} {model} price",
        "best {syn_category} {year}",
        "{color} {syn_category} with {tech}",
    ],
}

# Hard negative templates (confusing near-misses)
HARD_NEGATIVE_TEMPLATES = [
    # Same brand, wrong model
    {"query": "{brand} {wrong_model} {category}", "reason": "wrong_model"},
    # Same category, wrong brand
    {"query": "{wrong_brand} {model} {category}", "reason": "wrong_brand"},
    # Previous generation
    {"query": "{brand} {model} {prev_year} {category}", "reason": "old_generation"},
    # Different size/color (for exact match seekers)
    {"query": "{brand} {model} {wrong_color} {category} size {wrong_size}", "reason": "wrong_variant"},
    # Similar product type, wrong category
    {"query": "{brand} {category} accessories for {model}", "reason": "accessory_not_product"},
    # Competitor equivalent
    {"query": "{wrong_brand} equivalent to {brand} {model}", "reason": "competitor"},
]

# Synonym expansion maps
_SYNONYM_MAP = {
    "shoes": ["shoes", "footwear", "sneakers", "running shoes", "trainers", "kicks", "runners"],
    "electronics": ["electronics", "gadgets", "devices", "tech", "consumer electronics"],
    "clothing": ["clothing", "apparel", "garments", "fashion", "wear"],
    "home": ["home", "household", "home goods", "interior", "domestic"],
}

_BRAND_SYNONYM_MAP = {
    "nike": ["nike", "nik", "nike inc"],
    "sony": ["sony", "sony corp", "sony electronics"],
    "samsung": ["samsung", "samsung electronics", "samsung corp"],
    "apple": ["apple", "apple inc", "apple computer"],
}

@dataclass
class AugmentedProduct:
    """Rich product representation with realistic titles."""
    product_id: int
    title_en: str
    title_local: Dict[str, str] = field(default_factory=dict)
    brand: str = ""
    category: str = ""
    attributes: Dict[str, str] = field(default_factory=dict)


class SyntheticAugmentor:
    """
    LLM-backed synthetic data augmentor with rich local fallback.

    When an LLM API (Grok/OpenAI/etc.) is available, it generates high-quality
    realistic data. Otherwise, it uses template-based generation with deep
    variation to simulate LLM diversity.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_endpoint: Optional[str] = None,
        cache_path: Optional[str] = None,
        use_llm: bool = False,
        llm_model_name: str = "grok-2-latest",
        queries_per_product: int = 50,
        hard_negative_ratio: float = 0.15,
        attribute_noise_ratio: float = 0.20,
        synonym_injection_ratio: float = 0.30,
        seed: int = 42,
    ):
        # Only fall back to env vars when api_key is literally None (not explicit '')
        key_val = api_key if api_key is not None else (os.getenv("GROQ_API_KEY") or os.getenv("GROK_API_KEY") or os.getenv("OPENAI_API_KEY"))
        self.api_key = key_val.strip() if isinstance(key_val, str) else key_val
        self.api_endpoint = api_endpoint or os.getenv("GROK_API_ENDPOINT")
        self.llm_model_name = llm_model_name
        self.use_llm = use_llm and bool(self.api_key)
        self._llm_available = self.use_llm  # circuit-breaker: flipped to False after first total failure
        self.cache_path = Path(cache_path) if cache_path else None
        self.queries_per_product = queries_per_product
        self.hard_negative_ratio = hard_negative_ratio
        self.attribute_noise_ratio = attribute_noise_ratio
        self.synonym_injection_ratio = synonym_injection_ratio
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

        # Load cache if exists
        self._cache: Dict = {}
        if self.cache_path and self.cache_path.exists():
            try:
                with open(self.cache_path, "rb") as f:
                    self._cache = pickle.load(f)
                log.info(f"Loaded augmentation cache: {self.cache_path}")
            except Exception as e:
                log.warning(f"Failed to load cache: {e}")

    def _save_cache(self):
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_path, "wb") as f:
                pickle.dump(self._cache, f)

    def _call_llm(self, prompt: str, max_tokens: int = 500) -> Optional[str]:
        """Call LLM API if available. Returns None on failure."""
        if not self.use_llm or not self._llm_available:
            return None
        try:
            import time
            import requests
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "CrossMarketRanking/1.0",
            }
            endpoint = self.api_endpoint or "https://api.x.ai/v1/chat/completions"
            payload = {
                "model": self.llm_model_name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0.7,
                "stream": False,
            }
            resp = requests.post(endpoint, headers=headers, json=payload, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                if "choices" in data and len(data["choices"]) > 0:
                    return data["choices"][0]["message"]["content"]
            else:
                log.warning(f"LLM {resp.status_code}: {resp.text[:200]} — falling back to templates")
                self._llm_available = False  # circuit-break: stop retrying
            return None
        except Exception as e:
            log.warning(f"LLM call failed: {e}")
            return None

    # ── Product Generation ────────────────────────────────────────────────

    def generate_product(self, product_id: int, brand: str, category: str) -> AugmentedProduct:
        """Generate a realistic product with rich attributes."""
        cache_key = f"prod_{brand}_{category}_{product_id}"
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            return AugmentedProduct(**cached)

        # Try LLM first
        if self.use_llm:
            llm_result = self._llm_generate_product(brand, category)
            if llm_result:
                self._cache[cache_key] = llm_result.__dict__
                self._save_cache()
                return llm_result

        # Fallback: rich template-based generation
        product = self._fallback_generate_product(product_id, brand, category)
        self._cache[cache_key] = product.__dict__
        self._save_cache()
        return product

    def _llm_generate_product(self, brand: str, category: str) -> Optional[AugmentedProduct]:
        prompt = (
            f"Generate a realistic e-commerce product title for a {brand} {category}. "
            f"Include model name, key features, color, and size. "
            f"Also provide translations in Hindi, Arabic, and Polish. "
            f"Return as JSON with keys: title_en, title_hi, title_ar, title_pl, model, color, size, tech."
        )
        response = self._call_llm(prompt)
        if not response:
            return None
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return AugmentedProduct(
                    product_id=-1,  # Will be set by caller
                    title_en=data.get("title_en", f"{brand} {category}"),
                    title_local={
                        "hi": data.get("title_hi", ""),
                        "ar": data.get("title_ar", ""),
                        "pl": data.get("title_pl", ""),
                    },
                    brand=brand,
                    category=category,
                    attributes={
                        "model": data.get("model", ""),
                        "color": data.get("color", ""),
                        "size": data.get("size", ""),
                        "tech": data.get("tech", ""),
                    },
                )
        except Exception as e:
            log.warning(f"Failed to parse LLM product response: {e}")
        return None

    def _fallback_generate_product(self, product_id: int, brand: str, category: str) -> AugmentedProduct:
        """Rich template-based product generation."""
        templates = PRODUCT_TEMPLATES.get(category, PRODUCT_TEMPLATES["Electronics"])
        template = self.rng.choice(templates)

        # Fill slots
        slots = {}
        for slot_name, options in _SLOT_FILLERS.items():
            slots[slot_name] = self.rng.choice(options)

        title_en = template.format(brand=brand, category=category, **slots)

        # Generate local titles with realistic translations
        title_local = {
            "hi": self._translate_to_hindi(title_en, brand, category),
            "ar": self._translate_to_arabic(title_en, brand, category),
            "pl": self._translate_to_polish(title_en, brand, category),
        }

        return AugmentedProduct(
            product_id=product_id,
            title_en=title_en,
            title_local=title_local,
            brand=brand,
            category=category,
            attributes=slots,
        )

    def _translate_to_hindi(self, title: str, brand: str, category: str) -> str:
        """Generate realistic Hindi product title."""
        translations = {
            "Footwear": "जूते",
            "Electronics": "इलेक्ट्रॉनिक्स",
            "Clothing": "कपड़े",
            "Home": "घर",
            "Men's": "पुरुषों के लिए",
            "Women's": "महिलाओं के लिए",
            "Running": "दौड़ने",
            "Training": "प्रशिक्षण",
        }
        result = title
        for en, hi in translations.items():
            result = result.replace(en, hi)
        return result if result != title else f"{brand} {category} हिंदी में"

    def _translate_to_arabic(self, title: str, brand: str, category: str) -> str:
        """Generate realistic Arabic product title."""
        translations = {
            "Footwear": "أحذية",
            "Electronics": "إلكترونيات",
            "Clothing": "ملابس",
            "Home": "منزل",
            "Men's": "رجالي",
            "Women's": "نسائي",
            "Running": "الجري",
            "Training": "التدريب",
        }
        result = title
        for en, ar in translations.items():
            result = result.replace(en, ar)
        return result if result != title else f"{brand} {category} بالعربية"

    def _translate_to_polish(self, title: str, brand: str, category: str) -> str:
        """Generate realistic Polish product title."""
        translations = {
            "Footwear": "obuwie",
            "Electronics": "elektronika",
            "Clothing": "odzież",
            "Home": "dom",
            "Men's": "męskie",
            "Women's": "damskie",
            "Running": "biegania",
            "Training": "treningu",
        }
        result = title
        for en, pl in translations.items():
            result = result.replace(en, pl)
        return result if result != title else f"{brand} {category} po polsku"

    # ── Query Generation ──────────────────────────────────────────────────

    def generate_queries(self, product: AugmentedProduct, n: int = None) -> List[Dict[str, str]]:
        """Generate diverse queries for a product."""
        n = n or self.queries_per_product
        cache_key = f"queries_{product.brand}_{product.category}_{product.product_id}_{n}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        if self.use_llm:
            llm_queries = self._llm_generate_queries(product, n)
            if llm_queries:
                self._cache[cache_key] = llm_queries
                self._save_cache()
                return llm_queries

        queries = self._fallback_generate_queries(product, n)
        self._cache[cache_key] = queries
        self._save_cache()
        return queries

    def _llm_generate_queries(self, product: AugmentedProduct, n: int) -> Optional[List[Dict]]:
        prompt = (
            f"Generate {n} diverse user search queries for this product: {product.title_en}. "
            f"Include: generic queries, brand-specific queries, SKU-seeking queries, "
            f"descriptive queries, and queries in Hindi, Arabic, and Polish. "
            f"Return as JSON list with fields: text, lang, intent (generic/brand/sku/descriptive)."
        )
        response = self._call_llm(prompt, max_tokens=1000)
        if not response:
            return None
        try:
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return [{"text": q["text"], "lang": q.get("lang", "en"), "intent": q.get("intent", "generic")} for q in data]
        except Exception as e:
            log.warning(f"Failed to parse LLM query response: {e}")
        return None

    def _fallback_generate_queries(self, product: AugmentedProduct, n: int) -> List[Dict[str, str]]:
        """Generate diverse queries using templates with synonym injection and noise."""
        queries = []
        slots = product.attributes.copy()
        slots["brand"] = product.brand
        slots["category"] = product.category
        slots["year"] = str(2023 + self.rng.randint(0, 2))
        slots["price"] = f"${self.rng.randint(50, 500)}"
        slots["location"] = self.rng.choice(["India", "Egypt", "Poland", "USA", "UK"])

        # Generate from each intent type
        all_templates = []
        for intent, templates in QUERY_TEMPLATES.items():
            for template in templates:
                all_templates.append((intent, template))

        # Add multilingual queries
        for lang, templates in MULTILINGUAL_QUERIES.items():
            for template in templates:
                all_templates.append((f"multilingual_{lang}", template))

        self.rng.shuffle(all_templates)

        for intent, template in all_templates[:n]:
            # Synonym injection
            syn_brand = self._inject_synonym(product.brand, "brand")
            syn_category = self._inject_synonym(product.category, "category")

            query_text = template.format(
                syn_brand=syn_brand,
                syn_category=syn_category,
                **slots
            )

            # Attribute noise injection
            if self.np_rng.random() < self.attribute_noise_ratio:
                query_text = self._inject_attribute_noise(query_text, product)

            # Determine language
            if intent.startswith("multilingual_"):
                lang = intent.split("_")[1]
                intent_type = "multilingual"
            else:
                lang = self._detect_query_lang(query_text)
                intent_type = intent

            queries.append({
                "text": query_text,
                "lang": lang,
                "intent": intent_type,
            })

        return queries[:n]

    def _inject_synonym(self, term: str, entity_type: str) -> str:
        """Replace term with a synonym with probability."""
        if self.np_rng.random() > self.synonym_injection_ratio:
            return term

        if entity_type == "brand":
            canon = normalize_entity(term.lower(), "brand")
            syns = _BRAND_SYNONYM_MAP.get(canon, [term])
        else:
            canon = normalize_entity(term.lower(), "category")
            syns = _SYNONYM_MAP.get(canon, [term])

        return self.rng.choice(syns) if syns else term

    def _inject_attribute_noise(self, query: str, product: AugmentedProduct) -> str:
        """Randomly mask brand or category in query."""
        noise_type = self.rng.choice(["mask_brand", "mask_category", "misspell_brand"])

        if noise_type == "mask_brand" and product.brand.lower() in query.lower():
            return re.sub(
                re.escape(product.brand),
                self.rng.choice(["[BRAND]", "some brand", ""]),
                query,
                flags=re.IGNORECASE,
            )
        elif noise_type == "mask_category" and product.category.lower() in query.lower():
            return re.sub(
                re.escape(product.category),
                self.rng.choice(["[CATEGORY]", "something", ""]),
                query,
                flags=re.IGNORECASE,
            )
        elif noise_type == "misspell_brand":
            # Simple misspelling: swap adjacent letters
            brand = product.brand
            if len(brand) > 3:
                idx = self.rng.randint(1, len(brand) - 2)
                misspelled = brand[:idx] + brand[idx + 1] + brand[idx] + brand[idx + 2:]
                return query.replace(brand, misspelled)

        return query

    def _detect_query_lang(self, query: str) -> str:
        """Simple language detection based on script."""
        if any("\u0900" <= c <= "\u097f" for c in query):
            return "hi"
        if any("\u0600" <= c <= "\u06ff" for c in query):
            return "ar"
        if any("\u0100" <= c <= "\u017f" for c in query):  # Polish diacritics
            return "pl"
        return "en"

    # ── Smart Relevance Assignment (Synonym-Aware) ────────────────────────

    def assign_relevance(self, query_text: str, product: AugmentedProduct) -> int:
        """
        Assign relevance 0-3 using synonym-aware matching.

        Unlike strict string matching, this understands that:
          - "trainers" = "shoes" = "footwear" = "sneakers"
          - "affordable" implies purchase intent (complement)
          - Hard negatives are always 0
        """
        q_lower = query_text.lower()

        # Hard negatives are always irrelevant
        if 'hard_negative' in q_lower or 'irrelevant' in q_lower:
            return 0

        q_tokens = set(q_lower.split())

        # Get canonical forms
        brand_canon = normalize_entity(product.brand.lower(), 'brand')
        cat_canon = normalize_entity(product.category.lower(), 'category')

        # Build synonym sets for matching
        brand_syns = set(BRAND_SYNONYMS.get(brand_canon, [brand_canon]))
        cat_syns = set(CATEGORY_SYNONYMS.get(cat_canon, [cat_canon]))

        # Also check if query contains ANY known brand/category variant
        # and expand the query's matching vocabulary accordingly
        for canon, syns in BRAND_SYNONYMS.items():
            if any(s.lower() in q_lower for s in syns):
                q_tokens.update(s.lower() for s in syns)
        for canon, syns in CATEGORY_SYNONYMS.items():
            if any(s.lower() in q_lower for s in syns):
                q_tokens.update(s.lower() for s in syns)

        # Check for attribute matches (model, color, tech, etc.)
        attr_matches = 0
        for attr_val in product.attributes.values():
            if attr_val and attr_val.lower() in q_lower:
                attr_matches += 1

        # Match detection: does query contain brand or category (or their synonyms)?
        brand_match = bool(q_tokens & brand_syns)
        category_match = bool(q_tokens & cat_syns)

        # SKU-level match: model number in query
        model = product.attributes.get("model", "")
        sku_match = model and model.lower() in q_lower

        # Exact: brand + category (classic exact match)
        if brand_match and category_match:
            return 3

        # Exact: brand + SKU/model (specific product seek, e.g. "Nike Air Max 270 specs")
        if brand_match and sku_match:
            return 3

        # Exact: category + SKU/model (e.g. "Air Max 270 running shoes")
        if category_match and sku_match:
            return 3

        # Strong exact: brand + category + attributes
        if brand_match and category_match and attr_matches >= 2:
            return 3

        # Substitute: brand OR category (with some context)
        if brand_match or category_match:
            # If there are attribute hints, boost to substitute
            if attr_matches >= 1:
                return 2
            return 2

        # Complement: generic purchase intent without exact brand/category
        purchase_signals = ['best', 'cheap', 'buy', 'top', 'affordable', 'price', 'deal', 'review', 'online']
        if any(w in q_lower for w in purchase_signals):
            return 1

        # Activity match without brand/category (e.g., "running shoes" for Nike product)
        activity = product.attributes.get("activity", "").lower()
        if activity and activity in q_lower:
            return 1

        return 0

    # ── Hard Negative Generation ──────────────────────────────────────────

    def generate_hard_negatives(self, product: AugmentedProduct, query: Dict[str, str]) -> List[Dict[str, str]]:
        """Generate confusing near-miss queries for a product."""
        cache_key = f"hardneg_{product.brand}_{product.category}_{product.product_id}_{query['text'][:30]}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        negatives = self._fallback_generate_hard_negatives(product, query)
        self._cache[cache_key] = negatives
        self._save_cache()
        return negatives

    def _fallback_generate_hard_negatives(self, product: AugmentedProduct, query: Dict[str, str]) -> List[Dict[str, str]]:
        """Generate hard negatives using templates."""
        negatives = []
        slots = product.attributes.copy()
        slots["brand"] = product.brand
        slots["category"] = product.category
        slots["wrong_brand"] = self._get_wrong_brand(product.brand)
        slots["wrong_model"] = self._get_wrong_model(slots.get("model", "X1000"))
        slots["wrong_color"] = self.rng.choice(_SLOT_FILLERS["color"])
        slots["wrong_size"] = self.rng.choice(_SLOT_FILLERS["size"])
        slots["prev_year"] = str(int(2023 + self.rng.randint(-2, 0)))

        n_hard = max(1, int(self.queries_per_product * self.hard_negative_ratio))
        templates = self.rng.sample(HARD_NEGATIVE_TEMPLATES, min(n_hard, len(HARD_NEGATIVE_TEMPLATES)))

        for template in templates:
            neg_query = template["query"].format(**slots)
            negatives.append({
                "text": neg_query,
                "lang": query.get("lang", "en"),
                "intent": f"hard_negative_{template['reason']}",
                "relevance": 0,  # Explicitly irrelevant
            })

        return negatives

    def _get_wrong_brand(self, brand: str) -> str:
        """Select a different brand from the same category space."""
        brands = list(BRAND_SYNONYMS.keys())
        wrong = self.rng.choice(brands)
        while wrong == brand.lower():
            wrong = self.rng.choice(brands)
        return wrong.title()

    def _get_wrong_model(self, model: str) -> str:
        """Generate a similar but wrong model number."""
        models = _SLOT_FILLERS["model"]
        wrong = self.rng.choice(models)
        while wrong == model:
            wrong = self.rng.choice(models)
        return wrong

    # ── Batch Generation ──────────────────────────────────────────────────

    def generate_catalog(self, n: int, categories: List[str], brands: List[str]) -> List[AugmentedProduct]:
        """Generate a full catalog of realistic products."""
        products = []
        cats = self.np_rng.choice(categories, n)
        brs = self.np_rng.choice(brands, n)
        for i in range(n):
            product = self.generate_product(i, brs[i], cats[i])
            product.product_id = i
            products.append(product)
        return products

