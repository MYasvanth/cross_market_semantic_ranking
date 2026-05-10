"""Catalog Normalization — Synonym mapping and canonical form resolution."""

import re
from typing import Dict, List

# ── Brand Synonyms ──────────────────────────────────────────────────────────
# Maps canonical brand -> list of known variants (all lower-cased)
BRAND_SYNONYMS: Dict[str, List[str]] = {
    "sony":      ["sony", "sony corp", "sony corporation", "sony electronics", "ソニー"],
    "samsung":   ["samsung", "samsung electronics", "삼성"],
    "apple":     ["apple", "apple inc", "apple computer"],
    "nike":      ["nike", "nike inc", "ナイキ"],
    "adidas":    ["adidas", "adidas ag"],
    "lg":        ["lg", "lg electronics", "lucky goldstar"],
    "hp":        ["hp", "hewlett-packard", "hewlett packard"],
    "dell":      ["dell", "dell technologies"],
    "lenovo":    ["lenovo", "lenovo group"],
    "xiaomi":    ["xiaomi", "mi", "小米"],
    "huawei":    ["huawei", "华为"],
    "google":    ["google", "google inc", "alphabet"],
    "microsoft": ["microsoft", "microsoft corp", "msft"],
    "amazon":    ["amazon", "amazon.com", "amaz"],
    "asics":     ["asics", "アシックス"],
    "puma":      ["puma", "puma se"],
    "reebok":    ["reebok"],
    "under armour": ["under armour", "underarmour", "ua"],
    "new balance":  ["new balance", "newbalance", "nb"],
}

# Reverse lookup: variant -> canonical
_BRAND_REV: Dict[str, str] = {}
for canonical, variants in BRAND_SYNONYMS.items():
    for v in variants:
        _BRAND_REV[v] = canonical

# ── Category Synonyms ───────────────────────────────────────────────────────
CATEGORY_SYNONYMS: Dict[str, List[str]] = {
    "shoes":      ["shoes", "footwear", "sneakers", "running shoes", "trainers", "靴"],
    "electronics":["electronics", "consumer electronics", "gadgets", "electronic devices", "elektronika", "إلكترونيات", "इलेक्ट्रॉनिक्स"],
    "clothing":   ["clothing", "apparel", "garments", "fashion", "odzież", "ملابس", "कपड़े"],
    "laptops":    ["laptops", "notebook", "notebook computers", "portable computers", "laptop pc"],
    "phones":     ["phones", "smartphones", "mobile phones", "cell phones", "handsets"],
    "accessories":["accessories", "accessory", "peripherals", "add-ons", "extras"],
    "home":       ["home", "household", "home goods", "home products"],
    "sports":     ["sports", "sporting goods", "fitness", "athletic"],
}

_CATEGORY_REV: Dict[str, str] = {}
for canonical, variants in CATEGORY_SYNONYMS.items():
    for v in variants:
        _CATEGORY_REV[v] = canonical

# ── Unit Standardization ────────────────────────────────────────────────────
_UNIT_PATTERNS = [
    (re.compile(r"(\d+(?:\.\d+)?)\s*inch(?:es)?", re.I), r"\1_in"),
    (re.compile(r"(\d+(?:\.\d+)?)\s*gb", re.I),        r"\1_gb"),
    (re.compile(r"(\d+(?:\.\d+)?)\s*tb", re.I),        r"\1_tb"),
    (re.compile(r"(\d+(?:\.\d+)?)\s*mb", re.I),        r"\1_mb"),
    (re.compile(r"(\d+(?:\.\d+)?)\s*kg", re.I),        r"\1_kg"),
    (re.compile(r"(\d+(?:\.\d+)?)\s*lbs?", re.I),       r"\1_lb"),
    (re.compile(r"(\d+(?:\.\d+)?)\s*ml", re.I),         r"\1_ml"),
    (re.compile(r"(\d+(?:\.\d+)?)\s*oz", re.I),         r"\1_oz"),
]


def normalize_entity(text: str, entity_type: str = "brand") -> str:
    """
    Map a raw brand/category string to its canonical form.

    Args:
        text: Raw input (e.g., 'Sony Corp', 'sneakers').
        entity_type: 'brand' or 'category'.

    Returns:
        Canonical string or lower-cased original if no mapping exists.
    """
    if not text:
        return ""
    key = text.lower().strip()
    lookup = _BRAND_REV if entity_type == "brand" else _CATEGORY_REV
    return lookup.get(key, key)


def normalize_units(text: str) -> str:
    """Standardize common unit expressions for cleaner lexical matching."""
    if not text:
        return ""
    out = text.lower()
    for pattern, replacement in _UNIT_PATTERNS:
        out = pattern.sub(replacement, out)
    return out


def normalize_query(text: str) -> str:
    """Full normalization pipeline for user queries: lower-case, units, strip."""
    if not text:
        return ""
    return normalize_units(text.lower().strip())


# Convenience: expose canonical brand/category sets for intent detection
KNOWN_BRANDS: List[str] = list(BRAND_SYNONYMS.keys())
KNOWN_CATEGORIES: List[str] = list(CATEGORY_SYNONYMS.keys())

