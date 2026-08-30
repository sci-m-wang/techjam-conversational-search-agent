"""Catalog index: load once, answer fast.

Pure stdlib. Builds:
- padded token text per product  (" t1 t2 ... "  -> exact token membership via substring)
- log-popularity per product     (log1p(rating_number), normalized)
- category pools                 (coarse category -> asins sorted by popularity)
- token document frequency       (for IDF weighting in the ranker)
- parsed price per product       (float | None; handles float and string forms)

The coarse-category rule mirrors the public competition kit's session generator
(last two comma-split segments of the category path, generic roots excluded) so
that a customer's stated category maps onto one pool.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

TOKEN_RE = re.compile(r"[a-z0-9]+")
_GENERIC = {"clothing", "clothing shoes & jewelry", "clothing, shoes & jewelry"}


def tokens(text: str) -> list[str]:
    return [t for t in TOKEN_RE.findall(text.lower()) if len(t) > 1]


def coarse_category(values: list[str]) -> str:
    parts: list[str] = []
    for value in values:
        for part in str(value).split(","):
            part = part.strip()
            if part and part.lower() not in _GENERIC:
                parts.append(part)
    return " ".join(parts[-2:]) if parts else "clothing item"


def flat_text(product: dict) -> str:
    parts: list[str] = []
    for field in ("title", "features", "details", "description", "categories", "store"):
        value = product.get(field)
        if isinstance(value, dict):
            parts.extend(f"{k} {v}" for k, v in value.items())
        elif isinstance(value, list):
            parts.extend(str(v) for v in value)
        elif value is not None:
            parts.append(str(value))
    return " ".join(parts)


def parse_price(value) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        m = re.search(r"[\d.]+", value.replace(",", ""))
        if m:
            try:
                return float(m.group())
            except ValueError:
                return None
    return None


class Catalog:
    def __init__(self, catalog_path: str | Path):
        self.padded: dict[str, str] = {}          # asin -> " tok tok ... "
        self.pop: dict[str, float] = {}           # asin -> log1p(rating_number)
        self.price: dict[str, float | None] = {}
        self.title: dict[str, str] = {}
        self.pools: dict[str, list[str]] = {}     # coarse category -> asins
        self.pool_tokens: dict[str, frozenset] = {}
        df: dict[str, int] = {}
        with Path(catalog_path).open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                product = json.loads(line)
                asin = str(product["parent_asin"])
                toks = tokens(flat_text(product))
                self.padded[asin] = " " + " ".join(toks) + " "
                self.pop[asin] = math.log1p(product.get("rating_number") or 0)
                self.price[asin] = parse_price(product.get("price"))
                self.title[asin] = str(product.get("title") or "")
                for t in set(toks):
                    df[t] = df.get(t, 0) + 1
                cc = coarse_category(product.get("categories") or [])
                self.pools.setdefault(cc, []).append(asin)
        self.n_docs = len(self.padded)
        self.max_pop = max(self.pop.values()) if self.pop else 1.0
        self.df = df
        for cc, pool in self.pools.items():
            pool.sort(key=lambda a: (-self.pop[a], a))
            self.pool_tokens[cc] = frozenset(tokens(cc))
        self.pool_vocab = frozenset(t for toks in self.pool_tokens.values() for t in toks)
        # popularity-ordered global slice for category-less fallback
        self.global_head = sorted(self.padded, key=lambda a: (-self.pop[a], a))[:8000]

    def idf(self, token: str) -> float:
        return math.log((self.n_docs + 1) / (self.df.get(token, 0) + 1))

    def has_token(self, asin: str, token: str) -> bool:
        return f" {token} " in self.padded[asin]

    def has_phrase(self, asin: str, phrase_tokens: list[str]) -> bool:
        return f" {' '.join(phrase_tokens)} " in self.padded[asin]

    def match_pool(self, category_text: str) -> list[str]:
        """Exact pool hit, else best token-overlap pools (paraphrase-tolerant)."""
        if category_text in self.pools:
            return self.pools[category_text]
        want = frozenset(tokens(category_text)) & self.pool_vocab
        if not want:
            return self.global_head
        scored: list[tuple[float, str]] = []
        for cc, ptoks in self.pool_tokens.items():
            if not ptoks:
                continue
            inter = len(want & ptoks)
            if inter:
                scored.append((inter / len(want | ptoks), cc))
        if not scored:
            return self.global_head
        scored.sort(reverse=True)
        best = scored[0][0]
        merged: list[str] = []
        for share, cc in scored:
            if share < best * 0.75 and len(merged) >= 200:
                break
            merged.extend(self.pools[cc])
            if len(merged) >= 2000:
                break
        return merged or self.global_head
