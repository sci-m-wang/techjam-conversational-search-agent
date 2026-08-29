from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path


TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "with", "would", "you", "looking",
    "those", "what", "matters", "still", "exploring", "additional", "preference",
}


def flatten(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def terms(text: str) -> list[str]:
    return list(dict.fromkeys(
        token.lower()
        for token in TOKEN_RE.findall(text)
        if len(token) > 1 and token.lower() not in STOPWORDS
    ))[:60]


class CatalogRetriever:
    def __init__(self, catalog_path: str | Path) -> None:
        self.catalog_path = Path(catalog_path)
        self.connection = sqlite3.connect(":memory:")
        self.products: dict[str, dict] = {}
        self._build_index()

    @property
    def valid_ids(self) -> set[str]:
        return set(self.products)

    def _build_index(self) -> None:
        self.connection.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "parent_asin UNINDEXED, title, categories, features, details, store, description, price, "
            "tokenize='unicode61 remove_diacritics 2')"
        )
        batch: list[tuple[str, ...]] = []
        with self.catalog_path.open(encoding="utf-8") as handle:
            for line in handle:
                product = json.loads(line)
                parent_asin = str(product["parent_asin"])
                compact = {
                    "parent_asin": parent_asin,
                    "title": flatten(product.get("title"))[:300],
                    "categories": flatten(product.get("categories"))[:300],
                    "features": flatten(product.get("features"))[:900],
                    "details": flatten(product.get("details"))[:600],
                    "store": flatten(product.get("store"))[:160],
                    "description": flatten(product.get("description"))[:700],
                    "price": flatten(product.get("price"))[:80],
                    "average_rating": product.get("average_rating"),
                    "rating_number": product.get("rating_number"),
                }
                self.products[parent_asin] = compact
                batch.append((
                    parent_asin, compact["title"], compact["categories"], compact["features"],
                    compact["details"], compact["store"], compact["description"], compact["price"],
                ))
                if len(batch) >= 1000:
                    self.connection.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?, ?)", batch)
                    batch.clear()
        if batch:
            self.connection.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?, ?)", batch)
        self.connection.commit()

    def search(self, query: str, limit: int = 20) -> list[dict]:
        query_terms = terms(query)
        if not query_terms:
            return []
        precise = " AND ".join(f'"{term}"' for term in query_terms[:20])
        rows = self._search_expression(precise, limit)
        if not rows:
            broad = " OR ".join(f'"{term}"' for term in query_terms[:40])
            rows = self._search_expression(broad, limit)
        return [self.products[parent_asin] for parent_asin in rows]

    def fallback_search(self, context: str, limit: int = 10) -> list[dict]:
        query_terms = terms(context)
        if not query_terms:
            return []
        expression = " OR ".join(f'"{term}"' for term in query_terms[:40])
        rows = self._search_expression(expression, limit)
        return [self.products[parent_asin] for parent_asin in rows]

    def _search_expression(self, expression: str, limit: int) -> list[str]:
        rows = self.connection.execute(
            "SELECT parent_asin FROM products WHERE products MATCH ? "
            "ORDER BY bm25(products, 0.0, 8.0, 5.0, 3.0, 2.5, 2.0, 1.0, 1.0) LIMIT ?",
            (expression, max(1, min(limit, 50))),
        ).fetchall()
        return [str(row[0]) for row in rows]
