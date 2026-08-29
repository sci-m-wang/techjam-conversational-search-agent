from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from starter.retrieval import CatalogRetriever
from starter.session import SessionState


def write_catalog(root: Path) -> Path:
    rows = [
        {
            "parent_asin": "BLUE",
            "title": "Blue wool winter sweater",
            "categories": ["Clothing", "Sweaters"],
            "features": ["100% wool", "button closure"],
            "details": {"Department": "Women"},
            "description": ["Warm winter knit"],
            "store": "Example Blue",
            "price": 40.0,
            "average_rating": 4.2,
            "rating_number": 10,
        },
        {
            "parent_asin": "RED",
            "title": "Red cotton summer shirt",
            "categories": ["Clothing", "Shirts"],
            "features": ["100% cotton", "pull on closure"],
            "details": {"Department": "Women"},
            "description": ["Lightweight summer top"],
            "store": "Example Red",
            "price": 25.0,
            "average_rating": 4.5,
            "rating_number": 30,
        },
    ]
    path = root / "catalog.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


class SessionAndRetrievalTest(unittest.TestCase):
    def test_override_replaces_active_preference_but_keeps_category(self) -> None:
        state = SessionState({"summary": "prefers comfort", "preference_tags": ["comfort"]})
        state.observe("I'm looking for Women Sweaters. I prefer blue wool.")
        state.observe("Actually, ignore my earlier preference. What I need is: red cotton.")

        context = state.context_text().lower()
        self.assertIn("women sweaters", context)
        self.assertIn("red cotton", context)
        self.assertNotIn("blue wool", context)

    def test_declined_attribute_is_recorded_from_follow_up(self) -> None:
        state = SessionState({})
        state.observe("I'm looking for shirts, but I'm still exploring.")
        state.record_response("Which color?", "color")
        state.observe("I don't have an additional preference for color.")
        self.assertIn("color", state.declined_attributes)

    def test_catalog_search_returns_exact_matching_product(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            retriever = CatalogRetriever(write_catalog(Path(directory)))
            results = retriever.search("red cotton shirt", limit=2)
        self.assertEqual(results[0]["parent_asin"], "RED")


if __name__ == "__main__":
    unittest.main()
