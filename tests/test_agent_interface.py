from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from starter.agent import Agent
from starter.model_client import ModelResult, ModelUsage
from tests.test_retrieval import write_catalog


class FakeModelClient:
    def __init__(self, results: list[ModelResult]) -> None:
        self.results = list(results)
        self.calls: list[tuple[list[dict[str, str]], int]] = []

    def complete_json(self, messages: list[dict[str, str]], max_tokens: int) -> ModelResult:
        self.calls.append((messages, max_tokens))
        return self.results.pop(0)


class AgentInterfaceTest(unittest.TestCase):
    def test_model_plans_search_and_ranks_only_catalog_candidates(self) -> None:
        client = FakeModelClient([
            ModelResult(
                {"intent_mode": "buying", "search_queries": ["red cotton shirt"], "ask_attribute": "feature"},
                ModelUsage(prompt_tokens=11, completion_tokens=3),
            ),
            ModelResult(
                {
                    "message": "The red cotton shirt is the strongest match. Which style do you prefer?",
                    "ask_attribute": "style",
                    "recommendations": ["RED", "RED", "NOT-IN-CATALOG"],
                },
                ModelUsage(prompt_tokens=17, completion_tokens=5),
            ),
        ])
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {}, clear=True):
                agent = Agent(write_catalog(Path(directory)), model_client=client)
            agent.reset("session", {"summary": "prefers comfort", "preference_tags": ["comfort"]})
            response = agent.respond("session", "I need a red cotton shirt.", 1, 2)

        self.assertEqual(response["recommendations"][0]["parent_asin"], "RED")
        self.assertLessEqual(len(response["recommendations"]), 2)
        self.assertEqual(
            len({item["parent_asin"] for item in response["recommendations"]}),
            len(response["recommendations"]),
        )
        self.assertEqual(response["ask_attribute"], "style")
        self.assertEqual(response["usage"], {"prompt_tokens": 28, "completion_tokens": 8})
        self.assertEqual(len(client.calls), 2)

    def test_missing_key_uses_valid_stateful_offline_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {}, clear=True):
                agent = Agent(write_catalog(Path(directory)))
            agent.reset("offline", {"summary": "", "preference_tags": []})
            response = agent.respond(
                "offline", "I'm looking for Shirts, but I'm still exploring.", 1, 2
            )

        self.assertEqual(response["ask_attribute"], "material")
        self.assertLessEqual(len(response["recommendations"]), 2)
        self.assertEqual(response["usage"], {"prompt_tokens": 0, "completion_tokens": 0})
        self.assertIsInstance(response["message"], str)


if __name__ == "__main__":
    unittest.main()
