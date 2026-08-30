from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from starter.model_client import (
    ModelSettings,
    compatibility_retry_body,
    extract_json_object,
)


class ModelClientTest(unittest.TestCase):
    def test_settings_read_credentials_only_from_environment_and_redact_repr(self) -> None:
        with patch.dict(os.environ, {
            "TECHJAM_LLM_API_KEY": "secret-test-value",
            "TECHJAM_LLM_MODEL": "example-model",
            "TECHJAM_LLM_BASE_URL": "https://example.invalid/v1",
            "TECHJAM_LLM_TIMEOUT": "12",
        }, clear=True):
            settings = ModelSettings.from_env()

        self.assertTrue(settings.enabled)
        self.assertEqual(settings.api_key, "secret-test-value")
        self.assertEqual(settings.model, "example-model")
        self.assertEqual(settings.timeout_seconds, 12.0)
        self.assertNotIn("secret-test-value", repr(settings))
        self.assertIn("<set>", repr(settings))

    def test_missing_key_disables_live_model(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            settings = ModelSettings.from_env()
        self.assertFalse(settings.enabled)

    def test_extract_json_accepts_fenced_content(self) -> None:
        self.assertEqual(
            extract_json_object('```json\n{"search_queries":["cotton shirt"]}\n```'),
            {"search_queries": ["cotton shirt"]},
        )

    def test_new_reasoning_model_token_parameter_is_negotiated(self) -> None:
        original = {
            "max_tokens": 50,
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
        }
        retried = compatibility_retry_body(original, "max_tokens")

        self.assertIsNotNone(retried)
        self.assertNotIn("max_tokens", retried)
        self.assertEqual(retried["max_completion_tokens"], 50)
        self.assertEqual(original["max_tokens"], 50)

    def test_unsupported_optional_parameter_is_removed(self) -> None:
        original = {"temperature": 0.1, "model": "example-model"}
        self.assertEqual(
            compatibility_retry_body(original, "temperature"),
            {"model": "example-model"},
        )


if __name__ == "__main__":
    unittest.main()
