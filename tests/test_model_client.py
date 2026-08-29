from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from starter.model_client import ModelSettings, extract_json_object


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


if __name__ == "__main__":
    unittest.main()
