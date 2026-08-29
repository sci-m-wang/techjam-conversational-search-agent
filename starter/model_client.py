from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Protocol


class ModelClientError(RuntimeError):
    """Raised when a model request cannot produce a usable JSON object."""


@dataclass(frozen=True)
class ModelUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass(frozen=True)
class ModelResult:
    payload: dict
    usage: ModelUsage = ModelUsage()


class JsonModelClient(Protocol):
    def complete_json(self, messages: list[dict[str, str]], max_tokens: int) -> ModelResult:
        ...


@dataclass(frozen=True)
class ModelSettings:
    api_key: str
    model: str
    base_url: str = "https://api.openai.com/v1"
    timeout_seconds: float = 45.0

    @classmethod
    def from_env(cls) -> "ModelSettings":
        return cls(
            api_key=os.environ.get("TECHJAM_LLM_API_KEY", os.environ.get("OPENAI_API_KEY", "")).strip(),
            model=os.environ.get("TECHJAM_LLM_MODEL", os.environ.get("OPENAI_MODEL", "")).strip(),
            base_url=os.environ.get(
                "TECHJAM_LLM_BASE_URL",
                os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            ).strip(),
            timeout_seconds=float(os.environ.get("TECHJAM_LLM_TIMEOUT", "45")),
        )

    @property
    def enabled(self) -> bool:
        return bool(self.api_key and self.model and self.base_url)

    def __repr__(self) -> str:
        redacted = "<set>" if self.api_key else "<unset>"
        return (
            "ModelSettings(api_key="
            f"{redacted!r}, model={self.model!r}, base_url={self.base_url!r}, "
            f"timeout_seconds={self.timeout_seconds!r})"
        )


JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def extract_json_object(content: str) -> dict:
    cleaned = JSON_FENCE_RE.sub("", content.strip())
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end <= start:
            raise ModelClientError("Model response did not contain a JSON object") from None
        try:
            value = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError as error:
            raise ModelClientError("Model response contained invalid JSON") from error
    if not isinstance(value, dict):
        raise ModelClientError("Model response JSON must be an object")
    return value


class OpenAICompatibleClient:
    """Small standard-library client for OpenAI-compatible Chat Completions APIs."""

    def __init__(self, settings: ModelSettings) -> None:
        if not settings.enabled:
            raise ValueError("A model name, base URL, and API key are required")
        self.settings = settings

    def complete_json(self, messages: list[dict[str, str]], max_tokens: int) -> ModelResult:
        request_body = {
            "model": self.settings.model,
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }
        response = self._post(request_body, allow_json_mode_retry=True)
        try:
            content = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise ModelClientError("Model response omitted message content") from error
        usage = response.get("usage") if isinstance(response, dict) else None
        usage = usage if isinstance(usage, dict) else {}
        return ModelResult(
            payload=extract_json_object(str(content)),
            usage=ModelUsage(
                prompt_tokens=max(0, int(usage.get("prompt_tokens", 0) or 0)),
                completion_tokens=max(0, int(usage.get("completion_tokens", 0) or 0)),
            ),
        )

    def _post(self, body: dict, allow_json_mode_retry: bool) -> dict:
        url = self.settings.base_url.rstrip("/") + "/chat/completions"
        request = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.settings.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.settings.timeout_seconds) as response:
                decoded = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            if error.code == 400 and allow_json_mode_retry and "response_format" in body:
                retry_body = dict(body)
                retry_body.pop("response_format", None)
                return self._post(retry_body, allow_json_mode_retry=False)
            raise ModelClientError(f"Model API returned HTTP {error.code}") from error
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
            raise ModelClientError("Model API request failed") from error
        if not isinstance(decoded, dict):
            raise ModelClientError("Model API response must be a JSON object")
        return decoded
