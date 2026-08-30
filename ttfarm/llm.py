"""Optional LLM layer (Layer 3 NLU). OFF by default; the agent never requires it.

Enable with COPILOT_LLM=1 plus an OpenAI-compatible endpoint via COPILOT_LLM_URL,
COPILOT_LLM_MODEL, COPILOT_LLM_KEY. Used ONLY when layers 1-2 parse a mid-session
message with low confidence. Every failure falls back to the deterministic parse;
official scoring may run with network disabled and loses nothing.
Token usage is counted and reported through the response "usage" field.
"""
from __future__ import annotations

import json
import os
import urllib.request

USAGE = {"prompt_tokens": 0, "completion_tokens": 0}

_PROMPT = (
    "Extract shopping intent from this customer message as compact JSON with keys "
    '"category" (string|null), "constraints" (list of short strings), '
    '"override" (bool: did they replace an earlier preference?). Message: ')


def enabled() -> bool:
    return os.environ.get("COPILOT_LLM") == "1" and bool(os.environ.get("COPILOT_LLM_URL"))


def extract(message: str, timeout: float = 4.0) -> dict | None:
    if not enabled():
        return None
    try:
        body = json.dumps({
            "model": os.environ.get("COPILOT_LLM_MODEL", ""),
            "messages": [{"role": "user", "content": _PROMPT + json.dumps(message)}],
            "temperature": 0,
        }).encode()
        req = urllib.request.Request(
            os.environ["COPILOT_LLM_URL"].rstrip("/") + "/chat/completions",
            data=body, headers={"Content-Type": "application/json",
                                "Authorization": f"Bearer {os.environ.get('COPILOT_LLM_KEY','')}"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.load(resp)
        usage = payload.get("usage") or {}
        USAGE["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
        USAGE["completion_tokens"] += int(usage.get("completion_tokens") or 0)
        content = payload["choices"][0]["message"]["content"]
        data = json.loads(content[content.find("{"): content.rfind("}") + 1])
        if isinstance(data, dict):
            return data
    except Exception:
        return None
    return None
