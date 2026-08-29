from __future__ import annotations

import re
from dataclasses import dataclass, field


OVERRIDE_RE = re.compile(
    r"\b(actually|ignore (?:my )?earlier|instead|what i need is|changed my mind)\b",
    re.IGNORECASE,
)
NO_PREFERENCE_RE = re.compile(
    r"\b(?:do not|don't|no|not)\s+(?:have\s+)?(?:an?\s+)?(?:additional\s+)?preference\b",
    re.IGNORECASE,
)
CATEGORY_RE = re.compile(
    r"^\s*(i(?:'m| am) looking for\s+.*?)(?:,\s+but\b|\.\s+|$)",
    re.IGNORECASE,
)


def _clean(text: str, limit: int = 600) -> str:
    return re.sub(r"\s+", " ", text).strip()[:limit]


@dataclass
class SessionState:
    profile: dict
    category_message: str = ""
    active_user_messages: list[str] = field(default_factory=list)
    assistant_messages: list[str] = field(default_factory=list)
    asked_attributes: list[str] = field(default_factory=list)
    declined_attributes: set[str] = field(default_factory=set)

    def observe(self, user_message: str) -> None:
        message = _clean(user_message)
        if not self.category_message:
            match = CATEGORY_RE.search(message)
            self.category_message = _clean(match.group(1) if match else message)
        if NO_PREFERENCE_RE.search(message) and self.asked_attributes:
            self.declined_attributes.add(self.asked_attributes[-1])
        if OVERRIDE_RE.search(message):
            self.active_user_messages = [message]
        else:
            self.active_user_messages.append(message)
            self.active_user_messages = self.active_user_messages[-6:]

    def record_response(self, message: str, ask_attribute: str | None) -> None:
        self.assistant_messages.append(_clean(message, 300))
        self.assistant_messages = self.assistant_messages[-6:]
        if ask_attribute:
            self.asked_attributes.append(ask_attribute)
            self.asked_attributes = self.asked_attributes[-10:]

    def context_text(self) -> str:
        preference_tags = self.profile.get("preference_tags") or []
        profile_summary = _clean(str(self.profile.get("summary") or ""), 300)
        parts = [self.category_message, *self.active_user_messages]
        if profile_summary:
            parts.append(profile_summary)
        if isinstance(preference_tags, list):
            parts.extend(str(tag) for tag in preference_tags[:8])
        return _clean(" ".join(part for part in parts if part), 1800)

    def prompt_context(self, turn: int, user_message: str) -> dict:
        return {
            "turn": turn,
            "profile": {
                "summary": self.profile.get("summary", ""),
                "preference_tags": self.profile.get("preference_tags", []),
            },
            "category_message": self.category_message,
            "active_user_messages": self.active_user_messages[-6:],
            "latest_user_message": user_message,
            "asked_attributes": self.asked_attributes,
            "declined_attributes": sorted(self.declined_attributes),
        }
