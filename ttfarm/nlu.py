"""Layered NLU: turn a customer message into structured observations.

Layer 1 - exact templates of the public session generator (fast path).
Layer 2 - structural/fuzzy parse so paraphrased messages still yield the same
          kind of observations (the public spec reserves the right to add
          natural-language paraphrasing; revealed facts stay deterministic).
Layer 3 - optional LLM extraction (src/llm.py), used only when enabled AND
          layers 1-2 are low-confidence. Off by default; never required.

An observation never decides correctness by wording: downstream scoring is
token-weighted, so imperfect segmentation degrades gracefully.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ttfarm.catalog import tokens

# ---------------------------------------------------------------- Layer 1
RE_INIT_BUY = re.compile(r"^I'm looking for (?P<cat>.+?)\. A key requirement is: (?P<c>.+)\.$")
RE_INIT_BROWSE = re.compile(r"^I'm looking for (?P<cat>.+?), but I'm still exploring\.$")
RE_INIT_OTHER = re.compile(r"^I'm looking for (?P<cat>.+?)\. (?P<c>.+?)\.?$")
RE_OVERRIDE = re.compile(r"^Actually, ignore my earlier preference\. What I need is: (?P<c>.+)\.$")
RE_REPLY = re.compile(r"^For that, what matters is: (?P<cs>.+)\.$")
RE_NOPREF = re.compile(r"^I don't have (an additional preference|a preference) for .+", re.I)
RE_NUDGE = re.compile(r"^Those options are not quite right yet\.", re.I)

# ---------------------------------------------------------------- Layer 2
_OVERRIDE_CUES = ("ignore my earlier", "forget", "changed my mind", "change of plan",
                  "instead", "scratch that", "never mind", "on second thought",
                  "new plan", "drop my earlier")
_NOPREF_CUES = ("no preference", "no strong preference", "anything works", "you decide",
                "whatever you think", "use your judgment", "don't have a preference",
                "don't have an additional")
_NUDGE_CUES = ("not quite", "ask me about", "try asking", "something specific", "not right")
_LOOKING_CUES = re.compile(
    r"(?:looking at|looking for|shopping for|searching for|browsing|in the market for"
    r"|after|want|need)\s+(?:to buy\s+|some\s+|a\s+|an\s+)?(?P<rest>.+)", re.I)
_REQ_CUES = re.compile(
    r"(?:must have|really needs|non-negotiable(?: for me)?|key requirement[^:]*"
    r"|main thing(?: for me)?|it must|has to have)\s*[:\-]?\s*(?P<c>.+?)[.!]?$", re.I)
_NEED_TAIL = re.compile(r"(?:need is|needs?|matters is|priority is|care about is)\s+(?P<c>.+?)[.!]?$", re.I)
_OVERRIDE_TAIL = re.compile(r"\s*(?:is what matters(?: now)?|matters now|is the priority)[.!]?$", re.I)
_JUNK = frozenset(tokens("preference decide judgment best works quite asking ask specific "
                         "thing honestly whatever anything strong options mostly care"))
_STOP = frozenset(tokens(
    "i'm im looking for a an the but still exploring key requirement is what matters "
    "actually ignore my earlier preference need those options are not quite right yet "
    "ask me about one specific attribute don't have additional please use your judgment "
    "that this it want with and or of in on to me you your now days these"))


@dataclass
class Observation:
    kind: str                                # open | reply | override | nopref | nudge
    scenario_hint: str = ""                  # buying | browsing | override_likely | ""
    category: str | None = None
    constraints: list[str] = field(default_factory=list)
    loose_tokens: list[str] = field(default_factory=list)
    layer: int = 1                           # which layer produced it


def _content_tokens(text: str) -> list[str]:
    return [t for t in tokens(text) if t not in _STOP]


def _override_value(msg: str) -> str:
    """Best-effort extraction of the replacement requirement from free text."""
    if ":" in msg:
        tail = msg.split(":", 1)[1]
    else:
        m = _NEED_TAIL.search(msg)
        tail = m["c"] if m else re.split(r"[,;-]", msg)[-1]
    return _OVERRIDE_TAIL.sub("", tail).strip(" .!-")


def parse(message: str, turn: int) -> Observation:
    msg = " ".join(str(message).split())

    # ---- Layer 1: exact templates ----
    m = RE_OVERRIDE.match(msg)
    if m:
        return Observation("override", "override_likely", None, [m["c"]])
    m = RE_REPLY.match(msg)
    if m:
        return Observation("reply", "", None, m["cs"].split("; "))
    if RE_NOPREF.match(msg):
        return Observation("nopref")
    if RE_NUDGE.match(msg):
        return Observation("nudge")
    m = RE_INIT_BUY.match(msg)
    if m:
        return Observation("open", "buying", m["cat"], [m["c"]])
    m = RE_INIT_BROWSE.match(msg)
    if m:
        return Observation("open", "browsing", m["cat"], [])
    if turn == 1:
        m = RE_INIT_OTHER.match(msg)
        if m:  # override-style opener: category + a current (soft) preference
            return Observation("open", "override_likely", m["cat"], [m["c"]])

    # ---- Layer 2: structural fallback ----
    lower = msg.lower()
    if any(cue in lower for cue in _NOPREF_CUES):
        return Observation("nopref", layer=2)
    if turn > 1 and any(cue in lower for cue in _OVERRIDE_CUES):
        value = _override_value(msg)
        return Observation("override", "override_likely", None,
                           [value] if value else [], _content_tokens(msg), layer=2)
    if turn == 1:
        m = _LOOKING_CUES.search(msg)
        category = None
        if m:
            head = re.split(r"[,.;]|\s[-–—]\s|\s(?:it|that|which|but)\s", m["rest"])[0]
            category = head.strip(" .-") or None
        r = _REQ_CUES.search(msg)
        constraint = r["c"].strip(" .-") if r else None
        return Observation("open", "buying" if constraint else "browsing", category,
                           [constraint] if constraint else [], _content_tokens(msg), layer=2)
    if any(cue in lower for cue in _NUDGE_CUES):
        return Observation("nudge", layer=2)
    # mid-session free text: split on separators; keep informative, non-junk chunks
    chunks: list[str] = []
    for chunk in re.split(r"[;:.]|\s+and\s+|\s+plus\s+", msg):
        content = _content_tokens(chunk)
        if content and not all(t in _JUNK for t in content):
            chunks.append(chunk.strip(" .-"))
    return Observation("reply", "", None, chunks[:4], _content_tokens(msg), layer=2)
