"""TechJam Track 4 submission - conversational shopping agent.

Entry point exporting `Agent` per docs/submission_rules.md. Drop-in compatible
with the official evaluator (same constructor signature as the starter).

Offline, deterministic, stdlib-only by default. Optional LLM parse layer is
documented in src/llm.py and disabled unless COPILOT_LLM=1.
"""
from __future__ import annotations

from pathlib import Path

from ttfarm import llm
from ttfarm.catalog import Catalog
from ttfarm.nlu import Observation, parse
from ttfarm.policy import choose_ask, choose_k, compose_message
from ttfarm.search import rank
from ttfarm.state import Session


class Agent:
    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        self.catalog = Catalog(catalog_path)
        self.sessions: dict[str, Session] = {}

    def reset(self, session_id: str, user_profile: dict) -> None:
        self.sessions[session_id] = Session(profile=dict(user_profile or {}))

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        try:
            return self._respond(session_id, user_message, int(turn), int(top_k))
        except Exception:
            # Contract safety net: a turn must never be voided by an exception.
            return {"message": "Could you tell me one specific requirement?",
                    "ask_attribute": "other", "recommendations": [],
                    "usage": dict(llm.USAGE)}

    def _respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        session = self.sessions.setdefault(session_id, Session())
        session.turn = turn
        obs = parse(user_message, turn)
        if obs.layer == 2 and llm.enabled():          # Layer 3: optional, flagged
            data = llm.extract(user_message)
            if data:
                obs = Observation(
                    "override" if data.get("override") else obs.kind,
                    obs.scenario_hint, data.get("category") or obs.category,
                    [str(c) for c in (data.get("constraints") or [])][:4] or obs.constraints,
                    obs.loose_tokens, layer=3)
        session.observe(obs)
        k = choose_k(session, turn, top_k)
        # Turn 10 safety net: re-rank over the full pool ignoring the exclusion
        # memory, in case an unparsed override made us exclude the true target.
        recs = rank(self.catalog, session, limit=k, exclude_shown=turn < 10) if k else []
        session.shown.update(recs)
        ask = choose_ask(session, turn)
        return {
            "message": compose_message(session, ask, len(recs)),
            "ask_attribute": ask,
            "recommendations": [{"parent_asin": a} for a in recs],
            "usage": dict(llm.USAGE),
        }
