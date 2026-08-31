"""Escalation: ttfarm drives by default; a struggling session can hand over.

Tier ladder (see ttfarm/README.md):
  0  ttfarm alone (always, and the only tier that exists without a model key)
  1  optional LLM parse assist (ttfarm/llm.py layer 3 - already wired in nlu)
  2  full handover: the starter LLM agent takes the remaining turns of a
     losing session, seeded with everything ttfarm learned.

Design rules:
- Without a reachable model endpoint, tier 2 cannot engage. A dead endpoint
  would put starter into its FTS-only fallback, which scores WORSE than
  wounded ttfarm - so reachability is proven by a one-time ping first.
- Handover passes the case file: profile, full transcript (replayed through
  starter's own offline observe/record API), and the shown-set of proven-wrong
  products, which is filtered out of starter's output and backfilled from
  ttfarm's next-best candidates.
- Run-level regime detection: sessions are independent games but the judge is
  one world. Early misses across sessions lower the takeover threshold; a
  takeover tier that underperforms ttfarm is switched back off.
- Budgets: bounded takeover sessions and a token ceiling. Insurance must not
  cost more than the house.

All knobs are environment variables so experiments never require code edits.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except ValueError:
        return default


@dataclass
class SessionRecord:
    """What the controller remembers about one session."""
    transcript: list = field(default_factory=list)   # (customer_msg, our_msg, ask)
    last_turn: int = 0
    tier2: bool = False                              # handover engaged
    ended_by_miss: bool | None = None                # filled when the next reset arrives


class EscalationController:
    """Per-turn tier decisions plus run-level regime tracking."""

    def __init__(self) -> None:
        self.enabled = os.environ.get("COPILOT_ESCALATION", "1") == "1"
        self.force_tier = os.environ.get("COPILOT_FORCE_TIER", "")
        self.t2_turn = _int_env("COPILOT_T2_TURN", 7)
        self.t2_max_sessions = _int_env("COPILOT_T2_MAX_SESSIONS", 400)
        self.token_budget = _int_env("COPILOT_TOKEN_BUDGET", 3_000_000)
        # run-level state
        self.finished = 0
        self.misses = 0
        self.t2_sessions = 0
        self.t2_wins = 0
        self.hostile = False
        self.t2_disabled = False
        self.tokens_spent = 0

    # ---------------- run-level bookkeeping ----------------

    def note_session_end(self, record: SessionRecord) -> None:
        """Called when the NEXT session's reset arrives (or at shutdown)."""
        if record.last_turn == 0:
            return
        miss = record.last_turn >= 10
        record.ended_by_miss = miss
        self.finished += 1
        self.misses += miss
        if record.tier2:
            self.t2_sessions += 1
            self.t2_wins += not miss
        # Regime detection: the public judge yields ~0 misses. A hostile world
        # shows up immediately and everywhere.
        if not self.hostile and self.finished >= 12 and self.misses >= 4:
            self.hostile = True
            self.t2_turn = min(self.t2_turn, 4)
        # A takeover that loses more than it wins is worse than staying home.
        if (not self.t2_disabled and self.t2_sessions >= 8
                and self.t2_wins * 2 < self.t2_sessions):
            self.t2_disabled = True

    def spend(self, usage: dict) -> None:
        self.tokens_spent += int(usage.get("prompt_tokens", 0) or 0)
        self.tokens_spent += int(usage.get("completion_tokens", 0) or 0)

    # ---------------- per-turn decision ----------------

    def wants_tier2(self, session, record: SessionRecord, turn: int,
                    handover_available) -> bool:
        """`handover_available` is a CALLABLE, checked last: the reachability
        ping must never fire unless every cheap condition already passed."""
        if not self.enabled or self.t2_disabled:
            return False
        if self.force_tier == "2":
            return bool(handover_available())
        if self.t2_sessions >= self.t2_max_sessions:
            return False
        if self.tokens_spent >= self.token_budget:
            return False
        if record.tier2:                       # already handed over: stay
            return bool(handover_available())
        if turn < self.t2_turn:
            return False
        # Two conditions, both required. (1) The WORLD looks abnormal: the
        # parser had to fall back to layer 2, or the run-level regime is
        # hostile. On the template judge everything parses at layer 1, so a
        # merely-slow session never hands over. (2) This session is losing:
        # clue harvest is spent, or we are well past the takeover turn.
        weak_world = (2 in session.parse_layers) or self.hostile
        if not weak_world:
            return False
        losing = session.harvest_exhausted or session.fruitless_asks >= 1
        if not (losing or turn >= self.t2_turn + 1):
            return False
        return bool(handover_available())

    def stats(self) -> dict:
        return {
            "finished": self.finished, "misses": self.misses,
            "tier2_sessions": self.t2_sessions, "tier2_wins": self.t2_wins,
            "hostile": self.hostile, "tier2_disabled": self.t2_disabled,
            "tokens_spent": self.tokens_spent,
        }


class StarterHandover:
    """Lazy bridge to the starter LLM agent. Never touches starter's code."""

    def __init__(self, catalog_path, client_factory=None) -> None:
        self.catalog_path = catalog_path
        self.client_factory = client_factory   # test hook: inject a mock client
        self._starter = None
        self._available: bool | None = None
        self._failures = 0

    # -- availability -------------------------------------------------

    def available(self) -> bool:
        """True only when a model endpoint answered a real ping."""
        if self._failures >= 2:
            return False
        if self._available is None:
            self._available = self._probe()
        return self._available

    def _build_client(self):
        if self.client_factory is not None:
            return self.client_factory()
        from starter.model_client import ModelSettings, OpenAICompatibleClient
        base = ModelSettings.from_env()
        if not base.enabled:
            return None
        timeout = float(os.environ.get("COPILOT_T2_TIMEOUT", "12"))
        return OpenAICompatibleClient(ModelSettings(
            api_key=base.api_key, model=base.model,
            base_url=base.base_url, timeout_seconds=timeout))

    def _probe(self) -> bool:
        try:
            client = self._build_client()
            if client is None:
                return False
            client.complete_json(
                [{"role": "user", "content": 'Return exactly this JSON object: {"ok": true}'}],
                max_tokens=16)
            return True
        except Exception:
            return False

    # -- the handover itself ------------------------------------------

    def _ensure_starter(self):
        if self._starter is None:
            from starter.agent import Agent as StarterAgent
            self._starter = StarterAgent(self.catalog_path,
                                         model_client=self._build_client())
        return self._starter

    def respond(self, session_id: str, profile: dict, record: SessionRecord,
                shown: set, user_message: str, turn: int, top_k: int,
                backfill) -> dict | None:
        """One handed-over turn. Returns None on failure (caller falls back)."""
        try:
            starter = self._ensure_starter()
            if session_id not in starter._sessions:
                # Case file: replay every past turn through starter's own
                # offline state API (pure string processing, zero tokens).
                starter.reset(session_id, profile)
                state = starter._sessions[session_id]
                for cust, ours, ask in record.transcript:
                    state.observe(cust)
                    state.record_response(ours, ask)
            response = starter.respond(session_id, user_message, turn, top_k)
            # Filter proven-wrong products, top up from ttfarm's next best.
            recs = [r["parent_asin"] for r in response.get("recommendations", [])
                    if isinstance(r, dict)]
            recs = [a for a in recs if a and a not in shown]
            for asin in backfill(top_k * 2):
                if len(recs) >= top_k:
                    break
                if asin not in recs and asin not in shown:
                    recs.append(asin)
            message = response.get("message")
            if not isinstance(message, str) or not message.strip():
                message = "Here are the strongest remaining matches."
            ask = response.get("ask_attribute")
            usage = response.get("usage") or {}
            return {"message": message, "ask_attribute": ask,
                    "recommendations": [{"parent_asin": a} for a in recs[:top_k]],
                    "usage": {"prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
                              "completion_tokens": int(usage.get("completion_tokens", 0) or 0)}}
        except Exception:
            self._failures += 1
            if self._failures >= 2:
                self._available = False
            return None
