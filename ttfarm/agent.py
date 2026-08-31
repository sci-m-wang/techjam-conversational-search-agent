"""TT Farm conversational shopping agent - combined architecture.

Tier 0 (default, and the only tier without a model key): the deterministic
ttfarm pipeline - parse, remember, narrow, rank, decide. Offline, stdlib-only.

Tier 1: when a message needed the structural (layer-2) parser and a model is
configured, ttfarm/llm.py may refine the parse. ttfarm still drives.

Tier 2: a session that is clearly losing in an abnormal-looking world hands its
remaining turns to the LLM agent in starter/, seeded with ttfarm's case file
(profile, transcript, proven-wrong products). See ttfarm/escalation.py.

Without a key the escalation machinery is inert and behavior is byte-identical
to the pure deterministic agent (regression-pinned by tests).
"""
from __future__ import annotations

from pathlib import Path

from ttfarm import llm
from ttfarm.catalog import Catalog
from ttfarm.escalation import EscalationController, SessionRecord, StarterHandover
from ttfarm.nlu import _OVERRIDE_CUES, Observation, parse
from ttfarm.policy import choose_ask, choose_k, compose_message
from ttfarm.search import rank
from ttfarm.state import Session


class Agent:
    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl",
                 handover_client_factory=None) -> None:
        self.catalog = Catalog(catalog_path)
        self.sessions: dict[str, Session] = {}
        self.records: dict[str, SessionRecord] = {}
        self.controller = EscalationController()
        self.handover = StarterHandover(catalog_path, client_factory=handover_client_factory)
        self._last_sid: str | None = None
        self.tier2_turns = 0

    # ------------------------------------------------------------------

    def reset(self, session_id: str, user_profile: dict) -> None:
        if self._last_sid is not None and self._last_sid in self.records:
            self.controller.note_session_end(self.records[self._last_sid])
        self._last_sid = session_id
        self.sessions[session_id] = Session(profile=dict(user_profile or {}))
        self.records[session_id] = SessionRecord()

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        try:
            return self._respond(session_id, str(user_message), int(turn), int(top_k))
        except Exception:
            # Contract safety net: a turn must never be voided by an exception.
            return {"message": "Could you tell me one specific requirement?",
                    "ask_attribute": "other", "recommendations": [],
                    "usage": dict(llm.USAGE)}

    # ------------------------------------------------------------------

    def _respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        session = self.sessions.setdefault(session_id, Session())
        record = self.records.setdefault(session_id, SessionRecord())
        session.turn = turn
        record.last_turn = max(record.last_turn, turn)

        obs = parse(user_message, turn)
        if obs.layer == 2 and llm.enabled():          # Tier 1: optional, flagged
            data = llm.extract(user_message)
            if data:
                # Accept the model's override signal only when the message
                # carries an override cue. Measured on the hostile harness:
                # ungated, the model over-flags overrides, wiping the
                # exclusion memory - 7 of 30 override sessions flipped from
                # win to loss. An override without any cue word is far more
                # likely a model hallucination than a real change of mind.
                cued = any(c in user_message.lower() for c in _OVERRIDE_CUES)
                obs = Observation(
                    "override" if (data.get("override") and cued) else obs.kind,
                    obs.scenario_hint, data.get("category") or obs.category,
                    [str(c) for c in (data.get("constraints") or [])][:4] or obs.constraints,
                    obs.loose_tokens, layer=3)
        session.observe(obs)

        # ---- Tier 2: escalate a losing session (alternate or takeover) ----
        escalated = False
        if self.controller.wants_tier2(session, record, turn,
                                       self.handover.available):
            record.tier2 = True
            escalated = True
            if self.controller.starter_drives(record):
                response = self.handover.respond(
                    session_id, session.profile, record, session.shown,
                    user_message, turn, top_k,
                    backfill=lambda n: rank(self.catalog, session, limit=n))
                if response is not None:
                    record.tier2_turns += 1
                    self.tier2_turns += 1
                    self.controller.spend(response.get("usage", {}))
                    shown_now = [r["parent_asin"] for r in response["recommendations"]]
                    session.shown.update(shown_now)
                    record.transcript.append(
                        (user_message, response["message"], response["ask_attribute"]))
                    return response
                # handover failed: fall through to the deterministic path
            else:
                record.tier2_turns += 1        # ttfarm's alternation turn

        # ---- Tier 0: the deterministic ttfarm pipeline ----
        # On an escalated ttfarm turn, show the full list: in a hostile world
        # coverage beats sniping, and starter covers the other turns.
        k = top_k if escalated else choose_k(session, turn, top_k)
        # Turn 10 safety net: re-rank over the full pool ignoring the exclusion
        # memory, in case an unparsed override made us exclude the true target.
        recs = rank(self.catalog, session, limit=k, exclude_shown=turn < 10) if k else []
        session.shown.update(recs)
        ask = choose_ask(session, turn)
        message = compose_message(session, ask, len(recs))
        record.transcript.append((user_message, message, ask))
        return {
            "message": message,
            "ask_attribute": ask,
            "recommendations": [{"parent_asin": a} for a in recs],
            "usage": dict(llm.USAGE),
        }

    # ------------------------------------------------------------------

    @property
    def stats(self) -> dict:
        out = self.controller.stats()
        out["tier2_turns"] = self.tier2_turns
        out["handover_probed"] = self.handover._available is not None
        return out
