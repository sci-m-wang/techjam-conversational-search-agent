"""Session state: typed slots with override semantics that keep evidence.

Design decision (measured, notes/team-briefing.md §5): on an intent override we
demote earlier slots instead of deleting them. The public generator's "ignored"
preference still truthfully describes the same hidden target, and information
once revealed is never revealed again - erasure is unrecoverable. Demotion
implements the brief's "slot decay over time" (§4.3 in-scope list) and is
paraphrase-robust by construction.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ttfarm.catalog import tokens
from ttfarm.nlu import Observation

FRESH, DECAYED = 1.0, 0.35   # slot weights: current vs pre-override evidence


@dataclass
class Slot:
    text: str
    weight: float = FRESH

    @property
    def toks(self) -> list[str]:
        return tokens(self.text)


@dataclass
class Session:
    profile: dict = field(default_factory=dict)
    category: str | None = None
    slots: list[Slot] = field(default_factory=list)
    loose: dict[str, float] = field(default_factory=dict)   # token -> weight
    scenario: str = "browsing"        # buying | browsing | override_likely
    override_seen: bool = False
    fruitless_asks: int = 0           # consecutive no-new-info replies
    turn: int = 0
    turn_just_overridden: bool = False
    shown: set = field(default_factory=set)
    parse_layers: set[int] = field(default_factory=set)

    def _add_slot(self, text: str, weight: float = FRESH) -> None:
        text = text.strip()
        if not text:
            return
        for slot in self.slots:
            if slot.text == text:
                # Re-mentioned (e.g. an override naming an already-known clue):
                # restore its weight - the customer just re-asserted it.
                slot.weight = max(slot.weight, weight)
                return
        self.slots.append(Slot(text, weight))

    def observe(self, obs: Observation) -> None:
        self.parse_layers.add(obs.layer)
        self.turn_just_overridden = obs.kind == "override"
        if obs.kind == "open":
            self.category = obs.category or self.category
            self.scenario = obs.scenario_hint or self.scenario
            for c in obs.constraints:
                self._add_slot(c)
        elif obs.kind == "reply":
            if obs.constraints:
                self.fruitless_asks = 0
                for c in obs.constraints:
                    self._add_slot(c)
            else:
                self.fruitless_asks += 1
        elif obs.kind == "override":
            self.override_seen = True
            self.shown.clear()        # pre-override exposures were unscored
            self.scenario = "buying"          # post-override: locked requirement
            for s in self.slots:              # demote, never erase
                s.weight = min(s.weight, DECAYED)
            for c in obs.constraints:
                self._add_slot(c, FRESH)
        elif obs.kind in ("nopref", "nudge"):
            self.fruitless_asks += 1
        for t in obs.loose_tokens:
            self.loose[t] = max(self.loose.get(t, 0.0), 0.5 * (obs.layer == 2) + 0.0)

    @property
    def harvest_exhausted(self) -> bool:
        return self.fruitless_asks >= 2
