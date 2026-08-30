"""Candidate generation + ranking.

Dual-track (Pillar I):
- category pool  = high-precision track (measured: target inside its exact pool
  200/200 on the public set; median pool 184 of 50,000)
- token evidence = graduated scoring, so paraphrased or partial constraints
  still rank (hard AND is a special case of full-credit)

Score per product (weights tuned by self-play; see tools/selfplay.py):
    W_PHRASE * sum(slot.weight for slots whose token sequence appears verbatim)
  + W_FULL   * sum(slot.weight for slots with all tokens present)
  + W_COVER  * IDF-weighted fraction of slot tokens present
  + W_POP    * normalized log-popularity   (targets are popularity-skewed:
               median rating_number 6,846 vs 12 for the catalog)
  + W_PRICE  * price-band bonus when a budget slot is present
Deterministic tiebreak: (-score, -popularity, asin).
"""
from __future__ import annotations

import re

import json
import os

from ttfarm.catalog import Catalog
from ttfarm.state import Session

WEIGHTS = {"phrase": 4.0, "full": 2.0, "cover": 1.5, "pop": 1.0, "price": 0.5}
if os.environ.get("COPILOT_WEIGHTS"):    # experiment override (tools/ sweeps)
    WEIGHTS.update(json.loads(os.environ["COPILOT_WEIGHTS"]))
_BUDGET_RE = re.compile(r"(?:budget around|around|under|below|less than)?\s*\$\s*([\d.]+)", re.I)


def _budget_from_slots(session: Session) -> float | None:
    for slot in session.slots:
        if "$" in slot.text or "budget" in slot.text.lower():
            m = _BUDGET_RE.search(slot.text)
            if m:
                try:
                    return float(m.group(1))
                except ValueError:
                    continue
    return None


def rank(catalog: Catalog, session: Session, limit: int = 10, exclude_shown: bool = True) -> list[str]:
    pool = catalog.match_pool(session.category) if session.category else catalog.global_head
    budget = _budget_from_slots(session)
    slot_data = []
    for slot in session.slots:
        toks = slot.toks
        if toks:
            idf_sum = sum(catalog.idf(t) for t in toks)
            slot_data.append((slot, toks, idf_sum))
    loose = [(t, w) for t, w in session.loose.items()]
    loose_idf = sum(catalog.idf(t) * w for t, w in loose) or 1.0

    scored: list[tuple[float, float, str]] = []
    shown = session.shown if exclude_shown else frozenset()
    for asin in pool:
        if asin in shown:
            continue
        phrase = full = 0.0
        cover_num = cover_den = 0.0
        for slot, toks, idf_sum in slot_data:
            present = [t for t in toks if catalog.has_token(asin, t)]
            cover_den += slot.weight * (idf_sum or 1.0)
            cover_num += slot.weight * sum(catalog.idf(t) for t in present)
            if len(present) == len(toks):
                full += slot.weight
                if len(toks) > 1 and catalog.has_phrase(asin, toks):
                    phrase += slot.weight
                elif len(toks) == 1:
                    phrase += 0.25 * slot.weight   # single-token slot: no order info
        cover = (cover_num / cover_den) if cover_den else 0.0
        if loose:
            cover += 0.3 * sum(catalog.idf(t) * w for t, w in loose
                               if catalog.has_token(asin, t)) / loose_idf
        pop = catalog.pop[asin] / catalog.max_pop
        price_bonus = 0.0
        if budget is not None:
            p = catalog.price.get(asin)
            if p is not None and 0.5 * budget <= p <= 1.5 * budget:
                price_bonus = 1.0
        score = (WEIGHTS["phrase"] * phrase + WEIGHTS["full"] * full
                 + WEIGHTS["cover"] * cover + WEIGHTS["pop"] * pop
                 + WEIGHTS["price"] * price_bonus)
        scored.append((-score, -catalog.pop[asin], asin))
    scored.sort()
    return [asin for _, _, asin in scored[:limit]]
