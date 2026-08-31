"""Constraint-vocabulary damage curve: how fast does the score fall when the
customer's constraint WORDS change (synonyms we can't match), not just the
sentence templates around them?

Each token inside a constraint-bearing span is replaced with an
out-of-vocabulary word with probability p - the worst case for lexical
matching (a synonym we know nothing about). Templates and category stay
intact; this isolates vocabulary overfit from template overfit
(tools/paraphrase_eval.py).

    python3 tools/damage_eval.py --kit ../kit --rates 0.1 0.25 0.5
"""
from __future__ import annotations

import argparse
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

RE_BUY = re.compile(r"^(I'm looking for .+?\. A key requirement is: )(.+)(\.)$")
RE_OV = re.compile(r"^(Actually, ignore my earlier preference\. What I need is: )(.+)(\.)$")
RE_REPLY = re.compile(r"^(For that, what matters is: )(.+)(\.)$")
WORD = re.compile(r"[A-Za-z0-9%]+")


def damage_span(text: str, p: float, rng: random.Random) -> str:
    return WORD.sub(lambda m: "zzqx" if rng.random() < p else m.group(0), text)


class DamageProxy:
    def __init__(self, inner, p: float, seed: int = 11):
        self.inner, self.p, self.rng = inner, p, random.Random(seed)

    def reset(self, session_id, user_profile):
        return self.inner.reset(session_id, user_profile)

    def respond(self, session_id, user_message, turn, top_k):
        msg = user_message
        for pattern in (RE_BUY, RE_OV, RE_REPLY):
            m = pattern.match(msg)
            if m:
                msg = m.group(1) + damage_span(m.group(2), self.p, self.rng) + m.group(3)
                break
        return self.inner.respond(session_id, msg, turn, top_k)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kit", default=str(ROOT))
    ap.add_argument("--rates", type=float, nargs="+", default=[0.1, 0.25, 0.5])
    args = ap.parse_args()
    kit = Path(args.kit).resolve()
    sys.path.insert(0, str(kit))
    from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl

    from ttfarm.agent import Agent

    samples = load_jsonl(kit / "data/public_set.jsonl")
    catalog_ids, categories, products = catalog_index(kit / "data/catalog.jsonl")
    print(f"{'p_damage':>9} {'TS':>8} {'HR@10':>7} {'MRR':>7} {'MTTC':>6}")
    for p in [0.0] + list(args.rates):
        agent = DamageProxy(Agent(kit / "data/catalog.jsonl"), p)
        r = evaluate(agent, samples, catalog_ids, categories, products)
        print(f"{p:>9.2f} {r['recommended_technical_score']:>8.4f} "
              f"{r['hit_rate_at_10']:>7.3f} {r['mrr']:>7.4f} {r['mttc']:>6.2f}")


if __name__ == "__main__":
    main()
