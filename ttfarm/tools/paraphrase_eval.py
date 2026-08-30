"""Robustness harness: paraphrase every customer message before the agent sees
it, evaluator untouched. Facts (category text, constraint text) are preserved -
exactly the guarantee in the public spec ("paraphrasing ... cannot decide
correctness"); surface wording changes.

    python3 tools/paraphrase_eval.py --kit ../kit --level 2
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

BUY = ["I want {cat} - it must have {c}.",
       "Shopping for {cat}. Non-negotiable for me: {c}.",
       "I need {cat} and it really needs {c}."]
BROWSE = ["Just browsing {cat} for now.",
          "I want some {cat} but haven't decided yet.",
          "Casually looking at {cat}, nothing fixed."]
OPEN_OTHER = ["I want {cat}. {c}", "Looking at {cat} these days. {c}"]
REPLY = ["Mostly I care about {a}.", "It should have {a}.", "Main thing for me: {a}."]
NOPREF = ["No strong preference there, honestly.", "Anything works for that."]
BOUNDARY = ["No preference for that - you decide.", "Whatever you think is best there."]
NUDGE = ["Not quite it. Try asking me something specific.",
         "Still not right - ask me about one thing."]
OVERRIDE = ["You know what, forget that - what I really need is {c}.",
            "Change of plans: {c} is what matters now.",
            "On second thought, drop my earlier preference. I need {c}."]

RE_BUY = re.compile(r"^I'm looking for (?P<cat>.+?)\. A key requirement is: (?P<c>.+)\.$")
RE_BROWSE = re.compile(r"^I'm looking for (?P<cat>.+?), but I'm still exploring\.$")
RE_OTHER = re.compile(r"^I'm looking for (?P<cat>.+?)\. (?P<c>.+)$")
RE_OV = re.compile(r"^Actually, ignore my earlier preference\. What I need is: (?P<c>.+)\.$")
RE_REPLY = re.compile(r"^For that, what matters is: (?P<cs>.+)\.$")
RE_NOPREF = re.compile(r"^I don't have an additional preference for .+")
RE_BOUND = re.compile(r"^I don't have a preference for .+judgment\.$")
RE_NUDGE = re.compile(r"^Those options are not quite right yet\.")


def paraphrase(msg: str, level: int, rng: random.Random) -> str:
    if level == 0:
        return msg
    pick = lambda options: rng.choice(options)
    m = RE_OV.match(msg)
    if m:
        return pick(OVERRIDE).format(c=m["c"])
    m = RE_REPLY.match(msg)
    if m:
        joiner = " and " if level >= 2 else "; "
        return pick(REPLY).format(a=joiner.join(m["cs"].split("; ")))
    if RE_BOUND.match(msg):
        return pick(BOUNDARY)
    if RE_NOPREF.match(msg):
        return pick(NOPREF)
    if RE_NUDGE.match(msg):
        return pick(NUDGE)
    m = RE_BUY.match(msg)
    if m:
        return pick(BUY).format(cat=m["cat"], c=m["c"])
    m = RE_BROWSE.match(msg)
    if m:
        return pick(BROWSE).format(cat=m["cat"])
    m = RE_OTHER.match(msg)
    if m:
        return pick(OPEN_OTHER).format(cat=m["cat"], c=m["c"])
    return msg


class ParaphrasingProxy:
    def __init__(self, inner, level: int, seed: int = 7):
        self.inner, self.level, self.rng = inner, level, random.Random(seed)
        self.rewrites = 0

    def reset(self, session_id, user_profile):
        return self.inner.reset(session_id, user_profile)

    def respond(self, session_id, user_message, turn, top_k):
        new = paraphrase(user_message, self.level, self.rng)
        if new != user_message:
            self.rewrites += 1
        return self.inner.respond(session_id, new, turn, top_k)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kit", default=str(ROOT))
    ap.add_argument("--level", type=int, default=1)
    args = ap.parse_args()
    kit = Path(args.kit).resolve()
    sys.path.insert(0, str(kit))
    from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl

    from ttfarm.agent import Agent

    samples = load_jsonl(kit / "data/public_set.jsonl")
    catalog_ids, categories, products = catalog_index(kit / "data/catalog.jsonl")
    proxy = ParaphrasingProxy(Agent(kit / "data/catalog.jsonl"), args.level)
    result = evaluate(proxy, samples, catalog_ids, categories, products)
    out = {k: v for k, v in result.items() if k not in ("sessions", "reported_token_usage")}
    out["paraphrase_level"] = args.level
    out["messages_rewritten"] = proxy.rewrites
    print(json.dumps(out, indent=2))
    print("misses:", [s["sample_id"] for s in result["sessions"] if not s["hit"]])


if __name__ == "__main__":
    main()
