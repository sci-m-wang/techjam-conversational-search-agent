"""Submission tests: contract safety, behaviors, and a mini end-to-end run.

Run:  python3 -m pytest tests/ -q   (or python3 -m unittest discover tests)
No external dependencies; a tiny synthetic catalog is built per test class.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ttfarm.agent import Agent  # noqa: E402
from ttfarm.nlu import parse  # noqa: E402

CATALOG = [
    {"parent_asin": "A1", "title": "Blue leather belt with buckle", "features": ["100% Leather", "Buckle closure"],
     "details": {"Department": "mens"}, "categories": ["Clothing", "Accessories, Belts"],
     "store": "BeltCo", "rating_number": 5000, "average_rating": 4.5, "price": 25.0},
    {"parent_asin": "A2", "title": "Red cotton hoodie sweatshirt", "features": ["90% Cotton", "Pull On closure"],
     "details": {"Department": "womens"}, "categories": ["Clothing", "Fashion Hoodies & Sweatshirts"],
     "store": "HoodieCo", "rating_number": 900, "average_rating": 4.2, "price": 30.0},
    {"parent_asin": "A3", "title": "Green nylon running shorts", "features": ["Nylon", "Drawstring closure"],
     "details": {"Department": "mens"}, "categories": ["Clothing", "Active, Shorts"],
     "store": "RunCo", "rating_number": 12000, "average_rating": 4.7, "price": "19.99"},
    {"parent_asin": "A4", "title": "Black leather belt classic", "features": ["Full grain leather"],
     "details": {"Department": "mens"}, "categories": ["Clothing", "Accessories, Belts"],
     "store": "BeltCo", "rating_number": 300, "average_rating": 4.0, "price": 22.0},
]


def make_agent() -> Agent:
    tmp = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8")
    for row in CATALOG:
        tmp.write(json.dumps(row) + "\n")
    tmp.close()
    return Agent(tmp.name)


class ContractTest(unittest.TestCase):
    """The evaluator voids any turn whose response breaks shape - never break it."""

    @classmethod
    def setUpClass(cls):
        cls.agent = make_agent()

    def check(self, response):
        self.assertIsInstance(response, dict)
        self.assertIsInstance(response["message"], str)
        self.assertTrue(response["message"])
        self.assertIn(response["ask_attribute"],
                      {"category", "material", "color", "size", "style", "brand",
                       "budget", "feature", "use_case", "other", None})
        self.assertIsInstance(response["recommendations"], list)
        asins = [r["parent_asin"] for r in response["recommendations"]]
        self.assertEqual(len(asins), len(set(asins)))       # unique
        self.assertLessEqual(len(asins), 10)
        usage = response["usage"]
        self.assertGreaterEqual(usage["prompt_tokens"], 0)
        self.assertGreaterEqual(usage["completion_tokens"], 0)

    def test_hostile_inputs_never_break_contract(self):
        hostile = ["", "   ", "????", "x" * 20000, "I'm looking for \x00\x01.",
                   "🎉🎉🎉 unicode storm ☃", "For that, what matters is: .",
                   "Actually, ignore my earlier preference. What I need is: .",
                   "'; DROP TABLE products; --", "\n\n\n", "None", "nan"]
        self.agent.reset("s1", {})
        for turn, msg in enumerate(hostile, start=1):
            self.check(self.agent.respond("s1", msg, min(turn, 10), 10))

    def test_respond_without_reset_still_answers(self):
        self.check(self.agent.respond("never-reset", "I'm looking for belts.", 1, 10))

    def test_missing_profile_fields(self):
        self.agent.reset("s2", None)
        self.check(self.agent.respond("s2", "I'm looking for Accessories Belts, but I'm still exploring.", 1, 10))


class BehaviorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.agent = make_agent()

    def test_buying_flow_finds_target_and_unveils(self):
        self.agent.reset("b1", {})
        r1 = self.agent.respond("b1", "I'm looking for Accessories Belts. A key requirement is: Full grain leather.", 1, 10)
        self.assertEqual([x["parent_asin"] for x in r1["recommendations"]], ["A4"])  # k=1 sniper, exact phrase wins
        r2 = self.agent.respond("b1", "Those options are not quite right yet. Ask me about one specific attribute.", 2, 10)
        self.assertNotIn("A4", [x["parent_asin"] for x in r2["recommendations"]])    # unveiling excludes shown

    def test_override_keeps_evidence_and_clears_exclusions(self):
        self.agent.reset("o1", {})
        self.agent.respond("o1", "I'm looking for Accessories Belts. Buckle closure", 1, 10)
        r = self.agent.respond("o1", "Actually, ignore my earlier preference. What I need is: leather.", 3, 10)
        session = self.agent.sessions["o1"]
        self.assertTrue(session.override_seen)
        # Exclusion memory was cleared on override: only THIS turn's (scored,
        # post-override) recommendations may be in it - pre-override exposures
        # are re-eligible again.
        self.assertLessEqual(session.shown, {x["parent_asin"] for x in r["recommendations"]})
        texts = [s.text for s in session.slots]
        self.assertIn("leather", texts)                           # new value present
        self.assertIn("Buckle closure", texts)                    # old evidence retained (demoted)
        self.assertTrue(r["recommendations"])                     # still recommending

    def test_paraphrased_messages_parse_to_same_kinds(self):
        pairs = [
            ("I want Accessories Belts - it must have leather.", "open"),
            ("Mostly I care about Nylon and Drawstring closure.", "reply"),
            ("Change of plans: leather is what matters now.", "override"),
            ("No preference for that - you decide.", "nopref"),
        ]
        for msg, kind in pairs:
            self.assertEqual(parse(msg, 1 if kind == "open" else 3).kind, kind, msg)

    def test_deterministic(self):
        outs = []
        for run in range(2):
            agent = make_agent()
            agent.reset("d", {})
            outs.append(json.dumps([
                agent.respond("d", "I'm looking for Accessories Belts, but I'm still exploring.", 1, 10),
                agent.respond("d", "For that, what matters is: leather; Buckle closure.", 2, 10),
            ], sort_keys=True))
        self.assertEqual(outs[0], outs[1])


if __name__ == "__main__":
    unittest.main()


class _MockJsonClient:
    """Minimal JsonModelClient: proves the tier-2 handover plumbing end to end."""

    def complete_json(self, messages, max_tokens):
        from starter.model_client import ModelResult, ModelUsage
        text = " ".join(m.get("content", "") for m in messages)
        usage = ModelUsage(prompt_tokens=10, completion_tokens=5)
        if "search_queries" in text or "Plan this turn" in text:
            return ModelResult({"search_queries": ["leather belt"],
                                "ask_attribute": "feature"}, usage)
        if "Choose recommendations" in text:
            return ModelResult({"message": "How about these?",
                                "ask_attribute": "feature",
                                "recommendations": [{"parent_asin": "A1"}]}, usage)
        return ModelResult({"ok": True}, usage)


class EscalationTest(unittest.TestCase):
    def _agent(self, factory=None):
        import tempfile
        tmp = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8")
        for row in CATALOG:
            tmp.write(json.dumps(row) + "\n")
        tmp.close()
        return Agent(tmp.name, handover_client_factory=factory)

    def test_no_key_means_fully_inert(self):
        import os
        saved = {k: os.environ.pop(k, None) for k in
                 ("TECHJAM_LLM_API_KEY", "OPENAI_API_KEY", "COPILOT_FORCE_TIER")}
        try:
            agent = self._agent()
            agent.reset("s", {})
            for turn in range(1, 11):
                out = agent.respond("s", "totally freeform message about cowhide things", turn, 10)
                self.assertIsInstance(out["message"], str)
            self.assertEqual(agent.stats["tier2_turns"], 0)
        finally:
            for k, v in saved.items():
                if v is not None:
                    os.environ[k] = v

    def test_forced_tier2_uses_handover_and_respects_exclusions(self):
        import os
        os.environ["COPILOT_FORCE_TIER"] = "2"
        try:
            agent = self._agent(factory=lambda: _MockJsonClient())
            agent.reset("s", {"summary": "x"})
            first = agent.respond("s", "I'm looking for Accessories Belts. Buckle closure", 1, 10)
            self.assertIsInstance(first["message"], str)
            self.assertTrue(agent.stats["tier2_turns"] >= 1)
            shown_first = {r["parent_asin"] for r in first["recommendations"]}
            second = agent.respond("s", "For that, what matters is: leather.", 2, 10)
            shown_second = [r["parent_asin"] for r in second["recommendations"]]
            # proven-wrong products from turn 1 must not be shown again
            self.assertFalse(shown_first & set(shown_second))
            self.assertGreater(agent.stats["tokens_spent"], 0)
        finally:
            os.environ.pop("COPILOT_FORCE_TIER", None)

    def test_handover_failure_falls_back_to_deterministic_path(self):
        import os
        os.environ["COPILOT_FORCE_TIER"] = "2"

        class _Boom:
            def complete_json(self, messages, max_tokens):
                raise RuntimeError("endpoint down mid-run")
        try:
            agent = self._agent(factory=lambda: _Boom())
            agent.reset("s", {})
            out = agent.respond("s", "I'm looking for Accessories Belts. Buckle closure", 1, 10)
            # ping fails -> handover unavailable -> deterministic path answers
            self.assertTrue(out["recommendations"])
            self.assertEqual(agent.stats["tier2_turns"], 0)
        finally:
            os.environ.pop("COPILOT_FORCE_TIER", None)
