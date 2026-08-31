# Devpost description — paste-ready (Team TT Farm)

## Project name

Autonomous Conversational Search Agent (TT Farm)

## Elevator pitch

An AI shopping agent with fast reflexes: an autonomous LLM planner-ranker that
takes over conversations the moment a lightweight reflex layer stops being
enough — 0.9748 on the official benchmark for zero tokens, and a 0.71 → 0.905
hit-rate rescue when customers go off-script.

## The story

A customer hides one product among 50,000 and gives you ten turns. Easy
customers speak predictably; real ones don't. So we built two minds and taught
them to share the case.

The **AI agent** is the deep thinker: an autonomous two-stage loop in which an
LLM plans its own catalog searches, a local FTS5 index retrieves bounded
candidates, and the LLM reasons over them to pick the Top-10 and the next smart
question — stateful, provider-agnostic, and validated at every boundary. The
**reflex layer** is its lightweight partner: a stdlib-only fast path that
answers a routine turn in 3 milliseconds for free.

Between them sits a **confidence controller**. It watches parser fallbacks,
category-match quality, clue-harvest progress, and the run-level miss rate;
when a session is losing in a world that doesn't match expectations — and a
live model answers a reachability ping — it hands the conversation to the AI
agent with everything learned so far: the profile, every clue heard, and the
products already ruled out. From then on the two minds alternate turns over a
shared exclusion list until the product is found. This is the brief's "runtime
workflow re-orchestration", implemented and measured.

## How it addresses the four pillars

- **Intent routing & hybrid pipeline** — multi-route retrieval (keyword FTS5,
  category pools, evidence scoring) feeding **LLM semantic ranking** in the AI
  agent's decision stage.
- **Dialog strategy** — slot accumulation, override handling with decay (a
  disclosed clue is never re-disclosed, so evidence is demoted, not deleted),
  and proactive clarification every turn.
- **Self-evolution / dynamic context programming** — the confidence controller
  re-orchestrates the pipeline at runtime: per-session escalation, run-level
  regime detection, and a governor that disables a takeover that stops
  winning.
- **Evaluation matrix** — everything is measured through the unmodified
  official evaluator: paired per-session records across five configurations,
  robustness suites (template paraphrase, constraint-vocabulary damage), and
  regression pins that prove the offline path never changes.

## Results (unmodified official evaluator, full 200-session runs)

Official public benchmark: **TechnicalScore 0.9748** (HitRate@10 1.000), zero
tokens — the reflex layer alone suffices, and final scoring with network
disabled loses nothing.

Hostile stress benchmark (private harness, human-like customers, identical
scoring mechanics): combined agent **0.8100 / 0.905 hit rate** vs 0.7483 for
the flagship model running full-time and 0.6477 for the reflex layer alone.
The AI agent wins 70% of the sessions it takes over — and the combined system
beats always-on flagship thinking by +0.06 at under a third of the tokens.

## Development tools used

- Python 3.10+ (standard library only at runtime)
- _(fill: editors/IDEs; AI-assisted development tools if the team wishes to
  disclose them — the rules require the list to be accurate)_

## APIs used

- OpenAI-compatible Chat Completions endpoints for the AI agent's planning and
  ranking calls (measured with `gpt-5.6-sol` and `gpt-5.4-mini`). Optional at
  runtime: with no key configured the agent runs fully offline.

## Libraries and frameworks used

- Python standard library only (sqlite3 FTS5, urllib, unittest). No
  third-party runtime dependencies.

## Datasets and assets used

- The official TechJam participant kit only: frozen 50,000-product catalog and
  200 public sessions, derived from Amazon Reviews 2023 (McAuley Lab, UCSD).
  No external training data; no pretrained weights.

## Team

TT Farm — _(fill: member names as registered)_

## Links

- Public repository: https://github.com/sci-m-wang/techjam-conversational-search-agent
- Demo video (YouTube, public): _(fill)_
