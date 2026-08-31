# Autonomous Conversational Search Agent — TechJam 2026, Team TT Farm

**Fast reflexes, deep thinking: an AI shopping agent that knows when to think harder.**

📺 **Demo video:** https://www.youtube.com/watch?v=DxUX4gwlFUk

A customer hides one product in a 50,000-item catalog and gives you ten turns to
find it. Our answer is an **autonomous AI agent** — an LLM-driven planning and
ranking loop that reads any customer, plans its own catalog searches, and
reasons over candidates — wrapped around a **lightweight reflex layer** that
answers routine turns in 3 milliseconds for free.

A confidence controller sits between them and decides, turn by turn, when a
conversation needs the AI's full attention. Easy conversations never wake the
model. Hard ones are handed over — with the full case file.

## The story in one diagram

```
customer message
      │
      ▼
┌─────────────────────┐   routine turn: answered instantly, $0
│  reflex layer        │──────────────────────────────►  reply
│  (ttfarm/, stdlib,   │
│   3 ms per turn)     │   confidence low? session losing?
└─────────┬───────────┘   world doesn't match expectations?
          │ hand over the case file:
          │ profile · every clue heard · products already ruled out
          ▼
┌─────────────────────┐
│  AI agent            │   plans 1-3 catalog searches,
│  (starter/, LLM      │   reasons over 30 candidates,
│   planner + ranker)  │   asks the next smart question
└─────────────────────┘──────────────────────────────►  reply
          ▲
          └── from then on the two take alternating turns,
              sharing one exclusion list, until the product is found
```

The interesting engineering is the **handover**: knowing *when* the AI is
needed, and giving it everything the reflex layer learned so it starts smart,
not blind. That is the brief's "runtime workflow re-orchestration", built and
measured.

## Results

All numbers come from the **unmodified official evaluator**, full 200-session
runs.

**Official public benchmark** — templated customers, exactly as shipped:

| Configuration | TechnicalScore | HitRate@10 | Tokens |
| --- | ---: | ---: | ---: |
| Combined agent (reflex layer handles everything) | **0.9748** | 1.000 | 0 |
| Official weak baseline | 0.1067 | 0.125 | 0 |

**Hostile stress benchmark** — a private harness where the customer behaves
like a real human (free-form phrasing, synonyms for every clue, vague
categories, moods), scoring mechanics identical to the official evaluator:

| Configuration | TechnicalScore | HitRate@10 | Tokens | Wall clock |
| --- | ---: | ---: | ---: | ---: |
| **Combined agent (AI takes over when needed)** | **0.8100** | **0.905** | 1.3M | 28 min |
| AI agent full-time, `gpt-5.6-sol` | 0.7483 | 0.840 | 4.5M | 3 h 44 m |
| AI agent full-time, `gpt-5.4-mini` | 0.7233 | 0.850 | 5.5M | 42 min |
| Reflex layer alone (no model) | 0.6477 | 0.710 | 0 | 1 min |

Two sentences the tables earn:

- **When the customer goes off-script, the AI agent is what saves the game**:
  it takes over the losing sessions and wins 70% of them, lifting the hit rate
  from 0.71 to 0.905.
- **Thinking hard only when needed beats thinking hard always**: the combined
  agent outscores the always-on flagship model by +0.06 while spending less
  than a third of the tokens.

## Run it

Python 3.10+ recommended. No third-party runtime dependencies.

**1. Get the catalog** (a release asset of the official kit):

```bash
curl -LO https://github.com/TechJam2026/techjam-conversational-search/releases/download/participant-kit/catalog.jsonl.gz
curl -LO https://github.com/TechJam2026/techjam-conversational-search/releases/download/participant-kit/SHA256SUMS
shasum -a 256 -c SHA256SUMS --ignore-missing     # catalog.jsonl.gz: OK
gzip -dk catalog.jsonl.gz && mv catalog.jsonl data/catalog.jsonl
```

**2. Score it with the official evaluator** — no key needed for the public
benchmark:

```bash
python3 -m evaluator.local_evaluator            # scores starter/ (the AI agent path)
python3 -m ttfarm.tools.run_eval                # scores the combined agent -> 0.9748
python3 -m unittest discover tests              # 23 tests
```

**3. Enable the AI agent** (optional — used when conversations get hard):

```bash
export TECHJAM_LLM_API_KEY='<from your secret manager - never commit it>'
export TECHJAM_LLM_MODEL='gpt-5.4-mini'         # or any OpenAI-compatible model
export TECHJAM_LLM_BASE_URL='https://api.openai.com/v1'
```

`OPENAI_API_KEY/MODEL/BASE_URL` work as aliases; `TECHJAM_LLM_TIMEOUT` sets the
per-request timeout. For local development copy `.env.example` to the ignored
`.env`. If a request fails with `CERTIFICATE_VERIFY_FAILED` on a python.org
macOS build, set `SSL_CERT_FILE=/etc/ssl/cert.pem` locally; never disable TLS
verification. **If the model configuration is absent or unreachable, the agent
runs fully offline and still satisfies the official interface** — final
scoring may disable network access, and this repository loses nothing when it
does.

## What lives where

```
starter/    the AI agent: LLM planner -> FTS5 catalog retrieval -> LLM ranker,
            stateful across turns, provider-agnostic, validated JSON I/O
ttfarm/     the reflex layer + the confidence controller and handover
            (escalation.py), plus evaluation tools and robustness suites
evaluator/  the official local evaluator - unmodified
tests/      23 tests: both agents, the handover, contract fuzzing, regression pins
docs/       submission report, Devpost text, results, explainer PDFs (EN + CN)
```

Deep dives: [`docs/submission_report.md`](docs/submission_report.md) (method,
disclosure, limitations) and [`ttfarm/README.md`](ttfarm/README.md) (the reflex
layer and the escalation design, with every measurement).

## Honesty notes

- The evaluator and public labels were never modified; every score above comes
  from the official scorer. The hostile harness reuses its exact scoring
  mechanics and is kept out of this repository (it is a private stress tool).
- Public-set results are development measurements, not hidden-set predictions.
  With n=200 the sampling noise is roughly ±0.02–0.03; our honest expectation
  for the official hidden set is **0.95 ± 0.03**.
- Escalation decisions were calibrated on paired per-session records from the
  tune split; the details and the two calibrations that came out of it are in
  `ttfarm/README.md`.

## Team

Team **TT Farm**.

| Member | Contribution |
| --- | --- |
| Ming Wang | Autonomous AI agent (`starter/`): LLM planning and ranking loop, model client, live evaluation runs; repository owner |
| Wenjie Huang | Reflex layer and escalation (`ttfarm/`): fast path, confidence controller and handover, evaluation and robustness tooling; submission writings |
| Yang Xu | Early-stage brainstorming and track research; QA design; data preparation; independent reproduction of the evaluation pipeline from a clean clone |
| jayde zhang | Early-stage brainstorming and pre-research; submission QA — demo-video fact-check, documentation proofreading, registration and publication verification; data preparation support |

## Data

The catalog and sessions derive from Amazon Reviews 2023 (McAuley Lab, UCSD) —
see [`DATA_ATTRIBUTION.md`](DATA_ATTRIBUTION.md). Competition data is used only
for this competition and will be deleted when it ends, per the event rules.
