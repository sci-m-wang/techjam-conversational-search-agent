# Submission Report — Team TT Farm

## System

The submission is one `Agent` implementing the official Python interface, built
from two cooperating minds and a controller that decides which one answers.

**The AI agent** (`starter/agent.py`) is stateful across turns and uses a
two-stage autonomous decision loop:

1. a planner converts the anonymized profile and active conversation state into
   one to three catalog queries and selects the next clarification attribute;
2. local SQLite FTS5 retrieval produces a bounded candidate set from the frozen
   50,000-product catalog;
3. a decision call ranks only retrieved catalog identifiers and returns both a
   Top-10 list and a natural-language clarification question.

The agent tracks earlier questions, declined preferences, and explicit intent
overrides. Model output is parsed and validated before it reaches the
evaluator: unknown ASINs, duplicate ASINs, invalid attributes, and malformed
JSON are discarded or replaced by deterministic fallback behavior.

**The reflex layer** (`ttfarm/`) is a deliberately light deterministic path —
standard library only, ~3 ms per turn, zero tokens — that parses templated and
mildly paraphrased customers, accumulates constraint slots, retrieves through a
category-pool index, and sizes its recommendation lists decision-theoretically.
On the official public benchmark it resolves every session on its own.

**The escalation controller** (`ttfarm/escalation.py`) joins them. Confidence
signals (parser fallback depth, category-match quality, clue-harvest progress,
run-level miss rate) mark a session as *losing in an abnormal world*; a
reachability ping confirms a live model; then the session is handed to the AI
agent **with the full case file** — profile, every clue heard so far (replayed
through the AI agent's own state API at zero token cost), and the list of
products already ruled out, which is filtered from its answers. From the
handover onward the two minds take alternating turns over a shared exclusion
list. Run-level governance caps sessions and tokens and disables a takeover
that is not winning.

## Model and configuration

Measured live runs used `gpt-5.6-sol` and `gpt-5.4-mini` through
OpenAI-compatible Chat Completions endpoints. The source remains provider
agnostic: model, endpoint, and credentials are read only from
`TECHJAM_LLM_API_KEY`, `TECHJAM_LLM_MODEL`, and `TECHJAM_LLM_BASE_URL` (with
`OPENAI_*` aliases). Secrets are not present in this repository or its Git
history.

## Network behavior and offline fallback

Live planning and the escalation path require outbound access to the
configured API, gated by a reachability ping. If the model variables are
missing, the request fails, or the response is unusable, the same `Agent`
remains fully runnable: the reflex layer answers every turn with no network,
no third-party package, and no token budget. This matters because final
scoring may disable network access — offline, this submission still scores
its full official-benchmark result.

## Evaluation

All results come from the unmodified official evaluator.

**Official public benchmark (200 sessions):**

| Mode | Hit Rate@10 | MRR | MTTC | TechnicalScore |
| --- | ---: | ---: | ---: | ---: |
| Combined agent (offline; reflex layer suffices) | 1.000 | 0.995 | 2.19 | **0.9748** |
| AI agent, `gpt-5.6-sol` live | 0.935 | 0.673 | 3.77 | 0.8140 |
| AI agent, offline fallback | 0.850 | 0.464 | 4.61 | 0.6922 |
| Official weak starter | 0.125 | 0.068 | 9.81 | 0.1067 |

**Hostile stress benchmark** — a private harness with human-like customers
(free-form phrasing, synonym substitution on every clue, vague categories,
difficult moods), scoring mechanics identical to the official evaluator; full
200 sessions per run:

| Mode | Hit Rate@10 | MRR | MTTC | TechnicalScore |
| --- | ---: | ---: | ---: | ---: |
| **Combined agent (`gpt-5.4-mini` when escalated)** | **0.905** | 0.790 | 4.98 | **0.8100** |
| AI agent full-time, `gpt-5.6-sol` | 0.840 | 0.657 | 4.44 | 0.7483 |
| AI agent full-time, `gpt-5.4-mini` | 0.850 | 0.550 | 4.33 | 0.7233 |
| Reflex layer alone | 0.710 | 0.672 | 6.45 | 0.6477 |

The escalation engaged in 98 of 200 hostile sessions and won 70% of the
sessions it took over. Public-set results are development measurements, not
estimates of the organizer's hidden-set result.

## Latency, tokens, and estimated cost

- Offline path: zero model tokens, USD 0, ~3 ms per turn after a one-time
  ~2 s index build; the full 200-session official benchmark evaluates in
  under a second of agent time.
- Combined agent under hostile load: 1.30M tokens over 200 sessions
  (~6.5k/session), 28 minutes wall clock. Escalated turns make two model calls
  (planning and decision); Tier-1 parse assists are single small calls.
- Reference full-time AI-agent runs: `gpt-5.6-sol` official benchmark
  3,164,704 tokens in 4 h 13 m; hostile benchmark 4,476,571 tokens in
  3 h 44 m. `gpt-5.4-mini` hostile benchmark 5,508,981 tokens in 42 m.
- Provider token prices vary by account; with input price `Pi` and output
  price `Po` (USD per million tokens), the combined hostile run cost is
  `1.277 × Pi + 0.022 × Po` — at typical mini-tier prices, well under one US
  dollar. The agent reports the provider's token counters through the official
  `usage` field.
- Request timeout defaults to 45 s (`TECHJAM_LLM_TIMEOUT`); escalated calls
  default to 12 s (`COPILOT_T2_TIMEOUT`).

## Reproduction

Python 3.10+ recommended; no third-party runtime dependencies. Download and
decompress the official catalog as described in the top-level README, then:

```bash
python3 -m evaluator.local_evaluator      # official evaluator over starter/
python3 -m ttfarm.tools.run_eval          # official evaluator over the combined agent
python3 -m ttfarm.tools.paraphrase_eval --level 2
python3 -m ttfarm.tools.damage_eval --rates 0.25 0.5
python3 -m unittest discover tests        # 23 tests
```

Escalation knobs (`COPILOT_ESCALATION`, `COPILOT_T2_TURN`, `COPILOT_T2_MODE`,
`COPILOT_T2_MAX_SESSIONS`, `COPILOT_TOKEN_BUDGET`) are environment variables
documented in `ttfarm/README.md`.

## Limitations

- Candidate recall is bounded by local lexical retrieval; the ranking model
  cannot select a product that was not retrieved.
- The compact session-state logic recognizes common explicit overrides and
  no-preference replies, but does not implement unrestricted semantic memory.
- Live performance depends on model quality, provider latency, JSON adherence,
  and network availability; the offline path bounds the downside.
- Escalation thresholds were calibrated on paired per-session records from a
  tune split of the public sessions; they are environment-tunable, and the
  worst measured configuration error costs single points of hit rate.
- The public labels are development data. Public-set performance is not an
  estimate of the organizer's hidden-set result; our honest hidden-set
  expectation for the official benchmark is 0.95 ± 0.03.
