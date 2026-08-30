# Submission Report

## System

This submission implements the official Python `Agent` interface in
`starter/agent.py`. It is stateful across turns and uses a two-stage autonomous
decision loop:

1. a planner converts the anonymized profile and active conversation state into
   one to three catalog queries and selects the next clarification attribute;
2. local SQLite FTS5 retrieval produces a bounded candidate set from the frozen
   50,000-product catalog;
3. a decision call ranks only retrieved catalog identifiers and returns both a
   Top-10 list and a natural-language clarification question.

The agent tracks earlier questions, declined preferences, and explicit intent
overrides. Model output is parsed and validated before it reaches the evaluator:
unknown ASINs, duplicate ASINs, invalid attributes, and malformed JSON are
discarded or replaced by deterministic fallback behavior.

## Model and configuration

The measured live run used `gpt-5.6-sol` through an Azure-hosted,
OpenAI-compatible Chat Completions endpoint. The source remains provider
agnostic: model, endpoint, and credential values are read only from
`TECHJAM_LLM_API_KEY`, `TECHJAM_LLM_MODEL`, and `TECHJAM_LLM_BASE_URL` (with
`OPENAI_*` compatibility aliases). Secrets are not included in this repository
or its Git history.

## Network behavior and offline fallback

Live LLM planning requires outbound access to the configured API. If the model
variables are missing, the request fails, or the response is unusable, the same
`Agent` remains runnable and uses stateful local FTS5 retrieval. The fallback
requires no network, third-party Python package, model credential, or model
token budget.

## Public-set evaluation

The no-key fallback was evaluated on all 200 released sessions with the official
local evaluator:

| Mode | Sessions | Hit Rate@10 | MRR | MTTC | TechnicalScore |
| --- | ---: | ---: | ---: | ---: | ---: |
| Offline fallback | 200 | 0.850000 | 0.464188 | 4.605 | 0.692156 |
| `gpt-5.6-sol` live path | 200 | 0.935000 | 0.672671 | 3.765 | 0.814001 |

The weak starter reference in the official kit reports Hit Rate@10 0.125, MRR
0.068034, and MTTC 9.81.

The same aggregate result is available in machine-readable form at
`docs/live_results_summary.json`. Per-session evaluator records remain local
and are not part of the submission.

Live-model results by scenario:

| Scenario | Sessions | Hit Rate@10 | MRR | MTTC |
| --- | ---: | ---: | ---: | ---: |
| Buying | 80 | 0.950000 | 0.668646 | 3.312500 |
| Browsing | 80 | 0.987500 | 0.712703 | 3.000000 |
| Intent override | 30 | 0.766667 | 0.560873 | 6.566667 |
| Boundary | 10 | 0.900000 | 0.720000 | 5.100000 |

## Latency, tokens, and estimated cost

- Offline fallback: two local retrieval stages at most, zero model tokens, and
  USD 0 API cost.
- Live mode: normally two model calls per turn (planning and decision). The
  agent reports the provider's prompt and completion token counters through the
  official `usage` field.
- The 200-session live run recorded 2,788,427 prompt tokens and 376,277
  completion tokens, or 3,164,704 total (15,823.52 per session on average).
- The official serial evaluator completed in 15,174.44 seconds (4 hours,
  12 minutes, 54 seconds), averaging 75.87 seconds per session. There were 740
  actual conversation turns, averaging 3.70 per session.
- The Azure deployment's negotiated token prices were not available to the
  evaluator, so a fabricated USD total is not reported. Given input price `Pi`
  and output price `Po` in USD per million tokens, the measured run cost is
  `2.788427 × Pi + 0.376277 × Po`.
- Request timeout defaults to 45 seconds and is configurable with
  `TECHJAM_LLM_TIMEOUT`; the measured long-running evaluation used 600 seconds.

## Reproduction

Python 3.10 or later is recommended. There are no third-party runtime
dependencies. Download and decompress the official catalog as described in the
top-level README, optionally configure the model environment variables, then
run:

```bash
python3 -m evaluator.local_evaluator
```

## Limitations

- Candidate recall is bounded by local lexical retrieval; the LLM cannot select
  a product that was not retrieved.
- The compact session-state logic recognizes common explicit overrides and
  no-preference replies, but does not implement unrestricted semantic memory.
- Live performance depends on model quality, provider latency, JSON adherence,
  and network availability.
- The public labels are development data. Public-set performance is not an
  estimate of the organizer's hidden-set result.
