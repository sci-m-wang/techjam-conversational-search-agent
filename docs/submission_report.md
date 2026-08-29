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

The live path uses an OpenAI-compatible Chat Completions API. The exact provider
and model are deployment configuration, not source-code constants. They are
read only from `TECHJAM_LLM_API_KEY`, `TECHJAM_LLM_MODEL`, and
`TECHJAM_LLM_BASE_URL` (with `OPENAI_*` compatibility aliases). Secrets are not
included in this repository or its Git history.

The final model name, measured token usage, latency, and provider-priced cost
will be recorded here after the submitter configures the local environment and
runs the public evaluator. Until then, the live-model result is intentionally
reported as **not measured**, rather than being inferred from a manual study or
from the offline fallback.

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
| Configured live model | 200 | Pending | Pending | Pending | Pending |

The weak starter reference in the official kit reports Hit Rate@10 0.125, MRR
0.068034, and MTTC 9.81.

## Latency, tokens, and estimated cost

- Offline fallback: two local retrieval stages at most, zero model tokens, and
  USD 0 API cost.
- Live mode: normally two model calls per turn (planning and decision). The
  agent reports the provider's prompt and completion token counters through the
  official `usage` field.
- Live cost is calculated from the measured totals as
  `(prompt_tokens × provider_input_price + completion_tokens × provider_output_price) / 1,000,000`.
  A numeric estimate is pending the configured model and full public run.
- Request timeout defaults to 45 seconds and is configurable with
  `TECHJAM_LLM_TIMEOUT`.

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
