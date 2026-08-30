# Autonomous Conversational Search Agent — TechJam 2026

This repository is a participant solution built on the official
[`TechJam2026/techjam-conversational-search`](https://github.com/TechJam2026/techjam-conversational-search)
kit. The solution adds an autonomous, stateful LLM planning and ranking loop
while retaining a network-free retrieval fallback.

Build an AI shopping agent that asks useful follow-up questions and recommends the customer's hidden target product within at most 10 turns.

## What You Receive

- A frozen catalog of 50,000 products from the `Clothing_Shoes_and_Jewelry` category of Amazon Reviews 2023.
- 200 labeled public sessions for local development.
- A weak BM25 starter agent and deterministic local evaluator.
- The Agent API contract and scoring rules.

The organizer keeps 800 additional sessions private for final evaluation.

## Task

For each session, your agent receives an anonymized preference profile and a short customer message. Raw user IDs, review text, timestamps, and purchase history are never disclosed. On every turn the agent may:

- ask a natural clarification question in `message` and identify one requested field in `ask_attribute`;
- return a ranked list of up to 10 catalog `parent_asin` values;
- do both in the same response.

The session ends when the target product appears in the scored Top 10 or after turn 10. Sessions cover Buying, Browsing, Intent Override, and Boundary behavior.

## Download the Catalog

Download `catalog.jsonl.gz` from the
[official Participant Kit release](https://github.com/TechJam2026/techjam-conversational-search/releases/tag/participant-kit),
then run:

```bash
gzip -dk catalog.jsonl.gz
mv catalog.jsonl data/catalog.jsonl
```

Verify the downloaded file using the published `SHA256SUMS` file.

## Run the Starter

Python 3.10 or later is recommended. The starter uses only the Python standard library.

```bash
python3 -m evaluator.local_evaluator
```

Edit `starter/agent.py` to implement your system. Do not edit the evaluator or public labels when reporting your local score.
The command writes per-session results and aggregate metrics to `results.json`.

## Autonomous LLM Agent

The editable Agent now supports an autonomous two-stage model loop on every
turn:

1. the model reads the safe conversation state and plans one to three local
   catalog searches;
2. SQLite FTS5 retrieves a bounded candidate pool from the frozen catalog;
3. the model ranks only those candidates and returns both Top-10
   recommendations and the next clarification question.

Credentials must be injected through environment variables or the secret
manager provided by the execution platform. Never put an API key in source
code, `.env.example`, README text, logs, or Git history.

```bash
export TECHJAM_LLM_API_KEY='<provided-by-secret-manager>'
export TECHJAM_LLM_MODEL='your-model-name'
export TECHJAM_LLM_BASE_URL='https://api.openai.com/v1'
python3 -m evaluator.local_evaluator
```

For local development, you may copy `.env.example` to the ignored `.env`
file, fill it on your own machine, and load it into the current shell before
running the evaluator:

```bash
set -a
source .env
set +a
python3 -m evaluator.local_evaluator
```

`OPENAI_API_KEY`, `OPENAI_MODEL`, and `OPENAI_BASE_URL` are accepted as
compatibility aliases. `TECHJAM_LLM_TIMEOUT` controls the per-request timeout
and defaults to 45 seconds. The client uses an OpenAI-compatible Chat
Completions endpoint and reports the prompt/completion token counts returned by
the provider. It automatically negotiates the `max_tokens` versus
`max_completion_tokens` difference used by newer reasoning models.

Some Python.org macOS installations do not have a default CA certificate file.
If a request fails with `CERTIFICATE_VERIFY_FAILED`, add the following local-only
line to `.env`; do not disable TLS verification:

```dotenv
SSL_CERT_FILE=/etc/ssl/cert.pem
```

If the model configuration is absent, unreachable, or returns unusable JSON,
the Agent automatically falls back to stateful offline catalog retrieval and
continues to satisfy the official interface. This matters because final
scoring may disable network access.

The no-key fallback was verified on all 200 public sessions with Hit Rate@10
`0.85`, MRR `0.464188`, MTTC `4.605`, and TechnicalScore `0.692156`. This is a
fallback reference result, not the score of the live LLM path.

See [`docs/submission_report.md`](docs/submission_report.md) for the method,
evaluation status, network behavior, model-cost disclosure, and limitations.

The competition submission is the source bundle exporting the Python `Agent`
class, not a hosted HTTP endpoint. A web API may be added for a demo, but the
official evaluator must still be able to import and call `reset(...)` and
`respond(...)` locally. Copy `.env.example` only for local configuration; the
real `.env` file is ignored by Git and must not be submitted.

The included weak BM25 starter scores Hit Rate@10 `0.125`, MRR `0.068034`, and
MTTC `9.81` on the released public set. See `docs/baseline_results.json`.

## Agent Interface

```python
class Agent:
    def reset(self, session_id: str, user_profile: dict) -> None:
        ...

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        return {
            "message": "Do you have a material preference?",
            "ask_attribute": "material",
            "recommendations": [
                {"parent_asin": "B000..."},
                {"parent_asin": "B001..."}
            ],
            "usage": {"prompt_tokens": 120, "completion_tokens": 30}
        }
```

`ask_attribute` is one of `category`, `material`, `color`, `size`, `style`, `brand`, `budget`, `feature`, `use_case`, `other`, or `null`. See `docs/agent_api_contract.json`.

## Technical Metrics

- **Hit Rate@10:** fraction of sessions that find the target within 10 turns.
- **MRR:** mean reciprocal rank of the target; a miss contributes zero.
- **MTTC:** mean first-hit turn; a miss is assigned turn 11.
- **Reported token usage:** prompt and completion tokens returned by the team's model client.

```text
TechnicalScore = 0.50 × HitRate@10 + 0.30 × MRR + 0.20 × Efficiency
Efficiency = clip((11 - MTTC) / 10, 0, 1)
```

`TechnicalScore` is an objective input to the `Technical Execution` assessment. It is not a separate judging criterion and does not represent the entire `Technical Execution` score.

Only exact `parent_asin` equality produces a hit. Core metrics are also reported by scenario.

## Model Choice and Cost

Teams may use any legally accessible LLM API or local model. Teams manage their own credentials and must never commit API keys. Model choice, estimated cost, token usage, and latency must be disclosed. Token usage is a feasibility metric, not part of the core technical score. The organizer does not provide or reimburse model API credits; teams are responsible for any costs incurred through optional external services.

## Files

```text
data/public_set.jsonl             200 labeled development sessions
docs/competition_specification.md participant rules and evaluation protocol
docs/agent_api_contract.json      machine-readable Agent contract
docs/evaluation_config.json       scoring configuration
docs/baseline_results.json        reproducible weak-starter reference score
starter/agent.py                  editable weak starter
evaluator/local_evaluator.py      public-set simulator and scorer
```

## Judging and Submission Policy

- Participant submission requirements: `docs/submission_rules.md`
- Organizer-only final judging controls: `organizer/JUDGING_RUNBOOK.md`
- Organizer private release checklist: `organizer/private_release_checklist.md`
- Judging day operations SOP: `organizer/JUDGING_DAY_SOP.md`

## Data Source

The catalog and sessions are derived from Amazon Reviews 2023 by McAuley Lab, UCSD. See `DATA_ATTRIBUTION.md` before using or redistributing the data.
Sessions are sampled deterministically from the official Clothing 5-core leave-last-out split and joined to the frozen catalog.
