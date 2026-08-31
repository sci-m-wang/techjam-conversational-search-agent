"""Score an agent with the UNMODIFIED official evaluator in this repository.

    python3 -m ttfarm.tools.run_eval                      # TT Farm agent, full public set
    python3 -m ttfarm.tools.run_eval --split holdout      # 50-session holdout
    python3 -m ttfarm.tools.run_eval --agent starter      # the LLM agent, for comparison

Splits are deterministic by sample index (idx % 4 == 3 -> holdout). Tune on
`tune`, verify on `holdout`, quote `all` once before submitting.

The evaluator and the public labels are never modified; this script only
imports them.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def load_agent(name: str, catalog_path: Path):
    if name == "ttfarm":
        from ttfarm.agent import Agent
    elif name == "starter":
        from starter.agent import Agent
    else:  # pragma: no cover - argparse restricts the choices
        raise SystemExit(f"unknown agent: {name}")
    return Agent(catalog_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kit", default=str(ROOT),
                        help="repository root holding data/ and evaluator/ (default: this repo)")
    parser.add_argument("--agent", default="ttfarm", choices=["ttfarm", "starter"])
    parser.add_argument("--split", default="all", choices=["tune", "holdout", "all"])
    parser.add_argument("--output", default=None, help="optional path for the full JSON result")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    kit = Path(args.kit).resolve()
    sys.path.insert(0, str(kit))
    from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl

    catalog_path = kit / "data/catalog.jsonl"
    if not catalog_path.exists():
        raise SystemExit(
            f"catalog not found at {catalog_path}\n"
            "Download it from the official participant-kit release and place it there "
            "(see the repository README).")

    samples = load_jsonl(kit / "data/public_set.jsonl")
    if args.split == "tune":
        samples = [s for i, s in enumerate(samples) if i % 4 != 3]
    elif args.split == "holdout":
        samples = [s for i, s in enumerate(samples) if i % 4 == 3]

    catalog_ids, categories, products = catalog_index(catalog_path)
    t0 = time.time()
    agent = load_agent(args.agent, catalog_path)
    build = time.time() - t0

    t0 = time.time()
    result = evaluate(agent, samples, catalog_ids, categories, products)
    elapsed = time.time() - t0

    summary = {k: v for k, v in result.items() if k != "sessions"}
    summary["agent"] = args.agent
    summary["split"] = args.split
    summary["timing"] = {
        "index_build_s": round(build, 2),
        "eval_s": round(elapsed, 2),
        "s_per_session": round(elapsed / max(1, len(samples)), 4),
    }
    print(json.dumps(summary, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if not args.quiet:
        print("misses:", [s["sample_id"] for s in result["sessions"] if not s["hit"]])


if __name__ == "__main__":
    main()
