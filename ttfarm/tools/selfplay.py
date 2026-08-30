"""Self-play generalization check: does the ranker find ARBITRARY catalog
products, not just the 200 public targets?

Samples products (uniform + popularity-weighted, mirroring the 5-core
purchase-skew of real targets), builds each product's hidden intent card with
the official public generator (read-only import from --kit), simulates the
harvest states our agent reaches at turn 1 / turn 3, and measures where the
product ranks. Private targets are catalog products too, so this bounds
expected private behavior for the retrieval+ranking stack.

    python3 tools/selfplay.py --kit ../kit --n 2000
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kit", default=str(ROOT))
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()
    kit = Path(args.kit).resolve()
    sys.path.insert(0, str(kit))
    from evaluator.local_evaluator import catalog_index, intent_card, coarse_category

    from ttfarm.catalog import Catalog
    from ttfarm.search import rank
    from ttfarm.state import Session, Slot

    catalog = Catalog(kit / "data/catalog.jsonl")
    _, categories, products = catalog_index(kit / "data/catalog.jsonl")
    rng = random.Random(args.seed)
    asins = list(products)
    weights = [max(products[a].get("rating_number") or 0, 1) for a in asins]

    def measure(sample: list[str], label: str) -> None:
        stats = {("t1", k): 0 for k in (1, 3, 10)} | {("t3", k): 0 for k in (1, 3, 10)}
        for target in sample:
            card = intent_card(products[target])
            constraints = card["hard_constraints"] + card["soft_preferences"]
            cc = coarse_category(categories.get(target, []))
            for stage, n_slots in (("t1", 1), ("t3", 4)):
                session = Session(category=cc)
                session.slots = [Slot(c) for c in constraints[:n_slots]]
                top = rank(catalog, session, limit=10)
                if target in top:
                    r = top.index(target) + 1
                    for k in (1, 3, 10):
                        if r <= k:
                            stats[(stage, k)] += 1
        n = len(sample)
        print(f"{label} (n={n}):")
        for stage, desc in (("t1", "turn-1 info (1 constraint)"), ("t3", "full harvest (4)")):
            print(f"  {desc:<28} top1={stats[(stage,1)]/n:6.1%}  top3={stats[(stage,3)]/n:6.1%}  top10={stats[(stage,10)]/n:6.1%}")

    measure(rng.sample(asins, args.n), "uniform product sample")
    measure(rng.choices(asins, weights=weights, k=args.n), "popularity-weighted sample (mirrors 5-core purchase skew)")


if __name__ == "__main__":
    main()
