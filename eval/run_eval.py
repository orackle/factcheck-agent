"""Runs the agent over `eval/dataset.jsonl` and reports verdict accuracy.

This is intentionally not part of the pytest suite — it makes real LLM and
search calls, costs money/time depending on provider, and its "correct
answer" is a label a human assigned, not a deterministic assertion. Run it
by hand after changing prompts or swapping models/providers to check you
haven't regressed accuracy:

    python -m eval.run_eval
    python -m eval.run_eval --limit 3          # quick smoke check
    python -m eval.run_eval --dataset path.jsonl
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from factcheck.agent.graph import run_fact_check
from factcheck.config import load_settings
from factcheck.llm import get_chat_model
from factcheck.tools.search import get_search_provider

_DEFAULT_DATASET = Path(__file__).parent / "dataset.jsonl"

# "mixed" and "unverified" are treated as adjacent to each other and to
# either side, since real-world claims that are genuinely disputed can
# reasonably land on either label — only a flat contradiction between
# "supported" and "refuted" counts as a clear miss.
_INCOMPATIBLE = {
    ("supported", "refuted"),
    ("refuted", "supported"),
}


def _load_dataset(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=_DEFAULT_DATASET)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    settings = load_settings()
    llm = get_chat_model(settings)
    search_provider = get_search_provider(settings)

    cases = _load_dataset(args.dataset)
    if args.limit:
        cases = cases[: args.limit]

    correct = 0
    for i, case in enumerate(cases, start=1):
        claim = case["claim"]
        expected = case["expected_verdict"]

        start = time.time()
        try:
            result = run_fact_check(claim, llm, search_provider, settings)
            actual = result.verdict.value
            elapsed = time.time() - start
            is_miss = (expected, actual) in _INCOMPATIBLE
            correct += 0 if is_miss else 1
            status = "MISS" if is_miss else "ok"
            print(
                f"[{i}/{len(cases)}] {status:4s}  expected={expected:11s} "
                f"actual={actual:11s} ({elapsed:5.1f}s)  {claim}"
            )
        except Exception as e:  # noqa: BLE001 - eval script should keep going on one bad case
            print(f"[{i}/{len(cases)}] ERROR  {claim}  -> {e}")

    print(f"\n{correct}/{len(cases)} within expected range")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
