"""Command-line entrypoint.

    python -m factcheck.cli "The Eiffel Tower is taller than the Statue of Liberty"
    python -m factcheck.cli --json "..."
"""
from __future__ import annotations

import argparse
import json
import sys

from factcheck.agent.graph import run_fact_check
from factcheck.config import load_settings
from factcheck.llm import get_chat_model
from factcheck.tools.search import get_search_provider


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify a claim against live web sources.")
    parser.add_argument("claim", help="The claim or question to fact-check")
    parser.add_argument("--json", action="store_true", help="Print raw JSON instead of a formatted report")
    parser.add_argument("--trace", action="store_true", help="Include the step-by-step agent trace")
    args = parser.parse_args(argv)

    settings = load_settings()
    llm = get_chat_model(settings)
    search_provider = get_search_provider(settings)

    result = run_fact_check(args.claim, llm, search_provider, settings)

    if args.json:
        payload = result.model_dump(mode="json")
        if not args.trace:
            payload.pop("trace", None)
        print(json.dumps(payload, indent=2))
        return 0

    print(f"\nClaim: {result.claim}")
    print(f"Verdict: {result.verdict.value.upper()} (confidence {result.confidence:.0%})")
    print(f"\n{result.summary}\n")

    if result.citations:
        print("Sources:")
        for c in result.citations:
            print(f"  - {c.source_url}")
            print(f"    \"{c.quote}\"")

    if result.contradictions:
        print("\nContradictions found:")
        for c in result.contradictions:
            print(f"  - {c}")

    print(f"\n({result.sources_consulted} sources consulted, {result.iterations} search iteration(s))")

    if args.trace:
        print("\nTrace:")
        for line in result.trace:
            print(f"  {line}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
