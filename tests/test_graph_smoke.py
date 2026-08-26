"""End-to-end run of the graph with every external dependency faked out.

This is the test that proves the wiring in `agent/graph.py` is actually
correct — node order, the reducer-based state merging, and critically the
reflect -> search loop-back edge, which only a real graph execution (not a
unit test of a single node) can exercise.
"""
from __future__ import annotations

import json

from factcheck.agent.graph import run_fact_check
from factcheck.schemas import FetchedDoc, SearchResult
from tests.conftest import FakeLLM, FakeSearchProvider

_PLAN_JSON = json.dumps({"queries": ["round one query"]})

_EXTRACT_SUPPORTS = json.dumps(
    {"quote": "Official records confirm this is true.", "stance": "supports", "relevance": 0.9}
)
_EXTRACT_REFUTES = json.dumps(
    {"quote": "A follow-up report disputes the original claim.", "stance": "refutes", "relevance": 0.8}
)

_REFLECT_INSUFFICIENT = json.dumps(
    {"sufficient": False, "reasoning": "only one source so far", "follow_up_queries": ["round two query"]}
)
_REFLECT_SUFFICIENT = json.dumps(
    {"sufficient": True, "reasoning": "corroborating and conflicting evidence both found", "follow_up_queries": []}
)

_SYNTHESIZE_JSON = json.dumps(
    {
        "verdict": "mixed",
        "confidence": 0.6,
        "summary": "One source supports the claim, another disputes it.",
        "citations": [
            {"source_url": "https://source-one.example", "quote": "Official records confirm this is true."},
            {"source_url": "https://source-two.example", "quote": "A follow-up report disputes the original claim."},
        ],
        "contradictions": ["source-one and source-two disagree on the outcome"],
    }
)


def _make_responder():
    reflect_calls = {"n": 0}

    def responder(messages) -> str:
        system = messages[0][1]
        if "planning step" in system:
            return _PLAN_JSON
        if "evidence-extraction step" in system:
            user = messages[1][1]
            if "source-one" in user:
                return _EXTRACT_SUPPORTS
            return _EXTRACT_REFUTES
        if "reflection step" in system:
            reflect_calls["n"] += 1
            return _REFLECT_INSUFFICIENT if reflect_calls["n"] == 1 else _REFLECT_SUFFICIENT
        if "synthesis step" in system:
            return _SYNTHESIZE_JSON
        raise AssertionError(f"unexpected system prompt: {system!r}")

    return responder


def test_full_graph_loops_then_synthesizes(settings):
    llm = FakeLLM(_make_responder())
    search = FakeSearchProvider(
        {
            "round one query": [SearchResult(title="One", url="https://source-one.example", snippet="s1")],
            "round two query": [SearchResult(title="Two", url="https://source-two.example", snippet="s2")],
        }
    )

    def fake_fetch_doc(url, timeout_seconds, max_chars):
        return FetchedDoc(url=url, title="Fake", text=f"body text for {url}")

    import factcheck.agent.nodes as nodes_mod

    original_fetch_doc = nodes_mod.fetch_doc
    nodes_mod.fetch_doc = fake_fetch_doc
    try:
        result = run_fact_check("Some testable claim", llm, search, settings)
    finally:
        nodes_mod.fetch_doc = original_fetch_doc

    assert result.verdict.value == "mixed"
    assert result.confidence == 0.6
    assert result.sources_consulted == 2
    assert result.iterations == 2
    assert len(result.citations) == 2
    assert result.contradictions
    assert search.queries_seen == ["round one query", "round two query"]
    assert any("plan:" in line for line in result.trace)
    assert any("reflect:" in line for line in result.trace)
