"""Wires the node functions into a LangGraph state graph.

    plan -> search -> fetch -> extract -> reflect --(insufficient)--> search
                                              |
                                              --(sufficient)--> synthesize -> END

The loop back from `reflect` to `search` is the actual "agentic" part: the
graph decides for itself, based on what it has read so far, whether it needs
another round of search before it's willing to commit to a verdict — capped
by `max_search_iterations` so a stubborn claim can't loop forever.
"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from factcheck.agent.nodes import (
    build_extract_node,
    build_fetch_node,
    build_plan_node,
    build_reflect_node,
    build_search_node,
    build_synthesize_node,
)
from factcheck.agent.state import GraphState
from factcheck.config import Settings
from factcheck.schemas import FactCheckResult
from factcheck.tools.search import SearchProvider


def _route_after_reflect(state: GraphState) -> str:
    return "synthesize" if state["sufficient"] else "search"


def build_graph(llm, search_provider: SearchProvider, settings: Settings):
    graph = StateGraph(GraphState)

    graph.add_node("plan", build_plan_node(llm))
    graph.add_node("search", build_search_node(search_provider, settings))
    graph.add_node("fetch", build_fetch_node(settings))
    graph.add_node("extract", build_extract_node(llm))
    graph.add_node("reflect", build_reflect_node(llm))
    graph.add_node("synthesize", build_synthesize_node(llm))

    graph.add_edge(START, "plan")
    graph.add_edge("plan", "search")
    graph.add_edge("search", "fetch")
    graph.add_edge("fetch", "extract")
    graph.add_edge("extract", "reflect")
    graph.add_conditional_edges(
        "reflect", _route_after_reflect, {"search": "search", "synthesize": "synthesize"}
    )
    graph.add_edge("synthesize", END)

    return graph.compile()


def initial_state(claim: str, settings: Settings) -> GraphState:
    return GraphState(
        claim=claim,
        queries=[],
        searched_queries=[],
        search_results=[],
        fetched_urls=[],
        fetched=[],
        evidence=[],
        iterations=0,
        max_iterations=settings.max_search_iterations,
        sufficient=False,
        synthesis=None,
        trace=[],
    )


def run_fact_check(claim: str, llm, search_provider: SearchProvider, settings: Settings) -> FactCheckResult:
    graph = build_graph(llm, search_provider, settings)
    final_state = graph.invoke(initial_state(claim, settings))

    synthesis = final_state["synthesis"]
    return FactCheckResult(
        claim=claim,
        verdict=synthesis.verdict,
        confidence=synthesis.confidence,
        summary=synthesis.summary,
        citations=synthesis.citations,
        contradictions=synthesis.contradictions,
        sources_consulted=len({e.source_url for e in final_state["evidence"]}),
        iterations=final_state["iterations"],
        trace=final_state["trace"],
    )
