"""Node functions for the fact-checking graph.

Each `build_*_node` factory closes over its dependencies (an LLM, a search
provider, settings) and returns a plain `state -> dict` callable, which is
what LangGraph expects. Keeping dependencies as closures rather than module
globals is what makes `tests/test_graph_smoke.py` possible without any real
network or model calls: swap in fakes at construction time.
"""
from __future__ import annotations

from factcheck.agent.state import GraphState
from factcheck.config import Settings
from factcheck.llm import LLMError, call_structured
from factcheck.schemas import (
    Evidence,
    PlanOutput,
    ReflectionOutput,
    Stance,
    SynthesisOutput,
)
from factcheck.tools.fetch import fetch_doc
from factcheck.tools.search import SearchError, SearchProvider

_PLAN_SYSTEM = (
    "You are the planning step of a fact-checking agent. Given a claim, "
    "produce 2-4 short, specific web search queries that together would "
    "surface primary sources or reputable reporting to verify or refute it. "
    "Avoid vague queries; include names, dates, and numbers from the claim "
    "where present."
)

_EXTRACT_SYSTEM = (
    "You are the evidence-extraction step of a fact-checking agent. Given a "
    "claim and the text of one web page, decide whether the page contains "
    "information relevant to the claim. If it does, quote the most relevant "
    "sentence or two verbatim and classify whether it supports, refutes, or "
    "is neutral toward the claim. If the page is irrelevant, say so."
)

_REFLECT_SYSTEM = (
    "You are the reflection step of a fact-checking agent. Given a claim and "
    "the evidence gathered so far, decide whether there is enough evidence "
    "to reach a confident verdict. Require at least one source that directly "
    "supports or refutes the claim, not just tangentially related material. "
    "If evidence conflicts, that is fine — note it, but more corroboration "
    "is still useful. If insufficient, propose up to 3 new search queries "
    "that target the specific gap (e.g. a missing date, a disputed number, "
    "an unconfirmed party) rather than repeating the original queries."
)

_SYNTHESIZE_SYSTEM = (
    "You are the synthesis step of a fact-checking agent. Given a claim and "
    "all evidence gathered, produce a final verdict. Use 'supported' only if "
    "credible sources clearly back the claim, 'refuted' only if they clearly "
    "contradict it, 'mixed' if sources disagree, and 'unverified' if the "
    "evidence is too thin either way. Cite specific sources by URL with the "
    "quote that backs each citation. Note any contradictions between "
    "sources explicitly — do not paper over disagreement.\n\n"
    "The 'verdict' field and your 'summary' text must agree with each other "
    "and with the stance of the evidence you cite. Before answering, check: "
    "if the evidence is labeled 'refutes' the claim, the verdict must be "
    "'refuted', not 'supported' — do not confuse 'the evidence is real and "
    "well-sourced' with 'the evidence supports the claim being true'. Those "
    "are independent: strong evidence can point either for or against the "
    "claim."
)


def build_plan_node(llm):
    def plan_node(state: GraphState) -> dict:
        try:
            result = call_structured(
                llm,
                _PLAN_SYSTEM,
                f"Claim: {state['claim']}",
                PlanOutput,
                example={"queries": ["first specific search query", "second specific search query"]},
            )
            queries = result.queries
        except LLMError as e:
            queries = [state["claim"]]
            return {
                "queries": queries,
                "trace": [f"plan: LLM failed ({e}), falling back to raw claim as query"],
            }
        return {"queries": queries, "trace": [f"plan: generated queries {queries}"]}

    return plan_node


def build_search_node(search_provider: SearchProvider, settings: Settings):
    def search_node(state: GraphState) -> dict:
        pending = [q for q in state["queries"] if q not in state["searched_queries"]]
        new_results = []
        trace = []
        for query in pending:
            try:
                hits = search_provider.search(query, settings.results_per_query)
                new_results.extend(hits)
                trace.append(f"search: {query!r} -> {len(hits)} results")
            except SearchError as e:
                trace.append(f"search: {query!r} failed ({e})")

        seen = {r.url for r in state["search_results"]}
        deduped = [r for r in new_results if r.url not in seen and not seen.add(r.url)]

        return {
            "search_results": deduped,
            "searched_queries": state["searched_queries"] + pending,
            "trace": trace,
        }

    return search_node


def build_fetch_node(settings: Settings):
    def fetch_node(state: GraphState) -> dict:
        already_fetched = set(state["fetched_urls"])
        slots_left = max(0, settings.max_docs_to_fetch - len(already_fetched))
        candidates = [
            r.url for r in state["search_results"]
            if r.url not in already_fetched
        ][:slots_left]

        fetched = []
        trace = []
        for url in candidates:
            doc = fetch_doc(url, settings.fetch_timeout_seconds, settings.max_doc_chars)
            fetched.append(doc)
            trace.append(f"fetch: {url} -> {'ok' if doc.ok else doc.error}")

        return {
            "fetched": fetched,
            "fetched_urls": state["fetched_urls"] + candidates,
            "trace": trace,
        }

    return fetch_node


def build_extract_node(llm):
    def extract_node(state: GraphState) -> dict:
        covered = {e.source_url for e in state["evidence"]}
        to_process = [d for d in state["fetched"] if d.ok and d.url not in covered]

        evidence: list[Evidence] = []
        trace = []
        for doc in to_process:
            prompt = (
                f"Claim: {state['claim']}\n\n"
                f"Page URL: {doc.url}\n"
                f"Page title: {doc.title}\n"
                f"Page text (truncated):\n{doc.text}"
            )
            try:
                extraction = call_structured(
                    llm,
                    _EXTRACT_SYSTEM,
                    prompt,
                    _ExtractionOutput,
                    example={
                        "quote": "an exact sentence copied from the page text above",
                        "stance": "supports",
                        "relevance": 0.8,
                    },
                )
            except LLMError as e:
                trace.append(f"extract: {doc.url} failed ({e})")
                continue

            if extraction.stance != Stance.IRRELEVANT:
                evidence.append(
                    Evidence(
                        source_url=doc.url,
                        quote=extraction.quote,
                        stance=extraction.stance,
                        relevance=extraction.relevance,
                    )
                )
            trace.append(f"extract: {doc.url} -> stance={extraction.stance.value}")

        return {"evidence": evidence, "trace": trace}

    return extract_node


def build_reflect_node(llm):
    def reflect_node(state: GraphState) -> dict:
        iterations = state["iterations"] + 1
        if iterations >= state["max_iterations"]:
            return {
                "iterations": iterations,
                "sufficient": True,
                "trace": [f"reflect: hit max_iterations={state['max_iterations']}, stopping"],
            }

        evidence_summary = "\n".join(
            f"- [{e.stance.value}] {e.source_url}: {e.quote}" for e in state["evidence"]
        ) or "(none yet)"
        prompt = f"Claim: {state['claim']}\n\nEvidence so far:\n{evidence_summary}"

        try:
            reflection = call_structured(
                llm,
                _REFLECT_SYSTEM,
                prompt,
                ReflectionOutput,
                example={
                    "sufficient": False,
                    "reasoning": "one sentence explaining what's missing or why it's enough",
                    "follow_up_queries": ["a more targeted search query"],
                },
            )
        except LLMError as e:
            return {
                "iterations": iterations,
                "sufficient": True,
                "trace": [f"reflect: LLM failed ({e}), stopping to avoid an infinite loop"],
            }

        return {
            "iterations": iterations,
            "sufficient": reflection.sufficient,
            "queries": reflection.follow_up_queries,
            "trace": [f"reflect: sufficient={reflection.sufficient} — {reflection.reasoning}"],
        }

    return reflect_node


def build_synthesize_node(llm):
    def synthesize_node(state: GraphState) -> dict:
        evidence_block = "\n".join(
            f"- [{e.stance.value}, relevance={e.relevance:.2f}] {e.source_url}: {e.quote}"
            for e in state["evidence"]
        ) or "(no evidence was found)"
        prompt = f"Claim: {state['claim']}\n\nEvidence gathered:\n{evidence_block}"

        try:
            synthesis = call_structured(
                llm,
                _SYNTHESIZE_SYSTEM,
                prompt,
                SynthesisOutput,
                example={
                    "verdict": "supported",
                    "confidence": 0.75,
                    "summary": "one or two sentences giving the final verdict and why",
                    "citations": [
                        {"source_url": "https://example.com/article", "quote": "the exact quote backing this"}
                    ],
                    "contradictions": [],
                },
            )
        except LLMError as e:
            from factcheck.schemas import Verdict

            synthesis = SynthesisOutput(
                verdict=Verdict.UNVERIFIED,
                confidence=0.0,
                summary=f"Synthesis failed: {e}",
            )

        return {"synthesis": synthesis, "trace": ["synthesize: verdict reached"]}

    return synthesize_node


class _ExtractionOutput(Evidence):
    """Same shape as `Evidence` but without `source_url` — the LLM shouldn't
    invent or repeat the URL, the graph already knows it."""

    source_url: str = ""
