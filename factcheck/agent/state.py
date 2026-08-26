"""Graph state.

A single TypedDict threaded through every node. LangGraph merges each node's
returned dict into this state, so list-valued fields below are always
returned as the *full new list*, not a delta — see the reducer note on
`search_results`/`fetched`/`evidence`, which use `operator.add` so nodes can
append incrementally across loop iterations instead of overwriting.
"""
from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from factcheck.schemas import Evidence, FetchedDoc, SearchResult, SynthesisOutput


class GraphState(TypedDict):
    claim: str

    queries: list[str]
    searched_queries: list[str]
    search_results: Annotated[list[SearchResult], operator.add]

    fetched_urls: list[str]
    fetched: Annotated[list[FetchedDoc], operator.add]

    evidence: Annotated[list[Evidence], operator.add]

    iterations: int
    max_iterations: int
    sufficient: bool

    synthesis: SynthesisOutput | None

    trace: Annotated[list[str], operator.add]
