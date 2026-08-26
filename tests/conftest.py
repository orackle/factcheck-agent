from __future__ import annotations

import pytest

from factcheck.config import Settings
from factcheck.schemas import SearchResult
from factcheck.tools.search import SearchProvider


class FakeMessage:
    def __init__(self, content: str):
        self.content = content


class FakeLLM:
    """Stands in for a LangChain chat model.

    `responder(messages) -> str` receives exactly what `call_structured`
    would pass a real model and returns the JSON string to hand back. This
    lets tests script different JSON per graph step by inspecting the system
    prompt, without touching any network or real provider.
    """

    def __init__(self, responder):
        self._responder = responder
        self.calls: list[list[tuple[str, str]]] = []

    def invoke(self, messages):
        self.calls.append(messages)
        return FakeMessage(self._responder(messages))


class FakeSearchProvider(SearchProvider):
    def __init__(self, results_by_query: dict[str, list[SearchResult]] | None = None):
        self._results_by_query = results_by_query or {}
        self.queries_seen: list[str] = []

    def search(self, query: str, max_results: int) -> list[SearchResult]:
        self.queries_seen.append(query)
        return self._results_by_query.get(query, [])[:max_results]


@pytest.fixture
def settings() -> Settings:
    return Settings(
        llm_provider="ollama",
        llm_model="test-model",
        ollama_base_url="http://localhost:11434",
        openai_api_key="",
        anthropic_api_key="",
        search_provider="duckduckgo",
        tavily_api_key="",
        max_search_iterations=2,
        results_per_query=4,
        max_docs_to_fetch=5,
        fetch_timeout_seconds=5.0,
        max_doc_chars=6000,
        request_timeout_seconds=30.0,
    )
