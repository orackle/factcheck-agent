"""Web search abstraction.

Two backends ship out of the box:

- DuckDuckGo (default): no API key, good enough for demoing and for the
  eval suite. Rate-limited and occasionally flaky, which is exactly what a
  production system would need to tolerate — see `SearchError` handling in
  the agent's search node.
- Tavily: a search API built for LLM agents (cleaner snippets, an
  LLM-oriented ranking). Swap in via `SEARCH_PROVIDER=tavily` +
  `TAVILY_API_KEY` once you have a key; no other code changes needed since
  both backends speak the same `SearchProvider` interface.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from factcheck.config import Settings
from factcheck.schemas import SearchResult


class SearchError(RuntimeError):
    pass


class SearchProvider(ABC):
    @abstractmethod
    def search(self, query: str, max_results: int) -> list[SearchResult]:
        raise NotImplementedError


class DuckDuckGoSearchProvider(SearchProvider):
    def search(self, query: str, max_results: int) -> list[SearchResult]:
        try:
            from ddgs import DDGS
        except ImportError as e:
            raise SearchError(
                "ddgs is not installed. Run: pip install ddgs"
            ) from e

        try:
            with DDGS() as ddgs:
                raw_results = list(ddgs.text(query, max_results=max_results))
        except Exception as e:  # noqa: BLE001 - third-party client raises various error types
            raise SearchError(f"DuckDuckGo search failed for {query!r}: {e}") from e

        return [
            SearchResult(
                title=r.get("title", ""),
                url=r.get("href") or r.get("link") or "",
                snippet=r.get("body", ""),
            )
            for r in raw_results
            if r.get("href") or r.get("link")
        ]


class TavilySearchProvider(SearchProvider):
    def __init__(self, api_key: str):
        if not api_key:
            raise SearchError("TAVILY_API_KEY is not set but SEARCH_PROVIDER=tavily")
        self._api_key = api_key

    def search(self, query: str, max_results: int) -> list[SearchResult]:
        try:
            from tavily import TavilyClient
        except ImportError as e:
            raise SearchError("tavily-python is not installed. Run: pip install tavily-python") from e

        try:
            client = TavilyClient(api_key=self._api_key)
            response = client.search(query, max_results=max_results)
        except Exception as e:  # noqa: BLE001
            raise SearchError(f"Tavily search failed for {query!r}: {e}") from e

        return [
            SearchResult(
                title=r.get("title", ""),
                url=r.get("url", ""),
                snippet=r.get("content", ""),
            )
            for r in response.get("results", [])
            if r.get("url")
        ]


def get_search_provider(settings: Settings) -> SearchProvider:
    provider = settings.search_provider.lower()
    if provider == "duckduckgo":
        return DuckDuckGoSearchProvider()
    if provider == "tavily":
        return TavilySearchProvider(settings.tavily_api_key)
    raise SearchError(f"Unknown SEARCH_PROVIDER: {settings.search_provider!r}")
