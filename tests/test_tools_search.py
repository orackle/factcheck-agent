from __future__ import annotations

import pytest

from factcheck.tools.search import (
    DuckDuckGoSearchProvider,
    SearchError,
    TavilySearchProvider,
    get_search_provider,
)


def test_duckduckgo_provider_maps_results(monkeypatch):
    class FakeDDGS:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def text(self, query, max_results):
            return [
                {"title": "A", "href": "https://a.example", "body": "snippet a"},
                {"title": "B", "href": "https://b.example", "body": "snippet b"},
            ]

    # ddgs is imported lazily inside the method, so patch the module import target
    import sys
    import types

    fake_module = types.ModuleType("ddgs")
    fake_module.DDGS = lambda: FakeDDGS()
    sys.modules["ddgs"] = fake_module

    provider = DuckDuckGoSearchProvider()
    results = provider.search("test query", 5)

    assert len(results) == 2
    assert results[0].url == "https://a.example"
    assert results[0].title == "A"
    assert results[0].snippet == "snippet a"


def test_duckduckgo_provider_wraps_exceptions(monkeypatch):
    import sys
    import types

    class ExplodingDDGS:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def text(self, query, max_results):
            raise RuntimeError("rate limited")

    fake_module = types.ModuleType("ddgs")
    fake_module.DDGS = lambda: ExplodingDDGS()
    sys.modules["ddgs"] = fake_module

    provider = DuckDuckGoSearchProvider()
    with pytest.raises(SearchError):
        provider.search("test query", 5)


def test_tavily_provider_requires_api_key():
    with pytest.raises(SearchError):
        TavilySearchProvider(api_key="")


def test_get_search_provider_unknown_raises(settings):
    bad_settings = settings.__class__(**{**settings.__dict__, "search_provider": "bing"})
    with pytest.raises(SearchError):
        get_search_provider(bad_settings)


def test_get_search_provider_returns_duckduckgo_by_default(settings):
    provider = get_search_provider(settings)
    assert isinstance(provider, DuckDuckGoSearchProvider)
