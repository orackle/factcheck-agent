"""Environment-driven configuration.

Nothing in this module reaches out to the network — it only reads env vars
so the rest of the codebase (and the test suite) can construct providers
without a live LLM or search backend present.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass  # python-dotenv is optional; env vars can be set any other way


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


@dataclass(frozen=True)
class Settings:
    llm_provider: str  # "ollama" | "openai" | "anthropic"
    llm_model: str
    ollama_base_url: str
    openai_api_key: str
    anthropic_api_key: str

    search_provider: str  # "duckduckgo" | "tavily"
    tavily_api_key: str

    max_search_iterations: int
    results_per_query: int
    max_docs_to_fetch: int
    fetch_timeout_seconds: float
    max_doc_chars: int
    request_timeout_seconds: float
    llm_max_output_tokens: int


def load_settings() -> Settings:
    return Settings(
        llm_provider=_env("LLM_PROVIDER", "ollama") or "ollama",
        llm_model=_env("LLM_MODEL", "llama3.1"),
        ollama_base_url=_env("OLLAMA_BASE_URL", "http://localhost:11434"),
        openai_api_key=_env("OPENAI_API_KEY"),
        anthropic_api_key=_env("ANTHROPIC_API_KEY"),
        search_provider=_env("SEARCH_PROVIDER", "duckduckgo") or "duckduckgo",
        tavily_api_key=_env("TAVILY_API_KEY"),
        max_search_iterations=int(_env("MAX_SEARCH_ITERATIONS", "2") or 2),
        results_per_query=int(_env("RESULTS_PER_QUERY", "4") or 4),
        max_docs_to_fetch=int(_env("MAX_DOCS_TO_FETCH", "5") or 5),
        fetch_timeout_seconds=float(_env("FETCH_TIMEOUT_SECONDS", "10") or 10),
        max_doc_chars=int(_env("MAX_DOC_CHARS", "3000") or 3000),
        request_timeout_seconds=float(_env("LLM_TIMEOUT_SECONDS", "60") or 60),
        llm_max_output_tokens=int(_env("LLM_MAX_OUTPUT_TOKENS", "512") or 512),
    )
