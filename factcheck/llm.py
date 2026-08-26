"""Provider-agnostic LLM access.

Design decision: rather than lean on each provider's native structured-output
/ function-calling API (which varies in quality across OpenAI, Anthropic, and
local Ollama models), every call here asks the model for JSON in the prompt
and validates the response against a Pydantic schema. If parsing fails, the
raw output and the validation error are fed back to the model once for a
repair attempt. This costs one extra round trip in the rare failure case, but
means swapping providers — including to whatever small local model is
running under Ollama — never breaks the agent's control flow.
"""
from __future__ import annotations

import json
import re
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from factcheck.config import Settings

T = TypeVar("T", bound=BaseModel)

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


class LLMError(RuntimeError):
    pass


def get_chat_model(settings: Settings):
    """Return a LangChain chat model for the configured provider.

    Imports are lazy and provider-scoped so installing only the packages you
    actually need (e.g. just `langchain-ollama` for local dev) is enough.
    """
    provider = settings.llm_provider.lower()

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=settings.llm_model,
            base_url=settings.ollama_base_url,
            temperature=0.1,
        )

    if provider == "openai":
        if not settings.openai_api_key:
            raise LLMError("OPENAI_API_KEY is not set but LLM_PROVIDER=openai")
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.llm_model or "gpt-4o-mini",
            api_key=settings.openai_api_key,
            temperature=0.1,
            timeout=settings.request_timeout_seconds,
        )

    if provider == "anthropic":
        if not settings.anthropic_api_key:
            raise LLMError("ANTHROPIC_API_KEY is not set but LLM_PROVIDER=anthropic")
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=settings.llm_model or "claude-haiku-4-5-20251001",
            api_key=settings.anthropic_api_key,
            temperature=0.1,
            timeout=settings.request_timeout_seconds,
        )

    raise LLMError(f"Unknown LLM_PROVIDER: {settings.llm_provider!r}")


def _extract_json(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:]
    match = _JSON_BLOCK_RE.search(raw)
    return match.group(0) if match else raw


def call_structured(llm, system_prompt: str, user_prompt: str, schema: type[T]) -> T:
    """Call `llm` and parse its response as JSON matching `schema`.

    Retries once with the validation error appended to the prompt if the
    first response doesn't parse. Raises `LLMError` if the repair attempt
    also fails, rather than silently returning a guessed/empty object —
    callers (the graph nodes) decide how to handle that.
    """
    schema_hint = json.dumps(schema.model_json_schema(), indent=2)
    full_prompt = (
        f"{user_prompt}\n\n"
        f"Respond with ONLY a JSON object matching this schema, no prose "
        f"before or after:\n{schema_hint}"
    )

    messages = [("system", system_prompt), ("human", full_prompt)]
    response = llm.invoke(messages)
    raw = response.content if hasattr(response, "content") else str(response)

    try:
        return schema.model_validate_json(_extract_json(raw))
    except (ValidationError, json.JSONDecodeError) as first_error:
        repair_prompt = (
            f"Your previous response could not be parsed as valid JSON matching "
            f"the required schema.\n\nYour response:\n{raw}\n\n"
            f"Validation error:\n{first_error}\n\n"
            f"Reply again with ONLY corrected JSON matching this schema:\n{schema_hint}"
        )
        repair_response = llm.invoke(messages + [("human", repair_prompt)])
        repair_raw = repair_response.content if hasattr(repair_response, "content") else str(repair_response)
        try:
            return schema.model_validate_json(_extract_json(repair_raw))
        except (ValidationError, json.JSONDecodeError) as second_error:
            raise LLMError(
                f"Model failed to produce valid {schema.__name__} JSON after repair attempt: {second_error}"
            ) from second_error
