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
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from factcheck.config import Settings

T = TypeVar("T", bound=BaseModel)


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
            num_predict=settings.llm_max_output_tokens,
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
            max_tokens=settings.llm_max_output_tokens,
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
            max_tokens=settings.llm_max_output_tokens,
        )

    raise LLMError(f"Unknown LLM_PROVIDER: {settings.llm_provider!r}")


def _extract_json(raw: str) -> str:
    """Pull the first balanced {...} object out of `raw`.

    Small local models sometimes echo fragments of the schema (which is
    itself JSON) back into their response alongside — or instead of — an
    actual answer. A greedy `\\{.*\\}` regex spans from the first `{` to the
    *last* `}` in the whole response, so if both the real answer and an
    echoed schema fragment are present, it concatenates them into one
    invalid blob. Scanning for the first properly-balanced object avoids
    that: it stops at the matching close brace instead of the last one.
    """
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    start = raw.find("{")
    if start == -1:
        return raw

    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(raw)):
        ch = raw[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return raw[start : i + 1]

    return raw[start:]  # unbalanced — let json parsing surface the real error


def _lowercase_keys(obj):
    """Recursively lowercase dict keys.

    Small models are inconsistent about key casing ("Verdict" vs "verdict")
    even when told the exact field name. Every schema in this project uses
    lowercase snake_case fields by convention, so normalizing case before
    validation is safe and turns a common cosmetic mistake into a non-issue
    instead of a hard failure.
    """
    if isinstance(obj, dict):
        return {(k.lower() if isinstance(k, str) else k): _lowercase_keys(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_lowercase_keys(v) for v in obj]
    return obj


def _parse(raw: str, schema: type[T]) -> T:
    data = json.loads(_extract_json(raw))
    return schema.model_validate(_lowercase_keys(data))


def _build_hint(schema: type[T], example: dict | None) -> str:
    if example is not None:
        example_json = json.dumps(example, separators=(",", ":"))
        return (
            f"Respond with ONLY a JSON object shaped exactly like this example "
            f"(these are placeholder values — replace them with your real answer, "
            f"but keep the same field names, casing, and structure):\n{example_json}"
        )
    # Fallback for schemas without a hand-written example: a raw JSON-Schema
    # dump works but is a harder target for small models, which sometimes
    # echo the schema itself back instead of an instance of it.
    schema_hint = json.dumps(schema.model_json_schema(), separators=(",", ":"))
    return f"Respond with ONLY a JSON object matching this schema, no prose before or after:\n{schema_hint}"


def call_structured(
    llm, system_prompt: str, user_prompt: str, schema: type[T], example: dict | None = None
) -> T:
    """Call `llm` and parse its response as JSON matching `schema`.

    `example` — a dict with the exact keys `schema` expects, filled with
    representative (not necessarily valid) values — is strongly preferred
    over letting this fall back to a raw JSON-Schema dump: a concrete
    worked example is a much easier target for small local models than an
    abstract schema, which they sometimes parrot back verbatim instead of
    instantiating.

    Retries once with the validation error appended to the prompt if the
    first response doesn't parse. Raises `LLMError` if the repair attempt
    also fails, rather than silently returning a guessed/empty object —
    callers (the graph nodes) decide how to handle that.
    """
    hint = _build_hint(schema, example)
    full_prompt = f"{user_prompt}\n\n{hint}"

    messages = [("system", system_prompt), ("human", full_prompt)]
    response = llm.invoke(messages)
    raw = response.content if hasattr(response, "content") else str(response)

    try:
        return _parse(raw, schema)
    except (ValidationError, json.JSONDecodeError) as first_error:
        repair_prompt = (
            f"Your previous response could not be parsed as valid JSON matching "
            f"the required shape.\n\nYour response:\n{raw}\n\n"
            f"Validation error:\n{first_error}\n\n"
            f"Reply again with ONLY corrected JSON. {hint}"
        )
        repair_response = llm.invoke(messages + [("human", repair_prompt)])
        repair_raw = repair_response.content if hasattr(repair_response, "content") else str(repair_response)
        try:
            return _parse(repair_raw, schema)
        except (ValidationError, json.JSONDecodeError) as second_error:
            raise LLMError(
                f"Model failed to produce valid {schema.__name__} JSON after repair attempt: {second_error}"
            ) from second_error
