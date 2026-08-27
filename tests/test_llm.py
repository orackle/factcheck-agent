from __future__ import annotations

import json

import pytest
from pydantic import BaseModel

from factcheck.llm import LLMError, _extract_json, call_structured
from tests.conftest import FakeLLM


class Toy(BaseModel):
    value: int


def test_extract_json_ignores_trailing_schema_echo():
    # Reproduces a real failure seen with a small local model: it answered
    # correctly, then echoed the schema hint from the prompt right after.
    # A greedy `\{.*\}` regex spans first `{` to *last* `}` and swallows
    # both into one invalid blob — the balanced-brace scan must not.
    raw = (
        '{"value": 7}\n\n'
        'Note: this matches the schema below:\n'
        '{"$defs": {"Toy": {"type": "object"}}, "properties": {"value": {"type": "integer"}}}'
    )
    assert json.loads(_extract_json(raw)) == {"value": 7}


def test_extract_json_handles_nested_braces_and_string_content():
    raw = '{"value": 1, "note": "a {literal brace} inside a string"}'
    assert json.loads(_extract_json(raw)) == {
        "value": 1,
        "note": "a {literal brace} inside a string",
    }


def test_call_structured_parses_clean_json():
    llm = FakeLLM(lambda messages: '{"value": 42}')
    result = call_structured(llm, "system", "user", Toy)
    assert result.value == 42


def test_call_structured_parses_json_in_code_fence():
    llm = FakeLLM(lambda messages: '```json\n{"value": 7}\n```')
    result = call_structured(llm, "system", "user", Toy)
    assert result.value == 7


def test_call_structured_repairs_after_bad_first_response():
    calls = {"n": 0}

    def responder(messages):
        calls["n"] += 1
        if calls["n"] == 1:
            return "not json at all"
        return '{"value": 99}'

    llm = FakeLLM(responder)
    result = call_structured(llm, "system", "user", Toy)
    assert result.value == 99
    assert calls["n"] == 2


def test_call_structured_raises_after_repair_also_fails():
    llm = FakeLLM(lambda messages: "still not json")
    with pytest.raises(LLMError):
        call_structured(llm, "system", "user", Toy)
