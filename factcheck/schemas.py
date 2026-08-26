"""Typed data contracts shared across the agent, tools, and API.

Every LLM call in this project is required to return JSON matching one of
these schemas. Providers differ in how well they support native structured
output, so the parsing layer (see `factcheck/llm.py`) treats these as the
source of truth and validates against them rather than trusting any one
provider's function-calling format.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class Stance(str, Enum):
    SUPPORTS = "supports"
    REFUTES = "refutes"
    NEUTRAL = "neutral"
    IRRELEVANT = "irrelevant"


class Verdict(str, Enum):
    SUPPORTED = "supported"
    REFUTED = "refuted"
    MIXED = "mixed"
    UNVERIFIED = "unverified"


class SearchResult(BaseModel):
    title: str
    url: str
    snippet: str = ""


class FetchedDoc(BaseModel):
    url: str
    title: str = ""
    text: str = ""
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.text.strip())


class Evidence(BaseModel):
    source_url: str
    quote: str = Field(description="Verbatim or near-verbatim quote from the source")
    stance: Stance
    relevance: float = Field(ge=0.0, le=1.0, description="0 = irrelevant, 1 = directly addresses the claim")


class PlanOutput(BaseModel):
    queries: list[str] = Field(min_length=1, max_length=5)


class ReflectionOutput(BaseModel):
    sufficient: bool
    reasoning: str
    follow_up_queries: list[str] = Field(default_factory=list, max_length=3)


class Citation(BaseModel):
    source_url: str
    quote: str


class SynthesisOutput(BaseModel):
    """What the LLM decides at the synthesis step.

    Deliberately excludes fields the graph tracks itself (claim text,
    iteration count, trace) so the model only has to reason about the
    verdict — those get merged in afterward by `agent/nodes.py`.
    """

    verdict: Verdict
    confidence: float = Field(ge=0.0, le=1.0)
    summary: str
    citations: list[Citation] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)


class FactCheckResult(BaseModel):
    claim: str
    verdict: Verdict
    confidence: float = Field(ge=0.0, le=1.0)
    summary: str
    citations: list[Citation] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list, description="Notable disagreements found across sources")
    sources_consulted: int = 0
    iterations: int = 0
    trace: list[str] = Field(default_factory=list, description="Human-readable log of what the agent did, in order")
