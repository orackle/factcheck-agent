"""Minimal FastAPI wrapper.

    uvicorn factcheck.api:app --reload

This is the seam meant for the portfolio embed later — a small hosted
instance behind `/verify` is what backs the "in progress" card's eventual
live demo. Kept intentionally thin: no auth, no rate limiting, no queueing.
Those are exactly the things to add before pointing real traffic at it,
called out in the README rather than half-implemented here.
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from factcheck.agent.graph import run_fact_check
from factcheck.config import load_settings
from factcheck.llm import LLMError, get_chat_model
from factcheck.schemas import FactCheckResult
from factcheck.tools.search import SearchError, get_search_provider

app = FastAPI(
    title="Fact-Check Agent API",
    description="Multi-step web research agent that verifies a claim against live sources.",
    version="0.1.0",
)


class VerifyRequest(BaseModel):
    claim: str = Field(min_length=3, max_length=500)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/verify", response_model=FactCheckResult)
def verify(request: VerifyRequest) -> FactCheckResult:
    settings = load_settings()
    try:
        llm = get_chat_model(settings)
        search_provider = get_search_provider(settings)
        return run_fact_check(request.claim, llm, search_provider, settings)
    except (LLMError, SearchError) as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
