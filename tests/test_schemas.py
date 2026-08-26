from __future__ import annotations

import pytest
from pydantic import ValidationError

from factcheck.schemas import Evidence, FactCheckResult, PlanOutput, Stance, Verdict


def test_evidence_relevance_bounds_enforced():
    Evidence(source_url="https://x.example", quote="q", stance=Stance.SUPPORTS, relevance=0.5)
    with pytest.raises(ValidationError):
        Evidence(source_url="https://x.example", quote="q", stance=Stance.SUPPORTS, relevance=1.5)


def test_plan_output_requires_at_least_one_query():
    with pytest.raises(ValidationError):
        PlanOutput(queries=[])


def test_fact_check_result_defaults_are_empty_not_none():
    result = FactCheckResult(
        claim="test claim",
        verdict=Verdict.UNVERIFIED,
        confidence=0.0,
        summary="no evidence found",
    )
    assert result.citations == []
    assert result.contradictions == []
    assert result.trace == []
