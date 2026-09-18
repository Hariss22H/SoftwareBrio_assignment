"""Unit tests for Pydantic company intelligence models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.models import UNKNOWN_TARGET_AUDIENCE, CompanyIntelligence, TeamMember


def _valid_company(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "domain": "postman.com",
        "company_name": "Postman",
        "company_overview": "Postman is an API platform for building and using APIs.",
        "target_audience": "Developers and API teams",
        "contact_points": ["support@postman.com"],
        "team_members": [{"name": "Jane Doe", "role": "CEO", "linkedin_url": None}],
        "confidence_score": 0.8,
        "source_urls": ["https://www.postman.com/"],
        "processing_status": "success",
        "error_message": None,
    }
    payload.update(overrides)
    return payload


def test_valid_company_intelligence_passes() -> None:
    record = CompanyIntelligence.model_validate(_valid_company())

    assert record.domain == "postman.com"
    assert record.company_name == "Postman"
    assert record.confidence_score == 0.8
    assert record.processing_status == "success"
    assert record.error_message is None
    assert record.contact_points == ["support@postman.com"]
    assert record.team_members[0].name == "Jane Doe"


def test_optional_fields_can_be_omitted() -> None:
    record = CompanyIntelligence.model_validate(
        {
            "domain": "vapi.ai",
            "company_overview": "Vapi helps teams build voice agents.",
            "target_audience": UNKNOWN_TARGET_AUDIENCE,
            "confidence_score": 0.5,
            "processing_status": "partial_success",
        }
    )

    assert record.company_name is None
    assert record.contact_points == []
    assert record.team_members == []
    assert record.source_urls == []
    assert record.error_message is None
    assert record.processing_status == "partial_success"


def test_confidence_score_boundaries_are_accepted() -> None:
    low = CompanyIntelligence.model_validate(_valid_company(confidence_score=0.0))
    high = CompanyIntelligence.model_validate(_valid_company(confidence_score=1.0))

    assert low.confidence_score == 0.0
    assert high.confidence_score == 1.0


@pytest.mark.parametrize("score", [-0.01, 1.01, 2, -1])
def test_invalid_confidence_score_is_rejected(score: float) -> None:
    with pytest.raises(ValidationError):
        CompanyIntelligence.model_validate(_valid_company(confidence_score=score))


def test_invalid_processing_status_is_rejected() -> None:
    with pytest.raises(ValidationError):
        CompanyIntelligence.model_validate(_valid_company(processing_status="ok"))


def test_blank_company_name_becomes_none() -> None:
    record = CompanyIntelligence.model_validate(_valid_company(company_name="   "))
    assert record.company_name is None


def test_contact_points_are_deduplicated_and_junk_is_dropped() -> None:
    record = CompanyIntelligence.model_validate(
        _valid_company(contact_points=["Sales@Example.com", "sales@example.com", "not-an-email", ""])
    )
    assert record.contact_points == ["Sales@Example.com"]


def test_team_members_are_deduplicated() -> None:
    record = CompanyIntelligence.model_validate(
        _valid_company(
            team_members=[
                {"name": "Ada Lovelace", "role": "Engineer"},
                {"name": "Ada Lovelace", "role": "Engineer"},
            ]
        )
    )
    assert len(record.team_members) == 1


def test_blank_team_member_name_is_rejected() -> None:
    with pytest.raises(ValidationError):
        TeamMember.model_validate({"name": "   "})


def test_non_linkedin_url_is_rejected() -> None:
    with pytest.raises(ValidationError, match="LinkedIn"):
        TeamMember.model_validate(
            {"name": "Ada Lovelace", "linkedin_url": "https://example.com/ada"}
        )


def test_valid_linkedin_url_is_kept() -> None:
    member = TeamMember.model_validate(
        {
            "name": "Ada Lovelace",
            "role": None,
            "linkedin_url": "https://www.linkedin.com/in/ada-lovelace",
        }
    )
    assert member.linkedin_url == "https://www.linkedin.com/in/ada-lovelace"
    assert member.role is None


def test_failure_helper_uses_honest_empty_defaults() -> None:
    record = CompanyIntelligence.failure("supabase.com", "Homepage timed out")

    assert record.domain == "supabase.com"
    assert record.company_name is None
    assert record.company_overview == ""
    assert record.target_audience == UNKNOWN_TARGET_AUDIENCE
    assert record.contact_points == []
    assert record.team_members == []
    assert record.source_urls == []
    assert record.confidence_score == 0.0
    assert record.processing_status == "failed"
    assert record.error_message == "Homepage timed out"
