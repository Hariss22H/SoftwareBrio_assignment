"""Unit tests for OpenAI structured extraction. These tests never call the API."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

from openai import APITimeoutError, AuthenticationError, RateLimitError

from app.config import Settings
from app.llm_extractor import SYSTEM_PROMPT, extract_company_intelligence
from app.models import UNKNOWN_TARGET_AUDIENCE, CompanyIntelligence


def _status_error(error_cls: type, message: str, status_code: int) -> Exception:
    request = Mock()
    response = Mock()
    response.request = request
    response.status_code = status_code
    response.headers = {}
    return error_cls(message, response=response, body=None)


def _parsed_completion(record: CompanyIntelligence) -> SimpleNamespace:
    message = SimpleNamespace(parsed=record, refusal=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _client(parse_impl: Any) -> SimpleNamespace:
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(parse=parse_impl)))


def _valid_record(**overrides: object) -> CompanyIntelligence:
    payload: dict[str, object] = {
        "domain": "wrong.com",
        "company_name": "Acme",
        "company_overview": "Acme builds API tools for developers. Teams use it to test APIs.",
        "target_audience": "Developers building backend applications",
        "contact_points": ["sales@acme.com"],
        "team_members": [
            {
                "name": "Ada Lovelace",
                "role": "Chief Executive Officer",
                "linkedin_url": "https://www.linkedin.com/in/ada-lovelace",
            }
        ],
        "confidence_score": 0.82,
        "source_urls": ["https://invented.example/about"],
        "processing_status": "success",
        "error_message": None,
    }
    payload.update(overrides)
    return CompanyIntelligence.model_validate(payload)


CONTENT = """
PAGE TITLE: About Acme
URL: https://acme.com/about

CONTENT:
Acme builds API tools for developers.
Ada Lovelace, Chief Executive Officer
sales@acme.com
https://www.linkedin.com/in/ada-lovelace
"""


def test_successful_extraction_uses_content_and_overrides_domain(
    settings: Settings,
) -> None:
    captured: dict[str, Any] = {}

    def parse(**kwargs: Any) -> SimpleNamespace:
        captured.update(kwargs)
        return _parsed_completion(_valid_record())

    result = extract_company_intelligence(
        domain="acme.com",
        cleaned_content=CONTENT,
        source_urls=["https://acme.com/about"],
        settings=settings,
        client=_client(parse),
    )

    assert result.processing_status == "success"
    assert result.domain == "acme.com"
    assert result.company_name == "Acme"
    assert "API tools" in result.company_overview
    assert result.contact_points == ["sales@acme.com"]
    assert result.team_members[0].name == "Ada Lovelace"
    assert result.team_members[0].linkedin_url == "https://www.linkedin.com/in/ada-lovelace"
    assert result.source_urls == ["https://acme.com/about"]
    assert result.error_message is None
    assert 0.0 <= result.confidence_score <= 1.0
    assert captured["model"] == "gpt-4o-mini"
    assert captured["response_format"] is CompanyIntelligence
    assert "Do not invent" in captured["messages"][0]["content"]
    assert "cautious inference" in captured["messages"][0]["content"]
    assert "Never use outside knowledge" in captured["messages"][0]["content"]
    assert CONTENT.strip()[:20] in captured["messages"][1]["content"]
    assert "sk-test-not-a-real-key" not in str(captured)


def test_missing_fields_are_allowed(settings: Settings) -> None:
    sparse = _valid_record(
        company_name=None,
        company_overview="Acme builds API tools for developers. Teams use it to test APIs.",
        target_audience=UNKNOWN_TARGET_AUDIENCE,
        contact_points=[],
        team_members=[],
        confidence_score=0.45,
    )

    result = extract_company_intelligence(
        domain="acme.com",
        cleaned_content=CONTENT,
        source_urls=["https://acme.com/about"],
        settings=settings,
        client=_client(lambda **kwargs: _parsed_completion(sparse)),
    )

    assert result.company_name is None
    assert result.contact_points == []
    assert result.team_members == []
    assert result.target_audience == UNKNOWN_TARGET_AUDIENCE
    assert result.processing_status == "partial_success"


def test_missing_emails_and_team_do_not_fail_when_core_fields_exist(settings: Settings) -> None:
    sparse = _valid_record(
        contact_points=[],
        team_members=[],
        confidence_score=0.78,
    )

    result = extract_company_intelligence(
        domain="acme.com",
        cleaned_content=CONTENT,
        source_urls=["https://acme.com/about"],
        settings=settings,
        client=_client(lambda **kwargs: _parsed_completion(sparse)),
    )

    assert result.contact_points == []
    assert result.team_members == []
    assert result.company_overview
    assert result.target_audience != UNKNOWN_TARGET_AUDIENCE
    assert result.processing_status == "success"


def test_unknown_audience_or_low_score_is_partial_success(settings: Settings) -> None:
    unknown = _valid_record(
        target_audience=UNKNOWN_TARGET_AUDIENCE,
        contact_points=[],
        team_members=[],
        confidence_score=0.8,
    )
    result = extract_company_intelligence(
        domain="acme.com",
        cleaned_content=CONTENT,
        source_urls=["https://acme.com/about", "https://acme.com/pricing"],
        settings=settings,
        client=_client(lambda **kwargs: _parsed_completion(unknown)),
    )
    assert result.processing_status == "partial_success"
    assert result.confidence_score <= 0.7

    low = _valid_record(confidence_score=0.4)
    low_result = extract_company_intelligence(
        domain="acme.com",
        cleaned_content=CONTENT,
        source_urls=["https://acme.com/about"],
        settings=settings,
        client=_client(lambda **kwargs: _parsed_completion(low)),
    )
    assert low_result.processing_status == "partial_success"


def test_invented_emails_and_people_are_dropped(settings: Settings) -> None:
    invented = _valid_record(
        contact_points=["secret@not-on-the-page.com", "sales@acme.com"],
        team_members=[
            {"name": "Fake Person", "role": "CEO", "linkedin_url": None},
            {
                "name": "Ada Lovelace",
                "role": "Chief Executive Officer",
                "linkedin_url": "https://www.linkedin.com/in/not-in-the-page",
            },
        ],
    )

    result = extract_company_intelligence(
        domain="acme.com",
        cleaned_content=CONTENT,
        source_urls=["https://acme.com/about"],
        settings=settings,
        client=_client(lambda **kwargs: _parsed_completion(invented)),
    )

    assert result.contact_points == ["sales@acme.com"]
    assert [member.name for member in result.team_members] == ["Ada Lovelace"]
    assert result.team_members[0].linkedin_url is None


def test_empty_content_skips_openai(settings: Settings) -> None:
    parse = Mock(side_effect=AssertionError("OpenAI should not be called"))

    result = extract_company_intelligence(
        domain="acme.com",
        cleaned_content="   ",
        source_urls=["https://acme.com"],
        settings=settings,
        client=_client(parse),
    )

    parse.assert_not_called()
    assert result.processing_status == "failed"
    assert result.contact_points == []
    assert result.team_members == []
    assert result.company_name is None
    assert result.source_urls == ["https://acme.com"]
    assert "cleaned website content" in (result.error_message or "").lower()


def test_invalid_structured_output_returns_failed_record(settings: Settings) -> None:
    empty = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(parsed=None, refusal=None))]
    )

    result = extract_company_intelligence(
        domain="acme.com",
        cleaned_content=CONTENT,
        source_urls=["https://acme.com/about"],
        settings=settings,
        client=_client(lambda **kwargs: empty),
    )

    assert result.processing_status == "failed"
    assert result.error_message is not None
    assert result.company_name is None
    assert result.contact_points == []


def test_api_timeout_returns_safe_failure(settings: Settings) -> None:
    def parse(**kwargs: Any) -> None:
        raise APITimeoutError(request=None)

    result = extract_company_intelligence(
        domain="acme.com",
        cleaned_content=CONTENT,
        settings=settings,
        client=_client(parse),
        max_retries=1,
    )

    assert result.processing_status == "failed"
    assert result.error_message == "OpenAI request timed out."
    assert "sk-" not in (result.error_message or "")


def test_authentication_error_is_not_retried(settings: Settings) -> None:
    parse = Mock(side_effect=_status_error(AuthenticationError, "bad key sk-secret-value", 401))

    result = extract_company_intelligence(
        domain="acme.com",
        cleaned_content=CONTENT,
        settings=settings,
        client=_client(parse),
        max_retries=3,
        sleeper=lambda _: None,
    )

    assert parse.call_count == 1
    assert result.processing_status == "failed"
    assert result.error_message == "OpenAI authentication failed."
    assert "sk-secret-value" not in (result.error_message or "")
    assert "sk-secret-value" not in SYSTEM_PROMPT


def test_retryable_error_then_success(settings: Settings) -> None:
    calls = {"count": 0}

    def parse(**kwargs: Any) -> SimpleNamespace:
        calls["count"] += 1
        if calls["count"] == 1:
            raise _status_error(RateLimitError, "rate limited", 429)
        return _parsed_completion(_valid_record())

    result = extract_company_intelligence(
        domain="acme.com",
        cleaned_content=CONTENT,
        source_urls=["https://acme.com/about"],
        settings=settings,
        client=_client(parse),
        sleeper=lambda _: None,
    )

    assert calls["count"] == 2
    assert result.processing_status == "success"
    assert result.company_name == "Acme"


def test_confidence_is_capped_when_evidence_is_thin(settings: Settings) -> None:
    thin = _valid_record(
        company_name=None,
        company_overview="",
        contact_points=[],
        team_members=[],
        confidence_score=0.99,
    )

    result = extract_company_intelligence(
        domain="acme.com",
        cleaned_content="Welcome to our website.",
        source_urls=["https://acme.com"],
        settings=settings,
        client=_client(lambda **kwargs: _parsed_completion(thin)),
    )

    assert result.confidence_score <= 0.3
    assert result.processing_status == "partial_success"
    assert result.contact_points == []
    assert result.team_members == []
