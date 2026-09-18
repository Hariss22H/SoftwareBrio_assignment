"""Send cleaned website text to OpenAI and return structured company data.

The extractor never logs API keys. It also drops names, emails, and LinkedIn
URLs that do not appear in the supplied website text.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable
from typing import Any

from openai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)
from pydantic import ValidationError

from app.config import Settings
from app.models import UNKNOWN_TARGET_AUDIENCE, CompanyIntelligence, ProcessingStatus

logger = logging.getLogger(__name__)

OPENAI_TIMEOUT_SECONDS = 60.0
MAX_OUTPUT_TOKENS = 1500
DEFAULT_MAX_RETRIES = 3
SECRET_RE = re.compile(r"sk-[A-Za-z0-9_\-]+")

SYSTEM_PROMPT = """You extract structured company intelligence from website text.

Follow these rules exactly:
1. Use only the supplied website content. Never use outside knowledge.
2. Do not invent or guess facts, company names, people, job titles, email addresses, or LinkedIn URLs.
3. Facts (name, emails, team, LinkedIn) must be explicitly present in the content.

4. Target audience is a cautious inference, not a fact you must quote:
   - If the content describes the product, infer the likely customer from that description.
   - Examples of valid inferences: "Developers building backend applications", "Product teams building AI voice agents", "API and platform engineering teams".
   - Do not name specific customer companies unless they appear in the content.
   - Use exactly "Not clearly identified from the available website content." only when the text does not describe what the product is or who it is for.

5. company_overview must be about two sentences and only include claims supported by the content. If the content is too thin, use an empty string.
6. contact_points: only public generic/business emails found in the content (sales@, support@, contact@, hello@, info@). Empty list if none. Never fabricate emails.
7. team_members: only people named in the content. Use null for unknown role or LinkedIn URL. Empty list if none. A LinkedIn URL is allowed only if that exact profile URL appears in the content.
8. Unknown optional fields must be null. Missing lists must be [].
9. confidence_score is 0.0 to 1.0 based on evidence quality and completeness, not your self-confidence.
   - 0.0-0.3: very little usable content
   - 0.4-0.6: partial information
   - 0.7-0.85: good coverage with some missing fields
   - 0.86-1.0: strong coverage from multiple relevant pages
10. processing_status must be success, partial_success, or failed.
11. error_message must be null unless extraction failed.
12. source_urls should list the URLs provided in the user message. Do not invent URLs.
13. Follow the JSON schema exactly.
"""

RETRYABLE_ERRORS = (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
)


def extract_company_intelligence(
    *,
    domain: str,
    cleaned_content: str,
    settings: Settings,
    source_urls: list[str] | None = None,
    client: Any | None = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
    sleeper: Callable[[float], None] = time.sleep,
) -> CompanyIntelligence:
    """Extract a CompanyIntelligence record from cleaned website text."""
    urls = [url for url in (source_urls or []) if url]
    content = (cleaned_content or "").strip()
    logger.info(
        "LLM extraction starting for %s (model=%s, content_chars=%s)",
        domain,
        settings.openai_model,
        len(content),
    )

    if not content:
        logger.warning("No cleaned content for %s; skipping OpenAI call", domain)
        return _failed(domain, "No cleaned website content was available.", urls)

    truncated = _truncate_content(content, settings.max_total_content_chars_per_domain)
    active_client = client or _build_client(settings)

    last_error = "OpenAI extraction failed."
    attempts = max(1, max_retries)
    for attempt in range(1, attempts + 1):
        try:
            parsed = _parse_completion(
                client=active_client,
                settings=settings,
                domain=domain,
                content=truncated,
                source_urls=urls,
            )
            record = _finalize_record(parsed, domain=domain, content=truncated, source_urls=urls)
            logger.info(
                "LLM extraction finished for %s: status=%s confidence=%s",
                domain,
                record.processing_status,
                record.confidence_score,
            )
            return record
        except Exception as exc:
            last_error = _safe_error_message(exc)
            if _is_retryable(exc) and attempt < attempts:
                delay = min(2 ** (attempt - 1), 8)
                logger.warning(
                    "Temporary OpenAI error for %s (attempt %s/%s): %s",
                    domain,
                    attempt,
                    attempts,
                    last_error,
                )
                sleeper(delay)
                continue
            logger.error("LLM extraction failed for %s: %s", domain, last_error)
            return _failed(domain, last_error, urls)

    return _failed(domain, last_error, urls)


def _build_client(settings: Settings) -> OpenAI:
    return OpenAI(
        api_key=settings.openai_api_key.get_secret_value(),
        timeout=OPENAI_TIMEOUT_SECONDS,
    )


def _parse_completion(
    *,
    client: Any,
    settings: Settings,
    domain: str,
    content: str,
    source_urls: list[str],
) -> CompanyIntelligence:
    completion = client.chat.completions.parse(
        model=settings.openai_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": _user_prompt(domain, content, source_urls),
            },
        ],
        response_format=CompanyIntelligence,
        temperature=0,
        max_completion_tokens=MAX_OUTPUT_TOKENS,
        timeout=OPENAI_TIMEOUT_SECONDS,
    )
    choices = getattr(completion, "choices", None) or []
    if not choices:
        raise ValueError("OpenAI returned no choices.")
    message = choices[0].message
    refusal = getattr(message, "refusal", None)
    if refusal:
        raise ValueError("The model refused to extract company intelligence.")
    parsed = getattr(message, "parsed", None)
    if parsed is None:
        raise ValueError("The model returned empty structured output.")
    if not isinstance(parsed, CompanyIntelligence):
        parsed = CompanyIntelligence.model_validate(parsed)
    return parsed


def _user_prompt(domain: str, content: str, source_urls: list[str]) -> str:
    url_block = "\n".join(f"- {url}" for url in source_urls) or "- (none provided)"
    return (
        f"Domain: {domain}\n"
        f"Source URLs:\n{url_block}\n\n"
        "Website content:\n"
        f"{content}"
    )


def _finalize_record(
    record: CompanyIntelligence,
    *,
    domain: str,
    content: str,
    source_urls: list[str],
) -> CompanyIntelligence:
    """Keep only evidence-backed fields and set status in this process, not the model."""
    haystack = content.casefold()
    emails = [email for email in record.contact_points if email.casefold() in haystack]
    members = []
    for member in record.team_members:
        if member.name.casefold() not in haystack:
            continue
        linkedin_url = member.linkedin_url
        if linkedin_url and linkedin_url.casefold() not in haystack:
            linkedin_url = None
        members.append(member.model_copy(update={"linkedin_url": linkedin_url}))

    company_name = record.company_name
    if company_name and company_name.casefold() not in haystack:
        company_name = None

    overview = (record.company_overview or "").strip()
    audience = (record.target_audience or "").strip() or UNKNOWN_TARGET_AUDIENCE
    score = _cap_confidence(
        record.confidence_score,
        overview=overview,
        company_name=company_name,
        audience=audience,
        emails=emails,
        members=members,
        source_urls=source_urls,
    )
    status = _status_from_evidence(overview=overview, audience=audience, score=score)

    return record.model_copy(
        update={
            "domain": domain,
            "company_name": company_name,
            "company_overview": overview,
            "target_audience": audience,
            "contact_points": emails,
            "team_members": members,
            "confidence_score": score,
            "source_urls": source_urls,
            "processing_status": status,
            "error_message": None,
        }
    )


def _cap_confidence(
    score: float,
    *,
    overview: str,
    company_name: str | None,
    audience: str,
    emails: list[str],
    members: list[Any],
    source_urls: list[str],
) -> float:
    capped = min(max(score, 0.0), 1.0)
    if not overview:
        capped = min(capped, 0.3)
    elif not company_name and not emails and not members:
        capped = min(capped, 0.6)
    if audience == UNKNOWN_TARGET_AUDIENCE:
        capped = min(capped, 0.7)
    if len(source_urls) <= 1:
        capped = min(capped, 0.85)
    return round(capped, 3)


def _status_from_evidence(*, overview: str, audience: str, score: float) -> ProcessingStatus:
    """Mark partial_success when core copy exists but important fields are incomplete.

    Missing emails or team members alone does not make a result failed.
    """
    if not overview:
        return "partial_success"
    audience_unknown = audience.strip() == UNKNOWN_TARGET_AUDIENCE
    if audience_unknown or score < 0.7:
        return "partial_success"
    return "success"


def _truncate_content(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    marker = "\n[Truncated]"
    budget = max(1, max_chars - len(marker))
    return text[:budget].rstrip() + marker


def _failed(domain: str, message: str, source_urls: list[str]) -> CompanyIntelligence:
    record = CompanyIntelligence.failure(domain, message)
    if source_urls:
        return record.model_copy(update={"source_urls": source_urls})
    return record


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, RETRYABLE_ERRORS):
        return True
    if isinstance(exc, (AuthenticationError, BadRequestError, ValidationError)):
        return False
    return False


def _safe_error_message(exc: BaseException) -> str:
    if isinstance(exc, AuthenticationError):
        return "OpenAI authentication failed."
    if isinstance(exc, RateLimitError):
        return "OpenAI rate limit exceeded."
    if isinstance(exc, APITimeoutError):
        return "OpenAI request timed out."
    if isinstance(exc, APIConnectionError):
        return "Could not connect to OpenAI."
    if isinstance(exc, BadRequestError):
        return "OpenAI rejected the extraction request."
    if isinstance(exc, ValidationError):
        return "OpenAI returned data that did not match the schema."
    raw = str(exc).splitlines()[0].strip() if str(exc) else exc.__class__.__name__
    cleaned = SECRET_RE.sub("[redacted]", raw)
    if len(cleaned) > 200:
        cleaned = cleaned[:197] + "..."
    return cleaned or "OpenAI extraction failed."
