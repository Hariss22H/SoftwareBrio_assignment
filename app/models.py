"""Pydantic models for structured company intelligence.

Unknown facts stay empty or null. These models do not invent company names,
emails, people, or LinkedIn URLs.
"""

from __future__ import annotations

from typing import Literal
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ProcessingStatus = Literal["success", "partial_success", "failed"]

UNKNOWN_TARGET_AUDIENCE = "Not clearly identified from the available website content."


def _blank_to_none(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned if cleaned else None


class TeamMember(BaseModel):
    """A person listed on the company website. Omit anyone not found in the text."""

    model_config = ConfigDict(extra="ignore")

    name: str = Field(
        min_length=1,
        description="Full name as shown on the website. Do not invent people.",
    )
    role: str | None = Field(
        default=None,
        description="Job title if clearly listed. Use null when unknown.",
    )
    linkedin_url: str | None = Field(
        default=None,
        description=(
            "Public LinkedIn profile URL only if it appears in the website content. "
            "Use null when unknown. Do not invent or guess LinkedIn URLs."
        ),
    )

    @field_validator("name")
    @classmethod
    def name_must_not_be_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Team member name cannot be empty.")
        return cleaned

    @field_validator("role", mode="before")
    @classmethod
    def blank_role_is_none(cls, value: object) -> object:
        if isinstance(value, str):
            return _blank_to_none(value)
        return value

    @field_validator("linkedin_url", mode="before")
    @classmethod
    def linkedin_url_must_be_linkedin_or_null(cls, value: object) -> object:
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError("linkedin_url must be a string or null.")
        cleaned = _blank_to_none(value)
        if cleaned is None:
            return None
        parsed = urlparse(cleaned)
        host = (parsed.netloc or parsed.path).lower()
        if "linkedin.com" not in host:
            raise ValueError("linkedin_url must be a LinkedIn URL or null.")
        return cleaned


class CompanyIntelligence(BaseModel):
    """Structured company record produced from public website evidence only."""

    model_config = ConfigDict(extra="ignore")

    domain: str = Field(description="Normalized company domain, such as postman.com.")
    company_name: str | None = Field(
        default=None,
        description="Official company name if clearly stated. Use null when unknown. Do not invent.",
    )
    company_overview: str = Field(
        description=(
            "About two sentences on what the company does, using only the supplied "
            "website content. Do not invent claims."
        ),
    )
    target_audience: str = Field(
        description=(
            "Likely customers based on website evidence. If unclear, use "
            f"'{UNKNOWN_TARGET_AUDIENCE}'."
        ),
    )
    contact_points: list[str] = Field(
        default_factory=list,
        description=(
            "Public business emails found in the website content, such as sales@ "
            "or support@. Empty list if none. Do not fabricate emails."
        ),
    )
    team_members: list[TeamMember] = Field(
        default_factory=list,
        description="People named on the website. Empty list if none. Do not invent people.",
    )
    confidence_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Evidence quality from 0.0 to 1.0, not the model's self-confidence.",
    )
    source_urls: list[str] = Field(
        default_factory=list,
        description="Page URLs used as evidence. Empty list if none were collected.",
    )
    processing_status: ProcessingStatus = Field(
        description="success, partial_success, or failed.",
    )
    error_message: str | None = Field(
        default=None,
        description="Short safe error text on failure. Use null when processing succeeds.",
    )

    @field_validator("domain")
    @classmethod
    def domain_must_not_be_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("domain cannot be empty.")
        return cleaned

    @field_validator("company_name", "error_message", mode="before")
    @classmethod
    def blank_optional_text_is_none(cls, value: object) -> object:
        if isinstance(value, str):
            return _blank_to_none(value)
        return value

    @field_validator("contact_points", mode="before")
    @classmethod
    def keep_unique_public_emails(cls, value: object) -> object:
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError("contact_points must be a list of emails.")
        unique: list[str] = []
        seen: set[str] = set()
        for item in value:
            email = str(item).strip()
            if not email or "@" not in email:
                continue
            key = email.lower()
            if key in seen:
                continue
            seen.add(key)
            unique.append(email)
        return unique

    @field_validator("source_urls", mode="before")
    @classmethod
    def keep_unique_source_urls(cls, value: object) -> object:
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError("source_urls must be a list of URLs.")
        unique: list[str] = []
        seen: set[str] = set()
        for item in value:
            url = str(item).strip()
            if not url:
                continue
            key = url.rstrip("/").lower()
            if key in seen:
                continue
            seen.add(key)
            unique.append(url)
        return unique

    @model_validator(mode="after")
    def deduplicate_team_members(self) -> CompanyIntelligence:
        unique: list[TeamMember] = []
        seen: set[tuple[str, str | None]] = set()
        for member in self.team_members:
            key = (
                member.name.casefold(),
                member.linkedin_url.lower() if member.linkedin_url else None,
            )
            if key in seen:
                continue
            seen.add(key)
            unique.append(member)
        self.team_members = unique
        return self

    @classmethod
    def failure(cls, domain: str, error_message: str) -> CompanyIntelligence:
        """Build an honest failed record with empty optional data."""
        return cls(
            domain=domain,
            company_name=None,
            company_overview="",
            target_audience=UNKNOWN_TARGET_AUDIENCE,
            contact_points=[],
            team_members=[],
            confidence_score=0.0,
            source_urls=[],
            processing_status="failed",
            error_message=error_message,
        )
