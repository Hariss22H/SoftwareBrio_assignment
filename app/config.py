"""Load and validate settings from environment variables.

This module is the only place that reads .env / process environment values.
It never logs the raw OpenAI API key.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Mapping

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError, field_validator

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ENV_FILE = PROJECT_ROOT / ".env"
PLACEHOLDER_API_KEY = "your_openai_api_key_here"
ALLOWED_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR"}
TRUE_VALUES = {"1", "true", "yes", "on"}
FALSE_VALUES = {"0", "false", "no", "off"}


class ConfigError(Exception):
    """Raised when required settings are missing or invalid."""


class Settings(BaseModel):
    """Typed runtime configuration for the lead enrichment agent."""

    model_config = ConfigDict(frozen=True)

    openai_api_key: SecretStr = Field(repr=False)
    openai_model: str = "gpt-4o-mini"
    headless: bool = True
    max_pages_per_domain: int = Field(default=6, ge=1)
    page_timeout_ms: int = Field(default=20000, ge=1)
    max_content_chars_per_page: int = Field(default=12000, ge=1)
    max_total_content_chars_per_domain: int = Field(default=50000, ge=1)
    log_level: str = "INFO"

    @field_validator("openai_api_key", mode="before")
    @classmethod
    def api_key_must_be_real(cls, value: object) -> object:
        """Reject missing keys and the .env.example placeholder."""
        if isinstance(value, SecretStr):
            raw = value.get_secret_value().strip()
        else:
            raw = str(value or "").strip()
        if not raw or raw == PLACEHOLDER_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY is missing. Copy .env.example to .env and set a real key."
            )
        return raw

    @field_validator("openai_model")
    @classmethod
    def model_name_must_not_be_empty(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("OPENAI_MODEL cannot be empty.")
        return cleaned

    @field_validator("log_level")
    @classmethod
    def log_level_must_be_known(cls, value: str) -> str:
        level = value.strip().upper()
        if level not in ALLOWED_LOG_LEVELS:
            allowed = ", ".join(sorted(ALLOWED_LOG_LEVELS))
            raise ValueError(f"LOG_LEVEL must be one of: {allowed}.")
        return level

    def redacted_dict(self) -> dict[str, object]:
        """Return settings that are safe to print in logs."""
        data = self.model_dump(exclude={"openai_api_key"})
        data["openai_api_key_set"] = True
        return data


def configure_logging(log_level: str = "INFO") -> None:
    """Attach a simple console logger if one is not already configured."""
    level = getattr(logging, log_level.upper(), logging.INFO)
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        )
    else:
        root.setLevel(level)


def _read_optional(source: Mapping[str, str], key: str) -> str | None:
    value = source.get(key)
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned if cleaned else None


def _parse_bool(value: str | None, field_name: str, default: bool) -> bool:
    if value is None:
        return default
    normalized = value.lower()
    if normalized in TRUE_VALUES:
        return True
    if normalized in FALSE_VALUES:
        return False
    raise ConfigError(f"{field_name} must be true or false.")


def _parse_int(value: str | None, field_name: str, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ConfigError(f"{field_name} must be an integer.") from exc


def _safe_config_error_message(exc: ValidationError) -> str:
    """Build an error string that never includes secret input values."""
    parts: list[str] = []
    for error in exc.errors():
        location = ".".join(str(item) for item in error.get("loc", ()))
        message = error.get("msg", "Invalid configuration")
        if message.startswith("Value error, "):
            message = message.removeprefix("Value error, ")
        if location == "openai_api_key":
            parts.append(message)
        elif location:
            parts.append(f"{location}: {message}")
        else:
            parts.append(message)
    return "Invalid configuration: " + "; ".join(parts)


def load_settings(
    *,
    environ: Mapping[str, str] | None = None,
    env_file: Path | None = DEFAULT_ENV_FILE,
) -> Settings:
    """Load settings from a mapping or from `.env` + the process environment.

    Pass `environ` in tests so a real `.env` file cannot leak into assertions.
    """
    if environ is None:
        if env_file is not None and env_file.exists():
            load_dotenv(env_file)
        source: Mapping[str, str] = os.environ
    else:
        source = environ

    raw_values = {
        "openai_api_key": _read_optional(source, "OPENAI_API_KEY") or "",
        "openai_model": _read_optional(source, "OPENAI_MODEL") or "gpt-4o-mini",
        "headless": _parse_bool(_read_optional(source, "HEADLESS"), "HEADLESS", True),
        "max_pages_per_domain": _parse_int(
            _read_optional(source, "MAX_PAGES_PER_DOMAIN"),
            "MAX_PAGES_PER_DOMAIN",
            6,
        ),
        "page_timeout_ms": _parse_int(
            _read_optional(source, "PAGE_TIMEOUT_MS"),
            "PAGE_TIMEOUT_MS",
            20000,
        ),
        "max_content_chars_per_page": _parse_int(
            _read_optional(source, "MAX_CONTENT_CHARS_PER_PAGE"),
            "MAX_CONTENT_CHARS_PER_PAGE",
            12000,
        ),
        "max_total_content_chars_per_domain": _parse_int(
            _read_optional(source, "MAX_TOTAL_CONTENT_CHARS_PER_DOMAIN"),
            "MAX_TOTAL_CONTENT_CHARS_PER_DOMAIN",
            50000,
        ),
        "log_level": _read_optional(source, "LOG_LEVEL") or "INFO",
    }

    try:
        return Settings.model_validate(raw_values)
    except ValidationError as exc:
        raise ConfigError(_safe_config_error_message(exc)) from exc
