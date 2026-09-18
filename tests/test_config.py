"""Unit tests for environment configuration loading."""

from __future__ import annotations

import pytest
from pydantic import SecretStr

from app.config import ConfigError, Settings, load_settings


def _settings(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {"OPENAI_API_KEY": "sk-test-not-a-real-key"}
    values.update(overrides)
    return values


def test_load_settings_uses_spec_defaults() -> None:
    settings = load_settings(environ=_settings())

    assert settings.openai_model == "gpt-4o-mini"
    assert settings.headless is True
    assert settings.max_pages_per_domain == 6
    assert settings.page_timeout_ms == 20000
    assert settings.max_content_chars_per_page == 12000
    assert settings.max_total_content_chars_per_domain == 50000
    assert settings.log_level == "INFO"
    assert isinstance(settings.openai_api_key, SecretStr)


def test_load_settings_parses_typed_overrides() -> None:
    settings = load_settings(
        environ=_settings(
            OPENAI_MODEL="gpt-4o",
            HEADLESS="false",
            MAX_PAGES_PER_DOMAIN="3",
            PAGE_TIMEOUT_MS="15000",
            MAX_CONTENT_CHARS_PER_PAGE="8000",
            MAX_TOTAL_CONTENT_CHARS_PER_DOMAIN="20000",
            LOG_LEVEL="warning",
        )
    )

    assert settings.openai_model == "gpt-4o"
    assert settings.headless is False
    assert settings.max_pages_per_domain == 3
    assert settings.page_timeout_ms == 15000
    assert settings.max_content_chars_per_page == 8000
    assert settings.max_total_content_chars_per_domain == 20000
    assert settings.log_level == "WARNING"


def test_missing_api_key_raises_safe_error() -> None:
    with pytest.raises(ConfigError, match="OPENAI_API_KEY is missing"):
        load_settings(environ={"OPENAI_MODEL": "gpt-4o-mini"})


def test_placeholder_api_key_is_rejected() -> None:
    with pytest.raises(ConfigError, match="OPENAI_API_KEY is missing"):
        load_settings(environ=_settings(OPENAI_API_KEY="your_openai_api_key_here"))


def test_invalid_headless_value_raises_config_error() -> None:
    with pytest.raises(ConfigError, match="HEADLESS"):
        load_settings(environ=_settings(HEADLESS="maybe"))


def test_non_integer_limit_raises_config_error() -> None:
    with pytest.raises(ConfigError, match="MAX_PAGES_PER_DOMAIN"):
        load_settings(environ=_settings(MAX_PAGES_PER_DOMAIN="six"))


def test_non_positive_limit_raises_config_error() -> None:
    with pytest.raises(ConfigError, match="max_pages_per_domain"):
        load_settings(environ=_settings(MAX_PAGES_PER_DOMAIN="0"))


def test_invalid_log_level_raises_config_error() -> None:
    with pytest.raises(ConfigError, match="LOG_LEVEL"):
        load_settings(environ=_settings(LOG_LEVEL="VERBOSE"))


def test_settings_repr_and_logs_hide_api_key() -> None:
    secret = "sk-super-secret-test-key"
    settings = load_settings(environ=_settings(OPENAI_API_KEY=secret))
    redacted = settings.redacted_dict()

    assert secret not in repr(settings)
    assert secret not in str(redacted)
    assert "openai_api_key" not in redacted
    assert redacted["openai_api_key_set"] is True
    assert settings.openai_api_key.get_secret_value() == secret


def test_settings_model_direct_validation_rejects_placeholder() -> None:
    with pytest.raises(Exception):
        Settings(openai_api_key="your_openai_api_key_here")
