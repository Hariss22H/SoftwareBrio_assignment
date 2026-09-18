"""Shared pytest fixtures for Phase 3 tests."""

from __future__ import annotations

import pytest

from app.config import Settings


@pytest.fixture
def settings() -> Settings:
    """Settings that never read the real .env file or a real API key."""
    return Settings.model_validate(
        {
            "openai_api_key": "sk-test-not-a-real-key",
            "openai_model": "gpt-4o-mini",
            "headless": True,
            "max_pages_per_domain": 6,
            "page_timeout_ms": 20000,
            "max_content_chars_per_page": 12000,
            "max_total_content_chars_per_domain": 50000,
            "log_level": "INFO",
        }
    )
