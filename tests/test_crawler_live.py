"""Optional live crawler tests. Skipped during normal pytest runs.

Enable with:

    set RUN_LIVE_CRAWLER_TESTS=1
    pytest tests/test_crawler_live.py -v
"""

from __future__ import annotations

import os

import pytest

from app.browser import BrowserManager
from app.config import Settings
from app.crawler import crawl_domain, crawl_domains

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_LIVE_CRAWLER_TESTS") != "1",
    reason="Live crawler tests are skipped. Set RUN_LIVE_CRAWLER_TESTS=1 to enable.",
)


def test_crawl_example_com_collects_homepage(settings: Settings) -> None:
    with BrowserManager(settings) as browser:
        result = crawl_domain("example.com", settings, browser)

    assert result.domain == "example.com"
    assert result.pages_crawled >= 1
    homepage = result.pages[0]
    assert homepage.status_code == 200
    assert "Example" in homepage.title
    assert homepage.html
    assert result.processing_status in {"success", "partial_success"}


def test_invalid_domain_does_not_stop_live_example_com(settings: Settings) -> None:
    results = crawl_domains(
        ["not a domain", "example.com"],
        settings,
    )

    assert len(results) == 2
    assert results[0].processing_status == "failed"
    assert results[1].domain == "example.com"
    assert results[1].pages_crawled >= 1
