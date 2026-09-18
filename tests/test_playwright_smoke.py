"""Minimal Playwright check for Phase 1.

This test only proves Chromium can launch, open a simple page, and close.
It does not crawl company sites or call the LLM.
"""

from __future__ import annotations

from playwright.sync_api import sync_playwright


def test_playwright_launches_chromium_and_opens_a_page() -> None:
    """Launch headless Chromium, open example.com, then close the browser."""
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        try:
            page = browser.new_page()
            response = page.goto("https://example.com", wait_until="domcontentloaded", timeout=20000)
            assert response is not None
            assert response.ok
            title = page.title()
            assert "Example" in title
        finally:
            browser.close()
