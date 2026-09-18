"""Playwright browser setup and teardown helpers."""

from __future__ import annotations

import logging
from types import TracebackType

from playwright.sync_api import Browser, BrowserContext, Playwright, sync_playwright

from app.config import Settings

logger = logging.getLogger(__name__)

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)


class BrowserManager:
    """Launch one Chromium process and create an isolated context per domain."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._playwright: Playwright | None = None
        self._browser: Browser | None = None

    def start(self) -> BrowserManager:
        """Start Playwright and launch Chromium if they are not already running."""
        if self._playwright is None:
            logger.info("Starting Playwright")
            self._playwright = sync_playwright().start()
        if self._browser is None:
            logger.info("Launching Chromium (headless=%s)", self._settings.headless)
            self._browser = self._playwright.chromium.launch(
                headless=self._settings.headless,
            )
        return self

    def new_context(self) -> BrowserContext:
        """Open a fresh browser context so cookies from one domain do not leak."""
        if self._browser is None:
            self.start()
        assert self._browser is not None
        context = self._browser.new_context(
            user_agent=DEFAULT_USER_AGENT,
            locale="en-US",
            viewport={"width": 1280, "height": 720},
            java_script_enabled=True,
        )
        context.set_default_timeout(self._settings.page_timeout_ms)
        context.set_default_navigation_timeout(self._settings.page_timeout_ms)
        return context

    def close(self) -> None:
        """Close the browser and stop Playwright, even if one step fails."""
        if self._browser is not None:
            try:
                self._browser.close()
            except Exception:
                logger.warning("Failed to close Chromium cleanly", exc_info=True)
            self._browser = None
        if self._playwright is not None:
            try:
                self._playwright.stop()
            except Exception:
                logger.warning("Failed to stop Playwright cleanly", exc_info=True)
            self._playwright = None

    def __enter__(self) -> BrowserManager:
        return self.start()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
