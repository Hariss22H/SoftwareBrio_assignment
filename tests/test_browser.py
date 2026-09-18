"""Local Playwright checks that do not visit company websites."""

from __future__ import annotations

from app.browser import BrowserManager
from app.config import Settings


def test_browser_manager_launches_context_and_closes(settings: Settings) -> None:
    with BrowserManager(settings) as manager:
        context = manager.new_context()
        try:
            page = context.new_page()
            page.goto("about:blank")
            assert page.url in {"about:blank", "about:blank/"}
        finally:
            context.close()
