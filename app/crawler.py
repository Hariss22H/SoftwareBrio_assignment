"""Visit a company website and collect a small set of relevant pages.

This module collects raw HTML only. BeautifulSoup cleaning and LLM extraction
are implemented in later phases.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Iterable
from urllib.parse import urlparse

from playwright.sync_api import Page, TimeoutError as PlaywrightTimeoutError

from app.browser import BrowserManager
from app.config import ConfigError, Settings, configure_logging, load_settings
from app.models import ProcessingStatus
from app.utils import (
    InvalidURLError,
    homepage_url,
    is_same_domain,
    normalize_domain,
    normalize_url,
    url_dedup_key,
)

logger = logging.getLogger(__name__)

TARGET_DOMAINS = [
    "postman.com",
    "supabase.com",
    "vapi.ai",
]

RELEVANT_KEYWORDS = (
    "about",
    "company",
    "team",
    "leadership",
    "founders",
    "contact",
    "pricing",
    "product",
    "customers",
    "solutions",
    "careers",
)

# Prefer company-intel pages over job boards. Careers still counts, but lower.
HIGH_PRIORITY_KEYWORDS = (
    "about",
    "company",
    "team",
    "leadership",
    "founders",
    "contact",
)
MEDIUM_PRIORITY_KEYWORDS = (
    "customers",
    "solutions",
    "product",
    "pricing",
)
LOW_PRIORITY_KEYWORDS = ("careers", "career")

# Bound stored raw HTML. Cleaned-text caps (12k/50k) are applied after cleaning.
MAX_RAW_HTML_CHARS_PER_PAGE = 400_000
MAX_RAW_HTML_CHARS_PER_DOMAIN = 1_500_000

PAGE_DELAY_SECONDS = 0.5
MAX_LINKS_TO_CONSIDER = 200
BOT_HINTS = (
    "just a moment",
    "attention required",
    "access denied",
    "verify you are human",
    "captcha",
    "cloudflare",
    "bot detection",
)


@dataclass
class CrawledPage:
    """One visited page. ``html`` is raw markup, not cleaned text."""

    url: str
    title: str = ""
    status_code: int | None = None
    html: str = field(default="", repr=False)
    error_message: str | None = None

    @property
    def html_chars(self) -> int:
        return len(self.html)

    @property
    def is_usable(self) -> bool:
        return bool(self.html) or (
            self.status_code is not None and 200 <= self.status_code < 400
        )


@dataclass
class DomainCrawlResult:
    """Bounded crawl outcome for a single company domain."""

    domain: str
    pages: list[CrawledPage] = field(default_factory=list)
    processing_status: ProcessingStatus = "failed"
    error_message: str | None = None

    @property
    def pages_crawled(self) -> int:
        return len(self.pages)

    @property
    def source_urls(self) -> list[str]:
        return [page.url for page in self.pages if page.url]

    @classmethod
    def failed(cls, domain: str, error_message: str) -> DomainCrawlResult:
        return cls(domain=domain, processing_status="failed", error_message=error_message)

    def summary(self) -> dict[str, object]:
        """JSON-safe summary that does not include raw HTML."""
        return {
            "domain": self.domain,
            "processing_status": self.processing_status,
            "error_message": self.error_message,
            "pages_crawled": self.pages_crawled,
            "pages": [
                {
                    "url": page.url,
                    "title": page.title,
                    "status_code": page.status_code,
                    "html_chars": page.html_chars,
                    "error_message": page.error_message,
                }
                for page in self.pages
            ],
        }


class _AnchorParser(HTMLParser):
    """Collect href/text pairs without BeautifulSoup."""

    def __init__(self) -> None:
        super().__init__()
        self.links: list[tuple[str, str]] = []
        self._href: str | None = None
        self._text_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        self._flush()
        attr_map = {key: value or "" for key, value in attrs}
        self._href = attr_map.get("href") or None
        self._text_parts = []

    def handle_data(self, data: str) -> None:
        if self._href is not None:
            self._text_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "a":
            self._flush()

    def _flush(self) -> None:
        if self._href is not None:
            self.links.append((self._href, "".join(self._text_parts).strip()))
        self._href = None
        self._text_parts = []


def score_link(url: str, anchor_text: str = "") -> int:
    """Higher scores mean the page is more likely to contain company intel."""
    parsed = urlparse(url)
    path = parsed.path.lower().replace("-", " ").replace("_", " ")
    text = anchor_text.lower().replace("-", " ").replace("_", " ")
    segments = [part for part in parsed.path.lower().split("/") if part]
    last_segment = (segments[-1] if segments else "").replace("-", " ").replace("_", " ")

    # /company/careers should not beat /about just because "company" is in the path.
    if any(keyword in last_segment for keyword in LOW_PRIORITY_KEYWORDS):
        score = 3
        if path.count("/") <= 2 and len(path) < 40:
            score += 1
        return score

    score = 0
    for keyword in HIGH_PRIORITY_KEYWORDS:
        if keyword in path:
            score += 25
        elif keyword in text:
            score += 16
    for keyword in MEDIUM_PRIORITY_KEYWORDS:
        if keyword in path:
            score += 12
        elif keyword in text:
            score += 8
    if score and path.count("/") <= 2 and len(path) < 40:
        score += 2
    return score


def extract_links_from_html(html: str, *, base_url: str) -> list[tuple[str, str]]:
    """Parse anchor tags from HTML using the standard library HTML parser."""
    parser = _AnchorParser()
    try:
        parser.feed(html or "")
        parser.close()
    except Exception:
        logger.warning("HTML link parsing failed for %s", base_url)
        return []
    return parser.links[:MAX_LINKS_TO_CONSIDER]


def select_relevant_urls(
    *,
    homepage: str,
    candidates: Iterable[tuple[str, str]],
    domain: str,
    max_pages: int,
) -> list[str]:
    """Return homepage first, then the highest-scoring same-domain pages."""
    selected: list[str] = []
    seen: set[str] = set()

    home = normalize_url(homepage) or homepage
    selected.append(home)
    seen.add(url_dedup_key(home))

    ranked: list[tuple[int, str]] = []
    for href, text in candidates:
        absolute = normalize_url(href, base_url=home)
        if absolute is None:
            continue
        if not is_same_domain(absolute, domain):
            continue
        key = url_dedup_key(absolute)
        if key in seen:
            continue
        score = score_link(absolute, text)
        if score <= 0:
            continue
        ranked.append((score, absolute))

    ranked.sort(key=lambda item: (-item[0], item[1]))
    for _, url in ranked:
        key = url_dedup_key(url)
        if key in seen:
            continue
        selected.append(url)
        seen.add(key)
        if len(selected) >= max_pages:
            break
    return selected


def crawl_domain(
    domain: str,
    settings: Settings,
    browser_manager: BrowserManager | None = None,
) -> DomainCrawlResult:
    """Crawl one domain inside an isolated error boundary."""
    raw_domain = (domain or "").strip() or domain
    logger.info("Processing domain: %s", raw_domain)

    try:
        normalized = normalize_domain(raw_domain)
    except InvalidURLError as exc:
        logger.warning("Invalid domain %r: %s", raw_domain, exc)
        return DomainCrawlResult.failed(raw_domain, str(exc))

    if browser_manager is None:
        return DomainCrawlResult.failed(
            normalized,
            "Browser manager is required to crawl a valid domain.",
        )

    homepage = homepage_url(normalized)
    context = None
    try:
        context = browser_manager.new_context()
        page = context.new_page()
        pages = _crawl_pages(page, normalized, homepage, settings)
        status, error = _status_from_pages(pages)
        logger.info(
            "Finished domain %s: status=%s pages=%s",
            normalized,
            status,
            len(pages),
        )
        return DomainCrawlResult(
            domain=normalized,
            pages=pages,
            processing_status=status,
            error_message=error,
        )
    except Exception as exc:
        message = _safe_error_message(exc)
        logger.error("Domain %s failed: %s", normalized, message, exc_info=True)
        return DomainCrawlResult.failed(normalized, message)
    finally:
        if context is not None:
            try:
                context.close()
            except Exception:
                logger.warning("Failed to close context for %s", normalized, exc_info=True)


def crawl_domains(
    domains: Iterable[str],
    settings: Settings,
    browser_manager: BrowserManager | None = None,
) -> list[DomainCrawlResult]:
    """Crawl many domains. One failure does not stop the rest."""
    owns_browser = browser_manager is None
    manager = browser_manager
    results: list[DomainCrawlResult] = []

    try:
        for domain in domains:
            try:
                needs_browser = True
                try:
                    normalize_domain(domain)
                except InvalidURLError:
                    needs_browser = False

                if needs_browser and manager is None:
                    manager = BrowserManager(settings).start()

                results.append(crawl_domain(domain, settings, manager))
            except Exception as exc:
                message = _safe_error_message(exc)
                logger.error(
                    "Unexpected error while processing %s: %s",
                    domain,
                    message,
                    exc_info=True,
                )
                results.append(DomainCrawlResult.failed((domain or "").strip() or str(domain), message))
        return results
    finally:
        if owns_browser and manager is not None:
            manager.close()


def _crawl_pages(
    page: Page,
    domain: str,
    homepage: str,
    settings: Settings,
) -> list[CrawledPage]:
    pages: list[CrawledPage] = []
    visited: set[str] = set()
    total_chars = 0

    first = _fetch_page(page, homepage, settings.page_timeout_ms)
    link_source_html = first.html
    first = _truncate_page(
        first,
        max_page_chars=MAX_RAW_HTML_CHARS_PER_PAGE,
        remaining_domain_chars=MAX_RAW_HTML_CHARS_PER_DOMAIN,
    )
    pages.append(first)
    visited.add(url_dedup_key(first.url or homepage))
    total_chars += first.html_chars
    logger.info(
        "Homepage crawled for %s: status=%s html_chars=%s",
        domain,
        first.status_code,
        first.html_chars,
    )

    if total_chars >= MAX_RAW_HTML_CHARS_PER_DOMAIN:
        logger.info("Reached raw HTML storage cap after homepage for %s", domain)
        return pages

    candidates = _collect_links(page, link_source_html, first.url or homepage)
    selected = select_relevant_urls(
        homepage=first.url or homepage,
        candidates=candidates,
        domain=domain,
        max_pages=settings.max_pages_per_domain,
    )

    for url in selected:
        if len(pages) >= settings.max_pages_per_domain:
            break
        if url_dedup_key(url) in visited:
            continue
        if total_chars >= MAX_RAW_HTML_CHARS_PER_DOMAIN:
            logger.info("Reached raw HTML storage cap for %s", domain)
            break

        time.sleep(PAGE_DELAY_SECONDS)
        crawled = _fetch_page(page, url, settings.page_timeout_ms)
        remaining = MAX_RAW_HTML_CHARS_PER_DOMAIN - total_chars
        crawled = _truncate_page(
            crawled,
            max_page_chars=MAX_RAW_HTML_CHARS_PER_PAGE,
            remaining_domain_chars=remaining,
        )
        pages.append(crawled)
        visited.add(url_dedup_key(crawled.url or url))
        total_chars += crawled.html_chars

        # A failed subpage must not stop the remaining URLs.
        if crawled.error_message:
            logger.warning(
                "Subpage error for %s (%s): %s",
                domain,
                crawled.url,
                crawled.error_message,
            )

    return pages


def _collect_links(page: Page, html: str, base_url: str) -> list[tuple[str, str]]:
    try:
        raw = page.evaluate(
            """() => Array.from(document.querySelectorAll("a[href]"))
                .slice(0, 200)
                .map(a => ({
                    href: a.href || "",
                    text: ((a.innerText || a.getAttribute("aria-label") || "") + "").trim()
                }))"""
        )
        if isinstance(raw, list):
            pairs: list[tuple[str, str]] = []
            for item in raw:
                if isinstance(item, dict):
                    pairs.append((str(item.get("href") or ""), str(item.get("text") or "")))
            if pairs:
                return pairs
    except Exception:
        logger.warning("Playwright link extraction failed for %s", base_url)
    return extract_links_from_html(html, base_url=base_url)


def _fetch_page(page: Page, url: str, timeout_ms: int) -> CrawledPage:
    logger.info("Visiting URL: %s", url)
    try:
        response = page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
        try:
            page.wait_for_load_state("networkidle", timeout=min(5000, timeout_ms))
        except PlaywrightTimeoutError:
            logger.info("Continued without networkidle for %s", url)

        status = response.status if response is not None else None
        final_url = page.url or url
        title = _safe_title(page)
        html = _safe_html(page)
        error = _status_error(status, title, html)
        logger.info(
            "Page status=%s url=%s title=%r html_chars=%s",
            status,
            final_url,
            title,
            len(html),
        )
        return CrawledPage(
            url=final_url,
            title=title,
            status_code=status,
            html=html,
            error_message=error,
        )
    except PlaywrightTimeoutError:
        logger.warning("Page timeout: %s", url)
        return CrawledPage(url=url, error_message="Page timed out")
    except Exception as exc:
        message = _safe_error_message(exc)
        logger.warning("Page failure for %s: %s", url, message)
        return CrawledPage(url=url, error_message=message)


def _safe_title(page: Page) -> str:
    try:
        return page.title() or ""
    except Exception:
        logger.warning("Could not read page title")
        return ""


def _safe_html(page: Page) -> str:
    try:
        return page.content() or ""
    except Exception:
        logger.warning("Could not read page HTML")
        return ""


def _status_error(status: int | None, title: str, html: str) -> str | None:
    errors: list[str] = []
    if status == 404:
        errors.append("HTTP 404 Not Found")
    elif status == 403:
        errors.append("HTTP 403 Forbidden")
    elif status is not None and status >= 400:
        errors.append(f"HTTP {status}")
    if _looks_like_bot_block(title, html, status):
        errors.append("Possible bot protection or challenge page")
    return "; ".join(errors) if errors else None


def _looks_like_bot_block(title: str, html: str, status: int | None) -> bool:
    if status in {401, 403, 429, 503}:
        snippet = f"{title} {html[:1500]}".lower()
        return any(hint in snippet for hint in BOT_HINTS) or status in {401, 403}
    snippet = f"{title} {html[:1500]}".lower()
    return any(hint in snippet for hint in BOT_HINTS)


def _truncate_page(
    page: CrawledPage,
    *,
    max_page_chars: int,
    remaining_domain_chars: int,
) -> CrawledPage:
    limit = max(0, min(max_page_chars, remaining_domain_chars))
    if len(page.html) > limit:
        page.html = page.html[:limit]
    return page


def _status_from_pages(pages: list[CrawledPage]) -> tuple[ProcessingStatus, str | None]:
    if not pages:
        return "failed", "No pages were crawled."
    usable = [page for page in pages if page.is_usable]
    if not usable:
        return "failed", pages[0].error_message or "No usable pages were crawled."
    if any(page.error_message for page in pages):
        return "partial_success", "One or more pages returned errors."
    return "success", None


def _safe_error_message(exc: BaseException) -> str:
    text = str(exc).splitlines()[0].strip() if str(exc) else exc.__class__.__name__
    if len(text) > 200:
        text = text[:197] + "..."
    return text or "Unexpected crawler error"


def main(argv: list[str] | None = None) -> None:
    """Manual runner: ``python -m app.crawler [domain ...]``."""
    configure_logging("INFO")
    logger_main = logging.getLogger("app.crawler")
    try:
        settings = load_settings()
    except ConfigError as exc:
        logger_main.error("%s", exc)
        raise SystemExit(1) from exc

    configure_logging(settings.log_level)
    domains = argv if argv is not None else sys.argv[1:]
    selected = list(domains) if domains else list(TARGET_DOMAINS)
    logger_main.info("Crawling domains: %s", selected)
    results = crawl_domains(selected, settings)
    payload = [result.summary() for result in results]
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
