"""Turn raw HTML into compact, readable text for later LLM extraction.

This module does not invent company facts. It only keeps visible text that was
already present in the HTML, then trims noise and size.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Iterable
from urllib.parse import urljoin

from bs4 import BeautifulSoup, FeatureNotFound, Tag

from app.config import Settings

logger = logging.getLogger(__name__)

DEFAULT_MAX_CHARS_PER_PAGE = 12000
DEFAULT_MAX_TOTAL_CHARS = 50000

REMOVE_TAGS = {
    "script",
    "style",
    "noscript",
    "iframe",
    "svg",
    "canvas",
    "video",
    "audio",
    "source",
    "template",
    "object",
    "embed",
    "link",
    "meta",
    "picture",
    "img",
    "path",
    "symbol",
    "use",
}

BLOCK_TAGS = {
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "p",
    "li",
    "dt",
    "dd",
    "th",
    "td",
    "blockquote",
    "pre",
}

COOKIE_HINTS = (
    "cookie",
    "consent",
    "gdpr",
    "onetrust",
    "cookiebot",
    "cc-banner",
    "osano",
    "privacy-banner",
    "cookie-banner",
)

CHROME_ROLES = {"navigation", "menu", "banner", "complementary"}
PROTECTED_TAGS = {"html", "body", "main", "article"}

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
ASSET_EMAIL_SUFFIXES = (".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".css", ".js")
LINKEDIN_SKIP = ("share", "login", "signup", "help.", "/legal", "feed")

MAIN_SELECTORS = (
    "main",
    "article",
    "[role=main]",
    "#main",
    "#content",
    ".main-content",
    "#main-content",
)


def clean_html(
    html: str | bytes | None,
    *,
    title: str = "",
    url: str = "",
    max_chars: int | None = None,
    settings: Settings | None = None,
) -> str:
    """Clean one HTML document into the spec's compact page format."""
    page_limit = _page_limit(max_chars, settings)
    raw = _as_text(html)
    title = " ".join((title or "").split())
    url = (url or "").strip()

    if not raw.strip():
        if not title and not url:
            return ""
        return _truncate(_format_page(title, url, ""), page_limit)

    try:
        soup = _parse_html(raw)
        if soup is None:
            return _truncate(_format_page(title, url, ""), page_limit)

        page_title = title or _tag_text(soup.title)
        meta_description = _meta_description(soup)

        _decompose_tags(soup, REMOVE_TAGS)
        emails, linkedin_urls = _harvest_contacts(soup, page_url=url)
        _decompose_chrome(soup)

        body = _extract_readable_text(_content_root(soup))
        body = _merge_extras(body, meta_description, emails, linkedin_urls)
        body = normalize_whitespace(body)
        document = _format_page(page_title, url, body)
        cleaned = _truncate(document, page_limit)
        logger.info(
            "Cleaned page url=%s input_chars=%s output_chars=%s",
            url or "(none)",
            len(raw),
            len(cleaned),
        )
        return cleaned
    except Exception:
        logger.warning("HTML cleaning failed for %s", url or "(no url)", exc_info=True)
        if not title and not url:
            return ""
        return _truncate(_format_page(title, url, ""), page_limit)


def clean_pages(
    pages: Iterable[Any],
    *,
    max_chars_per_page: int | None = None,
    max_total_chars: int | None = None,
    settings: Settings | None = None,
) -> str:
    """Clean several crawled pages and join them under the domain-wide cap."""
    page_limit = _page_limit(max_chars_per_page, settings)
    total_limit = _total_limit(max_total_chars, settings)

    chunks: list[str] = []
    total = 0
    for page in pages:
        remaining = total_limit - total
        if remaining <= 0:
            logger.info("Stopped cleaning more pages after reaching the domain content cap")
            break
        cleaned = clean_html(
            getattr(page, "html", "") or "",
            title=getattr(page, "title", "") or "",
            url=getattr(page, "url", "") or "",
            max_chars=min(page_limit, remaining),
        )
        if not cleaned:
            continue
        piece = cleaned if not chunks else "\n\n" + cleaned
        if len(piece) > remaining:
            piece = _truncate(piece, remaining)
        chunks.append(piece)
        total += len(piece)

    combined = "".join(chunks).strip()
    logger.info("Cleaned domain content chars=%s", len(combined))
    return combined


def normalize_whitespace(text: str) -> str:
    """Collapse messy spacing without deleting useful line breaks."""
    if not text:
        return ""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [" ".join(line.split()) for line in normalized.split("\n")]

    cleaned: list[str] = []
    pending_blank = False
    previous = None
    for line in lines:
        if not line:
            pending_blank = bool(cleaned)
            continue
        if previous is not None and line == previous:
            continue
        if pending_blank:
            cleaned.append("")
            pending_blank = False
        cleaned.append(line)
        previous = line
    return "\n".join(cleaned).strip()


def _page_limit(max_chars: int | None, settings: Settings | None) -> int:
    if max_chars is not None:
        return max(1, max_chars)
    if settings is not None:
        return settings.max_content_chars_per_page
    return DEFAULT_MAX_CHARS_PER_PAGE


def _total_limit(max_chars: int | None, settings: Settings | None) -> int:
    if max_chars is not None:
        return max(1, max_chars)
    if settings is not None:
        return settings.max_total_content_chars_per_domain
    return DEFAULT_MAX_TOTAL_CHARS


def _as_text(html: str | bytes | None) -> str:
    if html is None:
        return ""
    if isinstance(html, bytes):
        return html.decode("utf-8", errors="replace")
    return str(html)


def _parse_html(html: str) -> BeautifulSoup | None:
    for parser in ("lxml", "html.parser"):
        try:
            return BeautifulSoup(html, parser)
        except FeatureNotFound:
            continue
        except Exception:
            continue
    return None


def _decompose_tags(soup: BeautifulSoup, names: set[str]) -> None:
    for tag in soup.find_all(names):
        tag.decompose()


def _decompose_chrome(soup: BeautifulSoup) -> None:
    for tag in list(soup.find_all(True)):
        if not isinstance(tag, Tag):
            continue
        if getattr(tag, "decomposed", False) or tag.attrs is None:
            continue
        if tag.name in PROTECTED_TAGS:
            continue
        if _is_cookie_or_nav_chrome(tag):
            tag.decompose()


def _is_cookie_or_nav_chrome(tag: Tag) -> bool:
    if getattr(tag, "decomposed", False) or tag.attrs is None:
        return False
    identity = " ".join(
        [
            tag.name or "",
            str(tag.get("id") or ""),
            " ".join(tag.get("class") or []),
            str(tag.get("role") or ""),
            str(tag.get("aria-label") or ""),
        ]
    ).lower()

    if any(hint in identity for hint in COOKIE_HINTS):
        return True
    if tag.name == "nav" or tag.get("role") in CHROME_ROLES:
        return True
    if tag.name in {"header", "footer", "aside"} and _looks_like_nav_only(tag):
        return True
    if tag.get("hidden") is not None or str(tag.get("aria-hidden") or "").lower() == "true":
        return True
    style = str(tag.get("style") or "").lower()
    if "display:none" in style.replace(" ", "") or "visibility:hidden" in style.replace(" ", ""):
        return True
    return False


def _looks_like_nav_only(tag: Tag) -> bool:
    text = tag.get_text(" ", strip=True)
    lowered = text.lower()
    if "@" in text or "linkedin.com" in lowered:
        return False
    if any(word in lowered for word in ("founder", "ceo", "contact us", "sales@", "support@")):
        return False
    links = tag.find_all("a")
    if len(links) >= 8 and len(text) < 800:
        return True
    if len(links) >= 5 and len(text) < 200:
        return True
    return False


def _content_root(soup: BeautifulSoup) -> Tag:
    for selector in MAIN_SELECTORS:
        node = soup.select_one(selector)
        if isinstance(node, Tag) and len(node.get_text(" ", strip=True)) >= 40:
            return node
    body = soup.body
    if isinstance(body, Tag):
        return body
    return soup


def _extract_readable_text(root: Tag) -> str:
    blocks: list[str] = []
    for tag in root.find_all(BLOCK_TAGS):
        if not isinstance(tag, Tag):
            continue
        if tag.find_parent(BLOCK_TAGS):
            continue
        text = _tag_text(tag)
        if not text:
            continue
        name = tag.name or ""
        if name in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            hashes = "#" * min(int(name[1]), 6)
            blocks.append(f"{hashes} {text}")
        elif name == "li":
            blocks.append(f"- {text}")
        else:
            blocks.append(text)

    if not blocks:
        fallback = root.get_text("\n", strip=True)
        return fallback

    return "\n".join(blocks)


def _tag_text(tag: Tag | None) -> str:
    if tag is None:
        return ""
    return " ".join(tag.get_text(" ", strip=True).split())


def _meta_description(soup: BeautifulSoup) -> str:
    tag = soup.find("meta", attrs={"name": re.compile(r"^description$", re.I)})
    if not isinstance(tag, Tag):
        tag = soup.find("meta", attrs={"property": re.compile(r"^og:description$", re.I)})
    if not isinstance(tag, Tag):
        return ""
    content = tag.get("content")
    if isinstance(content, list):
        content = " ".join(str(item) for item in content)
    return " ".join(str(content or "").split())


def _harvest_contacts(soup: BeautifulSoup, *, page_url: str) -> tuple[list[str], list[str]]:
    emails: list[str] = []
    linkedin_urls: list[str] = []
    seen_emails: set[str] = set()
    seen_links: set[str] = set()

    for anchor in soup.find_all("a", href=True):
        if not isinstance(anchor, Tag):
            continue
        href = str(anchor.get("href") or "").strip()
        if href.lower().startswith("mailto:"):
            address = href.split(":", 1)[1].split("?", 1)[0].strip()
            _add_email(address, emails, seen_emails)
            continue
        absolute = urljoin(page_url, href) if page_url else href
        if _is_profile_linkedin(absolute):
            key = absolute.rstrip("/").lower()
            if key not in seen_links:
                seen_links.add(key)
                linkedin_urls.append(absolute)

    for match in EMAIL_RE.findall(soup.get_text(" ")):
        _add_email(match, emails, seen_emails)

    return emails, linkedin_urls


def _add_email(address: str, emails: list[str], seen: set[str]) -> None:
    cleaned = address.strip().strip(".,;:()<>[]")
    if not cleaned or not _is_likely_email(cleaned):
        return
    key = cleaned.lower()
    if key in seen:
        return
    seen.add(key)
    emails.append(cleaned)


def _is_likely_email(address: str) -> bool:
    if address.count("@") != 1:
        return False
    domain = address.rsplit("@", 1)[-1].lower()
    return not domain.endswith(ASSET_EMAIL_SUFFIXES)


def _is_profile_linkedin(url: str) -> bool:
    lowered = url.lower()
    if "linkedin.com" not in lowered:
        return False
    if any(part in lowered for part in LINKEDIN_SKIP):
        return False
    return "/in/" in lowered or "/company/" in lowered


def _merge_extras(
    body: str,
    meta_description: str,
    emails: list[str],
    linkedin_urls: list[str],
) -> str:
    parts: list[str] = []
    lowered = body.lower()
    if meta_description and meta_description.lower() not in lowered:
        parts.append(meta_description)
    if body:
        parts.append(body)

    missing_emails = [item for item in emails if item.lower() not in lowered]
    missing_links = [item for item in linkedin_urls if item.lower() not in lowered]
    if missing_emails or missing_links:
        extras = ["Contact details:"]
        extras.extend(missing_emails)
        extras.extend(missing_links)
        parts.append("\n".join(extras))
    return "\n\n".join(part for part in parts if part)


def _format_page(title: str, url: str, content: str) -> str:
    lines: list[str] = []
    if title:
        lines.append(f"PAGE TITLE: {title}")
    if url:
        lines.append(f"URL: {url}")
    if lines:
        lines.append("")
    lines.append("CONTENT:")
    lines.append(content.strip() if content else "")
    return "\n".join(lines).strip()


def _truncate(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    marker = "\n[Truncated]"
    budget = max(1, max_chars - len(marker))
    sliced = text[:budget]
    break_at = max(sliced.rfind("\n"), sliced.rfind(" "))
    if break_at >= int(budget * 0.8):
        sliced = sliced[:break_at]
    return sliced.rstrip() + marker
