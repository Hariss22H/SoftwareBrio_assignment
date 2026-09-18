"""Unit tests for crawler link scoring, selection, and domain isolation."""

from __future__ import annotations

from app.config import Settings
from app.crawler import (
    MAX_RAW_HTML_CHARS_PER_DOMAIN,
    MAX_RAW_HTML_CHARS_PER_PAGE,
    CrawledPage,
    crawl_domain,
    crawl_domains,
    extract_links_from_html,
    score_link,
    select_relevant_urls,
    _truncate_page,
)


HOMEPAGE = "https://postman.com"
SAMPLE_HTML = """
<html>
  <body>
    <a href="/about">About Us</a>
    <a href="/pricing">Pricing</a>
    <a href="https://postman.com/company/team">Leadership team</a>
    <a href="https://other.example/about">External about</a>
    <a href="/blog/release-notes">Release notes</a>
    <a href="mailto:sales@postman.com">Email</a>
    <a href="/images/hero.png">Hero</a>
  </body>
</html>
"""


def test_relevant_paths_score_higher_than_blog_posts() -> None:
    about = score_link("https://postman.com/about", "About Us")
    pricing = score_link("https://postman.com/pricing", "Plans")
    blog = score_link("https://postman.com/blog/release-notes", "Release notes")

    assert about > 0
    assert pricing > 0
    assert about > blog
    assert blog == 0


def test_about_and_team_outrank_careers_pages() -> None:
    about = score_link("https://postman.com/company/about-postman", "About")
    team = score_link("https://postman.com/company/team", "Leadership team")
    contact = score_link("https://postman.com/company/contact-us", "Contact")
    careers = score_link("https://postman.com/company/careers", "Careers")

    assert about > careers
    assert team > careers
    assert contact > careers


def test_select_relevant_urls_prefers_intel_pages_over_careers() -> None:
    html = """
    <html><body>
      <a href="/careers">Careers</a>
      <a href="/about">About Us</a>
      <a href="/company/team">Team</a>
      <a href="/contact">Contact</a>
    </body></html>
    """
    candidates = extract_links_from_html(html, base_url=HOMEPAGE)
    selected = select_relevant_urls(
        homepage=HOMEPAGE,
        candidates=candidates,
        domain="postman.com",
        max_pages=4,
    )

    assert selected[0] == HOMEPAGE
    assert "https://postman.com/about" in selected
    assert "https://postman.com/company/team" in selected
    assert "https://postman.com/contact" in selected
    assert "https://postman.com/careers" not in selected



def test_select_relevant_urls_keeps_homepage_and_same_domain_hits() -> None:
    candidates = extract_links_from_html(SAMPLE_HTML, base_url=HOMEPAGE)
    selected = select_relevant_urls(
        homepage=HOMEPAGE,
        candidates=candidates,
        domain="postman.com",
        max_pages=6,
    )

    assert selected[0] == HOMEPAGE
    assert "https://postman.com/about" in selected
    assert "https://postman.com/pricing" in selected
    assert "https://postman.com/company/team" in selected
    assert "https://other.example/about" not in selected
    assert "https://postman.com/blog/release-notes" not in selected
    assert all(not item.startswith("mailto:") for item in selected)
    assert len(selected) <= 6


def test_select_relevant_urls_respects_max_pages() -> None:
    candidates = extract_links_from_html(SAMPLE_HTML, base_url=HOMEPAGE)
    selected = select_relevant_urls(
        homepage=HOMEPAGE,
        candidates=candidates,
        domain="postman.com",
        max_pages=2,
    )

    assert len(selected) == 2
    assert selected[0] == HOMEPAGE


def test_invalid_domain_fails_without_a_browser(settings: Settings) -> None:
    result = crawl_domain("not a domain", settings, browser_manager=None)

    assert result.processing_status == "failed"
    assert result.pages == []
    assert result.error_message is not None


def test_failed_domain_does_not_stop_the_batch(settings: Settings) -> None:
    results = crawl_domains(
        ["not a domain", "", "also invalid domain"],
        settings,
        browser_manager=None,
    )

    assert len(results) == 3
    assert all(item.processing_status == "failed" for item in results)
    assert {item.domain for item in results} == {"not a domain", "", "also invalid domain"}


def test_raw_html_is_not_clipped_at_cleaned_text_limit() -> None:
    html = "x" * 50_000
    page = CrawledPage(url="https://postman.com/", html=html)
    kept = _truncate_page(
        page,
        max_page_chars=MAX_RAW_HTML_CHARS_PER_PAGE,
        remaining_domain_chars=MAX_RAW_HTML_CHARS_PER_DOMAIN,
    )
    assert len(kept.html) == 50_000


def test_raw_html_storage_is_still_bounded() -> None:
    html = "y" * (MAX_RAW_HTML_CHARS_PER_PAGE + 25)
    page = CrawledPage(url="https://postman.com/", html=html)
    kept = _truncate_page(
        page,
        max_page_chars=MAX_RAW_HTML_CHARS_PER_PAGE,
        remaining_domain_chars=MAX_RAW_HTML_CHARS_PER_DOMAIN,
    )
    assert len(kept.html) == MAX_RAW_HTML_CHARS_PER_PAGE
