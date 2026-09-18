"""Unit tests for HTML content cleaning."""

from __future__ import annotations

from types import SimpleNamespace

from app.config import Settings
from app.content_cleaner import clean_html, clean_pages, normalize_whitespace


ABOUT_HTML = """
<html>
  <head>
    <title>About Acme</title>
    <meta name="description" content="Acme builds API tools for developers.">
    <script>window.tracking = 'do-not-keep';</script>
    <style>body { color: red; }</style>
  </head>
  <body>
    <nav>
      <a href="/">Home</a>
      <a href="/login">Login</a>
      <a href="/products">Products</a>
    </nav>
    <div id="onetrust-banner-sdk">Accept all cookies to continue.</div>
    <main>
      <h1>About Acme</h1>
      <p>Acme builds API tools for developers.</p>
      <h2>Leadership</h2>
      <ul>
        <li>Ada Lovelace, Chief Executive Officer</li>
        <li>Grace Hopper, Chief Technology Officer</li>
      </ul>
      <p>Contact sales at sales@acme.com or support@acme.com.</p>
      <p>
        <a href="https://www.linkedin.com/in/ada-lovelace">Ada on LinkedIn</a>
      </p>
    </main>
    <iframe src="https://ads.example/tracker"></iframe>
    <svg><circle cx="1" cy="1" r="1"></circle></svg>
    <noscript>Enable JavaScript</noscript>
  </body>
</html>
"""


def test_clean_html_removes_noise_and_keeps_company_facts() -> None:
    cleaned = clean_html(
        ABOUT_HTML,
        title="About Acme",
        url="https://acme.com/about",
    )

    assert "PAGE TITLE: About Acme" in cleaned
    assert "URL: https://acme.com/about" in cleaned
    assert "CONTENT:" in cleaned
    assert "Acme builds API tools for developers." in cleaned
    assert "Ada Lovelace, Chief Executive Officer" in cleaned
    assert "Grace Hopper, Chief Technology Officer" in cleaned
    assert "sales@acme.com" in cleaned
    assert "support@acme.com" in cleaned
    assert "https://www.linkedin.com/in/ada-lovelace" in cleaned
    assert "window.tracking" not in cleaned
    assert "color: red" not in cleaned
    assert "Accept all cookies" not in cleaned
    assert "Enable JavaScript" not in cleaned
    assert "ads.example/tracker" not in cleaned
    assert "Login" not in cleaned


def test_normalize_whitespace_collapses_blank_lines_and_spaces() -> None:
    messy = "Hello    world\n\n\n\nHello    world\nNext   line"
    assert normalize_whitespace(messy) == "Hello world\n\nNext line"


def test_empty_and_missing_html_do_not_crash() -> None:
    assert clean_html(None) == ""
    assert clean_html("") == ""
    assert clean_html("   ") == ""
    empty_with_url = clean_html("", title="Empty", url="https://acme.com")
    assert "PAGE TITLE: Empty" in empty_with_url
    assert "URL: https://acme.com" in empty_with_url
    assert "CONTENT:" in empty_with_url


def test_malformed_html_is_handled_safely() -> None:
    cleaned = clean_html("<div><p>Broken", title="Broken page", url="https://acme.com")
    assert "Broken" in cleaned
    assert "PAGE TITLE: Broken page" in cleaned


def test_page_limit_truncates_long_content() -> None:
    html = "<p>" + ("Useful company text. " * 400) + "</p>"
    cleaned = clean_html(html, title="Long", url="https://acme.com", max_chars=180)

    assert len(cleaned) <= 180
    assert cleaned.endswith("[Truncated]")
    assert "PAGE TITLE: Long" in cleaned


def test_clean_pages_respects_settings_limits(settings: Settings) -> None:
    tiny = settings.model_copy(
        update={
            "max_content_chars_per_page": 90,
            "max_total_content_chars_per_domain": 150,
        }
    )
    pages = [
        SimpleNamespace(
            title="Home",
            url="https://acme.com",
            html="<h1>Acme</h1><p>" + ("Product platform. " * 40) + "</p>",
        ),
        SimpleNamespace(
            title="About",
            url="https://acme.com/about",
            html="<h1>About</h1><p>" + ("Company story. " * 40) + "</p>",
        ),
    ]

    combined = clean_pages(pages, settings=tiny)
    assert len(combined) <= 150
    assert "PAGE TITLE: Home" in combined
    assert combined.count("[Truncated]") >= 1


def test_footer_email_is_preserved_when_footer_looks_like_nav() -> None:
    html = """
    <html><body>
      <main><p>Acme makes developer tools.</p></main>
      <footer>
        <a href="/">Home</a>
        <a href="/docs">Docs</a>
        <a href="/blog">Blog</a>
        <a href="/legal">Legal</a>
        <a href="/privacy">Privacy</a>
        <a href="/status">Status</a>
        <a href="/careers">Careers</a>
        <a href="/press">Press</a>
        <a href="mailto:hello@acme.com">Email us</a>
      </footer>
    </body></html>
    """
    cleaned = clean_html(html, url="https://acme.com")
    assert "Acme makes developer tools." in cleaned
    assert "hello@acme.com" in cleaned


def test_cleaner_does_not_invent_missing_facts() -> None:
    cleaned = clean_html("<main><p>We make calendars.</p></main>", url="https://acme.com")
    assert "We make calendars." in cleaned
    assert "sales@acme.com" not in cleaned
    assert "Chief Executive Officer" not in cleaned
    assert "linkedin.com" not in cleaned
