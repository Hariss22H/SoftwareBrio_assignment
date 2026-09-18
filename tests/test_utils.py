"""Unit tests for URL and domain normalization."""

from __future__ import annotations

import pytest

from app.utils import (
    InvalidURLError,
    homepage_url,
    is_same_domain,
    normalize_domain,
    normalize_url,
    url_dedup_key,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("postman.com", "postman.com"),
        ("https://postman.com", "postman.com"),
        ("https://www.postman.com/", "postman.com"),
        ("http://Postman.com/about", "postman.com"),
        ("postman.com/", "postman.com"),
        ("  supabase.com  ", "supabase.com"),
        ("vapi.ai", "vapi.ai"),
        ("https://www.vapi.ai/", "vapi.ai"),
    ],
)
def test_normalize_domain_accepts_common_shapes(raw: str, expected: str) -> None:
    assert normalize_domain(raw) == expected


@pytest.mark.parametrize(
    "raw",
    ["", "   ", "not a domain", "http://", "javascript:void(0)", "localhost"],
)
def test_normalize_domain_rejects_malformed_values(raw: str) -> None:
    with pytest.raises(InvalidURLError):
        normalize_domain(raw)


def test_homepage_url_uses_https() -> None:
    assert homepage_url("www.postman.com/") == "https://postman.com"


def test_normalize_url_resolves_relative_links() -> None:
    assert (
        normalize_url("/about", base_url="https://postman.com")
        == "https://postman.com/about"
    )


def test_normalize_url_rejects_mailto_javascript_and_assets() -> None:
    base = "https://postman.com"
    assert normalize_url("mailto:sales@postman.com", base_url=base) is None
    assert normalize_url("javascript:void(0)", base_url=base) is None
    assert normalize_url("/brand/logo.png", base_url=base) is None
    assert normalize_url("#pricing", base_url=base) is None


def test_www_and_trailing_slash_are_the_same_page() -> None:
    left = url_dedup_key("https://www.postman.com/about/")
    right = url_dedup_key("https://postman.com/about")
    assert left == right


def test_same_domain_allows_www_but_not_other_hosts() -> None:
    assert is_same_domain("https://www.postman.com/company", "postman.com")
    assert is_same_domain("https://postman.com/pricing", "https://www.postman.com")
    assert not is_same_domain("https://docs.postman.com/docs", "postman.com")
    assert not is_same_domain("https://supabase.com/about", "postman.com")
