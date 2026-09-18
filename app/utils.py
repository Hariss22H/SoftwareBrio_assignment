"""Small shared helpers such as URL and domain normalization."""

from __future__ import annotations

import re
from urllib.parse import urljoin, urlparse, urlunparse

_DOMAIN_RE = re.compile(r"^[a-z0-9-]+(?:\.[a-z0-9-]+)+$", re.IGNORECASE)
ASSET_EXTENSIONS = {
    ".css",
    ".gif",
    ".ico",
    ".jpeg",
    ".jpg",
    ".js",
    ".mp3",
    ".mp4",
    ".pdf",
    ".png",
    ".svg",
    ".webp",
    ".woff",
    ".woff2",
    ".xml",
    ".zip",
}
SKIP_SCHEMES = {"javascript", "mailto", "tel", "data", "sms", "blob"}


class InvalidURLError(ValueError):
    """Raised when a domain or URL cannot be normalized safely."""


def _strip_www(host: str) -> str:
    host = host.lower().rstrip(".")
    return host[4:] if host.startswith("www.") else host


def normalize_domain(value: str) -> str:
    """Return a bare hostname such as ``postman.com``.

    Accepts values with or without ``https://``, ``www.``, and trailing slashes.
    """
    cleaned = (value or "").strip()
    if not cleaned:
        raise InvalidURLError("Domain is empty.")
    if any(char.isspace() for char in cleaned):
        raise InvalidURLError(f"Domain is malformed: {value!r}.")

    if "://" not in cleaned:
        cleaned = "https://" + cleaned

    parsed = urlparse(cleaned)
    host = parsed.netloc or parsed.path.split("/")[0]
    host = host.strip().lower().rstrip(".")
    if not host:
        raise InvalidURLError(f"Domain is malformed: {value!r}.")

    if host.count(":") == 1:
        name, port_text = host.rsplit(":", 1)
        if port_text.isdigit():
            host = name

    host = _strip_www(host)
    if not _DOMAIN_RE.match(host):
        raise InvalidURLError(f"Domain is malformed: {value!r}.")
    return host


def homepage_url(domain: str) -> str:
    """Build the HTTPS homepage URL for a normalized domain."""
    return f"https://{normalize_domain(domain)}"


def canonical_url(url: str) -> str:
    """Drop fragments and trailing slashes so the same page is not visited twice."""
    parsed = urlparse(url.strip())
    scheme = (parsed.scheme or "https").lower()
    netloc = parsed.netloc.lower()
    path = parsed.path.rstrip("/")
    return urlunparse((scheme, netloc, path, "", parsed.query, ""))


def host_from_url(url: str) -> str:
    parsed = urlparse(url)
    host = (parsed.netloc or "").lower()
    if host.count(":") == 1 and host.rsplit(":", 1)[1].isdigit():
        host = host.rsplit(":", 1)[0]
    return _strip_www(host)


def is_same_domain(url: str, domain: str) -> bool:
    """Return True when *url* belongs to *domain*, treating www as the same host."""
    try:
        target = normalize_domain(domain)
    except InvalidURLError:
        target = _strip_www(domain)
    return host_from_url(url) == target


def is_http_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def is_asset_url(url: str) -> bool:
    path = urlparse(url).path.lower()
    return any(path.endswith(ext) for ext in ASSET_EXTENSIONS)


def normalize_url(value: str, *, base_url: str | None = None) -> str | None:
    """Resolve a href into an absolute http(s) URL, or return None if it is unusable."""
    raw = (value or "").strip()
    if not raw:
        return None

    lowered = raw.lower()
    if lowered.split(":", 1)[0] in SKIP_SCHEMES:
        return None
    if lowered.startswith("#"):
        return None

    absolute = urljoin(base_url or "", raw) if base_url else raw
    if "://" not in absolute:
        return None
    parsed = urlparse(absolute)
    if parsed.scheme.lower() not in {"http", "https"}:
        return None
    if not parsed.netloc:
        return None
    if is_asset_url(absolute):
        return None
    return canonical_url(absolute)


def url_dedup_key(url: str) -> str:
    """Identity key that treats www and trailing slashes as the same page."""
    parsed = urlparse(canonical_url(url))
    host = _strip_www(parsed.netloc)
    return f"{host}{parsed.path.lower()}?{parsed.query.lower()}"
