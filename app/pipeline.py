"""Run the full domain-by-domain enrichment flow and write output JSON."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.browser import BrowserManager
from app.config import PROJECT_ROOT, Settings
from app.content_cleaner import clean_pages
from app.crawler import TARGET_DOMAINS, DomainCrawlResult, crawl_domain
from app.llm_extractor import extract_company_intelligence
from app.models import CompanyIntelligence

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "output" / "output.json"
SECRET_RE = re.compile(r"sk-[A-Za-z0-9_\-]+")

CrawlFn = Callable[..., DomainCrawlResult]
CleanFn = Callable[..., str]
ExtractFn = Callable[..., CompanyIntelligence]


@dataclass
class PipelineResult:
    """In-memory pipeline outcome plus the path of the written JSON file."""

    generated_at: str
    results: list[CompanyIntelligence]
    output_path: Path

    @property
    def total_domains(self) -> int:
        return len(self.results)

    @property
    def failed_domains(self) -> int:
        return sum(1 for item in self.results if item.processing_status == "failed")

    @property
    def successful_domains(self) -> int:
        return self.total_domains - self.failed_domains

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "total_domains": self.total_domains,
            "successful_domains": self.successful_domains,
            "failed_domains": self.failed_domains,
            "results": [_public_record(item) for item in self.results],
        }


def run_pipeline(
    domains: Sequence[str] | None = None,
    settings: Settings | None = None,
    *,
    output_path: Path | None = None,
    browser_manager: BrowserManager | None = None,
    crawl_domain_fn: CrawlFn = crawl_domain,
    clean_pages_fn: CleanFn = clean_pages,
    extract_fn: ExtractFn = extract_company_intelligence,
) -> PipelineResult:
    """Crawl, clean, extract, and save one result per domain.

    A failure in one domain does not stop the remaining domains. The JSON file
    is written even when some or all domains fail.
    """
    if settings is None:
        raise ValueError("Settings are required to run the pipeline.")

    selected = list(domains) if domains else list(TARGET_DOMAINS)
    destination = output_path or DEFAULT_OUTPUT_PATH
    logger.info("Pipeline starting for %s domain(s): %s", len(selected), selected)
    logger.info("Configuration: %s", settings.redacted_dict())

    owns_browser = False
    manager = browser_manager
    records: list[CompanyIntelligence] = []

    try:
        for domain in selected:
            if manager is None and crawl_domain_fn is crawl_domain:
                manager = BrowserManager(settings).start()
                owns_browser = True
            records.append(
                _process_domain(
                    domain,
                    settings,
                    browser_manager=manager,
                    crawl_domain_fn=crawl_domain_fn,
                    clean_pages_fn=clean_pages_fn,
                    extract_fn=extract_fn,
                )
            )
    finally:
        if owns_browser and manager is not None:
            manager.close()

    generated_at = (
        datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    )
    result = PipelineResult(
        generated_at=generated_at,
        results=records,
        output_path=destination,
    )
    save_output(result)
    logger.info(
        "Pipeline finished: total=%s successful=%s failed=%s output=%s",
        result.total_domains,
        result.successful_domains,
        result.failed_domains,
        destination,
    )
    return result


def save_output(result: PipelineResult) -> Path:
    """Write UTF-8 JSON without API keys or raw HTML."""
    payload = result.to_dict()
    result.output_path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    result.output_path.write_text(text, encoding="utf-8")
    logger.info("Wrote output JSON: %s", result.output_path)
    return result.output_path


def _process_domain(
    domain: str,
    settings: Settings,
    *,
    browser_manager: BrowserManager | None,
    crawl_domain_fn: CrawlFn,
    clean_pages_fn: CleanFn,
    extract_fn: ExtractFn,
) -> CompanyIntelligence:
    logger.info("Pipeline processing domain: %s", domain)
    try:
        crawled = crawl_domain_fn(domain, settings, browser_manager)
    except Exception as exc:
        message = _safe_error_message(exc)
        logger.error("Crawler crashed for %s: %s", domain, message, exc_info=True)
        return _failed(_domain_label(domain), message, [])

    label = crawled.domain or _domain_label(domain)
    crawled_urls = list(crawled.source_urls)
    logger.info(
        "Pipeline crawled %s: pages=%s status=%s",
        label,
        crawled.pages_crawled,
        crawled.processing_status,
    )

    try:
        cleaned = clean_pages_fn(crawled.pages, settings=settings)
    except Exception as exc:
        message = f"Content cleaning failed: {_safe_error_message(exc)}"
        logger.error("Cleaning crashed for %s: %s", label, message, exc_info=True)
        return _failed(label, message, crawled_urls)

    logger.info("Pipeline cleaned %s: content_chars=%s", label, len(cleaned or ""))
    if not (cleaned or "").strip():
        message = crawled.error_message or "No usable website content was extracted."
        logger.warning("Pipeline found empty content for %s", label)
        return _failed(label, message, crawled_urls)

    source_urls = filter_source_urls(crawled_urls, cleaned)
    logger.info("Pipeline using %s source URL(s) for %s", len(source_urls), label)

    try:
        record = extract_fn(
            domain=label,
            cleaned_content=cleaned,
            settings=settings,
            source_urls=source_urls,
        )
    except Exception as exc:
        message = f"LLM extraction failed: {_safe_error_message(exc)}"
        logger.error("Extractor crashed for %s: %s", label, message, exc_info=True)
        return _failed(label, message, source_urls)

    finalized = record.model_copy(
        update={
            "domain": label,
            "source_urls": source_urls,
        }
    )
    logger.info(
        "Pipeline finished domain %s: status=%s confidence=%s",
        label,
        finalized.processing_status,
        finalized.confidence_score,
    )
    return finalized


def _public_record(record: CompanyIntelligence) -> dict[str, Any]:
    data = record.model_dump(mode="json")
    data.pop("html", None)
    if data.get("error_message"):
        data["error_message"] = _safe_error_message(data["error_message"])
    return data


def filter_source_urls(crawled_urls: Iterable[str], cleaned_content: str) -> list[str]:
    """Keep crawled URLs that appear in the cleaned text sent to the LLM.

    Never invent URLs. If none of the crawled URLs are visible in the cleaned
    text, keep the original crawled list.
    """
    ordered = [url for url in crawled_urls if url]
    if not ordered:
        return []
    haystack = cleaned_content or ""
    used: list[str] = []
    for url in ordered:
        variants = (url, url.rstrip("/"))
        if any(variant and variant in haystack for variant in variants):
            used.append(url)
    return used or ordered


def _failed(domain: str, message: str, source_urls: Iterable[str]) -> CompanyIntelligence:
    record = CompanyIntelligence.failure(domain, _safe_error_message(message))
    urls = [url for url in source_urls if url]
    if urls:
        return record.model_copy(update={"source_urls": urls})
    return record


def _domain_label(domain: str) -> str:
    cleaned = (domain or "").strip()
    return cleaned or "unknown-domain"


def _safe_error_message(exc: BaseException | str) -> str:
    raw = str(exc).splitlines()[0].strip() if str(exc) else exc.__class__.__name__
    cleaned = SECRET_RE.sub("[redacted]", raw)
    if len(cleaned) > 200:
        cleaned = cleaned[:197] + "..."
    return cleaned or "Unexpected pipeline error."
