"""Unit tests for the end-to-end pipeline. These tests never crawl or call OpenAI."""

from __future__ import annotations

import json
from pathlib import Path

from app.crawler import TARGET_DOMAINS, CrawledPage, DomainCrawlResult
from app.models import CompanyIntelligence
from app.pipeline import filter_source_urls, run_pipeline


def _page(domain: str, html: str = "<p>Acme builds API tools for developers.</p>") -> CrawledPage:
    return CrawledPage(
        url=f"https://{domain}/about",
        title="About",
        status_code=200,
        html=html,
    )


def _crawl_ok(domain: str, settings: object, browser_manager: object = None) -> DomainCrawlResult:
    label = domain.replace("https://", "").rstrip("/")
    return DomainCrawlResult(
        domain=label,
        pages=[_page(label)],
        processing_status="success",
    )


def _clean_ok(pages: object, settings: object = None, **kwargs: object) -> str:
    first = list(pages)[0]
    return f"PAGE TITLE: About\nURL: {first.url}\n\nCONTENT:\nAcme builds API tools for developers."


def _extract_ok(
    *,
    domain: str,
    cleaned_content: str,
    settings: object,
    source_urls: list[str] | None = None,
    **kwargs: object,
) -> CompanyIntelligence:
    return CompanyIntelligence(
        domain=domain,
        company_name="Acme",
        company_overview="Acme builds API tools for developers. Teams use it to ship APIs faster.",
        target_audience="Developers building backend applications",
        contact_points=[],
        team_members=[],
        confidence_score=0.72,
        source_urls=source_urls or [],
        processing_status="success",
        error_message=None,
    )


def test_successful_orchestration_writes_valid_json(settings: object, tmp_path: Path) -> None:
    output_path = tmp_path / "output.json"
    calls = {"crawl": [], "clean": 0, "extract": []}

    def crawl_fn(domain: str, settings: object, browser_manager: object = None) -> DomainCrawlResult:
        calls["crawl"].append(domain)
        return _crawl_ok(domain, settings, browser_manager)

    def clean_fn(pages: object, settings: object = None, **kwargs: object) -> str:
        calls["clean"] += 1
        return _clean_ok(pages, settings)

    def extract_fn(**kwargs: object) -> CompanyIntelligence:
        calls["extract"].append(kwargs["domain"])
        return _extract_ok(**kwargs)

    result = run_pipeline(
        domains=["postman.com", "supabase.com", "vapi.ai"],
        settings=settings,
        output_path=output_path,
        crawl_domain_fn=crawl_fn,
        clean_pages_fn=clean_fn,
        extract_fn=extract_fn,
    )

    assert calls["crawl"] == ["postman.com", "supabase.com", "vapi.ai"]
    assert calls["clean"] == 3
    assert calls["extract"] == ["postman.com", "supabase.com", "vapi.ai"]
    assert result.total_domains == 3
    assert result.successful_domains == 3
    assert result.failed_domains == 0
    assert output_path.exists()

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["total_domains"] == 3
    assert payload["successful_domains"] == 3
    assert payload["failed_domains"] == 0
    assert "generated_at" in payload
    assert payload["generated_at"].endswith("Z")
    assert [item["domain"] for item in payload["results"]] == [
        "postman.com",
        "supabase.com",
        "vapi.ai",
    ]
    for item in payload["results"]:
        CompanyIntelligence.model_validate(item)
        assert item["processing_status"] == "success"
        assert item["source_urls"] == [f"https://{item['domain']}/about"]
        assert "html" not in item
        assert "openai_api_key" not in json.dumps(payload)
        assert "sk-" not in json.dumps(payload)


def test_one_domain_failure_does_not_stop_others(settings: object, tmp_path: Path) -> None:
    output_path = tmp_path / "output.json"

    def crawl_fn(domain: str, settings: object, browser_manager: object = None) -> DomainCrawlResult:
        if domain == "supabase.com":
            raise RuntimeError("simulated crawler crash")
        return _crawl_ok(domain, settings, browser_manager)

    result = run_pipeline(
        domains=["postman.com", "supabase.com", "vapi.ai"],
        settings=settings,
        output_path=output_path,
        crawl_domain_fn=crawl_fn,
        clean_pages_fn=_clean_ok,
        extract_fn=_extract_ok,
    )

    assert result.total_domains == 3
    assert result.failed_domains == 1
    assert result.successful_domains == 2
    by_domain = {item.domain: item for item in result.results}
    assert by_domain["supabase.com"].processing_status == "failed"
    assert by_domain["supabase.com"].error_message == "simulated crawler crash"
    assert by_domain["postman.com"].processing_status == "success"
    assert by_domain["vapi.ai"].processing_status == "success"

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["failed_domains"] == 1
    assert payload["successful_domains"] == 2


def test_empty_content_skips_extractor(settings: object, tmp_path: Path) -> None:
    output_path = tmp_path / "output.json"
    extract_calls: list[str] = []

    def crawl_fn(domain: str, settings: object, browser_manager: object = None) -> DomainCrawlResult:
        return DomainCrawlResult(
            domain=domain,
            pages=[_page(domain, html="")],
            processing_status="failed",
            error_message="Homepage timed out",
        )

    def extract_fn(**kwargs: object) -> CompanyIntelligence:
        extract_calls.append(str(kwargs["domain"]))
        raise AssertionError("Extractor should not run for empty content")

    result = run_pipeline(
        domains=["vapi.ai"],
        settings=settings,
        output_path=output_path,
        crawl_domain_fn=crawl_fn,
        clean_pages_fn=lambda pages, settings=None, **kwargs: "",
        extract_fn=extract_fn,
    )

    assert extract_calls == []
    record = result.results[0]
    assert record.domain == "vapi.ai"
    assert record.processing_status == "failed"
    assert record.company_name is None
    assert record.contact_points == []
    assert record.team_members == []
    assert record.source_urls == ["https://vapi.ai/about"]
    assert "Homepage timed out" in (record.error_message or "")


def test_extractor_failure_still_writes_json(settings: object, tmp_path: Path) -> None:
    output_path = tmp_path / "output.json"

    def extract_fn(**kwargs: object) -> CompanyIntelligence:
        if kwargs["domain"] == "postman.com":
            raise RuntimeError("OpenAI timeout with sk-should-not-leak")
        return _extract_ok(**kwargs)

    result = run_pipeline(
        domains=["postman.com", "vapi.ai"],
        settings=settings,
        output_path=output_path,
        crawl_domain_fn=_crawl_ok,
        clean_pages_fn=_clean_ok,
        extract_fn=extract_fn,
    )

    assert result.results[0].processing_status == "failed"
    assert result.results[1].processing_status == "success"
    assert "sk-should-not-leak" not in (result.results[0].error_message or "")
    assert result.results[0].source_urls == ["https://postman.com/about"]

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    dumped = json.dumps(payload)
    assert "sk-should-not-leak" not in dumped
    assert payload["results"][0]["processing_status"] == "failed"
    assert payload["results"][1]["domain"] == "vapi.ai"
    CompanyIntelligence.model_validate(payload["results"][0])
    CompanyIntelligence.model_validate(payload["results"][1])


def test_default_domains_are_the_assignment_targets(settings: object, tmp_path: Path) -> None:
    seen: list[str] = []

    def crawl_fn(domain: str, settings: object, browser_manager: object = None) -> DomainCrawlResult:
        seen.append(domain)
        return _crawl_ok(domain, settings, browser_manager)

    run_pipeline(
        domains=None,
        settings=settings,
        output_path=tmp_path / "output.json",
        crawl_domain_fn=crawl_fn,
        clean_pages_fn=_clean_ok,
        extract_fn=_extract_ok,
    )

    assert seen == TARGET_DOMAINS


def test_filter_source_urls_keeps_only_urls_in_cleaned_text() -> None:
    crawled = ["https://acme.com/", "https://acme.com/about", "https://acme.com/careers"]
    cleaned = "PAGE TITLE: Home\nURL: https://acme.com/\n\nCONTENT:\nWelcome\n\nPAGE TITLE: About\nURL: https://acme.com/about\n\nCONTENT:\nAbout Acme"
    assert filter_source_urls(crawled, cleaned) == ["https://acme.com/", "https://acme.com/about"]


def test_pipeline_passes_used_source_urls_not_unused_crawled_pages(
    settings: object, tmp_path: Path
) -> None:
    captured: dict[str, list[str]] = {}

    def crawl_fn(domain: str, settings: object, browser_manager: object = None) -> DomainCrawlResult:
        return DomainCrawlResult(
            domain=domain,
            pages=[
                CrawledPage(url=f"https://{domain}/", title="Home", status_code=200, html="<p>Home</p>"),
                CrawledPage(
                    url=f"https://{domain}/careers",
                    title="Careers",
                    status_code=200,
                    html="<p>Jobs</p>",
                ),
            ],
            processing_status="success",
        )

    def clean_fn(pages: object, settings: object = None, **kwargs: object) -> str:
        return f"PAGE TITLE: Home\nURL: https://{list(pages)[0].url.split('/')[2]}/\n\nCONTENT:\nHome"

    def extract_fn(**kwargs: object) -> CompanyIntelligence:
        captured["urls"] = list(kwargs.get("source_urls") or [])
        return _extract_ok(**kwargs)

    result = run_pipeline(
        domains=["vapi.ai"],
        settings=settings,
        output_path=tmp_path / "output.json",
        crawl_domain_fn=crawl_fn,
        clean_pages_fn=clean_fn,
        extract_fn=extract_fn,
    )

    assert captured["urls"] == ["https://vapi.ai/"]
    assert result.results[0].source_urls == ["https://vapi.ai/"]
    assert "https://vapi.ai/careers" not in result.results[0].source_urls
