# Autonomous Lead Enrichment Agent

Python pipeline for the SoftwareBrio AI Engineer Intern assignment. Given company domains, it crawls a small set of public pages with Playwright, strips the HTML down to readable text, and asks OpenAI for structured company intelligence validated by Pydantic.

Default targets:

- `postman.com`
- `supabase.com`
- `vapi.ai`

## Key features

- Headless Chromium crawl of the homepage plus a few same-domain pages (`/about`, `/company`, `/team`, `/contact`, `/pricing`, and similar)
- Link ranking that prefers about/company/team/contact pages over careers
- BeautifulSoup/lxml cleaning: scripts, styles, SVGs, navigation chrome, and cookie banners are removed before any LLM call
- OpenAI structured output (`chat.completions.parse`) into a Pydantic `CompanyIntelligence` schema
- Post-parse filters that drop emails, people, and LinkedIn URLs not present in the cleaned text
- Per-domain isolation: one failed site does not stop the remaining domains
- JSON written to `output/output.json` with no API keys and no raw HTML

This is a bounded crawl-and-extract pipeline, not a multi-agent tool loop. It does not use search APIs, LangGraph, or cost tracking.

## Technology stack

- Python 3.11
- Playwright (sync Chromium)
- BeautifulSoup + lxml
- OpenAI Python SDK
- Pydantic v2
- python-dotenv
- pytest

## Project structure

```text
SoftwareBrio_assignment/
├── app/
│   ├── __init__.py
│   ├── main.py              # CLI: python -m app.main
│   ├── config.py            # .env / environment settings
│   ├── models.py            # Pydantic output schema
│   ├── browser.py           # Playwright launch, context, teardown
│   ├── crawler.py           # Homepage + ranked subpages
│   ├── content_cleaner.py   # HTML → compact text
│   ├── llm_extractor.py     # OpenAI structured extraction
│   ├── pipeline.py          # Orchestration and JSON writer
│   └── utils.py             # URL helpers
├── tests/                   # Unit tests; OpenAI is mocked
├── output/
│   └── output.json          # Sample run against the three test domains
├── .env.example
├── .gitignore
├── pytest.ini
├── README.md
└── requirements.txt
```

## Architecture

```text
domains
   → Playwright (headless Chromium, isolated context per domain)
   → HTML cleaning (BeautifulSoup/lxml; no raw HTML to the LLM)
   → OpenAI extraction (cleaned text + JSON schema)
   → Pydantic validation and evidence filters
   → output/output.json
```

1. **Crawl.** One Chromium browser, a fresh context per domain, up to `MAX_PAGES_PER_DOMAIN` pages (default 6), 20s page timeout.
2. **Clean.** Visible text only, capped at 12k characters per page and 50k per domain.
3. **Extract.** OpenAI returns a `CompanyIntelligence` object. Names, emails, and LinkedIn URLs must appear in the cleaned text or they are discarded.
4. **Write.** Envelope JSON with one result per domain.

## Setup (Python 3.11)

From the project root.

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS / Linux:

```bash
source .venv/bin/activate
```

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Install Playwright Chromium:

```bash
playwright install chromium
```

Create a local env file from the template:

```bash
copy .env.example .env
```

macOS / Linux: `cp .env.example .env`

Set a real `OPENAI_API_KEY` in `.env`. The placeholder `your_openai_api_key_here` is rejected at startup. Never commit `.env`.

```env
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o-mini
HEADLESS=true
MAX_PAGES_PER_DOMAIN=6
PAGE_TIMEOUT_MS=20000
MAX_CONTENT_CHARS_PER_PAGE=12000
MAX_TOTAL_CONTENT_CHARS_PER_DOMAIN=50000
LOG_LEVEL=INFO
```

## Run

Default assignment domains:

```bash
python -m app.main
```

Optional subset:

```bash
python -m app.main postman.com supabase.com
```

Requires a valid OpenAI key, Playwright Chromium, and network access. The run overwrites `output/output.json`.

## Tests

```bash
python -m pytest tests -v
```

Unit tests mock OpenAI and do not charge the API. Live crawler tests are skipped unless `RUN_LIVE_CRAWLER_TESTS=1`. `tests/test_playwright_smoke.py` and `tests/test_browser.py` launch Chromium; they need `playwright install chromium` and network access to `https://example.com`.

## Output

Written to [`output/output.json`](output/output.json). Top-level fields:

| Field | Meaning |
|---|---|
| `generated_at` | UTC timestamp |
| `total_domains` | Number of domains in the run |
| `successful_domains` | Results that are not `failed` (`success` and `partial_success`) |
| `failed_domains` | Domains that produced no usable extract |
| `results` | One `CompanyIntelligence` object per domain |

Each result includes:

- `company_overview` — about two sentences from page content
- `target_audience` — cautious ICP inference from product copy, or an explicit unknown fallback
- `contact_points` — public emails found in the text (`[]` if none)
- `team_members` — names, roles, LinkedIn URLs only when those strings appear on crawled pages
- `confidence_score` — 0.0 to 1.0 based on evidence completeness
- `source_urls` — crawled URLs used as evidence
- `processing_status` — `success`, `partial_success`, or `failed`
- `error_message` — set only on failure

Empty `contact_points` or `team_members` means the fields were not found on the crawled pages, not that the company has none.

## Error handling

- Invalid or missing `OPENAI_API_KEY` exits before the crawl.
- Timeouts, HTTP 404s, and likely bot-challenge pages are recorded; remaining pages for that domain still run.
- A crawler, cleaner, or extractor exception for one domain writes a `failed` record and continues.
- OpenAI timeouts and rate limits are retried a few times. Auth failures are not retried.
- Error strings are truncated and API-key-shaped tokens are redacted.

## Known limitations

- Public emails often do not exist; many sites use forms only.
- Team and LinkedIn data depend on which pages the crawler actually opens. Careers pages are deprioritized but may still be crawled when better pages are not found.
- Website copy, URLs, and leadership listings change over time, so a later run can differ from the sample JSON.
- The crawl is same-domain and bounded. It does not search Google or invent LinkedIn profiles.
- Anti-bot walls can leave a domain with thin or empty content.

## Manual operations readiness

Yes, I am 100% comfortable spending roughly 40% of my working hours on manual lead prospecting, email discovery, and account handling alongside my AI engineering tasks.

## Privacy and security

- Do not commit `.env`. It is listed in `.gitignore`.
- Do not expose API keys. Settings logs only `openai_api_key_set: true`.
- Do not include raw HTML in output. The JSON writer drops HTML fields and writes structured text only.
