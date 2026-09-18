"""Application entry point.

Runs the end-to-end lead enrichment pipeline and writes output/output.json.

Usage:
    python -m app.main
    python -m app.main postman.com supabase.com
"""

from __future__ import annotations

import logging
import sys

from app.config import ConfigError, configure_logging, load_settings
from app.pipeline import run_pipeline


def main(argv: list[str] | None = None) -> None:
    """Load settings, run the pipeline, and write JSON output."""
    configure_logging("INFO")
    logger = logging.getLogger("app.main")

    try:
        settings = load_settings()
    except ConfigError as exc:
        logger.error("%s", exc)
        raise SystemExit(1) from exc

    configure_logging(settings.log_level)
    logger.info("Autonomous Lead Enrichment Agent")
    logger.info("Configuration loaded: %s", settings.redacted_dict())

    domains = list(argv) if argv is not None else sys.argv[1:]
    selected = domains or None
    try:
        result = run_pipeline(domains=selected, settings=settings)
    except Exception as exc:
        logger.error("Pipeline stopped before writing output: %s", exc)
        raise SystemExit(1) from exc

    logger.info(
        "Wrote %s (total=%s successful=%s failed=%s)",
        result.output_path,
        result.total_domains,
        result.successful_domains,
        result.failed_domains,
    )


if __name__ == "__main__":
    main()
