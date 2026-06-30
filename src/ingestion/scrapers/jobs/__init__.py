"""Celery jobs for the scraper framework.

Phase 0 stub — Celery tasks are wired up in Phase 1 once the framework is
exercised end-to-end. For now, ingestion runs synchronously via
`python -m src.ingestion.run_ingestion`.

Planned jobs:
    discover_and_ingest(source: str, **discover_kwargs)
        - calls scraper.discover() to get CrawlJobs
        - enqueues each job as an individual Celery task
    ingest_url(url: str, source: str, metadata: dict)
        - fetches one URL via the scraper framework
        - parses via the matching parser
        - embeds + stores via IngestionPipeline
    reingest_changed(since_hours: int = 168)
        - re-fetches URLs whose ETag/Last-Modified might have changed
        - scheduled weekly via Celery beat
"""
from __future__ import annotations
