"""OpenInsight scraper framework.

Public API:
    BaseScraper, SourceConfig, ScrapedDocument, ScrapeResult
    HttpClient, RobotsChecker, RateLimiter, ScrapeCache
    CrossSourceDedup, MetadataExtractor
    get_scraper, list_sources

The framework sits in front of existing parsers:
    Internet → [SCRAPER fetches] → bytes → [PARSER structures] → chunks → embed → store

See docs/05_SCRAPER_FRAMEWORK.md (added in this phase) for the full architecture.
"""
from __future__ import annotations

from src.ingestion.scrapers.framework.base import BaseScraper, SourceConfig
from src.ingestion.scrapers.framework.models import (
    ScrapeResult,
    ScrapedDocument,
    CrawlJob,
)
from src.ingestion.scrapers.framework.registry import get_scraper, list_sources, register_source

__all__ = [
    "BaseScraper",
    "SourceConfig",
    "ScrapeResult",
    "ScrapedDocument",
    "CrawlJob",
    "get_scraper",
    "list_sources",
    "register_source",
]
