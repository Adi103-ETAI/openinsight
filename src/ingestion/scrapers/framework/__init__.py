"""Scraper framework internals.

Import paths:
    from src.ingestion.scrapers.framework import HttpClient, BaseScraper, ...
"""
from __future__ import annotations

from src.ingestion.scrapers.framework.base import BaseScraper, register_source
from src.ingestion.scrapers.framework.cache import ScrapeCache
from src.ingestion.scrapers.framework.dedup import DedupIndex, DedupMatch, compute_content_hash
from src.ingestion.scrapers.framework.http_client import HttpClient
from src.ingestion.scrapers.framework.metadata_extractor import MetadataExtractor, find_pdf_links
from src.ingestion.scrapers.framework.models import CrawlJob, MetadataSelectors, ScrapeResult, ScrapedDocument, SourceConfig
from src.ingestion.scrapers.framework.rate_limiter import RateLimiter, TokenBucket
from src.ingestion.scrapers.framework.registry import get_scraper, list_sources
from src.ingestion.scrapers.framework.robots import RobotsChecker, RobotsFile, RobotsRule

__all__ = [
    "BaseScraper",
    "register_source",
    "HttpClient",
    "ScrapeCache",
    "RateLimiter",
    "TokenBucket",
    "RobotsChecker",
    "RobotsFile",
    "RobotsRule",
    "DedupIndex",
    "DedupMatch",
    "compute_content_hash",
    "MetadataExtractor",
    "find_pdf_links",
    "ScrapeResult",
    "ScrapedDocument",
    "SourceConfig",
    "MetadataSelectors",
    "CrawlJob",
    "get_scraper",
    "list_sources",
]
