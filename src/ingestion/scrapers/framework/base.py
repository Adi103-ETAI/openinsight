"""BaseScraper ABC + source registry.

Each source implements BaseScraper. The framework provides HttpClient,
RobotsChecker, RateLimiter, ScrapeCache, CrossSourceDedup, MetadataExtractor
as injected dependencies — sources focus on source-specific logic only.

A "source" is something like pubmed, indmed, cdsco, nfhs. Each source has:
1. A SourceConfig (declarative: URLs, rate limits, selectors)
2. A BaseScraper subclass (or just the base, if config-only is sufficient)
3. (Optional) A parser in src/ingestion/parsers/ for structured-data extraction

The registry (registry.py) maps source names to scraper classes. Sources
register themselves via the @register_source decorator.
"""
from __future__ import annotations

import abc
from typing import Any

from loguru import logger

from src.ingestion.scrapers.framework.cache import ScrapeCache
from src.ingestion.scrapers.framework.dedup import DedupIndex
from src.ingestion.scrapers.framework.http_client import HttpClient
from src.ingestion.scrapers.framework.metadata_extractor import MetadataExtractor, find_pdf_links
from src.ingestion.scrapers.framework.models import CrawlJob, MetadataSelectors, ScrapeResult, ScrapedDocument, SourceConfig
from src.ingestion.scrapers.framework.rate_limiter import RateLimiter
from src.ingestion.scrapers.framework.robots import RobotsChecker


class BaseScraper(abc.ABC):
    """Abstract base for all source scrapers.

    Subclasses must:
    1. Set `config` (a SourceConfig instance) as a class attribute
    2. Implement `discover()` to yield CrawlJobs (URLs to fetch)
    3. Optionally override `process()` to do source-specific post-processing
       (e.g., extract PDF URLs from an article landing page)

    The framework handles: robots.txt, rate limiting, caching, retries,
    metadata extraction, dedup. Sources only handle source-specific logic.
    """

    config: SourceConfig  # subclasses must override

    def __init__(
        self,
        http_client: HttpClient | None = None,
        cache: ScrapeCache | None = None,
        rate_limiter: RateLimiter | None = None,
        dedup_index: DedupIndex | None = None,
    ) -> None:
        self.cache = cache or ScrapeCache()
        self.rate_limiter = rate_limiter or RateLimiter(
            per_domain_overrides={self.config.domain: (self.config.rate_limit, max(self.config.rate_limit * 3, 1.0))},
        )
        self.robots = RobotsChecker(cache=self.cache)
        self.http = http_client or HttpClient(
            cache=self.cache,
            rate_limiter=self.rate_limiter,
            robots=self.robots,
            user_agent=self.config.user_agent,
        )
        self.dedup = dedup_index or DedupIndex()
        self.metadata_extractor = MetadataExtractor(self.config.metadata_selectors)

    @abc.abstractmethod
    async def discover(self, **kwargs: Any) -> list[CrawlJob]:
        """Yield CrawlJobs to fetch.

        Implementations should return a list of CrawlJob objects. Each job
        represents one URL to fetch + parse. Discovery methods vary by source:
        - PubMed: Entrez esearch → PMID list → efetch URLs
        - IndMED: crawl journal index pages → article URLs
        - CDSCO: scrape approved-drugs search → pagination
        """
        ...

    async def fetch_one(self, job: CrawlJob) -> ScrapedDocument | None:
        """Fetch one URL and convert to a ScrapedDocument.

        Returns None on failure (the failure is logged + recorded in the
        dead-letter queue by the caller, not here).
        """
        result = await self.http.fetch(job.url)
        if not result.ok or not result.content:
            logger.warning(f"[{self.config.name}] fetch failed: {job.url} — {result.error}")
            return None
        return await self.process(result, job)

    async def process(self, result: ScrapeResult, job: CrawlJob) -> ScrapedDocument | None:
        """Convert a raw ScrapeResult into a ScrapedDocument.

        Default implementation: extract HTML metadata (title, authors, DOI, etc.).
        Override for sources with non-HTML structure (XML APIs, JSON APIs, PDFs).
        """
        # Extract metadata from HTML
        metadata: dict[str, Any] = {}
        if result.is_html:
            metadata = self.metadata_extractor.extract(result.content, result.encoding or "utf-8")
        elif result.is_xml:
            # XML — let the parser handle it; just record content-type
            metadata = {"content_type": "xml"}

        return ScrapedDocument(
            url=result.url,
            source=self.config.name,
            content=result.content,
            content_type=result.content_type or "",
            title=metadata.get("title"),
            authors=metadata.get("authors", []),
            journal=metadata.get("journal"),
            doi=metadata.get("doi"),
            pmid=metadata.get("pmid"),
            pubdate=metadata.get("pubdate"),
            abstract=metadata.get("abstract"),
            metadata=metadata,
            fetched_at=result.fetched_at,
            trust_tier=self.config.trust_tier,
            india_relevant=self.config.india_relevant_default or None,
            indian_source=self.config.indian_source_default or None,
        )

    async def discover_pdf_links(self, html: bytes | str, base_url: str) -> list[str]:
        """Find PDF download links in an HTML page (for journal landing pages)."""
        return find_pdf_links(html, base_url)

    async def close(self) -> None:
        """Release HTTP client resources."""
        await self.http.close()


# Source registry — sources register themselves via decorator
_SOURCE_REGISTRY: dict[str, type[BaseScraper]] = {}


def register_source(name: str) -> Any:
    """Decorator to register a BaseScraper subclass under `name`.

    Usage:
        @register_source("indmed")
        class IndMEDScraper(BaseScraper):
            config = INDMED_CONFIG
            ...
    """
    def decorator(cls: type[BaseScraper]) -> type[BaseScraper]:
        _SOURCE_REGISTRY[name] = cls
        return cls
    return decorator
