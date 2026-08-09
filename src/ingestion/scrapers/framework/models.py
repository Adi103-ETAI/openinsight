"""Core dataclasses for the scraper framework.

All framework modules exchange data through these typed structures — no raw
dicts cross module boundaries. This keeps the type checker useful and the
runtime behavior predictable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def _utcnow() -> datetime:
    """Timezone-aware UTC now (avoids the deprecated datetime.utcnow())."""
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class ScrapeResult:
    """Outcome of a single fetch attempt.

    `ok=False` does NOT raise — the framework returns failed ScrapeResult objects
    so callers can batch-process success/failure uniformly. The `dead_letter.py`
    module consumes failed results for retry scheduling.
    """
    url: str
    ok: bool
    status_code: int | None = None
    content: bytes | None = None
    content_type: str | None = None
    encoding: str | None = None
    error: str | None = None
    fetched_at: datetime = field(default_factory=_utcnow)
    from_cache: bool = False
    cache_layer: str | None = None  # "memory" | "redis" | "filesystem" | None
    elapsed_ms: float = 0.0
    headers: dict[str, str] = field(default_factory=dict)

    @property
    def is_pdf(self) -> bool:
        if self.content_type:
            return "application/pdf" in self.content_type.lower()
        if self.url.lower().endswith(".pdf"):
            return True
        return False

    @property
    def is_html(self) -> bool:
        if not self.content_type:
            return False
        ct = self.content_type.lower()
        return "text/html" in ct or "application/xhtml" in ct

    @property
    def is_xml(self) -> bool:
        if not self.content_type:
            return False
        ct = self.content_type.lower()
        return "xml" in ct


@dataclass
class ScrapedDocument:
    """A fetched + metadata-enriched document ready for parsing.

    The scraper framework produces ScrapedDocument instances; parsers consume
    them. This decouples parsers from HTTP concerns entirely.
    """
    url: str
    source: str  # source config name, e.g. "pubmed", "indmed"
    content: bytes
    content_type: str
    title: str | None = None
    authors: list[str] = field(default_factory=list)
    journal: str | None = None
    doi: str | None = None
    pmid: str | None = None
    pubdate: str | None = None  # ISO 8601 if extractable
    abstract: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)  # raw extraction dump
    fetched_at: datetime = field(default_factory=_utcnow)
    trust_tier: int = 3  # 1=highest (gov/ICMR), 5=lowest (news)
    india_relevant: bool | None = None
    indian_source: bool | None = None


@dataclass
class CrawlJob:
    """A unit of crawl work — one URL with provenance.

    CrawlJobs are queued by the scheduler and consumed by Celery workers.
    """
    url: str
    source: str
    discovered_from: str = ""  # "sitemap" | "seed" | "link-extraction" | "api"
    priority: int = 5  # 1=highest
    metadata: dict[str, Any] = field(default_factory=dict)
    queued_at: datetime = field(default_factory=_utcnow)

    def cache_key(self) -> str:
        """Stable cache key for this job (URL + source)."""
        return f"{self.source}:{self.url}"


@dataclass(frozen=True)
class MetadataSelectors:
    """Per-source CSS/HTML selectors for metadata extraction.

    Most academic journal sites expose Highwire Press citation_* meta tags,
    Dublin Core DC.* tags, or OpenGraph og:* tags. Each source config declares
    which tags to read for which field, in priority order.
    """
    title: list[str] = field(default_factory=lambda: [
        "citation_title", "DC.Title", "og:title", "DC.title",
    ])
    authors: list[str] = field(default_factory=lambda: [
        "citation_author", "DC.Creator", "DC.creator",
    ])
    journal: list[str] = field(default_factory=lambda: [
        "citation_journal_title", "DC.Source",
    ])
    doi: list[str] = field(default_factory=lambda: [
        "citation_doi", "DC.Identifier", "DOI",
    ])
    pmid: list[str] = field(default_factory=lambda: [
        "citation_pubmed_id", "PMID",
    ])
    pubdate: list[str] = field(default_factory=lambda: [
        "citation_publication_date", "citation_date", "DC.Date", "og:published_time",
    ])
    abstract: list[str] = field(default_factory=lambda: [
        "citation_abstract", "DC.Description", "description",
    ])


@dataclass(frozen=True)
class SourceConfig:
    """Declarative configuration for a single source.

    Adding a new source = writing one SourceConfig + (optionally) one parser.
    The framework reads the config and handles all HTTP concerns.
    """
    name: str
    base_url: str
    rate_limit: float = 1.0  # req/sec
    crawl_delay: float = 1.0  # seconds between requests
    user_agent_suffix: str = ""
    fetch_strategy: str = "http_first"  # "http_first" | "browser_first" | "http_only"
    expected_content_types: tuple[str, ...] = ("text/html", "application/pdf")
    sitemap_urls: tuple[str, ...] = ()
    metadata_selectors: MetadataSelectors = field(default_factory=MetadataSelectors)
    pdf_link_selectors: tuple[str, ...] = (
        "a[href$='.pdf']",
        "a:contains('Full Text')",
        "a:contains('PDF')",
    )
    trust_tier: int = 3
    india_relevant_default: bool = False
    indian_source_default: bool = False
    requires_api_key: bool = False
    api_key_env: str | None = None
    # Allow per-source custom config
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def user_agent(self) -> str:
        suffix = f" ({self.user_agent_suffix})" if self.user_agent_suffix else ""
        return f"OpenInsight-Bot/0.1 (clinical evidence indexing; contact: hello@openinsight.in){suffix}"

    @property
    def domain(self) -> str:
        """Extract domain from base_url for per-domain rate limiting."""
        from urllib.parse import urlparse
        return urlparse(self.base_url).netloc.lower()
