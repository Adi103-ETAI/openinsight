"""PubMed (NCBI Entrez) scraper.

Wraps the existing PubMedParser's Entrez-calling logic in the scraper
framework. The parser will be refactored in Phase 0.5 to take XML bytes
as input (no fetching); this scraper handles all Entrez API concerns.

API endpoints used:
- esearch.fcgi: search PubMed → returns PMID list
- efetch.fcgi: fetch full records by PMID → returns PubMed XML
- einfo.fcgi: database statistics (not used here, useful for monitoring)

Rate limits (NCBI):
- Without API key: 3 req/sec
- With NCBI_API_KEY: 10 req/sec
- 429 responses honor Retry-After (handled by HttpClient)

Caching:
- esearch results cached 7 days (PubMed corpus is append-only — old PMIDs
  don't disappear, but new ones appear; weekly re-crawl picks up new content)
- efetch results cached indefinitely (PubMed records are versioned via
  [PubMed-indexed] flag and rarely change)
"""
from __future__ import annotations

import asyncio
from typing import Any
from urllib.parse import quote_plus

from loguru import logger

from src.ingestion.scrapers.framework.base import BaseScraper, register_source
from src.ingestion.scrapers.framework.models import CrawlJob, MetadataSelectors, SourceConfig

PUBMED_CONFIG = SourceConfig(
    name="pubmed",
    base_url="https://eutils.ncbi.nlm.nih.gov/entrez/eutils",
    rate_limit=3.0,  # 3 req/sec without API key, 10 with
    crawl_delay=0.34,  # 1/3 sec
    user_agent_suffix="PubMed-indexer",
    fetch_strategy="http_only",  # NCBI API is HTTP-only, no JS
    expected_content_types=("application/xml", "text/xml"),
    trust_tier=4,  # PubMed = international, not India-specific
    india_relevant_default=False,
    indian_source_default=False,
    requires_api_key=False,  # API key is optional (raises rate limit if absent)
    api_key_env="NCBI_API_KEY",
    extra={
        "db": "pubmed",
        "retmax": 200,  # max records per efetch call
        "retmode": "xml",
    },
)


@register_source("pubmed")
class PubMedScraper(BaseScraper):
    """Scraper for PubMed via NCBI E-utilities."""

    config = PUBMED_CONFIG

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        import os
        self._api_key = os.environ.get(self.config.api_key_env or "", "")
        if self._api_key:
            # NCBI allows 10 req/sec with API key — bump the rate limit
            self.rate_limiter.set_domain_rate(self.config.domain, 10.0, 30.0)
            self._api_suffix = f"&api_key={self._api_key}"
        else:
            self._api_suffix = ""

    async def discover(
        self,
        query: str = "",
        max_results: int = 500,
        journal: str | None = None,
        date_range: str = "2015:2025[DP]",
        **kwargs: Any,
    ) -> list[CrawlJob]:
        """Discover PMIDs via esearch, return CrawlJobs (one per PMID).

        Args:
            query: PubMed query string (e.g., "TB drug-resistant India")
            max_results: max PMIDs to return
            journal: optional journal filter (e.g., "Indian J Med Res")
            date_range: optional date range filter (default: last 10 years)

        Returns:
            list of CrawlJobs, each pointing to an efetch URL for one PMID
            (efetch can fetch multiple PMIDs at once, but we use one-per-job
            for cleaner Celery parallelism + dedup)

        Example:
            jobs = await scraper.discover(journal="Indian J Med Res", max_results=500)
        """
        # Build the query
        full_query = query
        if journal:
            full_query = f'{journal}[TA] AND {date_range}' if not query else f'{query} AND {journal}[TA]'
        elif not query:
            full_query = date_range

        if not full_query:
            raise ValueError("Either `query` or `journal` must be provided")

        # esearch to get PMIDs
        esearch_url = (
            f"{self.config.base_url}/esearch.fcgi"
            f"?db=pubmed&term={quote_plus(full_query)}"
            f"&retmax={max_results}&retmode=json&usehistory=n"
            f"{self._api_suffix}"
        )
        result = await self.http.fetch(esearch_url, use_cache=True, cache_ttl=7 * 24 * 3600)
        if not result.ok or not result.content:
            logger.error(f"[pubmed] esearch failed: {result.error}")
            return []

        try:
            import json
            data = json.loads(result.content)
            pmids = data.get("esearchresult", {}).get("idlist", [])
        except Exception as e:
            logger.error(f"[pubmed] esearch JSON parse failed: {e}")
            return []

        logger.info(f"[pubmed] discovered {len(pmids)} PMIDs for query: {full_query[:80]}")

        # Build one CrawlJob per PMID (efetch URL)
        jobs: list[CrawlJob] = []
        for pmid in pmids:
            efetch_url = (
                f"{self.config.base_url}/efetch.fcgi"
                f"?db=pubmed&id={pmid}&retmode=xml&rettype=abstract"
                f"{self._api_suffix}"
            )
            jobs.append(CrawlJob(
                url=efetch_url,
                source=self.config.name,
                discovered_from="api",
                metadata={"pmid": pmid, "query": full_query},
            ))
        return jobs

    async def discover_by_journal(
        self,
        journal_abbrev: str,
        max_results: int = 500,
        date_range: str = "2015:2025[DP]",
    ) -> list[CrawlJob]:
        """Convenience: discover all articles from a specific journal.

        Args:
            journal_abbrev: PubMed journal abbreviation, e.g., "Indian J Med Res"
            max_results: cap on PMIDs returned
            date_range: date filter (default last 10 years)

        Returns:
            list of CrawlJobs
        """
        return await self.discover(
            journal=journal_abbrev,
            max_results=max_results,
            date_range=date_range,
        )
