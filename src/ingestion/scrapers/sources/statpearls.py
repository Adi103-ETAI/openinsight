"""StatPearls scraper — NCBI Bookshelf peer-reviewed clinical overviews.

StatPearls (https://www.ncbi.nlm.nih.gov/books/NBK430685/) is a peer-reviewed
open-access clinical reference published by NCBI Bookshelf. ~3K articles
covering diseases, drugs, procedures, and anatomy. Each article has a
standardized structure:

- Introduction / Definition
- Etiology
- Epidemiology
- History and Physical
- Evaluation (Diagnosis)
- Treatment / Management
- Differential Diagnosis
- Prognosis
- Complications
- Deterrence and Patient Education
- Pearls and Other Issues

This structured sectioning is GOLD for RAG — a chunk from "Treatment" should
not be confused with a chunk from "Differential Diagnosis".

StatPearls is also continuously updated (each article has a "Bookshelf ID"
like NBK430685 and a "Last Update Date"). Our scraper caches for 30 days
and re-fetches to pick up updates.

Discovery:
- NCBI E-utilities esearch on db=books with term=statpearls[book]
- Returns NBK IDs; efetch returns the full HTML

Politeness:
- 3 req/sec without API key, 10 with
- 30-day cache on individual articles
- 7-day cache on the search results page
"""
from __future__ import annotations

from typing import Any
from urllib.parse import quote_plus

from loguru import logger

from src.ingestion.scrapers.framework.base import BaseScraper, register_source
from src.ingestion.scrapers.framework.models import CrawlJob, SourceConfig

STATPEARLS_CONFIG = SourceConfig(
    name="statpearls",
    base_url="https://eutils.ncbi.nlm.nih.gov/entrez/eutils",
    rate_limit=3.0,
    crawl_delay=0.34,
    user_agent_suffix="StatPearls-indexer",
    fetch_strategy="http_only",
    expected_content_types=("text/html", "application/xml"),
    trust_tier=2,  # StatPearls = peer-reviewed, NCBI-hosted, regularly updated
    india_relevant_default=False,  # international reference, not India-specific
    indian_source_default=False,  # NCBI = US government
    requires_api_key=False,
    api_key_env="NCBI_API_KEY",
    extra={
        "db": "books",
        "book_term": "statpearls[book]",
        "retmax": 200,
    },
)


@register_source("statpearls")
class StatPearlsScraper(BaseScraper):
    """Scraper for StatPearls clinical overviews via NCBI Bookshelf."""

    config = STATPEARLS_CONFIG

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        import os
        self._api_key = os.environ.get(self.config.api_key_env or "", "")
        if self._api_key:
            self.rate_limiter.set_domain_rate(self.config.domain, 10.0, 30.0)
            self._api_suffix = f"&api_key={self._api_key}"
        else:
            self._api_suffix = ""

    async def discover(
        self,
        max_results: int = 500,
        **kwargs: Any,
    ) -> list[CrawlJob]:
        """Discover StatPearls article IDs via NCBI E-utilities.

        Args:
            max_results: cap on number of articles to discover (default 500)

        Returns:
            list of CrawlJobs, one per StatPearls article
        """
        # esearch on db=books with term=statpearls[book]
        esearch_url = (
            f"{self.config.base_url}/esearch.fcgi"
            f"?db=books&term={quote_plus(self.config.extra['book_term'])}"
            f"&retmax={max_results}&retmode=json&usehistory=n"
            f"{self._api_suffix}"
        )
        result = await self.http.fetch(esearch_url, use_cache=True, cache_ttl=7 * 24 * 3600)
        if not result.ok or not result.content:
            logger.error(f"[statpearls] esearch failed: {result.error}")
            return []

        try:
            import json
            data = json.loads(result.content)
            book_ids = data.get("esearchresult", {}).get("idlist", [])
        except Exception as e:
            logger.error(f"[statpearls] esearch JSON parse failed: {e}")
            return []

        logger.info(f"[statpearls] discovered {len(book_ids)} StatPearls articles")

        # Build CrawlJobs — efetch on db=books returns HTML
        jobs: list[CrawlJob] = []
        for book_id in book_ids:
            # NCBI Bookshelf URL pattern
            fetch_url = (
                f"https://www.ncbi.nlm.nih.gov/books/{book_id}/"
            )
            jobs.append(CrawlJob(
                url=fetch_url,
                source=self.config.name,
                discovered_from="api",
                metadata={
                    "book_id": book_id,
                    "bookshelf_url": fetch_url,
                },
            ))
        return jobs

    async def discover_by_topic(
        self,
        topic: str,
        max_results: int = 50,
    ) -> list[CrawlJob]:
        """Discover StatPearls articles for a specific topic.

        Args:
            topic: search term (e.g., "diabetes", "myocardial infarction")
            max_results: cap on results
        """
        # Search StatPearls with topic filter
        full_term = f"statpearls[book] AND {topic}"
        esearch_url = (
            f"{self.config.base_url}/esearch.fcgi"
            f"?db=books&term={quote_plus(full_term)}"
            f"&retmax={max_results}&retmode=json&usehistory=n"
            f"{self._api_suffix}"
        )
        result = await self.http.fetch(esearch_url, use_cache=True, cache_ttl=7 * 24 * 3600)
        if not result.ok or not result.content:
            return []

        try:
            import json
            data = json.loads(result.content)
            book_ids = data.get("esearchresult", {}).get("idlist", [])
        except Exception:
            return []

        logger.info(f"[statpearls] discovered {len(book_ids)} articles for topic: {topic}")
        jobs: list[CrawlJob] = []
        for book_id in book_ids:
            jobs.append(CrawlJob(
                url=f"https://www.ncbi.nlm.nih.gov/books/{book_id}/",
                source=self.config.name,
                discovered_from="api",
                metadata={
                    "book_id": book_id,
                    "topic": topic,
                },
            ))
        return jobs
