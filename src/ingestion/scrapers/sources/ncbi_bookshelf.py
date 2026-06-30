"""NCBI Bookshelf scraper — open-access medical books beyond StatPearls.

NCBI Bookshelf (https://www.ncbi.nlm.nih.gov/books/) is a free archive of
~400+ biomedical books including:
- GeneReviews (genetic disorders — ~800 chapters)
- Medical Genetics Summaries (~100 conditions)
- NCBI Handbook
- NIH-funded monographs
- Various specialty references

Unlike StatPearls (which has a standardized clinical structure), Bookshelf
books vary widely in format — some are textbook chapters, some are
monographs, some are reference manuals. This scraper handles the general
case; per-book structure parsing is delegated to the parser.

Discovery:
- Browse the Bookshelf catalog at /books/
- Or use E-utilities esearch on db=books with various filters:
  - "open_access[filter]" — only open-access books
  - "pubmed[book]" — PubMed-indexed books
  - Specific book filters: "genereviews[book]", "medicalgenetics[book]"

We focus on GeneReviews + Medical Genetics Summaries as the highest-value
Bookshelf content for clinical RAG.
"""
from __future__ import annotations

from typing import Any
from urllib.parse import quote_plus

from loguru import logger

from src.ingestion.scrapers.framework.base import BaseScraper, register_source
from src.ingestion.scrapers.framework.models import CrawlJob, SourceConfig

# High-value NCBI Bookshelf collections to index
BOOKSHELF_COLLECTIONS = {
    "genereviews": {
        "term": "genereviews[book]",
        "label": "GeneReviews",
        "description": "Genetic disease reviews (~800 chapters)",
        "trust_tier": 2,
    },
    "medical_genetics": {
        "term": "medicalgenetics[book]",
        "label": "Medical Genetics Summaries",
        "description": "Genetic condition summaries (~100)",
        "trust_tier": 2,
    },
    "ncbi_handbook": {
        "term": "handbook[book]",
        "label": "NCBI Handbook",
        "description": "Bioinformatics reference",
        "trust_tier": 3,
    },
    "consensus_study": {
        "term": "nap[book]",
        "label": "National Academies Consensus Studies",
        "description": "Health policy consensus reports",
        "trust_tier": 3,
    },
}

NCBI_BOOKSHELF_CONFIG = SourceConfig(
    name="ncbi_bookshelf",
    base_url="https://eutils.ncbi.nlm.nih.gov/entrez/eutils",
    rate_limit=3.0,
    crawl_delay=0.34,
    user_agent_suffix="Bookshelf-indexer",
    fetch_strategy="http_only",
    expected_content_types=("text/html", "application/xml"),
    trust_tier=2,  # Default; per-collection override applied in process()
    india_relevant_default=False,
    indian_source_default=False,
    requires_api_key=False,
    api_key_env="NCBI_API_KEY",
    extra={
        "db": "books",
        "collections": BOOKSHELF_COLLECTIONS,
        "retmax": 200,
    },
)


@register_source("ncbi_bookshelf")
class NCBIBookshelfScraper(BaseScraper):
    """Scraper for NCBI Bookshelf open-access medical books."""

    config = NCBI_BOOKSHELF_CONFIG

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
        collections: list[str] | None = None,
        max_per_collection: int = 200,
        **kwargs: Any,
    ) -> list[CrawlJob]:
        """Discover Bookshelf articles across collections.

        Args:
            collections: list of collection keys to discover (default: all).
                Options: "genereviews", "medical_genetics", "ncbi_handbook",
                "consensus_study"
            max_per_collection: cap per collection (default 200)

        Returns:
            list of CrawlJobs
        """
        target_collections = collections or list(self.config.extra["collections"].keys())
        jobs: list[CrawlJob] = []

        for collection_key in target_collections:
            collection = self.config.extra["collections"].get(collection_key)
            if not collection:
                logger.warning(f"[bookshelf] unknown collection: {collection_key}")
                continue
            try:
                collection_jobs = await self._discover_collection(
                    collection_key, collection, max_per_collection
                )
                jobs.extend(collection_jobs)
                logger.info(
                    f"[bookshelf] {collection['label']}: discovered {len(collection_jobs)} articles"
                )
            except Exception as e:
                logger.warning(f"[bookshelf] {collection_key}: discovery failed: {e}")

        return jobs

    async def _discover_collection(
        self,
        collection_key: str,
        collection: dict[str, Any],
        max_results: int,
    ) -> list[CrawlJob]:
        """Discover articles for one Bookshelf collection."""
        esearch_url = (
            f"{self.config.base_url}/esearch.fcgi"
            f"?db=books&term={quote_plus(collection['term'])}"
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
        except Exception as e:
            logger.error(f"[bookshelf] {collection_key}: JSON parse failed: {e}")
            return []

        jobs: list[CrawlJob] = []
        for book_id in book_ids:
            jobs.append(CrawlJob(
                url=f"https://www.ncbi.nlm.nih.gov/books/{book_id}/",
                source=self.config.name,
                discovered_from="api",
                metadata={
                    "book_id": book_id,
                    "collection": collection_key,
                    "collection_label": collection["label"],
                    "trust_tier": collection["trust_tier"],
                },
            ))
        return jobs

    async def discover_genereviews(self, max_results: int = 500) -> list[CrawlJob]:
        """Convenience: discover GeneReviews chapters."""
        return await self.discover(collections=["genereviews"], max_per_collection=max_results)

    async def discover_medical_genetics(self, max_results: int = 100) -> list[CrawlJob]:
        """Convenience: discover Medical Genetics Summaries."""
        return await self.discover(collections=["medical_genetics"], max_per_collection=max_results)

    async def process(self, result, job: CrawlJob) -> Any:
        """Process a fetched Bookshelf article.

        Applies per-collection trust_tier (from BOOKSHELF_COLLECTIONS map).
        """
        from src.ingestion.scrapers.framework.models import ScrapedDocument

        # Use collection-specific trust tier if available
        collection_tier = job.metadata.get("trust_tier", self.config.trust_tier)

        meta = self.metadata_extractor.extract(result.content, result.encoding or "utf-8") if result.is_html else {}

        return ScrapedDocument(
            url=result.url,
            source=self.config.name,
            content=result.content,
            content_type=result.content_type or "",
            title=meta.get("title"),
            authors=meta.get("authors", []),
            journal=meta.get("journal") or job.metadata.get("collection_label"),
            doi=meta.get("doi"),
            pmid=meta.get("pmid"),
            pubdate=meta.get("pubdate"),
            abstract=meta.get("abstract"),
            metadata={**meta, **job.metadata},
            fetched_at=result.fetched_at,
            trust_tier=collection_tier,
            india_relevant=False,
            indian_source=False,
        )
