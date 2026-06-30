"""PMC India subset scraper — full-text articles from PubMed Central with Indian affiliations.

PubMed Central (PMC) is NCBI's free full-text archive. Unlike PubMed (which
gives abstracts), PMC gives the COMPLETE article text — Methods, Results,
Discussion, References. This is high-value content for RAG because abstracts
often omit the specific dosing / population / outcome details that clinicians
actually need.

This scraper queries PMC for articles with Indian author affiliations:
    india[affiliation] AND open_access[filter] AND 2015:2025[DP]

API:
- esearch.fcgi (db=pmc) → PMC ID list
- efetch.fcgi (db=pmc)  → full-text XML (NLM Journal Publishing DTD)

The XML is structured — no HTML scraping. The parser (parsers/pmc_india.py)
extracts sections (Abstract, Introduction, Methods, Results, Discussion,
References) and chunks them with section metadata preserved.

Rate limits: same as PubMed (3 req/sec without API key, 10 with).
Cross-source dedup: PMC articles usually have a PMID too — dedup against
existing PubMed documents via PMID.
"""
from __future__ import annotations

from typing import Any
from urllib.parse import quote_plus

from loguru import logger

from src.ingestion.scrapers.framework.base import BaseScraper, register_source
from src.ingestion.scrapers.framework.models import CrawlJob, SourceConfig

PMC_INDIA_CONFIG = SourceConfig(
    name="pmc_india",
    base_url="https://eutils.ncbi.nlm.nih.gov/entrez/eutils",
    rate_limit=3.0,
    crawl_delay=0.34,
    user_agent_suffix="PMC-India-indexer",
    fetch_strategy="http_only",
    expected_content_types=("application/xml", "text/xml"),
    trust_tier=3,  # PMC = peer-reviewed, full-text
    india_relevant_default=True,  # query filters for India affiliations
    indian_source_default=False,  # PMC is international; articles have Indian authors
    requires_api_key=False,
    api_key_env="NCBI_API_KEY",
    extra={
        "db": "pmc",
        "retmax": 100,
        "retmode": "xml",
    },
)


@register_source("pmc_india")
class PMCIndiaScraper(BaseScraper):
    """Scraper for PubMed Central articles with Indian affiliations."""

    config = PMC_INDIA_CONFIG

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
        query: str = "",
        max_results: int = 500,
        date_range: str = "2015:2025[DP]",
        affiliation_filter: str = "india",
        **kwargs: Any,
    ) -> list[CrawlJob]:
        """Discover PMC articles with Indian affiliations.

        Args:
            query: optional additional query terms (e.g., "diabetes")
            max_results: cap on PMC IDs returned
            date_range: date filter (default last 10 years)
            affiliation_filter: country filter (default "india")

        Returns:
            list of CrawlJobs, one per PMC ID
        """
        # Build the query: <user_query> AND india[affiliation] AND open_access[filter] AND date
        parts = []
        if query:
            parts.append(query)
        if affiliation_filter:
            parts.append(f"{affiliation_filter}[affiliation]")
        parts.append("open_access[filter]")
        if date_range:
            parts.append(date_range)
        full_query = " AND ".join(parts)

        esearch_url = (
            f"{self.config.base_url}/esearch.fcgi"
            f"?db=pmc&term={quote_plus(full_query)}"
            f"&retmax={max_results}&retmode=json&usehistory=n"
            f"{self._api_suffix}"
        )
        result = await self.http.fetch(esearch_url, use_cache=True, cache_ttl=7 * 24 * 3600)
        if not result.ok or not result.content:
            logger.error(f"[pmc_india] esearch failed: {result.error}")
            return []

        try:
            import json
            data = json.loads(result.content)
            pmc_ids = data.get("esearchresult", {}).get("idlist", [])
        except Exception as e:
            logger.error(f"[pmc_india] esearch JSON parse failed: {e}")
            return []

        logger.info(
            f"[pmc_india] discovered {len(pmc_ids)} PMC IDs for query: {full_query[:80]}"
        )

        jobs: list[CrawlJob] = []
        for pmc_id in pmc_ids:
            efetch_url = (
                f"{self.config.base_url}/efetch.fcgi"
                f"?db=pmc&id={pmc_id}&retmode=xml"
                f"{self._api_suffix}"
            )
            jobs.append(CrawlJob(
                url=efetch_url,
                source=self.config.name,
                discovered_from="api",
                metadata={
                    "pmc_id": pmc_id,
                    "pmc_id_full": f"PMC{pmc_id}",
                    "query": full_query,
                },
            ))
        return jobs

    async def discover_by_specialty(
        self,
        specialty: str,
        max_results: int = 200,
        date_range: str = "2015:2025[DP]",
    ) -> list[CrawlJob]:
        """Convenience: discover PMC articles for a medical specialty.

        Args:
            specialty: one of "cardiology", "endocrinology", "pediatrics",
                "oncology", "neurology", "infectious_disease", "public_health"
            max_results: cap on results
            date_range: date filter
        """
        specialty_queries = {
            "cardiology": "cardiology OR myocardial infarction OR heart failure",
            "endocrinology": "endocrinology OR diabetes OR thyroid",
            "pediatrics": "pediatrics OR child health OR neonatal",
            "oncology": "oncology OR cancer OR chemotherapy",
            "neurology": "neurology OR stroke OR epilepsy",
            "infectious_disease": "tuberculosis OR malaria OR dengue OR HIV",
            "public_health": "public health OR epidemiology OR preventive medicine",
        }
        query = specialty_queries.get(specialty, specialty)
        return await self.discover(query=query, max_results=max_results, date_range=date_range)
