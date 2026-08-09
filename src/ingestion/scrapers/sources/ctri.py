"""CTRI scraper — Clinical Trials Registry India.

CTRI (ctri.nic.in) is India's mandatory clinical trial registry. All
clinical trials conducted in India must register here before enrolling
patients. Public searchable database.

Useful for queries like:
- "Are there ongoing Indian trials for X?"
- "What interventions are being studied for Y in India?"
- "Who is sponsoring trials for Z?"

Site structure:
- Search at ctri.nic.in/Clinicaltrials/trialsearch.aspx
- Returns HTML table of trials
- Each trial has a detail page with: trial ID, title, sponsor, phase,
  enrollment, status, condition, intervention, locations

Politeness:
- 0.5 req/sec (NIC server)
- 2s crawl delay
"""
from __future__ import annotations

import re
from typing import Any
from urllib.parse import urljoin, urlencode

from bs4 import BeautifulSoup
from loguru import logger

from src.ingestion.scrapers.framework.base import BaseScraper, register_source
from src.ingestion.scrapers.framework.models import CrawlJob, SourceConfig

CTRI_CONFIG = SourceConfig(
    name="ctri",
    base_url="http://ctri.nic.in",
    rate_limit=0.5,
    crawl_delay=2.0,
    user_agent_suffix="CTRI-trials-indexer",
    fetch_strategy="http_first",
    expected_content_types=("text/html",),
    trust_tier=2,  # Registry = authoritative but trials may not have published results
    india_relevant_default=True,
    indian_source_default=True,
    extra={
        "search_path": "/Clinicaltrials/trialsearch.aspx",
        "page_size": 50,
    },
)


@register_source("ctri")
class CTRIScraper(BaseScraper):
    """Scraper for Clinical Trials Registry India."""

    config = CTRI_CONFIG

    async def discover(
        self,
        condition: str | None = None,
        intervention: str | None = None,
        sponsor: str | None = None,
        max_results: int = 500,
        **kwargs: Any,
    ) -> list[CrawlJob]:
        """Discover CTRI trial records.

        Args:
            condition: medical condition to search (e.g., "diabetes")
            intervention: drug/intervention name (e.g., "metformin")
            sponsor: trial sponsor name
            max_results: cap on results

        Returns:
            list of CrawlJobs pointing to trial detail pages
        """
        if not any([condition, intervention, sponsor]):
            # Default: browse recent trials
            return await self._discover_recent(max_results)

        # Build search URL
        params: dict[str, str] = {}
        if condition:
            params["cond"] = condition
        if intervention:
            params["intv"] = intervention
        if sponsor:
            params["spon"] = sponsor

        search_url = f"{self.config.base_url}{self.config.extra['search_path']}?{urlencode(params)}"
        result = await self.http.fetch(search_url, use_cache=True, cache_ttl=7 * 24 * 3600)
        if not result.ok or not result.content:
            logger.warning(f"[ctri] search failed: {result.error}")
            return []

        html = result.content.decode("utf-8", errors="replace")
        trial_urls = self._extract_trial_urls(html)

        jobs: list[CrawlJob] = []
        for url, trial_id, title in trial_urls[:max_results]:
            jobs.append(CrawlJob(
                url=url,
                source=self.config.name,
                discovered_from="search",
                metadata={
                    "trial_id": trial_id,
                    "title": title,
                    "search_condition": condition,
                    "search_intervention": intervention,
                },
            ))
        return jobs

    async def _discover_recent(self, max_results: int) -> list[CrawlJob]:
        """Discover recently registered trials (no search filter)."""
        search_url = f"{self.config.base_url}{self.config.extra['search_path']}"
        result = await self.http.fetch(search_url, use_cache=True, cache_ttl=24 * 3600)
        if not result.ok or not result.content:
            return []

        html = result.content.decode("utf-8", errors="replace")
        trial_urls = self._extract_trial_urls(html)

        jobs: list[CrawlJob] = []
        for url, trial_id, title in trial_urls[:max_results]:
            jobs.append(CrawlJob(
                url=url,
                source=self.config.name,
                discovered_from="recent",
                metadata={"trial_id": trial_id, "title": title},
            ))
        return jobs

    def _extract_trial_urls(self, html: str) -> list[tuple[str, str, str]]:
        """Extract trial URLs, IDs, and titles from CTRI search results HTML.

        Returns list of (url, trial_id, title) tuples.
        """
        soup = BeautifulSoup(html, "lxml")
        results: list[tuple[str, str, str]] = []
        seen: set[str] = set()

        # CTRI trial detail URLs contain "trialview" or "ShowTrial"
        for a in soup.find_all("a", href=True):
            href = a["href"]
            text = a.get_text(" ", strip=True)

            if "trialview" not in href.lower() and "showtrial" not in href.lower():
                continue
            if not text or len(text) < 5:
                continue

            absolute = urljoin(self.config.base_url, href)
            if absolute in seen:
                continue
            seen.add(absolute)

            # Try to extract trial ID from text or href
            trial_id = self._extract_trial_id(text + " " + href)
            results.append((absolute, trial_id, text))

        return results

    @staticmethod
    def _extract_trial_id(text: str) -> str:
        """Extract CTRI trial ID from text.

        CTRI IDs look like: CTRI/2024/01/012345 or CTRI/2023/05/067890
        """
        match = re.search(r"CTRI[/\d]+", text, re.IGNORECASE)
        if match:
            return match.group(0).rstrip("/")
        return "unknown"

    async def process(self, result, job: CrawlJob) -> Any:
        """Process a fetched CTRI trial detail page."""
        from src.ingestion.scrapers.framework.models import ScrapedDocument

        if not result.is_html:
            return ScrapedDocument(
                url=result.url,
                source=self.config.name,
                content=result.content,
                content_type=result.content_type or "",
                fetched_at=result.fetched_at,
                trust_tier=self.config.trust_tier,
                india_relevant=True,
                indian_source=True,
                metadata=job.metadata,
            )

        meta = self.metadata_extractor.extract(result.content, result.encoding or "utf-8")
        structured = self._extract_trial_fields(result.content.decode("utf-8", errors="replace"))
        meta["ctri_fields"] = structured

        return ScrapedDocument(
            url=result.url,
            source=self.config.name,
            content=result.content,
            content_type=result.content_type or "",
            title=structured.get("title") or meta.get("title") or job.metadata.get("title"),
            authors=[],
            journal="CTRI",
            doi=None,
            pmid=None,
            pubdate=structured.get("registration_date"),
            abstract=structured.get("brief_summary"),
            metadata={**meta, **job.metadata, "structured": structured},
            fetched_at=result.fetched_at,
            trust_tier=self.config.trust_tier,
            india_relevant=True,
            indian_source=True,
        )

    @staticmethod
    def _extract_trial_fields(html: str) -> dict[str, str]:
        """Extract structured trial fields from CTRI detail page.

        Common fields: trial_id, title, sponsor, phase, enrollment, status,
        condition, intervention, locations, registration_date, brief_summary
        """
        soup = BeautifulSoup(html, "lxml")
        fields: dict[str, str] = {}

        label_map = {
            "trial_id": ["trial registration number", "ctri number", "trial id"],
            "title": ["title of trial", "study title", "official title"],
            "sponsor": ["sponsor", "primary sponsor"],
            "phase": ["phase"],
            "enrollment": ["target sample size", "enrollment", "sample size"],
            "status": ["recruitment status", "trial status", "status"],
            "condition": ["condition", "disease", "indication"],
            "intervention": ["intervention", "study intervention"],
            "locations": ["sites", "study sites", "locations"],
            "registration_date": ["date of registration", "registration date"],
            "brief_summary": ["brief summary", "summary", "objective"],
        }

        for cells in soup.find_all(["tr", "div", "dl"]):
            text = cells.get_text(" ", strip=True).lower()
            for field, labels in label_map.items():
                if field in fields:
                    continue
                for label in labels:
                    if label in text:
                        value = CTRIScraper._find_value_after_label(cells, label)
                        if value and len(value) < 2000:
                            fields[field] = value
                            break

        return fields

    @staticmethod
    def _find_value_after_label(container: Any, label: str) -> str | None:
        """Find the value that follows a label in the HTML container."""
        for header in container.find_all(["th", "dt", "strong", "b", "label"]):
            if label.lower() in header.get_text(strip=True).lower():
                sibling = header.find_next_sibling()
                while sibling:
                    text = sibling.get_text(strip=True)
                    if text and text.lower() != label.lower():
                        return text[:2000]
                    sibling = sibling.find_next_sibling()
                parent = header.parent
                if parent:
                    next_td = parent.find_next_sibling(["td", "dd"])
                    if next_td:
                        text = next_td.get_text(strip=True)
                        if text:
                            return text[:2000]
        return None
