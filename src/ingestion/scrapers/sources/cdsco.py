"""CDSCO scraper — Central Drugs Standard Control Organization approved drugs.

CDSCO (cdsco.gov.in) is India's FDA equivalent. The SUGAM portal
(cdscoonline.gov.in) has:
- Approved drugs database (searchable by drug name, manufacturer, etc.)
- New drug approvals (monthly circulars)
- Clinical trial approvals
- Ban/restriction notifications

This scraper focuses on the approved drugs database — the structured list
of every drug approved for marketing in India. Each record has:
- Drug name (generic + brand)
- Manufacturer
- Approval date
- Indication
- Strength / dosage form

Useful for queries like "is semaglutide approved in India?" and
"when was drug X approved?".

Site structure:
- Search form at cdscoonline.gov.in/DrugSearch
- POST returns HTML table with results
- Pagination via offset parameter

Politeness:
- 0.5 req/sec (Indian gov servers)
- 2s crawl delay
- 30-day cache on individual drug records (rarely change)
"""
from __future__ import annotations

import re
from typing import Any
from urllib.parse import urljoin, urlencode

from bs4 import BeautifulSoup
from loguru import logger

from src.ingestion.scrapers.framework.base import BaseScraper, register_source
from src.ingestion.scrapers.framework.models import CrawlJob, SourceConfig

CDSCO_CONFIG = SourceConfig(
    name="cdsco",
    base_url="https://cdscoonline.gov.in",
    rate_limit=0.5,
    crawl_delay=2.0,
    user_agent_suffix="CDSCO-drugs-indexer",
    fetch_strategy="http_first",
    expected_content_types=("text/html", "application/pdf"),
    trust_tier=1,  # CDSCO = India's FDA equivalent
    india_relevant_default=True,
    indian_source_default=True,
    extra={
        "search_path": "/DrugSearch",
        "approved_drugs_path": "/cdsDrugs/drugsController?drugSearch=approved",
        "new_drugs_path": "/cdsDrugs/drugsController?drugSearch=new",
        "banned_drugs_path": "/cdsDrugs/drugsController?drugSearch=banned",
        "page_size": 50,
    },
)


@register_source("cdsco")
class CDSCOScraper(BaseScraper):
    """Scraper for CDSCO approved drugs database."""

    config = CDSCO_CONFIG

    async def discover(
        self,
        search_type: str = "approved",
        max_results: int = 1000,
        **kwargs: Any,
    ) -> list[CrawlJob]:
        """Discover drug records from CDSCO.

        Args:
            search_type: "approved" (default), "new", or "banned"
            max_results: cap on total records to discover

        Returns:
            list of CrawlJobs, one per drug record URL
        """
        if search_type not in ("approved", "new", "banned"):
            raise ValueError(f"Invalid search_type: {search_type}")

        path_key = f"{search_type}_drugs_path"
        path = self.config.extra.get(path_key)
        if not path:
            logger.error(f"[cdsco] no path configured for search_type={search_type}")
            return []

        jobs: list[CrawlJob] = []
        offset = 0
        page_size = self.config.extra["page_size"]

        while len(jobs) < max_results:
            # Build search URL with pagination
            params = urlencode({
                "searchType": search_type,
                "page": offset // page_size,
                "size": page_size,
            })
            search_url = f"{self.config.base_url}{path}&{params}"

            result = await self.http.fetch(search_url, use_cache=True, cache_ttl=7 * 24 * 3600)
            if not result.ok or not result.content:
                logger.warning(f"[cdsco] page {offset}: fetch failed: {result.error}")
                break

            html = result.content.decode("utf-8", errors="replace")
            record_urls = self._extract_drug_record_urls(html)

            if not record_urls:
                logger.info(f"[cdsco] no more records at offset {offset}")
                break

            for url, drug_name in record_urls:
                if len(jobs) >= max_results:
                    break
                jobs.append(CrawlJob(
                    url=url,
                    source=self.config.name,
                    discovered_from="search",
                    metadata={
                        "drug_name": drug_name,
                        "search_type": search_type,
                    },
                ))

            offset += page_size
            logger.info(f"[cdsco] discovered {len(jobs)}/{max_results} records so far")

        return jobs

    async def discover_by_drug_name(
        self,
        drug_name: str,
        max_results: int = 50,
    ) -> list[CrawlJob]:
        """Discover CDSCO records for a specific drug name.

        Args:
            drug_name: drug name to search (e.g., "metformin", "semaglutide")
            max_results: cap on results
        """
        search_url = (
            f"{self.config.base_url}{self.config.extra['approved_drugs_path']}"
            f"&drugName={drug_name}"
        )
        result = await self.http.fetch(search_url, use_cache=True, cache_ttl=30 * 24 * 3600)
        if not result.ok or not result.content:
            logger.warning(f"[cdsco] drug search for '{drug_name}' failed: {result.error}")
            return []

        html = result.content.decode("utf-8", errors="replace")
        record_urls = self._extract_drug_record_urls(html)

        jobs: list[CrawlJob] = []
        for url, name in record_urls[:max_results]:
            jobs.append(CrawlJob(
                url=url,
                source=self.config.name,
                discovered_from="search",
                metadata={
                    "drug_name": name,
                    "search_type": "approved",
                    "search_query": drug_name,
                },
            ))
        return jobs

    def _extract_drug_record_urls(self, html: str) -> list[tuple[str, str]]:
        """Extract drug record URLs + drug names from CDSCO search results HTML.

        CDSCO search results are HTML tables where each row links to a
        drug detail page. The link text usually contains the drug name.
        """
        soup = BeautifulSoup(html, "lxml")
        records: list[tuple[str, str]] = []
        seen: set[str] = set()

        # Look for table rows with links to drug detail pages
        for a in soup.find_all("a", href=True):
            href = a["href"]
            text = a.get_text(" ", strip=True)

            # CDSCO drug detail URLs typically contain "drugDetail" or "viewDrug"
            if "drugDetail" not in href and "viewDrug" not in href and "drug-detail" not in href:
                continue
            if not text or len(text) < 2:
                continue

            absolute = urljoin(self.config.base_url, href)
            if absolute in seen:
                continue
            seen.add(absolute)
            records.append((absolute, text))

        return records

    async def process(self, result, job: CrawlJob) -> Any:
        """Process a fetched CDSCO drug record page.

        CDSCO drug detail pages show structured fields: drug name, manufacturer,
        approval date, strength, indication. We extract these and store
        them in ScrapedDocument.metadata for the parser to use.
        """
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
                metadata={**job.metadata, "is_pdf": True},
            )

        # Default metadata extraction (Highwire + JSON-LD + DC + OG)
        meta = self.metadata_extractor.extract(result.content, result.encoding or "utf-8")

        # CDSCO-specific: extract structured drug fields from the HTML table
        structured = self._extract_drug_fields(result.content.decode("utf-8", errors="replace"))
        meta["cdsco_fields"] = structured

        return ScrapedDocument(
            url=result.url,
            source=self.config.name,
            content=result.content,
            content_type=result.content_type or "",
            title=structured.get("drug_name") or meta.get("title") or job.metadata.get("drug_name"),
            authors=[],
            journal="CDSCO",
            doi=None,
            pmid=None,
            pubdate=structured.get("approval_date"),
            abstract=None,
            metadata={**meta, **job.metadata, "structured": structured},
            fetched_at=result.fetched_at,
            trust_tier=self.config.trust_tier,
            india_relevant=True,
            indian_source=True,
        )

    @staticmethod
    def _extract_drug_fields(html: str) -> dict[str, str]:
        """Extract structured drug fields from a CDSCO drug detail page.

        CDSCO pages use label-value pairs in a definition-list or table layout.
        We look for common field labels and extract the following text.
        """
        soup = BeautifulSoup(html, "lxml")
        fields: dict[str, str] = {}

        # Common CDSCO field labels (case-insensitive)
        label_map = {
            "drug_name": ["drug name", "name of drug", "generic name"],
            "brand_name": ["brand name", "trade name"],
            "manufacturer": ["manufacturer", "firm", "marketing company"],
            "approval_date": ["approval date", "date of approval", "approved on"],
            "indication": ["indication", "therapeutic indication", "use"],
            "strength": ["strength", "dosage form", "formulation"],
            "schedule": ["schedule", "drug schedule"],
            "batch_no": ["batch no", "batch number"],
        }

        # Look for <th> or <td> with label text, then extract the next cell
        for cells in soup.find_all(["tr", "div", "dl"]):
            text = cells.get_text(" ", strip=True).lower()
            for field, labels in label_map.items():
                if field in fields:
                    continue
                for label in labels:
                    if label in text:
                        # Try to find the value (next sibling or next cell)
                        value = CDSCOScraper._find_value_after_label(cells, label)
                        if value and len(value) < 500:
                            fields[field] = value
                            break

        return fields

    @staticmethod
    def _find_value_after_label(container: Any, label: str) -> str | None:
        """Find the value that follows a label in the HTML container.

        Handles table layouts (<th>label</th><td>value</td>) and
        definition lists (<dt>label</dt><dd>value</dd>).
        """
        # Method 1: look for <th> or <dt> with label text, then sibling
        for header in container.find_all(["th", "dt", "strong", "b", "label"]):
            if label.lower() in header.get_text(strip=True).lower():
                # Find next sibling that has text
                sibling = header.find_next_sibling()
                while sibling:
                    text = sibling.get_text(strip=True)
                    if text and text.lower() != label.lower():
                        return text[:500]
                    sibling = sibling.find_next_sibling()
                # Or parent's next cell
                parent = header.parent
                if parent:
                    next_td = parent.find_next_sibling(["td", "dd"])
                    if next_td:
                        text = next_td.get_text(strip=True)
                        if text:
                            return text[:500]
        return None
