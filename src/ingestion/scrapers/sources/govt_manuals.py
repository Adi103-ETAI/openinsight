"""Government public health manuals scraper — Indian national programme guidelines.

These cross Layer 1 (foundational — they define standard practice) and
Layer 3 (clinical guidelines). Sources:

- NTEP (National TB Elimination Programme) — tbcindia.gov.in
  → TB diagnosis, DOTS, MDR-TB, pediatric TB guidelines
- NVBDCP (National Vector Borne Disease Control Programme) — nvbdcp.gov.in
  → Malaria, dengue, kala-azar, chikungunya guidelines
- NHM (National Health Mission) — nrhm.gov.in
  → RCH (maternal/child health), immunization, family planning
- NPCDCS (National Programme for Prevention and Control of Cancer,
  Diabetes, Cardiovascular Diseases and Stroke) — npcpcds.nic.in
  → NCD screening, management protocols

All are government-published PDFs. Each gets its own source_type so retrieval
can filter by programme.

Discovery: scrape the guidelines/Manuals page on each site for PDF links.
Parsing: handled by the pipeline's GROBID/pdfplumber chain.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
from loguru import logger

from src.ingestion.scrapers.framework.base import BaseScraper, register_source
from src.ingestion.scrapers.framework.models import CrawlJob, SourceConfig


@dataclass(frozen=True)
class GovtProgramme:
    """Configuration for one government health programme."""
    source_type: str  # "ntep" | "nvbdcp" | "nhm" | "npcds"
    base_url: str
    guidelines_paths: tuple[str, ...]
    trust_tier: int = 1  # Govt programmes = Tier 1
    india_relevant: bool = True
    indian_source: bool = True


# Programme configurations
GOVT_PROGRAMMES: dict[str, GovtProgramme] = {
    "ntep": GovtProgramme(
        source_type="ntep",
        base_url="https://tbcindia.gov.in",
        guidelines_paths=(
            "/index.php?lang=1&level=1&sublinkid=5465&lid=3180",  # Guidelines page
            "/writeReadData/l2images/aboutus-2.jpg",  # Fallbacks if structure changes
        ),
    ),
    "nvbdcp": GovtProgramme(
        source_type="nvbdcp",
        base_url="https://nvbdcp.gov.in",
        guidelines_paths=(
            "/index1.php?lang=1&level=1&sublinkid=5775&lid=3649",
            "/Guidelines.html",
        ),
    ),
    "nhm": GovtProgramme(
        source_type="nhm",
        base_url="https://nhm.gov.in",
        guidelines_paths=(
            "/index1.php?lang=1&level=1&sublinkid=46&lid=46",
            "/guidelines.html",
        ),
    ),
    "npcds": GovtProgramme(
        source_type="npcds",
        base_url="https://npcdc.nic.in",
        guidelines_paths=(
            "/index1.php?lang=1&level=1&sublinkid=46&lid=46",
            "/WriteReadData/userfiles/4image/NPCDCS_Guidelines.pdf",
        ),
    ),
}

# Build a SourceConfig for each programme
for prog_key, prog in GOVT_PROGRAMMES.items():
    # Each programme gets its own registered scraper with its own config
    pass  # registration happens in the factory below


def make_govt_config(programme: GovtProgramme) -> SourceConfig:
    """Build a SourceConfig for a specific government programme."""
    return SourceConfig(
        name=programme.source_type,
        base_url=programme.base_url,
        rate_limit=0.5,  # Indian gov servers are slow
        crawl_delay=2.0,
        user_agent_suffix=f"{programme.source_type.upper()}-indexer",
        fetch_strategy="http_first",
        expected_content_types=("text/html", "application/pdf"),
        trust_tier=programme.trust_tier,
        india_relevant_default=programme.india_relevant,
        indian_source_default=programme.indian_source,
        extra={
            "guidelines_paths": programme.guidelines_paths,
            "programme_key": programme.source_type,
        },
    )


class GovtManualsScraper(BaseScraper):
    """Generic scraper for Indian government health programme manuals.

    Instantiate with a programme key ("ntep", "nvbdcp", "nhm", "npcds")
    to get a programme-specific scraper.
    """

    # Subclasses set this — base class is abstract
    config: SourceConfig  # type: ignore[assignment]

    async def discover(
        self,
        max_results: int = 50,
        **kwargs: Any,
    ) -> list[CrawlJob]:
        """Discover PDF guidelines from the programme's guidelines page."""
        jobs: list[CrawlJob] = []

        for path in self.config.extra["guidelines_paths"]:
            if len(jobs) >= max_results:
                break
            url = f"{self.config.base_url}{path}"
            try:
                pdf_jobs = await self._discover_from_page(url, max_results - len(jobs))
                jobs.extend(pdf_jobs)
                logger.info(
                    f"[{self.config.name}] {path}: discovered {len(pdf_jobs)} PDFs"
                )
            except Exception as e:
                logger.warning(f"[{self.config.name}] {path}: discovery failed: {e}")

        return jobs[:max_results]

    async def _discover_from_page(self, page_url: str, max_pdfs: int) -> list[CrawlJob]:
        """Discover PDF links from a single guidelines page."""
        result = await self.http.fetch(page_url, use_cache=True, cache_ttl=7 * 24 * 3600)
        if not result.ok or not result.content:
            return []

        html = result.content.decode("utf-8", errors="replace")
        pdf_links = self._extract_pdf_links(html, page_url)

        jobs: list[CrawlJob] = []
        for pdf_url, title in pdf_links[:max_pdfs]:
            jobs.append(CrawlJob(
                url=pdf_url,
                source=self.config.name,
                discovered_from="link-extraction",
                metadata={
                    "title": title,
                    "discovered_from_url": page_url,
                },
            ))
        return jobs

    def _extract_pdf_links(self, html: str, base_url: str) -> list[tuple[str, str]]:
        """Extract PDF links with link text as title."""
        soup = BeautifulSoup(html, "lxml")
        links: list[tuple[str, str]] = []
        seen: set[str] = set()

        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            text = a.get_text(" ", strip=True)

            if not href.lower().endswith(".pdf") and ".pdf" not in href.lower():
                continue

            absolute = urljoin(base_url, href)
            if absolute in seen:
                continue
            seen.add(absolute)
            links.append((absolute, text or "Untitled Guideline"))

        return links

    async def process(self, result, job: CrawlJob) -> Any:
        """Process a fetched government PDF."""
        from src.ingestion.scrapers.framework.models import ScrapedDocument

        return ScrapedDocument(
            url=result.url,
            source=self.config.name,
            content=result.content,
            content_type=result.content_type or "application/pdf",
            title=job.metadata.get("title"),
            authors=[],
            journal=self.config.name.upper(),
            doi=None,
            pmid=None,
            pubdate=None,
            abstract=None,
            metadata={**job.metadata, "is_pdf": True},
            fetched_at=result.fetched_at,
            trust_tier=self.config.trust_tier,
            india_relevant=True,
            indian_source=True,
        )


# Register a scraper class for each programme
def _register_govt_scrapers() -> None:
    """Dynamically register a scraper class for each government programme."""
    for prog_key, programme in GOVT_PROGRAMMES.items():
        config = make_govt_config(programme)

        # Create a subclass with the right config
        cls_name = f"{prog_key.upper()}Scraper"
        cls = type(cls_name, (GovtManualsScraper,), {"config": config})

        # Register under the programme's source_type
        from src.ingestion.scrapers.framework.base import _SOURCE_REGISTRY
        _SOURCE_REGISTRY[programme.source_type] = cls


_register_govt_scrapers()
