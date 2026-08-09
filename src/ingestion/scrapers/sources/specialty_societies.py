"""Specialty society guidelines scraper — RSSDI, CSI, ISCCM, IAP, FOGSI, AIOS, ISN.

Indian specialty societies publish consensus statements and clinical
practice guidelines. These are the India-specific clinical guidelines that
make OpenInsight useful for day-to-day practice.

Societies covered:
- RSSDI (Research Society for the Study of Diabetes in India) — rssdi.in
  → Annual diabetes management guidelines
- CSI (Cardiological Society of India) — csi-india.org
  → Hypertension, dyslipidemia, heart failure consensus
- ISCCM (Indian Society of Critical Care Medicine) — isccm.org
  → ICU guidelines, sepsis, ARDS
- IAP (Indian Academy of Pediatrics) — iapindia.org
  → Immunization, growth, pediatric infectious disease
- FOGSI (Federation of Obstetric and Gynaecological Societies of India) — fogsi.org
  → Maternal health, contraceptive guidelines
- AIOS (All India Ophthalmological Society) — aios.org
  → Eye care guidelines
- ISN (Indian Society of Nephrology) — isnindia.org
  → CKD, dialysis guidelines

All sites publish guidelines as PDFs. This scraper discovers PDF links
from each society's guidelines/resources page. PDF parsing handled by
the pipeline's GROBID chain.

Politeness:
- 1 req/sec (society sites are usually on commercial hosting)
- 1s crawl delay
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import urljoin

from bs4 import BeautifulSoup
from loguru import logger

from src.ingestion.scrapers.framework.base import BaseScraper, register_source
from src.ingestion.scrapers.framework.models import CrawlJob, SourceConfig


@dataclass(frozen=True)
class SpecialtySociety:
    """Configuration for one specialty society."""
    source_type: str
    base_url: str
    guidelines_paths: tuple[str, ...]
    full_name: str
    trust_tier: int = 1  # Specialty society guidelines = Tier 1
    india_relevant: bool = True
    indian_source: bool = True


SPECIALTY_SOCIETIES: dict[str, SpecialtySociety] = {
    "rssdi": SpecialtySociety(
        source_type="rssdi",
        base_url="https://rssdi.in",
        guidelines_paths=("/clinical-practice-guidelines/", "/guidelines/"),
        full_name="Research Society for the Study of Diabetes in India",
    ),
    "csi": SpecialtySociety(
        source_type="csi",
        base_url="https://csi-india.org",
        guidelines_paths=("/guidelines/", "/consensus-statements/"),
        full_name="Cardiological Society of India",
    ),
    "isccm": SpecialtySociety(
        source_type="isccm",
        base_url="https://isccm.org",
        guidelines_paths=("/guidelines/", "/publications/"),
        full_name="Indian Society of Critical Care Medicine",
    ),
    "iap": SpecialtySociety(
        source_type="iap",
        base_url="https://www.iapindia.org",
        guidelines_paths=("/guidelines/", "/protocols/"),
        full_name="Indian Academy of Pediatrics",
    ),
    "fogsi": SpecialtySociety(
        source_type="fogsi",
        base_url="https://www.fogsi.org",
        guidelines_paths=("/guidelines/", "/clinical-guidelines/"),
        full_name="Federation of Obstetric and Gynaecological Societies of India",
    ),
    "aios": SpecialtySociety(
        source_type="aios",
        base_url="https://www.aios.org",
        guidelines_paths=("/guidelines/", "/publications/"),
        full_name="All India Ophthalmological Society",
    ),
    "isn": SpecialtySociety(
        source_type="isn",
        base_url="https://isnindia.org",
        guidelines_paths=("/guidelines/", "/resources/"),
        full_name="Indian Society of Nephrology",
    ),
}


def make_society_config(society: SpecialtySociety) -> SourceConfig:
    """Build a SourceConfig for a specific specialty society."""
    return SourceConfig(
        name=society.source_type,
        base_url=society.base_url,
        rate_limit=1.0,  # Society sites are on commercial hosting, can handle 1 req/sec
        crawl_delay=1.0,
        user_agent_suffix=f"{society.source_type.upper()}-guidelines-indexer",
        fetch_strategy="http_first",
        expected_content_types=("text/html", "application/pdf"),
        trust_tier=society.trust_tier,
        india_relevant_default=society.india_relevant,
        indian_source_default=society.indian_source,
        extra={
            "guidelines_paths": society.guidelines_paths,
            "full_name": society.full_name,
        },
    )


class SpecialtySocietyScraper(BaseScraper):
    """Generic scraper for Indian specialty society guidelines.

    Instantiate with a society key ("rssdi", "csi", etc.) to get a
    society-specific scraper. Each society's guidelines page is scraped
    for PDF links.
    """

    config: SourceConfig  # type: ignore[assignment]

    async def discover(
        self,
        max_results: int = 50,
        **kwargs: Any,
    ) -> list[CrawlJob]:
        """Discover guideline PDFs from the society's guidelines page."""
        jobs: list[CrawlJob] = []

        for path in self.config.extra["guidelines_paths"]:
            if len(jobs) >= max_results:
                break
            url = f"{self.config.base_url}{path}"
            try:
                pdf_jobs = await self._discover_from_page(url, max_results - len(jobs))
                jobs.extend(pdf_jobs)
                logger.info(
                    f"[{self.config.name}] {path}: discovered {len(pdf_jobs)} guideline PDFs"
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
                    "society": self.config.extra["full_name"],
                    "discovered_from_url": page_url,
                    "is_pdf": True,
                },
            ))
        return jobs

    def _extract_pdf_links(self, html: str, base_url: str) -> list[tuple[str, str]]:
        """Extract PDF links with link text as title.

        Filters for guideline/consensus/protocol related PDFs to avoid
        downloading every PDF on the site (annual reports, membership forms, etc.)
        """
        soup = BeautifulSoup(html, "lxml")
        links: list[tuple[str, str]] = []
        seen: set[str] = set()

        # Patterns that indicate a guideline/consensus document
        guideline_patterns = [
            re.compile(p, re.IGNORECASE) for p in [
                r"guideline",
                r"consensus",
                r"protocol",
                r"recommendation",
                r"statement",
                r"position paper",
                r"practice parameter",
            ]
        ]

        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            text = a.get_text(" ", strip=True)

            # Must be a PDF link
            if not href.lower().endswith(".pdf") and ".pdf" not in href.lower():
                continue

            # Must match at least one guideline pattern (in text or href)
            combined = f"{text} {href}".lower()
            if not any(p.search(combined) for p in guideline_patterns):
                continue

            absolute = urljoin(base_url, href)
            if absolute in seen:
                continue
            seen.add(absolute)
            links.append((absolute, text or "Untitled Guideline"))

        return links

    async def process(self, result, job: CrawlJob) -> Any:
        """Process a fetched society guideline PDF."""
        from src.ingestion.scrapers.framework.models import ScrapedDocument

        return ScrapedDocument(
            url=result.url,
            source=self.config.name,
            content=result.content,
            content_type=result.content_type or "application/pdf",
            title=job.metadata.get("title"),
            authors=[],
            journal=self.config.extra["full_name"],
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


# Register a scraper class for each specialty society
def _register_society_scrapers() -> None:
    """Dynamically register a scraper class for each specialty society."""
    for society_key, society in SPECIALTY_SOCIETIES.items():
        config = make_society_config(society)

        # Create a subclass with the right config
        cls_name = f"{society_key.upper()}Scraper"
        cls = type(cls_name, (SpecialtySocietyScraper,), {"config": config})

        # Register under the society's source_type
        from src.ingestion.scrapers.framework.base import _SOURCE_REGISTRY
        _SOURCE_REGISTRY[society.source_type] = cls


_register_society_scrapers()
