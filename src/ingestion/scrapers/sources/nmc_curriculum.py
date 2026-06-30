"""NMC curriculum scraper — National Medical Commission competency frameworks.

NMC (nmc.org.in) publishes:
- PG curriculum guidelines (per specialty)
- UG competency-based curriculum
- Various training + assessment guidelines

These define what Indian doctors are taught — foundational knowledge
for the "school-level" Layer 1.

Source PDFs are on nmc.org.in/resources and linked subdomains. Structure:
- Browse /resources → list of guideline PDFs
- Each PDF has a title + publication date
- PDF content parsed via the pipeline's GROBID/pdfplumber chain

Discovery:
- Scrape the resources page for PDF links
- Filter for "curriculum" or "competency" in title/text

Politeness:
- 0.5 req/sec (Indian gov servers are slow)
- 2s crawl delay
"""
from __future__ import annotations

import re
from typing import Any
from urllib.parse import urljoin

from bs4 import BeautifulSoup
from loguru import logger

from src.ingestion.scrapers.framework.base import BaseScraper, register_source
from src.ingestion.scrapers.framework.models import CrawlJob, SourceConfig

NMC_CONFIG = SourceConfig(
    name="nmc_curriculum",
    base_url="https://www.nmc.org.in",
    rate_limit=0.5,
    crawl_delay=2.0,
    user_agent_suffix="NMC-curriculum-indexer",
    fetch_strategy="http_first",
    expected_content_types=("text/html", "application/pdf"),
    trust_tier=1,  # NMC = apex medical education body in India
    india_relevant_default=True,
    indian_source_default=True,
    extra={
        "resources_paths": [
            "/resources/medical-education",
            "/guidelines/pg-education",
            "/guidelines/ug-education",
        ],
        "pdf_link_patterns": [
            r"curriculum",
            r"competency",
            r"syllabus",
            r"training",
        ],
    },
)


@register_source("nmc_curriculum")
class NMCCurriculumScraper(BaseScraper):
    """Scraper for NMC competency frameworks and curriculum guidelines."""

    config = NMC_CONFIG

    async def discover(
        self,
        max_results: int = 100,
        **kwargs: Any,
    ) -> list[CrawlJob]:
        """Discover NMC curriculum + guideline PDFs.

        Args:
            max_results: cap on total PDFs to discover

        Returns:
            list of CrawlJobs pointing to PDF URLs
        """
        jobs: list[CrawlJob] = []

        for path in self.config.extra["resources_paths"]:
            if len(jobs) >= max_results:
                break
            url = f"{self.config.base_url}{path}"
            try:
                pdf_jobs = await self._discover_from_page(url, max_results - len(jobs))
                jobs.extend(pdf_jobs)
                logger.info(f"[nmc] {path}: discovered {len(pdf_jobs)} PDFs")
            except Exception as e:
                logger.warning(f"[nmc] {path}: discovery failed: {e}")

        return jobs[:max_results]

    async def _discover_from_page(self, page_url: str, max_pdfs: int) -> list[CrawlJob]:
        """Discover PDF links from a single NMC resources page."""
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
        """Extract PDF links with their link text (used as title).

        Filters for curriculum/competency-related PDFs using the
        pdf_link_patterns from config.
        """
        soup = BeautifulSoup(html, "lxml")
        links: list[tuple[str, str]] = []
        seen: set[str] = set()
        patterns = [re.compile(p, re.IGNORECASE) for p in self.config.extra["pdf_link_patterns"]]

        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            text = a.get_text(" ", strip=True)

            # Must be a PDF link
            if not href.lower().endswith(".pdf") and "pdf" not in href.lower():
                continue

            # Must match at least one pattern (curriculum/competency/etc.)
            combined = f"{text} {href}".lower()
            if not any(p.search(combined) for p in patterns):
                continue

            absolute = urljoin(base_url, href)
            if absolute in seen:
                continue
            seen.add(absolute)
            links.append((absolute, text or "Untitled"))

        return links

    async def process(self, result, job: CrawlJob) -> Any:
        """Process a fetched NMC PDF.

        NMC PDFs are parsed by the pipeline's GROBID/pdfplumber chain,
        so we just record the content + provenance.
        """
        from src.ingestion.scrapers.framework.models import ScrapedDocument

        return ScrapedDocument(
            url=result.url,
            source=self.config.name,
            content=result.content,
            content_type=result.content_type or "application/pdf",
            title=job.metadata.get("title"),
            authors=[],
            journal="NMC",
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
