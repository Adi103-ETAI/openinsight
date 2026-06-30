"""PvPI scraper — Pharmacovigilance Programme of India drug safety alerts.

PvPI (ipc.gov.in/PvPI) publishes:
- Monthly drug safety alerts (PDF) — list of drugs with new ADR signals
- Quarterly newsletters
- Signal detection reports

Useful for queries like "is there a safety alert for X?" and surfaces
recent safety concerns in RAG answers about drugs.

Site structure:
- Browse ipc.gov.in/PvPI/alerts
- Each month has a PDF download link
- PDF content: table of drug name + ADR + severity + action

Politeness:
- 0.5 req/sec
- 2s crawl delay
- 30-day cache on alert PDFs
"""
from __future__ import annotations

import re
from typing import Any
from urllib.parse import urljoin

from bs4 import BeautifulSoup
from loguru import logger

from src.ingestion.scrapers.framework.base import BaseScraper, register_source
from src.ingestion.scrapers.framework.models import CrawlJob, SourceConfig

PVPI_CONFIG = SourceConfig(
    name="pvpi",
    base_url="https://www.ipc.gov.in",
    rate_limit=0.5,
    crawl_delay=2.0,
    user_agent_suffix="PvPI-safety-indexer",
    fetch_strategy="http_first",
    expected_content_types=("text/html", "application/pdf"),
    trust_tier=1,  # PvPI = official pharmacovigilance authority
    india_relevant_default=True,
    indian_source_default=True,
    extra={
        "alerts_path": "/PvPI/alerts.html",
        "newsletter_path": "/PvPI/newsletter.html",
        "pdf_link_patterns": [
            r"alert",
            r"signal",
            r"safety",
            r"adr",
            r"adverse",
        ],
    },
)


@register_source("pvpi")
class PVPIScraper(BaseScraper):
    """Scraper for PvPI drug safety alerts."""

    config = PVPI_CONFIG

    async def discover(
        self,
        alert_type: str = "alerts",
        max_results: int = 100,
        **kwargs: Any,
    ) -> list[CrawlJob]:
        """Discover PvPI safety alert PDFs.

        Args:
            alert_type: "alerts" (monthly alerts) or "newsletter" (quarterly)
            max_results: cap on results

        Returns:
            list of CrawlJobs pointing to PDF URLs
        """
        if alert_type == "alerts":
            path = self.config.extra["alerts_path"]
        elif alert_type == "newsletter":
            path = self.config.extra["newsletter_path"]
        else:
            raise ValueError(f"Invalid alert_type: {alert_type}")

        page_url = f"{self.config.base_url}{path}"
        result = await self.http.fetch(page_url, use_cache=True, cache_ttl=7 * 24 * 3600)
        if not result.ok or not result.content:
            logger.warning(f"[pvpi] {path}: fetch failed: {result.error}")
            return []

        html = result.content.decode("utf-8", errors="replace")
        pdf_links = self._extract_pdf_links(html, page_url)

        jobs: list[CrawlJob] = []
        for pdf_url, title, month_year in pdf_links[:max_results]:
            jobs.append(CrawlJob(
                url=pdf_url,
                source=self.config.name,
                discovered_from="link-extraction",
                metadata={
                    "title": title,
                    "alert_type": alert_type,
                    "month_year": month_year,
                    "is_pdf": True,
                },
            ))
        return jobs

    def _extract_pdf_links(self, html: str, base_url: str) -> list[tuple[str, str, str]]:
        """Extract PDF links with title and optional month/year.

        Returns list of (url, title, month_year) tuples.
        """
        soup = BeautifulSoup(html, "lxml")
        links: list[tuple[str, str, str]] = []
        seen: set[str] = set()
        patterns = [re.compile(p, re.IGNORECASE) for p in self.config.extra["pdf_link_patterns"]]

        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            text = a.get_text(" ", strip=True)

            if not href.lower().endswith(".pdf") and "pdf" not in href.lower():
                continue

            # Must match at least one pattern (alert/signal/safety/adr)
            combined = f"{text} {href}".lower()
            if not any(p.search(combined) for p in patterns):
                continue

            absolute = urljoin(base_url, href)
            if absolute in seen:
                continue
            seen.add(absolute)

            # Try to extract month/year from link text (e.g., "January 2024 Alert")
            month_year = self._extract_month_year(text)

            links.append((absolute, text or "Untitled Alert", month_year))

        return links

    @staticmethod
    def _extract_month_year(text: str) -> str:
        """Extract month and year from link text.

        Examples: "January 2024 Drug Safety Alert" → "January 2024"
                  "Q1 2024 Newsletter" → "Q1 2024"
        """
        # Month name + year
        match = re.search(
            r"(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{4})",
            text, re.IGNORECASE,
        )
        if match:
            return f"{match.group(1).title()} {match.group(2)}"

        # Quarter + year
        match = re.search(r"(q[1-4])\s+(\d{4})", text, re.IGNORECASE)
        if match:
            return f"{match.group(1).upper()} {match.group(2)}"

        # Just year
        match = re.search(r"(20\d{2})", text)
        if match:
            return match.group(1)

        return ""

    async def process(self, result, job: CrawlJob) -> Any:
        """Process a fetched PvPI PDF alert."""
        from src.ingestion.scrapers.framework.models import ScrapedDocument

        return ScrapedDocument(
            url=result.url,
            source=self.config.name,
            content=result.content,
            content_type=result.content_type or "application/pdf",
            title=job.metadata.get("title"),
            authors=[],
            journal="PvPI Drug Safety Alerts",
            doi=None,
            pmid=None,
            pubdate=job.metadata.get("month_year"),
            abstract=None,
            metadata={**job.metadata, "is_pdf": True},
            fetched_at=result.fetched_at,
            trust_tier=self.config.trust_tier,
            india_relevant=True,
            indian_source=True,
        )
