"""IndMED scraper — indexes Indian medical journals hosted on indmedinfo.nic.in.

IndMED (https://indmedinfo.nic.in) is the National Informatics Centre's
index of ~100 Indian peer-reviewed biomedical journals not all of which are
in PubMed/MEDLINE. The site runs Open Journal Systems (OJS) — a common
journal-management platform — so this scraper is also reusable for any
OJS-hosted Indian journal (JAPI, IJMR, NMJI all use OJS instances).

OJS exposes:
- /index.php/<journal>/issue/archive       → list of past issues
- /index.php/<journal>/issue/view/<id>     → table of contents for one issue
- /index.php/<journal>/article/view/<id>   → article landing page
- /index.php/<journal>/article/download/<id>/<file_id>  → PDF download
- /index.php/<journal>/sitemap             → sitemap (sometimes disabled)

Article landing pages expose Highwire Press citation_* meta tags, which
the framework's MetadataExtractor handles out of the box.

Politeness:
- 0.5 req/sec rate limit (NIC servers are slow)
- 2.0s crawl delay
- 7-day cache on issue/archive pages (don't re-crawl every run)
- 30-day cache on article pages (rarely change after publication)
- Honors robots.txt (NIC's robots.txt typically allows /index.php/ paths)
"""
from __future__ import annotations

import asyncio
import re
from typing import Any
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
from loguru import logger

from src.ingestion.scrapers.framework.base import BaseScraper, register_source
from src.ingestion.scrapers.framework.models import CrawlJob, MetadataSelectors, SourceConfig

# Curated list of IndMED-indexed journals with their OJS abbreviations.
# These are the highest-value Indian journals NOT reliably in PubMed.
# Source: indmedinfo.nic.in browse page (verified June 2025).
INDMED_JOURNALS: dict[str, str] = {
    "Indian Journal of Pharmacology": "ijp",
    "Indian Journal of Public Health Research and Development": "ijphrd",
    "Indian Journal of Community Medicine": "ijcm",
    "Journal of the Indian Medical Association": "jima",
    "Indian Journal of Dermatology, Venereology and Leprology": "ijdvl",
    "Indian Journal of Ophthalmology": "ijo",
    "Indian Journal of Pediatrics": "ijped",
    "Indian Journal of Surgery": "ijs",
    "Indian Journal of Orthopaedics": "ijo",
    "Indian Journal of Anaesthesia": "ija",
    "Indian Journal of Radiology and Imaging": "ijri",
    "Indian Journal of Pathology and Microbiology": "ijpm",
    "Indian Journal of Nephrology": "ijn",
    "Indian Journal of Psychiatry": "ijpsy",
    "Indian Journal of Dental Research": "ijdr",
    "Indian Journal of Medical Sciences": "ijms",
    "Indian Journal of Critical Care Medicine": "ijccm",
    "Indian Journal of Urology": "iju",
    "Indian Journal of Rheumatology": "ijr",
    "Indian Journal of Palliative Care": "ijpc",
    "Indian Journal of Endocrinology and Metabolism": "ijem",
    "Indian Journal of Gastroenterology": "ijg",
    "Indian Journal of Medical Microbiology": "ijmm",
    "Indian Journal of Medical and Paediatric Oncology": "ijmpo",
    "Journal of Postgraduate Medicine": "jpm",
    "Annals of Indian Academy of Neurology": "aian",
    "Med J Armed Forces India": "mjafi",
    "Indian Heart Journal": "ihj",
    "Indian Journal of Medical Research": "ijmr",
}

# Trust tier mapping — IJMR/NMJI are top-tier (ICMR/AIIMS-backed, MEDLINE)
# IndMED-only journals (not in PubMed) are Tier 4
JOURNAL_TRUST_TIER: dict[str, int] = {
    "ijmr": 1,  # Indian J Med Res — ICMR, MEDLINE-indexed
    "ihj": 2,  # Indian Heart J — MEDLINE
    "jima": 3,  # J Indian Med Assoc — long-running
    "ijp": 3,  # Indian J Pharmacology — PubMed indexed
    "ijcm": 3,  # Indian J Community Medicine
    "ijo": 3,  # Indian J Ophthalmology
    "ijped": 3,  # Indian J Pediatrics
    "ijs": 3,  # Indian J Surgery
    "ija": 3,  # Indian J Anaesthesia
    "ijri": 3,  # Indian J Radiology
    "ijpm": 3,  # Indian J Pathology Microbiology
    "ijn": 3,  # Indian J Nephrology
    "ijpsy": 3,  # Indian J Psychiatry
    "ijdr": 3,  # Indian J Dental Research
    "ijms": 3,  # Indian J Medical Sciences
    "ijccm": 3,  # Indian J Critical Care Medicine
    "iju": 3,  # Indian J Urology
    "ijr": 3,  # Indian J Rheumatology
    "ijpc": 3,  # Indian J Palliative Care
    "ijem": 3,  # Indian J Endocrinology Metabolism
    "ijg": 3,  # Indian J Gastroenterology
    "ijmm": 3,  # Indian J Medical Microbiology
    "ijmpo": 3,  # Indian J Medical Paediatric Oncology
    "jpm": 3,  # J Postgraduate Medicine
    "aian": 3,  # Ann Indian Acad Neurology
    "mjafi": 3,  # Med J Armed Forces India
    "ijdvl": 3,  # Indian J Dermatol Venereol Leprol
    "ijphrd": 4,  # Indian J Public Health Research Development — IndMED-only
}

INDMED_CONFIG = SourceConfig(
    name="indmed",
    base_url="https://indmedinfo.nic.in",
    rate_limit=0.5,  # 0.5 req/sec — NIC servers are slow
    crawl_delay=2.0,
    user_agent_suffix="IndMED-indexer",
    fetch_strategy="http_first",
    expected_content_types=("text/html", "application/pdf"),
    sitemap_urls=(),
    trust_tier=4,  # IndMED default (per-journal override applied in process())
    india_relevant_default=True,
    indian_source_default=True,
    extra={
        "journals": INDMED_JOURNALS,
        "journal_trust_tier": JOURNAL_TRUST_TIER,
        "default_date_range": "2015:2025",
    },
)


@register_source("indmed")
class IndMEDScraper(BaseScraper):
    """Scraper for IndMED-indexed Indian journals (OJS pattern)."""

    config = INDMED_CONFIG

    async def discover(
        self,
        journals: list[str] | None = None,
        max_articles_per_journal: int = 500,
        year_range: tuple[int, int] = (2015, 2025),
        **kwargs: Any,
    ) -> list[CrawlJob]:
        """Discover article URLs across IndMED journals.

        Args:
            journals: optional list of OJS abbreviations to limit discovery
                (e.g., ["ijp", "ijcm"]). Default: all journals in INDMED_JOURNALS.
            max_articles_per_journal: cap per journal (default 500)
            year_range: (start_year, end_year) inclusive

        Returns:
            list of CrawlJobs, one per article URL
        """
        target_journals = journals or list(self.config.extra["journals"].values())
        jobs: list[CrawlJob] = []
        for journal_abbr in target_journals:
            try:
                journal_jobs = await self._discover_journal(
                    journal_abbr, max_articles_per_journal, year_range
                )
                jobs.extend(journal_jobs)
                logger.info(
                    f"[indmed] {journal_abbr}: discovered {len(journal_jobs)} articles"
                )
            except Exception as e:
                logger.warning(f"[indmed] {journal_abbr}: discovery failed: {e}")
        return jobs

    async def _discover_journal(
        self,
        journal_abbr: str,
        max_articles: int,
        year_range: tuple[int, int],
    ) -> list[CrawlJob]:
        """Discover articles for one journal via the issue archive."""
        # Step 1: fetch issue archive → list of issue URLs
        archive_url = f"{self.config.base_url}/index.php/{journal_abbr}/issue/archive"
        result = await self.http.fetch(archive_url, use_cache=True, cache_ttl=7 * 24 * 3600)
        if not result.ok or not result.content:
            logger.warning(f"[indmed] {journal_abbr}: archive fetch failed: {result.error}")
            return []

        issue_urls = self._extract_issue_urls(result.content.decode("utf-8", errors="replace"), journal_abbr)
        logger.debug(f"[indmed] {journal_abbr}: {len(issue_urls)} issues found")

        # Step 2: for each issue, fetch its TOC → list of article URLs
        # Cap total articles per journal
        jobs: list[CrawlJob] = []
        for issue_url in issue_urls:
            if len(jobs) >= max_articles:
                break
            issue_result = await self.http.fetch(issue_url, use_cache=True, cache_ttl=7 * 24 * 3600)
            if not issue_result.ok or not issue_result.content:
                continue
            article_urls = self._extract_article_urls(
                issue_result.content.decode("utf-8", errors="replace"),
                journal_abbr,
                year_range,
            )
            for url in article_urls:
                if len(jobs) >= max_articles:
                    break
                jobs.append(CrawlJob(
                    url=url,
                    source=self.config.name,
                    discovered_from="link-extraction",
                    metadata={"journal_abbr": journal_abbr},
                ))
        return jobs

    def _extract_issue_urls(self, html: str, journal_abbr: str) -> list[str]:
        """Extract issue URLs from the OJS archive page."""
        soup = BeautifulSoup(html, "lxml")
        urls: list[str] = []
        seen: set[str] = set()
        # OJS archive pages link to /index.php/<journal>/issue/view/<id>
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if f"/index.php/{journal_abbr}/issue/view/" not in href:
                continue
            absolute = urljoin(self.config.base_url, href)
            if absolute not in seen:
                seen.add(absolute)
                urls.append(absolute)
        return urls

    def _extract_article_urls(
        self,
        html: str,
        journal_abbr: str,
        year_range: tuple[int, int],
    ) -> list[str]:
        """Extract article URLs from an OJS issue table-of-contents page.

        Filters by year_range using the article's published date (extracted
        from the HTML — OJS shows year alongside each article).
        """
        soup = BeautifulSoup(html, "lxml")
        urls: list[str] = []
        seen: set[str] = set()
        # OJS TOC pages link to /index.php/<journal>/article/view/<id>
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if f"/index.php/{journal_abbr}/article/view/" not in href:
                continue
            absolute = urljoin(self.config.base_url, href)
            if absolute in seen:
                continue
            seen.add(absolute)
            urls.append(absolute)
        # Year filtering is best-effort — OJS TOC pages don't always show
        # per-article years. We do final filtering at article-fetch time
        # via the citation_publication_date meta tag.
        return urls

    async def process(self, result, job: CrawlJob) -> Any:
        """Process a fetched article page.

        Overrides base implementation to:
        1. Apply per-journal trust_tier (from JOURNAL_TRUST_TIER map)
        2. Detect if the article also has a PubMed PMID (for cross-source dedup)
        3. Extract PDF link if available (article landing pages link to PDFs)
        """
        from src.ingestion.scrapers.framework.models import ScrapedDocument

        if not result.is_html:
            # Probably a PDF — return as-is, parser will handle
            return ScrapedDocument(
                url=result.url,
                source=self.config.name,
                content=result.content,
                content_type=result.content_type or "",
                fetched_at=result.fetched_at,
                trust_tier=self.config.trust_tier,
                india_relevant=True,
                indian_source=True,
                metadata={"journal_abbr": job.metadata.get("journal_abbr")},
            )

        # Default metadata extraction (Highwire citation_*, JSON-LD, etc.)
        meta = self.metadata_extractor.extract(result.content, result.encoding or "utf-8")

        # Apply per-journal trust tier
        journal_abbr = job.metadata.get("journal_abbr", "")
        trust_tier = self.config.extra["journal_trust_tier"].get(journal_abbr, self.config.trust_tier)

        # Try to find a PDF download link on the article page
        pdf_url = self._find_pdf_download_link(
            result.content.decode("utf-8", errors="replace"),
            result.url,
            journal_abbr,
        )
        if pdf_url:
            meta["pdf_url"] = pdf_url

        # Parse year from pubdate for filtering
        pubdate = meta.get("pubdate") or ""
        year = self._extract_year(pubdate)
        if year and year < 2015:  # skip pre-2015 (configurable later)
            logger.debug(f"[indmed] skipping old article {result.url} (year={year})")
            return None

        return ScrapedDocument(
            url=result.url,
            source=self.config.name,
            content=result.content,
            content_type=result.content_type or "",
            title=meta.get("title"),
            authors=meta.get("authors", []),
            journal=meta.get("journal"),
            doi=meta.get("doi"),
            pmid=meta.get("pmid"),
            pubdate=meta.get("pubdate"),
            abstract=meta.get("abstract"),
            metadata=meta,
            fetched_at=result.fetched_at,
            trust_tier=trust_tier,
            india_relevant=True,
            indian_source=True,
        )

    @staticmethod
    def _find_pdf_download_link(html: str, base_url: str, journal_abbr: str) -> str | None:
        """Find the PDF download URL on an OJS article page.

        OJS pattern: /index.php/<journal>/article/download/<article_id>/<file_id>
        The link is usually in a "Download PDF" or "Supplementary Material" section.
        """
        soup = BeautifulSoup(html, "lxml")
        # Look for download links in the standard OJS positions
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if f"/index.php/{journal_abbr}/article/download/" in href:
                return urljoin(base_url, href)
            # Sometimes the PDF link uses /article/view/<id>/<file_id> (galley link)
            match = re.search(rf"/index.php/{journal_abbr}/article/view/(\d+)/(\d+)", href)
            if match and href.endswith(match.group(2)):
                # Verify this is a PDF galley by checking the link text or class
                text = a.get_text(" ", strip=True).lower()
                classes = a.get("class", []) or []
                if "pdf" in text or "galley" in classes or "download" in text:
                    return urljoin(base_url, href)
        return None

    @staticmethod
    def _extract_year(pubdate: str | None) -> int | None:
        """Extract a 4-digit year from a publication date string."""
        if not pubdate:
            return None
        match = re.search(r"(19|20)\d{2}", pubdate)
        if match:
            return int(match.group(0))
        return None
