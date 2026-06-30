"""Medknow scraper — enriches PubMed abstracts with full-text PDFs from Medknow.

Medknow (medknow.com / www.medknow.org) is Wolters Kluwer India's open-access
journal hosting platform. ~25 Indian journals are hosted here, many of which
are also PubMed-indexed. The value of this scraper is NOT new article
discovery (PubMed already has those) but full-text enrichment:

- PubMed gives us: title + authors + DOI + abstract (structured XML)
- Medknow gives us: full-text HTML body + PDF download link

When both sources cover the same article, cross-source dedup keeps the
PubMed version (Tier 4) but records Medknow in `also_indexed_in`. The
pipeline can then fetch the Medknow full-text PDF separately and create
additional full-text chunks linked to the same document.

Site structure (medknow.com):
- Journal landing: /journals/<journal_abbr>.htm
- Issue archive:   /journals/<journal_abbr>/archive.asp
- Issue TOC:       /journals/<journal_abbr>/<year>/<vol>_<issue>.htm
- Article page:    /article.asp?issn=<issn>;year=<year>;volume=<vol>;issue=<iss>;spage=<page>;epage=<page>;aulast=<author>
- PDF download:    /article.asp?issn=<issn>;year=<year>;volume=<vol>;issue=<iss>;spage=<page>;epage=<page>;aulast=<author>;type=2

Politeness:
- 1 req/sec (Wolters Kluwer has decent infrastructure)
- 1s crawl delay
- 7-day cache on archive/issue pages
- 30-day cache on article pages
"""
from __future__ import annotations

import re
from typing import Any
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
from loguru import logger

from src.ingestion.scrapers.framework.base import BaseScraper, register_source
from src.ingestion.scrapers.framework.models import CrawlJob, SourceConfig

# Medknow-hosted Indian journals (ISSN is the canonical identifier)
MEDKNOW_JOURNALS: dict[str, dict[str, str]] = {
    "Indian Journal of Anaesthesia": {
        "abbr": "ijan", "issn_print": "0019-5049", "issn_online": "0976-0019",
        "medknow_path": "ijan",
    },
    "Indian Journal of Pharmacology": {
        "abbr": "ijp", "issn_print": "0253-7613", "issn_online": "1998-3751",
        "medknow_path": "ijp",
    },
    "Indian Journal of Pathology and Microbiology": {
        "abbr": "ijpm", "issn_print": "0377-4927", "issn_online": "0973-2691",
        "medknow_path": "ijpm",
    },
    "Indian Journal of Ophthalmology": {
        "abbr": "ijo", "issn_print": "0301-4738", "issn_online": "1998-3673",
        "medknow_path": "ijo",
    },
    "Indian Journal of Dermatology, Venereology and Leprology": {
        "abbr": "ijdvl", "issn_print": "0378-6323", "issn_online": "0973-3922",
        "medknow_path": "ijdvl",
    },
    "Indian Journal of Orthopaedics": {
        "abbr": "ijo", "issn_print": "0019-5413", "issn_online": "1998-3743",
        "medknow_path": "ijoo",
    },
    "Indian Journal of Radiology and Imaging": {
        "abbr": "ijri", "issn_print": "0971-3026", "issn_online": "0973-3325",
        "medknow_path": "ijri",
    },
    "Indian Journal of Nephrology": {
        "abbr": "ijn", "issn_print": "0971-4065", "issn_online": "1998-3662",
        "medknow_path": "ijn",
    },
    "Indian Journal of Urology": {
        "abbr": "iju", "issn_print": "0251-5142", "issn_online": "0970-1591",
        "mednow_path": "iju", "medknow_path": "iju",
    },
    "Indian Journal of Palliative Care": {
        "abbr": "ijpc", "issn_print": "0973-1075", "issn_online": "1998-376X",
        "medknow_path": "ijpc",
    },
    "Indian Journal of Critical Care Medicine": {
        "abbr": "ijccm", "issn_print": "0972-5229", "issn_online": "1998-3678",
        "medknow_path": "ijccm",
    },
    "Indian Journal of Endocrinology and Metabolism": {
        "abbr": "ijem", "issn_print": "2230-8210", "issn_online": "2230-809X",
        "medknow_path": "ijem",
    },
    "Indian Journal of Medical Microbiology": {
        "abbr": "ijmm", "issn_print": "0255-0857", "issn_online": "1998-3679",
        "medknow_path": "ijmm",
    },
    "Indian Journal of Medical and Paediatric Oncology": {
        "abbr": "ijmpo", "issn_print": "0971-5851", "issn_online": "0976-8586",
        "medknow_path": "ijmpo",
    },
    "Indian Journal of Rheumatology": {
        "abbr": "ijr", "issn_print": "0973-3698", "issn_online": "1998-3681",
        "medknow_path": "ijr",
    },
    "Indian Journal of Community Medicine": {
        "abbr": "ijcm", "issn_print": "0970-0218", "issn_online": "1998-3581",
        "medknow_path": "ijcm",
    },
    "Indian Journal of Public Health": {
        "abbr": "ijph", "issn_print": "0019-557X", "issn_online": "1998-3618",
        "medknow_path": "ijph",
    },
    "Journal of Postgraduate Medicine": {
        "abbr": "jpm", "issn_print": "0022-3859", "issn_online": "0972-2823",
        "medknow_path": "jpgm",
    },
    "Med J Armed Forces India": {
        "abbr": "mjafi", "issn_print": "0377-1237", "issn_online": "0975-8533",
        "medknow_path": "mjafi",
    },
    "Annals of Indian Academy of Neurology": {
        "abbr": "aian", "issn_print": "0972-2327", "issn_online": "1998-3676",
        "medknow_path": "aian",
    },
}

MEDKNOW_CONFIG = SourceConfig(
    name="medknow",
    base_url="https://www.medknow.com",
    rate_limit=1.0,  # 1 req/sec — Wolters Kluwer infra is decent
    crawl_delay=1.0,
    user_agent_suffix="Medknow-enrichment",
    fetch_strategy="http_first",
    expected_content_types=("text/html", "application/pdf"),
    trust_tier=3,  # Medknow = peer-reviewed, PubMed-indexed journals
    india_relevant_default=True,
    indian_source_default=True,
    extra={
        "journals": MEDKNOW_JOURNALS,
    },
)


@register_source("medknow")
class MedknowScraper(BaseScraper):
    """Scraper for Medknow-hosted Indian journals.

    Primary use case: enrich PubMed-indexed articles with full-text PDFs.
    Discovery via journal archive pages; per-article fetch extracts
    full-text HTML body + PDF download link.
    """

    config = MEDKNOW_CONFIG

    async def discover(
        self,
        journals: list[str] | None = None,
        max_articles_per_journal: int = 500,
        year_range: tuple[int, int] = (2015, 2025),
        **kwargs: Any,
    ) -> list[CrawlJob]:
        """Discover article URLs across Medknow-hosted journals.

        Args:
            journals: optional list of journal abbreviations (e.g., ["ijp", "ijcm"]).
                Default: all journals in MEDKNOW_JOURNALS.
            max_articles_per_journal: cap per journal
            year_range: (start, end) inclusive — used for archive URL construction

        Returns:
            list of CrawlJobs
        """
        all_journals = journals or list(self.config.extra["journals"].keys())
        jobs: list[CrawlJob] = []
        for journal_name in all_journals:
            journal_info = self.config.extra["journals"].get(journal_name)
            if not journal_info:
                logger.warning(f"[medknow] unknown journal: {journal_name}")
                continue
            try:
                journal_jobs = await self._discover_journal(
                    journal_name, journal_info, max_articles_per_journal, year_range
                )
                jobs.extend(journal_jobs)
                logger.info(
                    f"[medknow] {journal_name}: discovered {len(journal_jobs)} articles"
                )
            except Exception as e:
                logger.warning(f"[medknow] {journal_name}: discovery failed: {e}")
        return jobs

    async def _discover_journal(
        self,
        journal_name: str,
        journal_info: dict[str, str],
        max_articles: int,
        year_range: tuple[int, int],
    ) -> list[CrawlJob]:
        """Discover articles for one journal via Medknow's archive page.

        Medknow archive pages list issues by year+volume. We extract article
        URLs from each issue's TOC page.
        """
        medknow_path = journal_info.get("medknow_path")
        if not medknow_path:
            return []

        # Step 1: fetch archive page
        archive_url = f"{self.config.base_url}/journals/{medknow_path}/archive.asp"
        result = await self.http.fetch(archive_url, use_cache=True, cache_ttl=7 * 24 * 3600)
        if not result.ok or not result.content:
            logger.warning(f"[medknow] {journal_name}: archive fetch failed: {result.error}")
            return []

        # Step 2: extract issue URLs from archive page
        issue_urls = self._extract_issue_urls(
            result.content.decode("utf-8", errors="replace"),
            medknow_path,
            year_range,
        )
        logger.debug(f"[medknow] {journal_name}: {len(issue_urls)} issues found")

        # Step 3: for each issue, fetch TOC and extract article URLs
        jobs: list[CrawlJob] = []
        for issue_url in issue_urls:
            if len(jobs) >= max_articles:
                break
            issue_result = await self.http.fetch(issue_url, use_cache=True, cache_ttl=7 * 24 * 3600)
            if not issue_result.ok or not issue_result.content:
                continue
            article_urls = self._extract_article_urls(
                issue_result.content.decode("utf-8", errors="replace"),
                issue_url,
                journal_info,
            )
            for url in article_urls:
                if len(jobs) >= max_articles:
                    break
                jobs.append(CrawlJob(
                    url=url,
                    source=self.config.name,
                    discovered_from="link-extraction",
                    metadata={
                        "journal_name": journal_name,
                        "journal_abbr": journal_info.get("abbr"),
                        "issn": journal_info.get("issn_online") or journal_info.get("issn_print"),
                    },
                ))
        return jobs

    def _extract_issue_urls(
        self,
        html: str,
        medknow_path: str,
        year_range: tuple[int, int],
    ) -> list[str]:
        """Extract issue URLs from Medknow's archive page.

        Medknow archive pages link to issue TOCs. URLs look like:
        /journals/<journal>/<year>/<vol>_<issue>.htm
        or /journals/<journal>/archive/<year>.asp
        """
        soup = BeautifulSoup(html, "lxml")
        urls: list[str] = []
        seen: set[str] = set()
        for a in soup.find_all("a", href=True):
            href = a["href"]
            text = a.get_text(strip=True)
            # Match year references in link text or href
            year_match = re.search(r"(19|20)\d{2}", text + " " + href)
            if not year_match:
                continue
            year = int(year_match.group(0))
            if year < year_range[0] or year > year_range[1]:
                continue
            absolute = urljoin(self.config.base_url, href)
            # Only accept URLs that look like issue pages
            if medknow_path in absolute or "/archive/" in absolute:
                if absolute not in seen:
                    seen.add(absolute)
                    urls.append(absolute)
        return urls

    def _extract_article_urls(
        self,
        html: str,
        issue_url: str,
        journal_info: dict[str, str],
    ) -> list[str]:
        """Extract article URLs from a Medknow issue TOC page.

        Medknow article URLs use a query string format:
        /article.asp?issn=<issn>;year=<year>;volume=<vol>;issue=<iss>;spage=<page>;epage=<page>;aulast=<author>
        """
        soup = BeautifulSoup(html, "lxml")
        urls: list[str] = []
        seen: set[str] = set()
        issn = journal_info.get("issn_online") or journal_info.get("issn_print", "")
        for a in soup.find_all("a", href=True):
            href = a["href"]
            # Medknow article.asp URLs with this journal's ISSN
            if "article.asp" not in href:
                continue
            if issn and issn.replace("-", "") not in href.replace("-", ""):
                # Only accept articles from this journal's ISSN
                continue
            absolute = urljoin(issue_url, href)
            # Normalize: strip fragment
            parsed = urlparse(absolute)
            absolute = parsed._replace(fragment="").geturl()
            if absolute not in seen and "type=2" not in absolute:  # type=2 = PDF
                seen.add(absolute)
                urls.append(absolute)
        return urls

    async def process(self, result, job: CrawlJob) -> Any:
        """Process a fetched Medknow article page.

        Medknow article pages expose Highwire Press citation_* meta tags
        (same as PubMed/Medknow journals). The default MetadataExtractor
        handles these. We override only to extract the PDF download URL
        (which uses ?type=2 query parameter) and apply per-journal trust tier.
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
                metadata=job.metadata,
            )

        meta = self.metadata_extractor.extract(result.content, result.encoding or "utf-8")

        # Find PDF download link (?type=2 on the article URL itself)
        pdf_url = self._find_pdf_url(result.url, result.content.decode("utf-8", errors="replace"))
        if pdf_url:
            meta["pdf_url"] = pdf_url

        return ScrapedDocument(
            url=result.url,
            source=self.config.name,
            content=result.content,
            content_type=result.content_type or "",
            title=meta.get("title"),
            authors=meta.get("authors", []),
            journal=meta.get("journal") or job.metadata.get("journal_name"),
            doi=meta.get("doi"),
            pmid=meta.get("pmid"),
            pubdate=meta.get("pubdate"),
            abstract=meta.get("abstract"),
            metadata={**meta, **job.metadata},
            fetched_at=result.fetched_at,
            trust_tier=self.config.trust_tier,
            india_relevant=True,
            indian_source=True,
        )

    @staticmethod
    def _find_pdf_url(article_url: str, html: str) -> str | None:
        """Find the PDF download URL for a Medknow article.

        Medknow PDF links use ?type=2 query parameter on the same article.asp URL.
        Also check for explicit PDF download links in the HTML.
        """
        # Method 1: try adding ?type=2 to the article URL (Medknow convention)
        # article.asp?issn=X;year=Y;volume=V;issue=I;spage=S;epage=E;aulast=A
        # becomes the same with ;type=2 appended
        if "article.asp" in article_url and "type=" not in article_url:
            # Append ;type=2 (Medknow uses ; as separator)
            pdf_url = article_url + ("&type=2" if "?" in article_url and ";" not in article_url else ";type=2")
            return pdf_url

        # Method 2: look for explicit PDF link in HTML
        soup = BeautifulSoup(html, "lxml")
        for a in soup.find_all("a", href=True):
            href = a["href"]
            text = a.get_text(" ", strip=True).lower()
            if "pdf" in text or "[pdf]" in text or "full text pdf" in text:
                if "type=2" in href or href.lower().endswith(".pdf"):
                    return urljoin(article_url, href)
        return None
