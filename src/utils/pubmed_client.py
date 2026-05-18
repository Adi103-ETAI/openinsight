"""
PubMed / NCBI Entrez Client Utility

A shared client for interacting with NCBI Entrez APIs (PubMed, PMC, etc.).
Handles configuration, rate limiting, fetching, and XML parsing of article data.

Eliminates duplicated Entrez setup, rate limiting, and abstract parsing logic
across: pubmed.py, who.py, cdc.py, cochrane.py, statpearls.py
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any

from Bio import Entrez
from loguru import logger
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from src.config.settings import Settings


@dataclass
class PubMedArticle:
    """Structured representation of a parsed PubMed article."""

    pmid: str
    title: str
    abstract: str
    journal: str = ""
    year: str = ""
    doi: str | None = None
    url: str | None = None
    authors: list[str] = field(default_factory=list)
    mesh_terms: list[str] = field(default_factory=list)
    raw_data: dict[str, Any] = field(default_factory=dict)


class PubMedClient:
    """
    Shared client for NCBI Entrez API interactions.

    Provides:
    - Centralized Entrez configuration (email, API key)
    - Configurable rate limiting based on API key presence
    - PubMed article search and fetch with retry logic
    - PMC article search and fetch
    - Structured article parsing from Entrez XML responses

    Usage:
        client = PubMedClient(settings)
        articles = client.fetch_articles_by_query("malaria treatment", max_results=50)
        for article in articles:
            print(article.title, article.abstract)
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._configure_entrez()

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def _configure_entrez(self) -> None:
        """Configure Bio.Entrez with email and API key from settings."""
        Entrez.email = self._settings.ncbi_email
        Entrez.api_key = self._settings.ncbi_api_key or None

    @property
    def has_api_key(self) -> bool:
        """Whether an NCBI API key is configured."""
        return bool(self._settings.ncbi_api_key)

    def get_rate_limit_delay(self) -> float:
        """
        Return the appropriate rate limit delay based on API key presence.

        With API key:  0.1s  (up to 10 requests/sec)
        Without key:   0.34s (up to 3 requests/sec)
        """
        if self.has_api_key:
            return self._settings.pubmed_rate_limit_with_key
        return self._settings.pubmed_rate_limit_seconds

    def apply_rate_limit(self) -> None:
        """Sleep for the configured rate limit duration."""
        delay = self.get_rate_limit_delay()
        logger.debug(f"[PubMedClient] Rate limiting: sleeping {delay}s")
        time.sleep(delay)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_fixed(2),
        retry=retry_if_exception_type(
            (RuntimeError, ValueError, TypeError, OSError)
        ),
        reraise=True,
    )
    def search_pubmed(
        self,
        query: str,
        max_results: int = 100,
        db: str = "pubmed",
    ) -> list[str]:
        """
        Search PubMed (or another Entrez database) and return a list of IDs.

        Args:
            query: Entrez query string.
            max_results: Maximum number of results to return.
            db: Entrez database name (default: "pubmed").

        Returns:
            List of ID strings (PMIDs, PMCIDs, etc.).
        """
        logger.debug(f"[PubMedClient] Searching {db}: query='{query}', max={max_results}")
        with Entrez.esearch(db=db, term=query, retmax=max_results) as handle:
            result = Entrez.read(handle)
        ids = result.get("IdList", [])
        logger.info(f"[PubMedClient] {db} search returned {len(ids)} IDs")
        return ids

    # ------------------------------------------------------------------
    # Fetch
    # ------------------------------------------------------------------

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_fixed(2),
        retry=retry_if_exception_type(
            (RuntimeError, ValueError, TypeError, OSError)
        ),
        reraise=True,
    )
    def fetch_pubmed_articles(
        self,
        pmids: list[str],
    ) -> list[dict[str, Any]]:
        """
        Fetch full PubMed XML records for a list of PMIDs.

        Args:
            pmids: List of PubMed IDs.

        Returns:
            List of article dicts from the Entrez XML response.
        """
        if not pmids:
            return []

        self.apply_rate_limit()
        logger.debug(f"[PubMedClient] Fetching {len(pmids)} PubMed articles")

        with Entrez.efetch(
            db="pubmed", id=pmids, rettype="xml", retmode="xml"
        ) as handle:
            data = Entrez.read(handle)

        return data.get("PubmedArticle", [])

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_fixed(2),
        retry=retry_if_exception_type(
            (RuntimeError, ValueError, TypeError, OSError)
        ),
        reraise=True,
    )
    def fetch_pmc_articles(
        self,
        pmcids: list[str],
    ) -> list[dict[str, Any]]:
        """
        Fetch full PMC XML records for a list of PMCIDs.

        Args:
            pmcids: List of PubMed Central IDs.

        Returns:
            List of article dicts from the Entrez XML response.
        """
        if not pmcids:
            return []

        self.apply_rate_limit()
        logger.debug(f"[PubMedClient] Fetching {len(pmcids)} PMC articles")

        with Entrez.efetch(
            db="pmc", id=pmcids, rettype="xml", retmode="xml"
        ) as handle:
            data = Entrez.read(handle)

        articles = data.get("article", [])
        if not articles and isinstance(data, list):
            articles = data
        return articles if isinstance(articles, list) else []

    # ------------------------------------------------------------------
    # Combined search + fetch
    # ------------------------------------------------------------------

    def search_and_fetch_pubmed(
        self,
        query: str,
        max_results: int = 100,
        db: str = "pubmed",
    ) -> list[dict[str, Any]]:
        """
        Search and fetch articles in one call.

        Args:
            query: Entrez query string.
            max_results: Maximum number of results.
            db: Entrez database name.

        Returns:
            List of raw article dicts.
        """
        ids = self.search_pubmed(query, max_results=max_results, db=db)
        if not ids:
            return []

        if db == "pmc":
            return self.fetch_pmc_articles(ids)
        return self.fetch_pubmed_articles(ids)

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def extract_abstract(article: dict[str, Any]) -> str:
        """
        Extract and join abstract text from a PubMed article dict.

        Handles both structured abstracts (list of AbstractText elements with
        optional labels) and plain abstracts.

        Args:
            article: A PubMed article dict from Entrez XML.

        Returns:
            Joined abstract text, or empty string if none found.
        """
        article_data = article.get("MedlineCitation", {}).get("Article", {})
        abstract = article_data.get("Abstract", {})

        if not isinstance(abstract, dict):
            return ""

        abstract_parts = abstract.get("AbstractText", [])
        if not abstract_parts:
            return ""

        text = " ".join(
            str(part).strip()
            for part in abstract_parts
            if str(part).strip()
        )
        return text

    @staticmethod
    def extract_article_title(article: dict[str, Any]) -> str:
        """Extract the article title from a PubMed article dict."""
        article_data = article.get("MedlineCitation", {}).get("Article", {})
        return str(article_data.get("ArticleTitle", "")).strip()

    @staticmethod
    def extract_pmid(article: dict[str, Any]) -> str:
        """Extract the PMID from a PubMed article dict."""
        citation = article.get("MedlineCitation", {})
        return str(citation.get("PMID", "")).strip()

    @staticmethod
    def extract_year(article: dict[str, Any]) -> str:
        """
        Extract the publication year from a PubMed article dict.

        Tries PubDate.Year first, then falls back to extracting from
        MedlineDate using regex.

        Args:
            article: A PubMed article dict from Entrez XML.

        Returns:
            Year string, or empty string if not found.
        """
        from src.utils.date_utils import extract_year_from_text

        article_data = article.get("MedlineCitation", {}).get("Article", {})
        journal = article_data.get("Journal", {})
        journal_issue = journal.get("JournalIssue", {})
        pub_date = journal_issue.get("PubDate", {})

        if not isinstance(pub_date, dict):
            return ""

        # Try direct Year field
        year = str(pub_date.get("Year", "")).strip()
        if year:
            return year

        # Fall back to MedlineDate (e.g., "2023 Jan-Feb")
        medline_date = str(pub_date.get("MedlineDate", "")).strip()
        return extract_year_from_text(medline_date)

    @staticmethod
    def extract_journal_name(article: dict[str, Any]) -> str:
        """Extract the journal name from a PubMed article dict."""
        article_data = article.get("MedlineCitation", {}).get("Article", {})
        journal = article_data.get("Journal", {})
        return str(journal.get("Title", "")).strip()

    @staticmethod
    def article_to_pubmed_article(
        article: dict[str, Any],
        journal_override: str | None = None,
    ) -> PubMedArticle | None:
        """
        Convert a raw Entrez article dict to a structured PubMedArticle.

        Returns None if title or abstract is missing.

        Args:
            article: Raw article dict from Entrez XML.
            journal_override: Optional journal name to override the extracted one.
        """
        title = PubMedClient.extract_article_title(article)
        if not title:
            return None

        abstract = PubMedClient.extract_abstract(article)
        if not abstract:
            return None

        pmid = PubMedClient.extract_pmid(article)
        year = PubMedClient.extract_year(article)
        journal = journal_override or PubMedClient.extract_journal_name(article)

        return PubMedArticle(
            pmid=pmid,
            title=title,
            abstract=abstract,
            journal=journal,
            year=year,
            doi=f"PMID:{pmid}" if pmid else None,
            url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else None,
            raw_data=article,
        )

    # ------------------------------------------------------------------
    # PMC-specific parsing
    # ------------------------------------------------------------------

    @staticmethod
    def extract_pmc_abstract(article: dict[str, Any]) -> str:
        """
        Extract abstract text from a PMC article dict (different XML structure).

        Args:
            article: A PMC article dict from Entrez XML (db=pmc).

        Returns:
            Abstract text, or empty string if not found.
        """
        front = article.get("front", {})
        article_meta = front.get("article-meta", {})
        abstract_section = article_meta.get("abstract", {})

        if isinstance(abstract_section, dict):
            paragraphs = abstract_section.get("p", [])
            if isinstance(paragraphs, str):
                return paragraphs.strip()
            if isinstance(paragraphs, list):
                return " ".join(str(p) for p in paragraphs).strip()
        elif isinstance(abstract_section, str):
            return abstract_section.strip()

        return ""

    @staticmethod
    def extract_pmc_title(article: dict[str, Any]) -> str:
        """Extract title from a PMC article dict."""
        front = article.get("front", {})
        article_meta = front.get("article-meta", {})
        title_group = article_meta.get("title-group", {})
        title_raw = title_group.get("article-title", "")
        return str(title_raw).strip() if title_raw else ""

    @staticmethod
    def extract_pmc_year(article: dict[str, Any]) -> str:
        """Extract publication year from a PMC article dict."""
        from src.utils.date_utils import extract_year_from_text

        front = article.get("front", {})
        article_meta = front.get("article-meta", {})
        pub_date = article_meta.get("pub-date", {})

        if isinstance(pub_date, dict):
            year = str(pub_date.get("year", "")).strip()
            if year:
                return year
        elif isinstance(pub_date, list) and pub_date:
            year = str(pub_date[0].get("year", "")).strip()
            if year:
                return year

        return ""

    @staticmethod
    def extract_pmc_ids(article: dict[str, Any]) -> dict[str, str]:
        """
        Extract PMC ID and PMID from a PMC article dict.

        Returns:
            Dict with 'pmcid' and 'pmid' keys.
        """
        front = article.get("front", {})
        article_meta = front.get("article-meta", {})
        article_ids = article_meta.get("article-id", [])

        pmcid = ""
        pmid = ""

        if isinstance(article_ids, list):
            for aid in article_ids:
                if isinstance(aid, dict):
                    if aid.get("pub-id-type") == "pmc":
                        pmcid = str(aid.get("#text", "")).strip()
                    elif aid.get("pub-id-type") == "pmid":
                        pmid = str(aid.get("#text", "")).strip()

        return {"pmcid": pmcid, "pmid": pmid}
