"""
CDC / NIH Guidelines Parser

Retrieves US Centers for Disease Control (CDC) and National Institutes of
Health (NIH) guidelines via:
  - PubMed search restricted to CDC/NIH corporate authors
  - CDC Public Health Publications API (https://tools.cdc.gov/api/v2/resources)

Uses shared utilities for PubMed fallback and date extraction.

Usage:
    parser = CDCParser(query="malaria prevention", max_results=30)
    documents = parser.parse()
"""

import requests
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_fixed

from src.config.settings import get_settings
from src.ingestion.document_db import DocumentRecord
from src.ingestion.parsers.base import BaseParser
from src.utils.date_utils import extract_year_from_text
from src.utils.pubmed_client import PubMedClient

_CDC_API_BASE = "https://tools.cdc.gov/api/v2/resources/media"
_REQUEST_TIMEOUT = 15


class CDCParser(BaseParser):
    """
    Fetches CDC and NIH guideline documents.

    Strategy:
      1. CDC Media API (JSON) — targeted health topics
      2. PubMed fallback — CDC/NIH corporate author records
    """

    def __init__(self, query: str, max_results: int = 50):
        self.query = query
        self.max_results = max_results
        self.settings = get_settings()
        self._pubmed_client = PubMedClient(self.settings)

    @property
    def source_type(self) -> str:
        return "cdc"

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(3))
    def _fetch_cdc_api(self) -> list[dict]:
        """Query CDC Public Health Media/Resource API."""
        params = {
            "q": self.query,
            "max": min(self.max_results, 50),
            "fields": "title,description,datePublished,sourceUrl",
            "mediaTypes": "PDF,HTML",
            "sort": "relevance",
        }
        resp = requests.get(
            _CDC_API_BASE,
            params=params,
            timeout=_REQUEST_TIMEOUT,
            headers={"Accept": "application/json"},
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("results", [])

    def _cdc_item_to_doc(self, item: dict) -> DocumentRecord | None:
        """Convert a CDC API item to a DocumentRecord."""
        title = (item.get("title") or "").strip()
        description = (item.get("description") or "").strip()

        if not title or not description:
            return None

        # Extract year using shared utility
        date_str = (item.get("datePublished") or "").strip()
        year_str = extract_year_from_text(date_str) if date_str else ""

        url = (item.get("sourceUrl") or "").strip() or None

        return DocumentRecord(
            source_type=self.source_type,
            title=title,
            content=description,
            url=url,
            published_date=date_str or None,
            year=int(year_str) if year_str.isdigit() else None,
            journal="CDC Health Guidelines",
            study_type="guideline",
            evidence_level=2,
            parser_version="v3",
        )

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    def _fetch_pubmed_fallback(self) -> list[DocumentRecord]:
        """PubMed search for CDC/NIH-authored records."""
        org_filter = (
            '("Centers for Disease Control and Prevention"[Corporate Author] OR '
            '"National Institutes of Health"[Corporate Author] OR '
            '"CDC"[Corporate Author] OR "NIH"[Corporate Author])'
        )
        full_query = f"({self.query}) AND {org_filter}"

        pmids = self._pubmed_client.search_pubmed(
            full_query, max_results=self.max_results
        )
        if not pmids:
            return []

        articles = self._pubmed_client.fetch_pubmed_articles(pmids)

        documents: list[DocumentRecord] = []
        for article in articles:
            pubmed_article = PubMedClient.article_to_pubmed_article(
                article, journal_override="CDC/NIH Publications"
            )
            if pubmed_article is None:
                continue

            documents.append(
                DocumentRecord(
                    source_type=self.source_type,
                    title=pubmed_article.title,
                    content=pubmed_article.abstract,
                    url=pubmed_article.url,
                    doi=pubmed_article.doi,
                    journal="CDC/NIH Publications",
                    study_type="guideline",
                    evidence_level=2,
                    parser_version="v3",
                )
            )
        return documents

    def parse(self) -> list[DocumentRecord]:
        documents: list[DocumentRecord] = []

        # Primary: CDC Public Media API
        try:
            items = self._fetch_cdc_api()
            for item in items:
                doc = self._cdc_item_to_doc(item)
                if doc:
                    documents.append(doc)
            logger.info(
                f"[CDC] API returned {len(documents)} documents for query='{self.query}'"
            )
        except requests.RequestException as exc:
            logger.warning(f"[CDC] API failed ({exc}), trying PubMed fallback")
            try:
                documents = self._fetch_pubmed_fallback()
                logger.info(
                    f"[CDC] PubMed fallback returned {len(documents)} documents"
                )
            except (RuntimeError, ValueError, TypeError, OSError) as exc2:
                logger.error(f"[CDC] Both strategies failed: {exc2}")

        return documents
