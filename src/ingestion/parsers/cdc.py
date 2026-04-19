"""
CDC / NIH Guidelines Parser

Retrieves US Centers for Disease Control (CDC) and National Institutes of
Health (NIH) guidelines via:
  - PubMed search restricted to CDC/NIH corporate authors
  - CDC Public Health Publications API (https://tools.cdc.gov/api/v2/resources)

Usage:
    parser = CDCParser(query="malaria prevention", max_results=30)
    documents = parser.parse()
"""

import re
import time

import requests
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_fixed

from src.core.config import get_settings
from src.ingestion.document_db import DocumentRecord
from src.ingestion.parsers.base import BaseParser

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

        date_str = (item.get("datePublished") or "").strip()
        year_str = ""
        if date_str:
            m = re.search(r"\b(19|20)\d{2}\b", date_str)
            if m:
                year_str = m.group(0)

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
        from Bio import Entrez

        Entrez.email = self.settings.ncbi_email
        Entrez.api_key = self.settings.ncbi_api_key or None

        org_filter = (
            '("Centers for Disease Control and Prevention"[Corporate Author] OR '
            '"National Institutes of Health"[Corporate Author] OR '
            '"CDC"[Corporate Author] OR "NIH"[Corporate Author])'
        )
        full_query = f"({self.query}) AND {org_filter}"

        with Entrez.esearch(
            db="pubmed", term=full_query, retmax=self.max_results
        ) as handle:
            search_result = Entrez.read(handle)

        pmids = search_result.get("IdList", [])
        if not pmids:
            return []

        sleep_seconds = 0.1 if self.settings.ncbi_api_key else 0.34
        time.sleep(sleep_seconds)

        with Entrez.efetch(
            db="pubmed", id=pmids, rettype="xml", retmode="xml"
        ) as handle:
            fetched = Entrez.read(handle)

        documents: list[DocumentRecord] = []
        for article in fetched.get("PubmedArticle", []):
            citation = article.get("MedlineCitation", {})
            article_data = citation.get("Article", {})
            title = str(article_data.get("ArticleTitle", "")).strip()
            if not title:
                continue
            abstract = article_data.get("Abstract", {})
            abstract_parts = (
                abstract.get("AbstractText", []) if isinstance(abstract, dict) else []
            )
            abstract_text = " ".join(
                str(p).strip() for p in abstract_parts if str(p).strip()
            )
            if not abstract_text:
                continue
            pmid = str(citation.get("PMID", "")).strip()
            documents.append(
                DocumentRecord(
                    source_type=self.source_type,
                    title=title,
                    content=abstract_text,
                    url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else None,
                    doi=f"PMID:{pmid}" if pmid else None,
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
