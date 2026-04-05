"""
WHO Guidelines Parser

Retrieves WHO guideline publications from the WHO IRIS open repository
(https://iris.who.int) using its public REST/OAI-PMH API and from
PubMed (WHO publications are partially indexed in MEDLINE).

Primary strategy: WHO IRIS REST API (JSON)
Fallback strategy: PubMed search restricted to WHO publications
"""
import re
import time
from typing import Optional

import requests
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_fixed

from src.core.config import get_settings
from src.ingestion.document_db import DocumentRecord
from src.ingestion.parsers.base import BaseParser

_WHO_IRIS_BASE = "https://iris.who.int/rest/items"
_WHO_IRIS_SEARCH = "https://iris.who.int/rest/search"
_REQUEST_TIMEOUT = 15


class WHOParser(BaseParser):
    """
    Fetches WHO guidelines and health documents via the WHO IRIS REST API.
    Falls back to PubMed for WHO-authored records when IRIS is unavailable.
    """

    def __init__(self, query: str, max_results: int = 50):
        self.query = query
        self.max_results = max_results
        self.settings = get_settings()

    @property
    def source_type(self) -> str:
        return "who"

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(3))
    def _fetch_iris(self) -> list[dict]:
        """Query WHO IRIS repository for guidelines matching the query."""
        params = {
            "query": self.query,
            "scope": "/handle/10665",  # WHO IRIS root handle
            "rpp": self.max_results,
            "sort_by": "score",
            "order": "desc",
        }
        resp = requests.get(
            _WHO_IRIS_SEARCH,
            params=params,
            timeout=_REQUEST_TIMEOUT,
            headers={"Accept": "application/json"},
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("_embedded", {}).get("items", [])

    def _parse_iris_item(self, item: dict) -> Optional[DocumentRecord]:
        """Convert a WHO IRIS item dict to a DocumentRecord."""
        name = item.get("name", "").strip()
        handle = item.get("handle", "")
        url = f"https://iris.who.int/handle/{handle}" if handle else None

        # Metadata
        metadata: list[dict] = item.get("metadata", [])
        meta_map: dict[str, list[str]] = {}
        for entry in metadata:
            key = entry.get("key", "")
            value = entry.get("value", "")
            if key and value:
                meta_map.setdefault(key, []).append(value)

        title = (
            (meta_map.get("dc.title") or meta_map.get("dc.title.alternative") or [name])[0].strip()
        )
        abstract_parts = meta_map.get("dc.description.abstract") or meta_map.get("dc.description") or []
        abstract = " ".join(abstract_parts).strip()

        if not title or not abstract:
            return None

        year_str = ""
        date_parts = meta_map.get("dc.date.issued") or meta_map.get("dc.date") or []
        if date_parts:
            m = re.search(r"\b(19|20)\d{2}\b", date_parts[0])
            if m:
                year_str = m.group(0)

        doi_parts = meta_map.get("dc.identifier.doi") or []
        doi = doi_parts[0].strip() if doi_parts else None

        return DocumentRecord(
            source_type=self.source_type,
            title=title,
            content=abstract,
            url=url,
            doi=doi,
            published_date=year_str or None,
            year=int(year_str) if year_str.isdigit() else None,
            journal="WHO Guidelines",
            study_type="guideline",
            evidence_level=2,
            parser_version="v3",
        )

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    def _fetch_pubmed_fallback(self) -> list[DocumentRecord]:
        """PubMed fallback: search for WHO-published records."""
        from Bio import Entrez

        Entrez.email = self.settings.ncbi_email
        Entrez.api_key = self.settings.ncbi_api_key or None

        full_query = f'({self.query}) AND ("World Health Organization"[Corporate Author])'
        with Entrez.esearch(db="pubmed", term=full_query, retmax=self.max_results) as handle:
            search_result = Entrez.read(handle)

        pmids = search_result.get("IdList", [])
        if not pmids:
            return []

        sleep_seconds = 0.1 if self.settings.ncbi_api_key else 0.34
        time.sleep(sleep_seconds)

        with Entrez.efetch(db="pubmed", id=pmids, rettype="xml", retmode="xml") as handle:
            fetched = Entrez.read(handle)

        documents: list[DocumentRecord] = []
        for article in fetched.get("PubmedArticle", []):
            citation = article.get("MedlineCitation", {})
            article_data = citation.get("Article", {})
            title = str(article_data.get("ArticleTitle", "")).strip()
            if not title:
                continue
            abstract = article_data.get("Abstract", {})
            abstract_parts = abstract.get("AbstractText", []) if isinstance(abstract, dict) else []
            abstract_text = " ".join(str(p).strip() for p in abstract_parts if str(p).strip())
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
                    journal="WHO Publications",
                    study_type="guideline",
                    evidence_level=2,
                    parser_version="v3",
                )
            )
        return documents

    def parse(self) -> list[DocumentRecord]:
        documents: list[DocumentRecord] = []

        # Primary: WHO IRIS
        try:
            items = self._fetch_iris()
            for item in items:
                doc = self._parse_iris_item(item)
                if doc:
                    documents.append(doc)
            logger.info(f"[WHO] IRIS returned {len(documents)} documents for query='{self.query}'")
        except Exception as exc:
            logger.warning(f"[WHO] IRIS fetch failed ({exc}), trying PubMed fallback")
            try:
                documents = self._fetch_pubmed_fallback()
                logger.info(f"[WHO] PubMed fallback returned {len(documents)} documents")
            except Exception as exc2:
                logger.error(f"[WHO] Both strategies failed: {exc2}")

        return documents
