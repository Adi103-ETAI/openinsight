"""
StatPearls Parser (NCBI Bookshelf)

StatPearls is a continuously-updated, peer-reviewed medical reference
published by StatPearls Publishing.  It is freely available through the
NCBI Bookshelf (https://www.ncbi.nlm.nih.gov/books/NBK430685/).

This parser queries the NCBI Bookshelf via the Entrez E-utilities API
(db=books) to retrieve StatPearls chapters matching a clinical query.

Usage:
    parser = StatPearlsParser(query="tuberculosis treatment", max_results=20)
    documents = parser.parse()
"""

import time
from typing import Optional

from Bio import Entrez
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_fixed

from src.config.settings import get_settings
from src.ingestion.document_db import DocumentRecord
from src.ingestion.parsers.base import BaseParser

# StatPearls book ID on NCBI Bookshelf
_STATPEARLS_BOOK_ID = "NBK430685"


class StatPearlsParser(BaseParser):
    """
    Fetches StatPearls book chapters from NCBI Bookshelf via Entrez.
    Each chapter is a concise, clinician-oriented review of a condition,
    drug, or procedure — high clinical relevance.
    """

    def __init__(self, query: str, max_results: int = 20):
        self.query = query
        self.max_results = max_results
        self.settings = get_settings()

    @property
    def source_type(self) -> str:
        return "statpearls"

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    def _search_pmc(self) -> list[str]:
        """
        Search PubMed Central for StatPearls articles.
        StatPearls chapters are indexed in PMC under the StatPearls book.
        """
        Entrez.email = self.settings.ncbi_email
        Entrez.api_key = self.settings.ncbi_api_key or None

        # Search PMC for StatPearls articles
        full_query = f'("{self.query}") AND ("StatPearls"[Book])'
        with Entrez.esearch(
            db="pmc", term=full_query, retmax=self.max_results
        ) as handle:
            search_result = Entrez.read(handle)

        pmcids = search_result.get("IdList", [])
        logger.info(
            f"[StatPearls] query='{self.query}' found {len(pmcids)} PMC results"
        )
        return pmcids

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    def _fetch_pmc_articles(self, pmcids: list[str]) -> list[dict]:
        """Fetch full PMC article records."""
        if not pmcids:
            return []

        sleep_seconds = 0.1 if self.settings.ncbi_api_key else 0.34
        time.sleep(sleep_seconds)

        with Entrez.efetch(db="pmc", id=pmcids, rettype="xml", retmode="xml") as handle:
            data = Entrez.read(handle)

        articles = data.get("article", [])
        if not articles and isinstance(data, list):
            articles = data
        return articles if isinstance(articles, list) else []

    def _pmc_article_to_doc(self, article: dict) -> Optional[DocumentRecord]:
        """Convert a PMC article dict to a DocumentRecord."""
        try:
            front = article.get("front", {})
            article_meta = front.get("article-meta", {})

            # Title
            title_group = article_meta.get("title-group", {})
            title_raw = title_group.get("article-title", "")
            title = str(title_raw).strip() if title_raw else ""

            if not title:
                return None

            # Abstract
            abstract_section = article_meta.get("abstract", {})
            abstract_text = ""
            if isinstance(abstract_section, dict):
                paragraphs = abstract_section.get("p", [])
                if isinstance(paragraphs, str):
                    abstract_text = paragraphs
                elif isinstance(paragraphs, list):
                    abstract_text = " ".join(str(p) for p in paragraphs)
            elif isinstance(abstract_section, str):
                abstract_text = abstract_section

            abstract_text = abstract_text.strip()
            if not abstract_text:
                return None

            # Year
            pub_date = article_meta.get("pub-date", {})
            year_str = ""
            if isinstance(pub_date, dict):
                year_str = str(pub_date.get("year", "")).strip()
            elif isinstance(pub_date, list) and pub_date:
                year_str = str(pub_date[0].get("year", "")).strip()

            # URL (PMCID)
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

            url = (
                f"https://www.ncbi.nlm.nih.gov/books/{_STATPEARLS_BOOK_ID}/"
                if not pmcid
                else f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmcid}/"
            )
            doi = f"PMID:{pmid}" if pmid else (f"PMC:{pmcid}" if pmcid else None)

            return DocumentRecord(
                source_type=self.source_type,
                title=title,
                content=abstract_text,
                url=url,
                doi=doi,
                published_date=year_str or None,
                year=int(year_str) if year_str.isdigit() else None,
                journal="StatPearls",
                study_type="review",
                evidence_level=3,  # Peer-reviewed book chapter
                parser_version="v3",
            )
        except (RuntimeError, ValueError, TypeError, OSError) as exc:
            logger.debug(f"[StatPearls] Error parsing article: {exc}")
            return None

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    def _pubmed_fallback(self) -> list[DocumentRecord]:
        """
        Fallback: search PubMed for StatPearls-associated records.
        StatPearls articles sometimes appear in PubMed with PMID.
        """
        Entrez.email = self.settings.ncbi_email
        Entrez.api_key = self.settings.ncbi_api_key or None

        full_query = f"({self.query}) AND (StatPearls[Title/Abstract])"
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
                    journal="StatPearls",
                    study_type="review",
                    evidence_level=3,
                    parser_version="v3",
                )
            )
        return documents

    def parse(self) -> list[DocumentRecord]:
        documents: list[DocumentRecord] = []

        # Primary: PMC full-text search
        try:
            pmcids = self._search_pmc()
            if pmcids:
                articles = self._fetch_pmc_articles(pmcids)
                for article in articles:
                    doc = self._pmc_article_to_doc(article)
                    if doc:
                        documents.append(doc)
                logger.info(
                    f"[StatPearls] Parsed {len(documents)} documents for query='{self.query}'"
                )
        except (RuntimeError, ValueError, TypeError, OSError) as exc:
            logger.warning(
                f"[StatPearls] PMC fetch failed ({exc}), trying PubMed fallback"
            )

        # Fallback if nothing found
        if not documents:
            try:
                documents = self._pubmed_fallback()
                logger.info(
                    f"[StatPearls] PubMed fallback returned {len(documents)} documents"
                )
            except (RuntimeError, ValueError, TypeError, OSError) as exc2:
                logger.error(f"[StatPearls] Both strategies failed: {exc2}")

        return documents
