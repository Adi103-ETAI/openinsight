"""
StatPearls Parser (NCBI Bookshelf)

StatPearls is a continuously-updated, peer-reviewed medical reference
published by StatPearls Publishing.  It is freely available through the
NCBI Bookshelf (https://www.ncbi.nlm.nih.gov/books/NBK430685/).

This parser queries the NCBI Bookshelf via the Entrez E-utilities API
(db=books) to retrieve StatPearls chapters matching a clinical query.

Uses shared PubMedClient for all Entrez interactions.

Usage:
    parser = StatPearlsParser(query="tuberculosis treatment", max_results=20)
    documents = parser.parse()
"""

from typing import Optional

from loguru import logger

from src.config.settings import get_settings
from src.ingestion.document_db import DocumentRecord
from src.ingestion.parsers.base import BaseParser
from src.utils.pubmed_client import PubMedClient

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
        self._client = PubMedClient(self.settings)

    @property
    def source_type(self) -> str:
        return "statpearls"

    def _pmc_article_to_doc(self, article: dict) -> Optional[DocumentRecord]:
        """Convert a PMC article dict to a DocumentRecord."""
        try:
            title = PubMedClient.extract_pmc_title(article)
            if not title:
                return None

            abstract_text = PubMedClient.extract_pmc_abstract(article)
            if not abstract_text:
                return None

            year_str = PubMedClient.extract_pmc_year(article)
            ids = PubMedClient.extract_pmc_ids(article)
            pmcid = ids["pmcid"]
            pmid = ids["pmid"]

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

    def parse(self) -> list[DocumentRecord]:
        documents: list[DocumentRecord] = []

        # Primary: PMC full-text search
        try:
            full_query = f'("{self.query}") AND ("StatPearls"[Book])'
            pmcids = self._client.search_pubmed(
                full_query, max_results=self.max_results, db="pmc"
            )
            logger.info(
                f"[StatPearls] query='{self.query}' found {len(pmcids)} PMC results"
            )

            if pmcids:
                articles = self._client.fetch_pmc_articles(pmcids)
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
                full_query = f"({self.query}) AND (StatPearls[Title/Abstract])"
                pmids = self._client.search_pubmed(
                    full_query, max_results=self.max_results
                )
                if pmids:
                    articles = self._client.fetch_pubmed_articles(pmids)
                    for article in articles:
                        pubmed_article = PubMedClient.article_to_pubmed_article(
                            article, journal_override="StatPearls"
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
                                journal="StatPearls",
                                study_type="review",
                                evidence_level=3,
                                parser_version="v3",
                            )
                        )
                logger.info(
                    f"[StatPearls] PubMed fallback returned {len(documents)} documents"
                )
            except (RuntimeError, ValueError, TypeError, OSError) as exc2:
                logger.error(f"[StatPearls] Both strategies failed: {exc2}")

        return documents
