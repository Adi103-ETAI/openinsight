"""
PubMed Parser

Retrieves articles from PubMed via the NCBI Entrez API.
Uses the shared PubMedClient for all Entrez interactions.

Usage:
    parser = PubMedParser(query="malaria treatment", max_results=100)
    documents = parser.parse()
"""

from loguru import logger

from src.config.settings import get_settings
from src.ingestion.document_db import DocumentRecord
from src.ingestion.parsers.base import BaseParser
from src.utils.pubmed_client import PubMedClient
from src.utils.text_utils import extract_keywords_from_query


class PubMedParser(BaseParser):
    """Fetches and parses PubMed articles via NCBI Entrez API."""

    def __init__(self, query: str, max_results: int = 100):
        self.query = query
        self.max_results = max_results
        self.settings = get_settings()
        self._client = PubMedClient(self.settings)

    @property
    def source_type(self) -> str:
        return "pubmed"

    def _extract_condition_tags(self) -> list[str]:
        """Extract meaningful condition keywords from the search query."""
        return extract_keywords_from_query(self.query)

    def parse(self) -> list[DocumentRecord]:
        try:
            pmids = self._client.search_pubmed(
                self.query, max_results=self.max_results
            )
            if not pmids:
                logger.info(
                    f"PubMed query='{self.query}' found 0 results"
                )
                return []

            articles = self._client.fetch_pubmed_articles(pmids)
            condition_tags = self._extract_condition_tags()
            documents: list[DocumentRecord] = []

            for article in articles:
                pubmed_article = PubMedClient.article_to_pubmed_article(article)
                if pubmed_article is None:
                    continue

                doc = DocumentRecord(
                    source_type=self.source_type,
                    title=pubmed_article.title,
                    content=pubmed_article.abstract,
                    url=pubmed_article.url,
                    doi=pubmed_article.doi,
                    published_date=pubmed_article.year or None,
                    condition_tags=condition_tags,
                )
                documents.append(doc)

            logger.info(
                f"PubMed query='{self.query}' parsed {len(documents)} documents with abstracts"
            )
            return documents

        except (RuntimeError, ValueError, TypeError, OSError) as exc:
            logger.error(f"PubMed parse failed for query='{self.query}': {exc}")
            return []
