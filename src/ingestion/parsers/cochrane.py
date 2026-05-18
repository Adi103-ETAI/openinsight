"""
Cochrane Systematic Reviews Parser

Retrieves Cochrane Database of Systematic Reviews (CDSR) records via
PubMed/Entrez.  Cochrane reviews are fully indexed in MEDLINE under
journal "Cochrane Database Syst Rev".

Uses shared PubMedClient for all Entrez interactions.

Usage:
    parser = CochraneParser(query="hypertension management", max_results=50)
    documents = parser.parse()
"""

from loguru import logger

from src.config.settings import get_settings
from src.ingestion.document_db import DocumentRecord
from src.ingestion.parsers.base import BaseParser
from src.utils.pubmed_client import PubMedClient

_COCHRANE_JOURNAL = "Cochrane Database Syst Rev"


class CochraneParser(BaseParser):
    """
    Fetches Cochrane systematic reviews through the NCBI Entrez API.
    Restricts the PubMed search to the Cochrane Database of Systematic
    Reviews journal and enriches metadata with evidence level 1.
    """

    def __init__(self, query: str, max_results: int = 50):
        self.query = query
        self.max_results = max_results
        self.settings = get_settings()
        self._client = PubMedClient(self.settings)

    @property
    def source_type(self) -> str:
        return "cochrane"

    def parse(self) -> list[DocumentRecord]:
        try:
            # Restrict query to Cochrane journal
            full_query = f'({self.query}) AND ("{_COCHRANE_JOURNAL}"[Journal])'
            pmids = self._client.search_pubmed(
                full_query, max_results=self.max_results
            )
            if not pmids:
                logger.info(f"[Cochrane] query='{self.query}' found 0 results")
                return []

            articles = self._client.fetch_pubmed_articles(pmids)
            documents: list[DocumentRecord] = []

            for article in articles:
                pubmed_article = PubMedClient.article_to_pubmed_article(
                    article, journal_override=_COCHRANE_JOURNAL
                )
                if pubmed_article is None:
                    continue

                doc = DocumentRecord(
                    source_type=self.source_type,
                    title=pubmed_article.title,
                    content=pubmed_article.abstract,
                    url=pubmed_article.url,
                    doi=pubmed_article.doi,
                    published_date=pubmed_article.year or None,
                    year=int(pubmed_article.year) if pubmed_article.year.isdigit() else None,
                    journal=_COCHRANE_JOURNAL,
                    study_type="meta_analysis",
                    evidence_level=1,  # Cochrane reviews are highest evidence
                    parser_version="v3",
                )
                documents.append(doc)

            logger.info(
                f"[Cochrane] Parsed {len(documents)} documents for query='{self.query}'"
            )
            return documents

        except (RuntimeError, ValueError, TypeError, OSError) as exc:
            logger.error(f"[Cochrane] parse failed for query='{self.query}': {exc}")
            return []
