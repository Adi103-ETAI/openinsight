"""
Cochrane Systematic Reviews Parser

Retrieves Cochrane Database of Systematic Reviews (CDSR) records via
PubMed/Entrez.  Cochrane reviews are fully indexed in MEDLINE under
journal "Cochrane Database Syst Rev".

Usage:
    parser = CochraneParser(query="hypertension management", max_results=50)
    documents = parser.parse()
"""
import re
import time

from Bio import Entrez
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_fixed

from src.core.config import get_settings
from src.ingestion.document_db import DocumentRecord
from src.ingestion.parsers.base import BaseParser

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

    @property
    def source_type(self) -> str:
        return "cochrane"

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    def _fetch_records(self) -> list[dict]:
        Entrez.email = self.settings.ncbi_email
        Entrez.api_key = self.settings.ncbi_api_key or None

        # Restrict query to Cochrane journal
        full_query = f'({self.query}) AND ("{_COCHRANE_JOURNAL}"[Journal])'

        with Entrez.esearch(db="pubmed", term=full_query, retmax=self.max_results) as handle:
            search_result = Entrez.read(handle)

        pmids = search_result.get("IdList", [])
        logger.info(f"[Cochrane] query='{self.query}' found {len(pmids)} results")
        if not pmids:
            return []

        sleep_seconds = 0.1 if self.settings.ncbi_api_key else 0.34
        time.sleep(sleep_seconds)

        with Entrez.efetch(db="pubmed", id=pmids, rettype="xml", retmode="xml") as handle:
            fetched = Entrez.read(handle)

        return fetched.get("PubmedArticle", [])

    def parse(self) -> list[DocumentRecord]:
        try:
            articles = self._fetch_records()
            documents: list[DocumentRecord] = []

            for article in articles:
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
                journal = article_data.get("Journal", {})
                year = ""
                journal_issue = journal.get("JournalIssue", {})
                pub_date = journal_issue.get("PubDate", {})
                if isinstance(pub_date, dict):
                    year = str(pub_date.get("Year", "")).strip()
                    if not year:
                        medline_date = str(pub_date.get("MedlineDate", "")).strip()
                        m = re.search(r"\b(19|20)\d{2}\b", medline_date)
                        if m:
                            year = m.group(0)

                doc = DocumentRecord(
                    source_type=self.source_type,
                    title=title,
                    content=abstract_text,
                    url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else None,
                    doi=f"PMID:{pmid}" if pmid else None,
                    published_date=year or None,
                    year=int(year) if year.isdigit() else None,
                    journal=_COCHRANE_JOURNAL,
                    study_type="meta_analysis",
                    evidence_level=1,  # Cochrane reviews are highest evidence
                    parser_version="v3",
                )
                documents.append(doc)

            logger.info(f"[Cochrane] Parsed {len(documents)} documents for query='{self.query}'")
            return documents

        except Exception as exc:
            logger.error(f"[Cochrane] parse failed for query='{self.query}': {exc}")
            return []
