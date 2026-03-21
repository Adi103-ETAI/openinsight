import re
import time

from Bio import Entrez
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_fixed

from src.core.config import get_settings
from src.ingestion.document_db import DocumentRecord
from src.ingestion.parsers.base import BaseParser

COMMON_QUERY_WORDS = {"india", "treatment", "management", "clinical", "study"}


class PubMedParser(BaseParser):
    def __init__(self, query: str, max_results: int = 100):
        self.query = query
        self.max_results = max_results
        self.settings = get_settings()

    @property
    def source_type(self) -> str:
        return "pubmed"

    def _extract_condition_tags(self) -> list[str]:
        tokens = re.findall(r"[A-Za-z0-9\-]+", self.query.lower())
        tags: list[str] = []
        for token in tokens:
            if len(token) < 4:
                continue
            if token in COMMON_QUERY_WORDS:
                continue
            if token not in tags:
                tags.append(token)
        return tags

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    def _fetch_records(self) -> list[dict]:
        Entrez.email = self.settings.ncbi_email
        Entrez.api_key = self.settings.ncbi_api_key or None

        with Entrez.esearch(db="pubmed", term=self.query, retmax=self.max_results) as handle:
            search_result = Entrez.read(handle)
        pmids = search_result.get("IdList", [])

        logger.info(f"PubMed query='{self.query}' found {len(pmids)} results")
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
            condition_tags = self._extract_condition_tags()

            for article in articles:
                citation = article.get("MedlineCitation", {})
                article_data = citation.get("Article", {})

                title = str(article_data.get("ArticleTitle", "")).strip()
                if not title:
                    continue

                abstract = article_data.get("Abstract", {})
                abstract_text_parts = abstract.get("AbstractText", []) if isinstance(abstract, dict) else []
                abstract_text = " ".join(str(part).strip() for part in abstract_text_parts if str(part).strip())
                if not abstract_text:
                    continue

                pmid = str(citation.get("PMID", "")).strip()

                journal = article_data.get("Journal", {})
                journal_title = str(journal.get("Title", "")).strip()

                year = ""
                journal_issue = journal.get("JournalIssue", {})
                pub_date = journal_issue.get("PubDate", {})
                if isinstance(pub_date, dict):
                    year = str(pub_date.get("Year", "")).strip()
                    if not year:
                        medline_date = str(pub_date.get("MedlineDate", "")).strip()
                        year_match = re.search(r"\b(19|20)\d{2}\b", medline_date)
                        if year_match:
                            year = year_match.group(0)

                doc = DocumentRecord(
                    source_type=self.source_type,
                    title=title,
                    content=abstract_text,
                    url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else None,
                    doi=f"PMID:{pmid}" if pmid else None,
                    published_date=year or None,
                    condition_tags=condition_tags,
                )
                documents.append(doc)

            logger.info(
                f"PubMed query='{self.query}' parsed {len(documents)} documents with abstracts"
            )
            return documents
        except Exception as exc:
            logger.error(f"PubMed parse failed for query='{self.query}': {exc}")
            return []
