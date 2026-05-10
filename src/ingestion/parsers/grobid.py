"""
GROBID Parser
Uses GROBID REST API to extract structured content from research PDFs.
Extracts: title, abstract, sections, full text with section labels.
Falls back to pdfplumber if GROBID is unavailable.
"""

import requests
from pathlib import Path
from loguru import logger
from bs4 import BeautifulSoup
from typing import Optional

from src.ingestion.document_db import DocumentRecord
from src.ingestion.parsers.base import BaseParser
from src.ml.ner import infer_study_type
from src.config.settings import get_settings

settings = get_settings()


class GROBIDParser(BaseParser):
    """
    Parser for research PDFs using GROBID.
    Best for: PubMed full-text PDFs, journal articles, research papers.
    Not ideal for: government guidelines, policy documents (use ICMRParser for those).
    """

    def __init__(self, file_path: str | Path, source_type: str = "research"):
        self.file_path = Path(file_path)
        self._source_type = source_type

    @property
    def source_type(self) -> str:
        return self._source_type

    def _call_grobid(self) -> Optional[str]:
        """Send PDF to GROBID and return TEI XML response."""
        url = f"{settings.grobid_url}/api/processFulltextDocument"
        try:
            with open(self.file_path, "rb") as f:
                response = requests.post(
                    url,
                    files={"input": f},
                    data={"consolidateHeader": "1", "consolidateCitations": "0"},
                    timeout=120,
                )
            if response.status_code == 200:
                return response.text
            else:
                logger.warning(
                    f"GROBID returned {response.status_code} for {self.file_path.name}"
                )
                return None
        except (
            requests.RequestException,
            RuntimeError,
            ValueError,
            TypeError,
            OSError,
        ) as e:
            logger.warning(f"GROBID call failed for {self.file_path.name}: {e}")
            return None

    def _parse_tei(self, tei_xml: str) -> dict:
        """Parse TEI XML from GROBID into structured dict."""
        soup = BeautifulSoup(tei_xml, "xml")
        result = {
            "title": "",
            "abstract": "",
            "sections": [],
            "year": None,
            "journal": "",
        }

        # Title
        title_tag = soup.find("titleStmt")
        if title_tag:
            result["title"] = title_tag.get_text(separator=" ").strip()

        # Abstract
        abstract_tag = soup.find("abstract")
        if abstract_tag:
            result["abstract"] = abstract_tag.get_text(separator=" ").strip()

        # Year
        date_tag = soup.find("date", {"type": "published"})
        if date_tag and date_tag.get("when"):
            try:
                result["year"] = int(date_tag["when"][:4])
            except (TypeError, ValueError):
                pass

        # Journal
        journal_tag = soup.find("title", {"level": "j"})
        if journal_tag:
            result["journal"] = journal_tag.get_text().strip()

        # Body sections
        body = soup.find("body")
        if body:
            for div in body.find_all("div"):
                head = div.find("head")
                section_title = head.get_text().strip() if head else None
                paragraphs = div.find_all("p")
                section_text = " ".join(
                    p.get_text(separator=" ").strip() for p in paragraphs
                )
                if section_text.strip():
                    result["sections"].append(
                        {
                            "title": section_title,
                            "text": section_text,
                        }
                    )

        return result

    def parse(self) -> list[DocumentRecord]:
        if not self.file_path.exists():
            logger.error(f"File not found: {self.file_path}")
            return []

        logger.info(f"Parsing with GROBID: {self.file_path.name}")
        tei_xml = self._call_grobid()

        if not tei_xml:
            logger.warning(
                f"GROBID failed, falling back to pdfplumber for: {self.file_path.name}"
            )
            from src.ingestion.parsers.icmr import ICMRParser

            fallback = ICMRParser(self.file_path)
            return fallback.parse()

        parsed = self._parse_tei(tei_xml)

        # Build full content: abstract + all section texts
        content_parts = []
        if parsed["abstract"]:
            content_parts.append(f"Abstract\n{parsed['abstract']}")
        for section in parsed["sections"]:
            if section["title"]:
                content_parts.append(f"{section['title']}\n{section['text']}")
            else:
                content_parts.append(section["text"])

        full_content = "\n\n".join(content_parts).strip()
        if not full_content:
            logger.warning(f"GROBID extracted no content from {self.file_path.name}")
            return []

        title = parsed["title"] or self.file_path.stem.replace("_", " ")
        study_type, evidence_level = infer_study_type(full_content[:500], title)

        doc = DocumentRecord(
            source_type=self.source_type,
            title=title,
            content=full_content,
            url=str(self.file_path.resolve()),
            published_date=str(parsed["year"]) if parsed["year"] else None,
            year=parsed["year"],
            journal=parsed["journal"],
            study_type=study_type,
            evidence_level=evidence_level,
            is_india_specific=False,
            parser_version="v2",
        )
        logger.info(f"GROBID parsed: {title[:60]} — {len(full_content)} chars")
        return [doc]
