"""Medknow parser — converts scraped Medknow article HTML into DocumentRecord + ChunkRecord.

Medknow (medknow.com) hosts Indian journals using a custom Wolters Kluwer
template (not OJS). Article pages expose Highwire Press citation_* meta tags,
which the scraper's MetadataExtractor handles. This parser:

1. Builds a DocumentRecord from extracted metadata
2. Extracts full-text body (Medknow article body — usually in <div id="article"> or similar)
3. Chunks the body into ChunkRecord entries
4. Records ISSN + journal name for cross-source dedup

When this parser produces a document that matches an existing PubMed document
(by DOI or PMID), the pipeline's cross-source dedup should:
- Keep the PubMed version as canonical (it has structured XML)
- Set `also_indexed_in: ["pubmed", "medknow"]` on the canonical doc
- Fetch the Medknow full-text PDF separately and create additional
  full-text chunks linked to the canonical document via `document_id`
"""
from __future__ import annotations

import hashlib
import re
from typing import Any

from bs4 import BeautifulSoup
from loguru import logger

from src.ingestion.document_db import ChunkRecord, DocumentRecord
from src.ingestion.scrapers.framework.models import ScrapedDocument


class MedknowParser:
    """Parses Medknow-hosted article HTML into DocumentRecord + ChunkRecord."""

    def __init__(self) -> None:
        self.source_type = "medknow"

    def parse(self, doc: ScrapedDocument) -> tuple[DocumentRecord, list[ChunkRecord]]:
        """Parse one scraped Medknow article."""
        if not doc.content:
            logger.warning(f"[medknow:parser] empty content for {doc.url}")
            return self._empty_record(doc), []

        try:
            html = doc.content.decode("utf-8", errors="replace")
        except Exception:
            html = doc.content.decode("latin-1", errors="replace")

        soup = BeautifulSoup(html, "lxml")
        body_text = self._extract_body_text(soup)
        abstract = doc.abstract or self._extract_abstract(soup)
        full_content = body_text or abstract or ""

        record = DocumentRecord(
            source_type=self.source_type,
            title=doc.title or "Untitled",
            content=full_content,
            url=doc.url,
            doi=doc.doi,
            published_date=doc.pubdate,
            year=self._extract_year(doc.pubdate),
            journal=doc.journal,
            is_india_specific=True,
            parser_version="medknow-v1",
            content_hash=self._content_hash(full_content),
            condition_tags=[],
            specialty_tags=doc.metadata.get("journal_abbr") and [doc.metadata["journal_abbr"]] or [],
        )

        chunks: list[ChunkRecord] = []
        if full_content:
            chunks = self._chunk_body(full_content, record, doc)

        logger.info(
            f"[medknow:parser] parsed {doc.url}: title='{(doc.title or '')[:60]}', "
            f"chunks={len(chunks)}"
        )
        return record, chunks

    def _extract_body_text(self, soup: BeautifulSoup) -> str:
        """Extract article body from Medknow's Wolters Kluwer template.

        Medknow templates vary, but the body usually lives in:
        - <div id="article">
        - <div class="article-content">
        - <div id="content">
        - <div class="fulltext">
        Falls back to all paragraphs if none match.
        """
        for selector in [
            {"name": "div", "id": "article"},
            {"name": "div", "class_": "article-content"},
            {"name": "div", "id": "content"},
            {"name": "div", "class_": "fulltext"},
            {"name": "div", "class_": "article-fulltext"},
            {"name": "article"},
        ]:
            container = soup.find(**selector)
            if container:
                text = container.get_text("\n", strip=True)
                if len(text) > 200:
                    return text

        paragraphs = []
        for p in soup.find_all("p"):
            text = p.get_text(strip=True)
            if len(text) > 50:
                paragraphs.append(text)
        return "\n\n".join(paragraphs)

    def _extract_abstract(self, soup: BeautifulSoup) -> str | None:
        """Extract abstract from Medknow article page."""
        meta = soup.find("meta", attrs={"name": "citation_abstract"})
        if meta and meta.get("content"):
            return BeautifulSoup(meta["content"], "lxml").get_text(" ", strip=True)
        for selector in [
            {"name": "div", "class_": "abstract"},
            {"name": "section", "class_": "abstract"},
            {"name": "p", "class_": "abstract"},
            {"name": "div", "id": "abstract"},
        ]:
            el = soup.find(**selector)
            if el:
                return el.get_text(" ", strip=True)
        return None

    def _chunk_body(
        self,
        body_text: str,
        record: DocumentRecord,
        doc: ScrapedDocument,
    ) -> list[ChunkRecord]:
        """Paragraph-based chunking with 200-char overlap."""
        target_chars = 1500
        overlap_chars = 200

        paragraphs = [p.strip() for p in body_text.split("\n\n") if p.strip()]
        if not paragraphs:
            paragraphs = [body_text]

        chunks: list[ChunkRecord] = []
        current = ""
        chunk_idx = 0

        for para in paragraphs:
            if len(current) + len(para) + 2 > target_chars and current:
                chunks.append(self._make_chunk(current, record, doc, chunk_idx))
                chunk_idx += 1
                if len(current) > overlap_chars:
                    current = current[-overlap_chars:] + "\n\n" + para
                else:
                    current = para
            else:
                current = (current + "\n\n" + para) if current else para

        if current:
            chunks.append(self._make_chunk(current, record, doc, chunk_idx))
        return chunks

    def _make_chunk(
        self,
        text: str,
        record: DocumentRecord,
        doc: ScrapedDocument,
        idx: int,
    ) -> ChunkRecord:
        return ChunkRecord(
            document_id=record.url or doc.url,
            source_type=self.source_type,
            title=record.title,
            chunk_text=text,
            chunk_index=idx,
            char_count=len(text),
            section="body",
            diseases=[],
            drugs=[],
            symptoms=[],
            dosages=[],
            contraindications=[],
            patient_populations=[],
            outcomes=[],
            has_safety_flag=False,
            content_type="text",
            content_weight=1.0,
            quality_score=1.0,
            is_india_specific=True,
            evidence_level=record.evidence_level,
            parser_version="medknow-v1",
            token_estimate=len(text) // 4,
        )

    def _extract_year(self, pubdate: str | None) -> int | None:
        if not pubdate:
            return None
        match = re.search(r"(19|20)\d{2}", pubdate)
        return int(match.group(0)) if match else None

    def _content_hash(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    def _empty_record(self, doc: ScrapedDocument) -> DocumentRecord:
        return DocumentRecord(
            source_type=self.source_type,
            title=doc.title or "Untitled",
            content="",
            url=doc.url,
            doi=doc.doi,
            published_date=doc.pubdate,
            year=self._extract_year(doc.pubdate),
            journal=doc.journal,
            is_india_specific=True,
            parser_version="medknow-v1",
            content_hash="",
            condition_tags=[],
            specialty_tags=[],
        )
