"""IndMED parser — converts scraped HTML article pages into DocumentRecord + ChunkRecord.

Used by the ingestion pipeline after IndMEDScraper has fetched the article.
This parser does NOT do any HTTP — it takes already-fetched bytes (from
ScrapedDocument.content) and produces structured records.

Most of the heavy lifting (title, authors, DOI, abstract extraction) is
done by the scraper framework's MetadataExtractor at fetch time. This parser:
1. Builds a DocumentRecord from the extracted metadata
2. Extracts the full-text body (OJS article body HTML)
3. Chunks the body into ChunkRecord entries
4. Records provenance (URL, journal, IndMED source)

If a PDF download URL is present in ScrapedDocument.metadata['pdf_url'],
this parser records it in DocumentRecord.url for the pipeline to fetch
separately (the pipeline's PDF-extraction step handles the PDF → text
conversion via GROBID).
"""
from __future__ import annotations

import hashlib
import re
from datetime import datetime
from typing import Any
from urllib.parse import urlparse

from bs4 import BeautifulSoup
from loguru import logger

from src.ingestion.document_db import ChunkRecord, DocumentRecord
from src.ingestion.scrapers.framework.models import ScrapedDocument


class IndMEDParser:
    """Parses IndMED (OJS-hosted) article HTML into DocumentRecord + ChunkRecord."""

    def __init__(self) -> None:
        self.source_type = "indmed"

    def parse(self, doc: ScrapedDocument) -> tuple[DocumentRecord, list[ChunkRecord]]:
        """Parse one scraped IndMED article into a document + its chunks.

        Args:
            doc: ScrapedDocument from IndMEDScraper.fetch_one()

        Returns:
            (DocumentRecord, list[ChunkRecord]) — empty list if no body text extracted
        """
        if not doc.content:
            logger.warning(f"[indmed:parser] empty content for {doc.url}")
            return self._empty_record(doc), []

        try:
            html = doc.content.decode("utf-8", errors="replace")
        except Exception:
            html = doc.content.decode("latin-1", errors="replace")

        soup = BeautifulSoup(html, "lxml")

        # Extract body text (OJS article body)
        body_text = self._extract_body_text(soup)

        # Extract abstract if not already in metadata
        abstract = doc.abstract or self._extract_abstract(soup)

        # Build the DocumentRecord (matches the actual Pydantic schema in document_db.py)
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
            is_india_specific=True,  # IndMED = Indian journals
            parser_version="indmed-v1",
            content_hash=self._content_hash(full_content),
            # Pydantic schema doesn't have these — store in condition_tags/specialty_tags
            condition_tags=[],
            specialty_tags=[doc.metadata.get("journal_abbr")] if doc.metadata.get("journal_abbr") else [],
        )

        # Chunk the body
        chunks: list[ChunkRecord] = []
        if full_content:
            chunks = self._chunk_body(full_content, record, doc)

        logger.info(
            f"[indmed:parser] parsed {doc.url}: title='{(doc.title or '')[:60]}', "
            f"chunks={len(chunks)}"
        )
        return record, chunks

    def _extract_body_text(self, soup: BeautifulSoup) -> str:
        """Extract the main article body text from OJS HTML.

        OJS templates vary, but the body usually lives in:
        - <div class="article-body">
        - <section class="main-entry">
        - <div id="content">
        We try each in order and fall back to concatenating paragraph text.
        """
        # Try OJS-specific containers
        for selector in [
            {"name": "div", "class_": "article-body"},
            {"name": "section", "class_": "main-entry"},
            {"name": "div", "id": "content"},
            {"name": "div", "class_": "content"},
            {"name": "article"},
        ]:
            container = soup.find(**selector)
            if container:
                text = container.get_text("\n", strip=True)
                if len(text) > 200:  # only accept if substantial
                    return text

        # Fallback: collect all paragraphs in the main body
        paragraphs = []
        for p in soup.find_all("p"):
            text = p.get_text(strip=True)
            if len(text) > 50:  # skip nav/footer paragraphs
                paragraphs.append(text)
        return "\n\n".join(paragraphs)

    def _extract_abstract(self, soup: BeautifulSoup) -> str | None:
        """Extract abstract from OJS article page."""
        # Try meta tag first (already extracted by MetadataExtractor, but
        # double-check in case the scraper didn't run it)
        meta = soup.find("meta", attrs={"name": "citation_abstract"})
        if meta and meta.get("content"):
            return BeautifulSoup(meta["content"], "lxml").get_text(" ", strip=True)
        # Try HTML elements
        for selector in [
            {"name": "div", "class_": "abstract"},
            {"name": "section", "class_": "abstract"},
            {"name": "p", "class_": "abstract"},
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
        """Chunk body text into ~350-token ChunkRecord entries.

        Simple paragraph-based chunking with overlap. The pipeline's
        HierarchicalChunkerV3 runs AFTER this for finer-grained semantic
        chunking — this is the coarse first pass.
        """
        target_chars = 1500  # ~350 tokens
        overlap_chars = 200  # ~50 tokens

        # Split on double newlines (paragraph boundaries)
        paragraphs = [p.strip() for p in body_text.split("\n\n") if p.strip()]
        if not paragraphs:
            paragraphs = [body_text]

        chunks: list[ChunkRecord] = []
        current = ""
        chunk_idx = 0

        for para in paragraphs:
            if len(current) + len(para) + 2 > target_chars and current:
                # Flush current chunk
                chunks.append(self._make_chunk(current, record, doc, chunk_idx))
                chunk_idx += 1
                # Start new chunk with overlap
                if len(current) > overlap_chars:
                    current = current[-overlap_chars:] + "\n\n" + para
                else:
                    current = para
            else:
                current = (current + "\n\n" + para) if current else para

        # Flush last chunk
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
        """Build a ChunkRecord from text + provenance."""
        return ChunkRecord(
            document_id=record.url or doc.url,  # use URL as document_id (Pydantic schema has no doc_id field)
            source_type=self.source_type,
            title=record.title,
            chunk_text=text,
            chunk_index=idx,
            char_count=len(text),
            section="body",
            diseases=[],  # populated by NER later in pipeline
            drugs=[],
            symptoms=[],
            dosages=[],
            contraindications=[],
            patient_populations=[],
            outcomes=[],
            has_safety_flag=False,
            content_type="text",
            content_weight=1.0,
            quality_score=1.0,  # populated by quality scorer later
            is_india_specific=True,
            evidence_level=record.evidence_level,
            parser_version="indmed-v1",
            token_estimate=len(text) // 4,  # rough estimate
        )

    def _extract_year(self, pubdate: str | None) -> int | None:
        if not pubdate:
            return None
        match = re.search(r"(19|20)\d{2}", pubdate)
        return int(match.group(0)) if match else None

    def _content_hash(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    def _empty_record(self, doc: ScrapedDocument) -> DocumentRecord:
        """Build a minimal DocumentRecord for empty/failed parses."""
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
            parser_version="indmed-v1",
            content_hash="",
            condition_tags=[],
            specialty_tags=[],
        )
