"""NCBI Bookshelf parser — converts Bookshelf HTML into DocumentRecord + ChunkRecord.

Handles various Bookshelf book formats:
- GeneReviews (standardized genetic disease review structure)
- Medical Genetics Summaries
- NCBI Handbook
- General NIH monographs

Each book has its own section structure, but all use NCBI Bookshelf's
common HTML pattern:
- <h1> for book/chapter title
- <h2> / <h3> for section headings
- <p> for paragraph content
- <meta name="citation_*"> for Highwire Press metadata

Section-aware chunking: each <h2> section becomes its own chunk with the
section title preserved in ChunkRecord.section.
"""
from __future__ import annotations

import hashlib
import re
from typing import Any

from bs4 import BeautifulSoup
from loguru import logger

from src.ingestion.document_db import ChunkRecord, DocumentRecord
from src.ingestion.scrapers.framework.models import ScrapedDocument


class NCBIBookshelfParser:
    """Parses NCBI Bookshelf HTML into DocumentRecord + ChunkRecord."""

    def __init__(self) -> None:
        self.source_type = "ncbi_bookshelf"

    def parse(self, doc: ScrapedDocument) -> tuple[DocumentRecord, list[ChunkRecord]]:
        """Parse one scraped Bookshelf article."""
        if not doc.content:
            logger.warning(f"[bookshelf:parser] empty content for {doc.url}")
            return self._empty_record(doc), []

        try:
            html = doc.content.decode("utf-8", errors="replace")
        except Exception:
            html = doc.content.decode("latin-1", errors="replace")

        soup = BeautifulSoup(html, "lxml")

        # Extract title
        title = doc.title or self._extract_title(soup)

        # Extract abstract
        abstract = doc.abstract or self._extract_abstract(soup)

        # Extract sections
        sections = self._extract_sections(soup)

        # Build full content
        full_content_parts = []
        if abstract:
            full_content_parts.append(f"Abstract\n\n{abstract}")
        for sec_title, sec_text in sections:
            full_content_parts.append(f"{sec_title}\n\n{sec_text}")
        full_content = "\n\n".join(full_content_parts) or abstract or ""

        # Determine collection from metadata (affects journal field)
        collection_label = doc.metadata.get("collection_label", "NCBI Bookshelf")

        record = DocumentRecord(
            source_type=self.source_type,
            title=title or "Untitled",
            content=full_content,
            url=doc.url,
            doi=doc.doi,
            published_date=doc.pubdate,
            year=self._extract_year(doc.pubdate),
            journal=collection_label,
            is_india_specific=False,
            parser_version="bookshelf-v1",
            content_hash=self._content_hash(full_content),
            condition_tags=[],
            specialty_tags=[doc.metadata.get("collection", "bookshelf")] if doc.metadata.get("collection") else [],
        )

        # Chunk by section
        chunks: list[ChunkRecord] = []
        chunk_idx = 0

        if abstract and len(abstract) > 80:
            chunks.append(self._make_chunk(
                abstract, record, doc, chunk_idx, "abstract"
            ))
            chunk_idx += 1

        for sec_title, sec_text in sections:
            if not sec_text or len(sec_text) < 80:
                continue
            if len(sec_text) > 2000:
                sub_chunks = self._split_long_section(sec_text, target_chars=1500, overlap=200)
                for i, sub in enumerate(sub_chunks):
                    section_label = f"{sec_title} (part {i+1}/{len(sub_chunks)})" if len(sub_chunks) > 1 else sec_title
                    chunks.append(self._make_chunk(
                        sub, record, doc, chunk_idx, section_label
                    ))
                    chunk_idx += 1
            else:
                chunks.append(self._make_chunk(
                    sec_text, record, doc, chunk_idx, sec_title
                ))
                chunk_idx += 1

        logger.info(
            f"[bookshelf:parser] parsed {doc.url}: title='{(title or '')[:60]}', "
            f"chunks={len(chunks)}"
        )
        return record, chunks

    def _extract_title(self, soup: BeautifulSoup) -> str | None:
        meta = soup.find("meta", attrs={"name": "citation_title"})
        if meta and meta.get("content"):
            return meta["content"].strip()
        h1 = soup.find("h1")
        if h1:
            return h1.get_text(strip=True)
        if soup.title:
            return soup.title.get_text(strip=True).split(" - ")[0]
        return None

    def _extract_abstract(self, soup: BeautifulSoup) -> str | None:
        meta = soup.find("meta", attrs={"name": "citation_abstract"})
        if meta and meta.get("content"):
            return BeautifulSoup(meta["content"], "lxml").get_text(" ", strip=True)
        for selector in [
            {"name": "div", "class_": "abstract"},
            {"name": "section", "class_": "abstract"},
        ]:
            el = soup.find(**selector)
            if el:
                return el.get_text(" ", strip=True)
        return None

    def _extract_sections(self, soup: BeautifulSoup) -> list[tuple[str, str]]:
        """Extract (section_title, section_text) tuples.

        Handles various Bookshelf section patterns:
        - <h2> with following <p> tags (most common)
        - <section> with <h2>/<h3> title
        - <div class="section"> with heading
        """
        sections: list[tuple[str, str]] = []
        headings = soup.find_all(["h2", "h3"])

        for heading in headings:
            sec_title = heading.get_text(strip=True)
            if not sec_title or len(sec_title) > 200:
                continue
            classes = heading.get("class", []) or []
            if any(c in classes for c in ["nav", "menu", "sidebar", "footer", "toc"]):
                continue

            paragraphs: list[str] = []
            for sibling in heading.find_next_siblings():
                if sibling.name in ["h2", "h3"] and sibling.name <= heading.name:
                    break
                if sibling.name == "p":
                    text = sibling.get_text(" ", strip=True)
                    if text and len(text) > 20:
                        paragraphs.append(text)
                elif sibling.name in ["ul", "ol"]:
                    for li in sibling.find_all("li", recursive=False):
                        text = li.get_text(" ", strip=True)
                        if text:
                            paragraphs.append(f"- {text}")

            sec_text = "\n\n".join(paragraphs)
            if sec_text and len(sec_text) > 80:
                sections.append((sec_title, sec_text))

        return sections

    def _split_long_section(
        self,
        text: str,
        target_chars: int = 1500,
        overlap: int = 200,
    ) -> list[str]:
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        if not paragraphs:
            return [text]
        if len(paragraphs) == 1 and len(paragraphs[0]) > target_chars:
            sentences = re.split(r'(?<=[.!?])\s+', paragraphs[0])
            if len(sentences) >= 2:
                paragraphs = sentences

        chunks: list[str] = []
        current = ""
        for para in paragraphs:
            if len(current) + len(para) + 2 > target_chars and current:
                chunks.append(current)
                if len(current) > overlap:
                    current = current[-overlap:] + "\n\n" + para
                else:
                    current = para
            else:
                current = (current + "\n\n" + para) if current else para
        if current:
            chunks.append(current)
        return chunks

    def _make_chunk(
        self,
        text: str,
        record: DocumentRecord,
        doc: ScrapedDocument,
        idx: int,
        section: str,
    ) -> ChunkRecord:
        return ChunkRecord(
            document_id=record.url or doc.url,
            source_type=self.source_type,
            title=record.title,
            chunk_text=text,
            chunk_index=idx,
            char_count=len(text),
            section=section,
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
            is_india_specific=False,
            evidence_level=record.evidence_level,
            parser_version="bookshelf-v1",
            token_estimate=len(text) // 4,
            trust_tier=doc.trust_tier,
            indian_source=False,
            also_indexed_in=[],
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
            journal=doc.metadata.get("collection_label", "NCBI Bookshelf"),
            is_india_specific=False,
            parser_version="bookshelf-v1",
            content_hash="",
            condition_tags=[],
            specialty_tags=[],
        )
