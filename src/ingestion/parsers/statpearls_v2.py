"""StatPearls parser — converts NCBI Bookshelf HTML into DocumentRecord + ChunkRecord.

StatPearls articles on NCBI Bookshelf use a consistent HTML structure:
- <h1 class="book-title"> or <title> → article title
- <meta name="citation_*"> → Highwire Press metadata
- <div class="section"> or <section> → each clinical section
- Section headings: Introduction, Etiology, Epidemiology, History and Physical,
  Evaluation, Treatment/Management, Differential Diagnosis, Prognosis,
  Complications, Deterrence, Pearls

Each section becomes its own chunk with the section title preserved in
ChunkRecord.section. This is critical for clinical accuracy — a "Treatment"
chunk should rank differently than a "Differential Diagnosis" chunk for a
therapeutic query.
"""
from __future__ import annotations

import hashlib
import re
from typing import Any

from bs4 import BeautifulSoup
from loguru import logger

from src.ingestion.document_db import ChunkRecord, DocumentRecord
from src.ingestion.scrapers.framework.models import ScrapedDocument


# Standard StatPearls sections in display order
STATPEARLS_SECTIONS = [
    "Introduction",
    "Definition",
    "Etiology",
    "Epidemiology",
    "Pathophysiology",
    "History and Physical",
    "Evaluation",
    "Treatment",
    "Management",
    "Differential Diagnosis",
    "Prognosis",
    "Complications",
    "Deterrence",
    "Patient Education",
    "Pearls",
    "Other Issues",
    "Enhancing Healthcare Team Outcomes",
]


class StatPearlsParser:
    """Parses StatPearls HTML into DocumentRecord + ChunkRecord."""

    def __init__(self) -> None:
        self.source_type = "statpearls"

    def parse(self, doc: ScrapedDocument) -> tuple[DocumentRecord, list[ChunkRecord]]:
        """Parse one scraped StatPearls article."""
        if not doc.content:
            logger.warning(f"[statpearls:parser] empty content for {doc.url}")
            return self._empty_record(doc), []

        try:
            html = doc.content.decode("utf-8", errors="replace")
        except Exception:
            html = doc.content.decode("latin-1", errors="replace")

        soup = BeautifulSoup(html, "lxml")

        # Extract abstract (StatPearls articles often have a "Summary" or
        # the Introduction serves as abstract)
        abstract = doc.abstract or self._extract_abstract(soup)

        # Extract sections — StatPearls uses <h2> or <h3> for section titles
        sections = self._extract_sections(soup)

        # Build full content for DocumentRecord
        full_content_parts = []
        if abstract:
            full_content_parts.append(f"Abstract\n\n{abstract}")
        for sec_title, sec_text in sections:
            full_content_parts.append(f"{sec_title}\n\n{sec_text}")
        full_content = "\n\n".join(full_content_parts) or abstract or ""

        record = DocumentRecord(
            source_type=self.source_type,
            title=doc.title or self._extract_title(soup) or "Untitled",
            content=full_content,
            url=doc.url,
            doi=doc.doi,
            published_date=doc.pubdate,
            year=self._extract_year(doc.pubdate),
            journal="StatPearls",
            is_india_specific=False,  # StatPearls is international
            parser_version="statpearls-v1",
            content_hash=self._content_hash(full_content),
            condition_tags=[],
            specialty_tags=["statpearls"],
        )

        # Chunk by section
        chunks: list[ChunkRecord] = []
        chunk_idx = 0

        # Abstract as first chunk
        if abstract and len(abstract) > 80:
            chunks.append(self._make_chunk(
                abstract, record, doc, chunk_idx, "abstract"
            ))
            chunk_idx += 1

        # Each body section as its own chunk(s)
        for sec_title, sec_text in sections:
            if not sec_text or len(sec_text) < 80:
                continue
            # Split long sections into sub-chunks
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
            f"[statpearls:parser] parsed {doc.url}: title='{(record.title or '')[:60]}', "
            f"chunks={len(chunks)}"
        )
        return record, chunks

    def _extract_title(self, soup: BeautifulSoup) -> str | None:
        """Extract article title from StatPearls HTML."""
        # Try Highwire Press meta first
        meta = soup.find("meta", attrs={"name": "citation_title"})
        if meta and meta.get("content"):
            return meta["content"].strip()
        # Try <h1> with book-title class
        h1 = soup.find("h1", class_="book-title") or soup.find("h1")
        if h1:
            return h1.get_text(strip=True)
        # Fallback to <title>
        if soup.title:
            return soup.title.get_text(strip=True).split(" - ")[0]
        return None

    def _extract_abstract(self, soup: BeautifulSoup) -> str | None:
        """Extract abstract from StatPearls HTML."""
        # StatPearls abstract is often in <div class="abstract"> or the
        # first paragraph of the Introduction section
        for selector in [
            {"name": "div", "class_": "abstract"},
            {"name": "section", "class_": "abstract"},
            {"name": "p", "class_": "abstract"},
        ]:
            el = soup.find(**selector)
            if el:
                return el.get_text(" ", strip=True)
        # Try meta tag
        meta = soup.find("meta", attrs={"name": "citation_abstract"})
        if meta and meta.get("content"):
            return BeautifulSoup(meta["content"], "lxml").get_text(" ", strip=True)
        return None

    def _extract_sections(self, soup: BeautifulSoup) -> list[tuple[str, str]]:
        """Extract (section_title, section_text) tuples from StatPearls HTML.

        StatPearls uses <h2> for major section titles, with content in
        the following <p> tags until the next <h2>.
        """
        sections: list[tuple[str, str]] = []

        # Find all <h2> headings (StatPearls section markers)
        headings = soup.find_all(["h2", "h3"])
        for heading in headings:
            sec_title = heading.get_text(strip=True)
            if not sec_title or len(sec_title) > 200:
                continue
            # Skip non-content headings (nav, footer, etc.)
            classes = heading.get("class", []) or []
            if any(c in classes for c in ["nav", "menu", "sidebar", "footer"]):
                continue

            # Collect all <p> tags until the next heading of same or higher level
            paragraphs: list[str] = []
            for sibling in heading.find_next_siblings():
                if sibling.name in ["h2", "h3"] and sibling.name <= heading.name:
                    break
                if sibling.name == "p":
                    text = sibling.get_text(" ", strip=True)
                    if text and len(text) > 20:
                        paragraphs.append(text)
                elif sibling.name in ["ul", "ol"]:
                    # Include list items as text
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
        """Split a long section into overlapping sub-chunks."""
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        if not paragraphs:
            return [text]
        # If single paragraph, split by sentences
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
            section=section,  # StatPearls preserves section structure!
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
            parser_version="statpearls-v1",
            token_estimate=len(text) // 4,
            trust_tier=2,  # StatPearls = peer-reviewed, Tier 2
            indian_source=False,  # NCBI = US
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
            journal="StatPearls",
            is_india_specific=False,
            parser_version="statpearls-v1",
            content_hash="",
            condition_tags=[],
            specialty_tags=["statpearls"],
        )
