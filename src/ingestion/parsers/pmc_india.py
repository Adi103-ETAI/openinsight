"""PMC India parser — converts PubMed Central XML into DocumentRecord + ChunkRecord.

PMC articles come as NLM Journal Publishing DTD XML (structured). Unlike
HTML scraping, this gives us reliable section boundaries:
- <article-meta> → title, authors, journal, DOI, PMID, PMC ID, pubdate
- <abstract> → abstract text
- <body> → full-text with <sec> sections (Introduction, Methods, Results, etc.)

Each <sec> in <body> becomes its own chunk with the section title preserved
in ChunkRecord.section. This is much better than paragraph-based chunking
because section context is critical for clinical accuracy (a Methods chunk
shouldn't be confused with a Results chunk).
"""
from __future__ import annotations

import hashlib
import re
from typing import Any

from bs4 import BeautifulSoup
from loguru import logger

from src.ingestion.document_db import ChunkRecord, DocumentRecord
from src.ingestion.scrapers.framework.models import ScrapedDocument


class PMCIndiaParser:
    """Parses PubMed Central XML into DocumentRecord + ChunkRecord."""

    def __init__(self) -> None:
        self.source_type = "pmc_india"

    def parse(self, doc: ScrapedDocument) -> tuple[DocumentRecord, list[ChunkRecord]]:
        """Parse one scraped PMC article."""
        if not doc.content:
            logger.warning(f"[pmc_india:parser] empty content for {doc.url}")
            return self._empty_record(doc), []

        try:
            xml_text = doc.content.decode("utf-8", errors="replace")
        except Exception:
            xml_text = doc.content.decode("latin-1", errors="replace")

        # Parse as XML (lxml-xml parser, not html)
        soup = BeautifulSoup(xml_text, "xml")

        # Extract article metadata from <article-meta>
        article_meta = soup.find("article-meta")
        title = self._extract_title(article_meta or soup)
        authors = self._extract_authors(article_meta or soup)
        journal = self._extract_journal(soup)
        doi = self._extract_id(article_meta, "doi")
        pmid = self._extract_id(article_meta, "pmid")
        pmc_id = doc.metadata.get("pmc_id_full") or self._extract_id(article_meta, "pmc")
        pubdate = self._extract_pubdate(article_meta)
        year = self._extract_year(pubdate)

        # Extract abstract
        abstract = self._extract_abstract(soup)

        # Extract body sections
        sections = self._extract_body_sections(soup)

        # Build full content for the DocumentRecord
        full_content_parts = []
        if abstract:
            full_content_parts.append(f"Abstract\n\n{abstract}")
        for sec_title, sec_text in sections:
            full_content_parts.append(f"{sec_title}\n\n{sec_text}")
        full_content = "\n\n".join(full_content_parts) or abstract or ""

        record = DocumentRecord(
            source_type=self.source_type,
            title=title or "Untitled",
            content=full_content,
            url=doc.url,
            doi=doi,
            published_date=pubdate,
            year=year,
            journal=journal,
            is_india_specific=True,  # query filtered for Indian affiliations
            parser_version="pmc_india-v1",
            content_hash=self._content_hash(full_content),
            condition_tags=[],
            specialty_tags=[],
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
            # If section is large, split into sub-chunks
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
            f"[pmc_india:parser] parsed {doc.url}: title='{(title or '')[:60]}', "
            f"chunks={len(chunks)}"
        )
        return record, chunks

    def _extract_title(self, article_meta: Any) -> str | None:
        """Extract article title from <article-title> element."""
        title_el = article_meta.find("article-title")
        if title_el:
            return title_el.get_text(" ", strip=True)
        return None

    def _extract_authors(self, article_meta: Any) -> list[str]:
        """Extract author names from <contrib> elements.

        Returns list of "Lastname, Firstname" strings.
        """
        authors: list[str] = []
        for contrib in article_meta.find_all("contrib", attrs={"contrib-type": "author"}):
            name = contrib.find("name")
            if name:
                surname = name.find("surname")
                given = name.find("given-names")
                if surname and given:
                    authors.append(f"{surname.get_text(strip=True)}, {given.get_text(strip=True)}")
                elif surname:
                    authors.append(surname.get_text(strip=True))
            else:
                # Try string-name fallback
                string_name = contrib.find("string-name")
                if string_name:
                    authors.append(string_name.get_text(", ", strip=True))
        return authors

    def _extract_journal(self, soup: Any) -> str | None:
        """Extract journal title from <journal-title> in <journal-meta>."""
        journal_meta = soup.find("journal-meta")
        if journal_meta:
            jt = journal_meta.find("journal-title")
            if jt:
                return jt.get_text(strip=True)
            # Abbreviated journal title
            jta = journal_meta.find("journal-id", attrs={"journal-id-type": "iso-abbrev"})
            if jta:
                return jta.get_text(strip=True)
        return None

    def _extract_id(self, article_meta: Any, id_type: str) -> str | None:
        """Extract an article identifier (doi, pmid, pmc) from <article-id>."""
        if not article_meta:
            return None
        id_el = article_meta.find("article-id", attrs={"pub-id-type": id_type})
        if id_el:
            return id_el.get_text(strip=True)
        return None

    def _extract_pubdate(self, article_meta: Any) -> str | None:
        """Extract publication date from <pub-date> element.

        Returns ISO 8601 date string (YYYY-MM-DD) if possible, else YYYY or None.
        """
        if not article_meta:
            return None
        # Try <pub-date pub-type="epub"> first, then any pub-date
        pub_date = article_meta.find("pub-date", attrs={"pub-type": "epub"}) or \
                   article_meta.find("pub-date", attrs={"date-type": "pub"}) or \
                   article_meta.find("pub-date")
        if not pub_date:
            return None
        year = pub_date.find("year")
        month = pub_date.find("month")
        day = pub_date.find("day")
        y = year.get_text(strip=True) if year else None
        m = month.get_text(strip=True).zfill(2) if month else None
        d = day.get_text(strip=True).zfill(2) if day else None
        if y and m and d:
            return f"{y}-{m}-{d}"
        if y and m:
            return f"{y}-{m}"
        if y:
            return y
        return None

    def _extract_abstract(self, soup: Any) -> str | None:
        """Extract abstract text from <abstract> element."""
        abstract = soup.find("abstract")
        if not abstract:
            return None
        # Strip nested <sec> titles for clean text
        return abstract.get_text(" ", strip=True)

    def _extract_body_sections(self, soup: Any) -> list[tuple[str, str]]:
        """Extract body sections as (section_title, section_text) tuples.

        PMC articles use <sec><title>...</title><p>...</p></sec> structure.
        Falls back to a single ("Body", full_text) if no sections found.
        """
        body = soup.find("body")
        if not body:
            return []

        sections: list[tuple[str, str]] = []
        # Find top-level <sec> elements in body
        top_secs = body.find_all("sec", recursive=False)
        if not top_secs:
            # No section structure — treat whole body as one section
            text = body.get_text("\n", strip=True)
            if text:
                sections.append(("Body", text))
            return sections

        for sec in top_secs:
            title_el = sec.find("title", recursive=False)
            sec_title = title_el.get_text(strip=True) if title_el else "Section"
            # Get all paragraph text in this section (excluding nested sec titles)
            paragraphs = []
            for p in sec.find_all("p"):
                # Skip paragraphs that are inside nested <sec> (we'll handle those separately)
                if p.find_parent("sec") != sec:
                    continue
                text = p.get_text(" ", strip=True)
                if text:
                    paragraphs.append(text)
            sec_text = "\n\n".join(paragraphs)
            if sec_text:
                sections.append((sec_title, sec_text))

        return sections

    def _split_long_section(
        self,
        text: str,
        target_chars: int = 1500,
        overlap: int = 200,
    ) -> list[str]:
        """Split a long section into overlapping sub-chunks.

        First tries paragraph boundaries (\n\n). If a section has only one
        long paragraph, falls back to sentence-boundary splitting to avoid
        creating one giant chunk.
        """
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        if not paragraphs:
            return [text]
        # If a single paragraph is longer than target_chars, split by sentences
        if len(paragraphs) == 1 and len(paragraphs[0]) > target_chars:
            import re
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
            section=section,  # PMC preserves section structure!
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
            parser_version="pmc_india-v1",
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
            parser_version="pmc_india-v1",
            content_hash="",
            condition_tags=[],
            specialty_tags=[],
        )
