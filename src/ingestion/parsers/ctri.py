"""CTRI parser — converts scraped trial records into DocumentRecord + ChunkRecord.

Each trial becomes ONE document with ONE chunk containing the structured
trial fields formatted as readable text. Trial records are highly structured
and don't benefit from paragraph-based chunking.
"""
from __future__ import annotations

import hashlib
import re
from typing import Any

from bs4 import BeautifulSoup
from loguru import logger

from src.ingestion.document_db import ChunkRecord, DocumentRecord
from src.ingestion.scrapers.framework.models import ScrapedDocument


class CTRIParser:
    """Parses CTRI trial HTML into DocumentRecord + ChunkRecord."""

    def __init__(self) -> None:
        self.source_type = "ctri"

    def parse(self, doc: ScrapedDocument) -> tuple[DocumentRecord, list[ChunkRecord]]:
        """Parse one scraped CTRI trial record."""
        if not doc.content:
            logger.warning(f"[ctri:parser] empty content for {doc.url}")
            return self._empty_record(doc), []

        try:
            html = doc.content.decode("utf-8", errors="replace")
        except Exception:
            html = doc.content.decode("latin-1", errors="replace")

        structured = doc.metadata.get("structured", {})
        if not structured:
            from src.ingestion.scrapers.sources.ctri import CTRIScraper
            structured = CTRIScraper._extract_trial_fields(html)

        trial_id = structured.get("trial_id") or doc.metadata.get("trial_id") or "unknown"
        title = structured.get("title") or doc.title or f"CTRI Trial {trial_id}"
        text_block = self._build_text_block(trial_id, title, structured, doc.url)

        record = DocumentRecord(
            source_type=self.source_type,
            title=f"CTRI: {trial_id} - {title[:80]}",
            content=text_block,
            url=doc.url,
            doi=None,
            published_date=structured.get("registration_date"),
            year=self._extract_year(structured.get("registration_date")),
            journal="CTRI",
            is_india_specific=True,
            parser_version="ctri-v1",
            content_hash=self._content_hash(text_block),
            condition_tags=[structured.get("condition", "")] if structured.get("condition") else [],
            specialty_tags=["clinical_trial"],
        )

        chunk = ChunkRecord(
            document_id=record.url or doc.url,
            source_type=self.source_type,
            title=record.title,
            chunk_text=text_block,
            chunk_index=0,
            char_count=len(text_block),
            section="trial_record",
            diseases=[structured.get("condition", "")] if structured.get("condition") else [],
            drugs=[structured.get("intervention", "")] if structured.get("intervention") else [],
            symptoms=[],
            dosages=[],
            contraindications=[],
            patient_populations=[],
            outcomes=[],
            has_safety_flag=False,
            content_type="trial_record",
            content_weight=1.0,
            quality_score=1.0,
            is_india_specific=True,
            evidence_level=2,  # Trials = high evidence but may not have published results
            parser_version="ctri-v1",
            token_estimate=len(text_block) // 4,
            trust_tier=2,
            indian_source=True,
            also_indexed_in=[],
        )

        logger.info(
            f"[ctri:parser] parsed {doc.url}: trial_id='{trial_id}', "
            f"fields={list(structured.keys())}"
        )
        return record, [chunk]

    def _build_text_block(
        self,
        trial_id: str,
        title: str,
        structured: dict[str, str],
        url: str,
    ) -> str:
        """Build readable text block from structured trial fields."""
        lines = [f"Trial ID: {trial_id}", f"Title: {title}"]

        field_order = [
            ("sponsor", "Sponsor"),
            ("phase", "Phase"),
            ("status", "Recruitment Status"),
            ("condition", "Condition"),
            ("intervention", "Intervention"),
            ("enrollment", "Target Enrollment"),
            ("locations", "Study Sites"),
            ("registration_date", "Registration Date"),
            ("brief_summary", "Summary"),
        ]

        for key, label in field_order:
            value = structured.get(key)
            if value:
                lines.append(f"{label}: {value}")

        lines.append("")
        lines.append("Source: CTRI (Clinical Trials Registry - India)")
        lines.append(f"URL: {url}")

        return "\n".join(lines)

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
            title=doc.title or "Untitled CTRI Trial",
            content="",
            url=doc.url,
            doi=None,
            published_date=doc.pubdate,
            year=self._extract_year(doc.pubdate),
            journal="CTRI",
            is_india_specific=True,
            parser_version="ctri-v1",
            content_hash="",
            condition_tags=[],
            specialty_tags=["clinical_trial"],
        )
