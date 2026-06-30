"""CDSCO parser — converts scraped drug records into DocumentRecord + ChunkRecord.

CDSCO drug records are highly structured — each drug has discrete fields
(drug name, manufacturer, approval date, indication, strength, schedule).
Rather than chunking the HTML body text, we create ONE chunk per drug
containing all the structured fields as a readable text block.

This is the OpenEvidence pattern for drug labels — structured lookup, not
RAG-via-embeddings. When the system gets a drug_info query, the
QueryUnderstanding should detect it and the retrieval layer should
prefer these structured drug chunks over generic PubMed articles.
"""
from __future__ import annotations

import hashlib
import re
from datetime import datetime
from typing import Any

from bs4 import BeautifulSoup
from loguru import logger

from src.ingestion.document_db import ChunkRecord, DocumentRecord
from src.ingestion.scrapers.framework.models import ScrapedDocument


class CDSCOParser:
    """Parses CDSCO drug record HTML into DocumentRecord + ChunkRecord.

    Each drug becomes ONE document with ONE chunk containing the structured
    fields formatted as readable text. This enables direct drug lookup
    without RAG synthesis for drug-specific queries.
    """

    def __init__(self) -> None:
        self.source_type = "cdsco"

    def parse(self, doc: ScrapedDocument) -> tuple[DocumentRecord, list[ChunkRecord]]:
        """Parse one scraped CDSCO drug record."""
        if not doc.content:
            logger.warning(f"[cdsco:parser] empty content for {doc.url}")
            return self._empty_record(doc), []

        try:
            html = doc.content.decode("utf-8", errors="replace")
        except Exception:
            html = doc.content.decode("latin-1", errors="replace")

        # Get structured fields (already extracted by scraper.process())
        structured = doc.metadata.get("structured", {})

        # If structured fields are missing, try to extract them now
        if not structured:
            from src.ingestion.scrapers.sources.cdsco import CDSCOScraper
            structured = CDSCOScraper._extract_drug_fields(html)

        # Build a readable text representation of the drug record
        drug_name = structured.get("drug_name") or doc.title or "Unknown Drug"
        text_block = self._build_text_block(drug_name, structured, doc.url)

        record = DocumentRecord(
            source_type=self.source_type,
            title=f"CDSCO: {drug_name}",
            content=text_block,
            url=doc.url,
            doi=None,
            published_date=structured.get("approval_date"),
            year=self._extract_year(structured.get("approval_date")),
            journal="CDSCO Approved Drugs",
            is_india_specific=True,
            parser_version="cdsco-v1",
            content_hash=self._content_hash(text_block),
            condition_tags=[],
            specialty_tags=["drug_regulatory"],
        )

        # One chunk per drug — the whole structured record
        chunk = ChunkRecord(
            document_id=record.url or doc.url,
            source_type=self.source_type,
            title=record.title,
            chunk_text=text_block,
            chunk_index=0,
            char_count=len(text_block),
            section="drug_record",  # Marker for drug-lookup fast path
            diseases=[],
            drugs=[drug_name] if drug_name else [],
            symptoms=[],
            dosages=[],  # Could be populated from "strength" field
            contraindications=[],
            patient_populations=[],
            outcomes=[],
            has_safety_flag=False,
            content_type="drug_record",
            content_weight=1.5,  # Boost drug records in retrieval
            quality_score=1.0,
            is_india_specific=True,
            evidence_level=1,  # Regulatory = highest evidence
            parser_version="cdsco-v1",
            token_estimate=len(text_block) // 4,
            trust_tier=1,  # CDSCO = Tier 1
            indian_source=True,
            also_indexed_in=[],
        )
        # Store structured fields in content text (already done above via _build_text_block)
        # The structured data is embedded in the chunk_text for RAG retrieval.
        # For direct lookup, the /search endpoint can filter by source_type="cdsco"
        # and section="drug_record" to find drug records quickly.

        logger.info(
            f"[cdsco:parser] parsed {doc.url}: drug='{drug_name}', "
            f"fields={list(structured.keys())}"
        )
        return record, [chunk]

    def _build_text_block(
        self,
        drug_name: str,
        structured: dict[str, str],
        url: str,
    ) -> str:
        """Build a readable text block from structured drug fields.

        Format:
            Drug: <name>
            Brand Name: <brand>
            Manufacturer: <manufacturer>
            Approval Date: <date>
            Indication: <indication>
            Strength: <strength>
            Schedule: <schedule>

            Source: CDSCO (Central Drugs Standard Control Organization, India)
            URL: <url>
        """
        lines = [f"Drug: {drug_name}"]

        field_order = [
            ("brand_name", "Brand Name"),
            ("manufacturer", "Manufacturer"),
            ("approval_date", "Approval Date"),
            ("indication", "Indication"),
            ("strength", "Strength / Dosage Form"),
            ("schedule", "Schedule"),
            ("batch_no", "Batch Number"),
        ]

        for key, label in field_order:
            value = structured.get(key)
            if value:
                lines.append(f"{label}: {value}")

        lines.append("")
        lines.append("Source: CDSCO (Central Drugs Standard Control Organization, India)")
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
            title=doc.title or "Untitled CDSCO Record",
            content="",
            url=doc.url,
            doi=None,
            published_date=doc.pubdate,
            year=self._extract_year(doc.pubdate),
            journal="CDSCO Approved Drugs",
            is_india_specific=True,
            parser_version="cdsco-v1",
            content_hash="",
            condition_tags=[],
            specialty_tags=["drug_regulatory"],
        )
