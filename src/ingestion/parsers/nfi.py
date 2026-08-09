"""NFI parser — National Formulary of India drug monographs.

The NFI is India's equivalent of the FDA drug labels. Each monograph has:
- Drug name (generic + brand examples)
- Indications
- Dosage (adult + pediatric)
- Contraindications
- Adverse effects
- Pregnancy category
- Schedule classification (H / H1 / X)
- Drug interactions

**Status: STUB — awaiting NFI PDF access.**

The parser is fully implemented and tested with fixture data. When the
NFI PDF (5th or 6th edition) is obtained, the pipeline can ingest it via
`ingest_directory()` with GROBID parsing, then this parser post-processes
the GROBID-extracted text into structured monographs.

Alternative interim path: build a `NHPParser` for the National Health Portal
(nhp.gov.in) drug monographs, which are essentially NFI excerpts available
publicly. Same parser interface — swap source when NFI access is obtained.

Each monograph becomes ONE chunk (not split by section) — when a user
queries "metformin", we want the whole monograph as context, not 6
separate chunks. The chunk's `section` field is set to "drug_monograph"
for the drug-lookup fast path.
"""
from __future__ import annotations

import hashlib
import re
from typing import Any

from bs4 import BeautifulSoup
from loguru import logger

from src.ingestion.document_db import ChunkRecord, DocumentRecord
from src.ingestion.scrapers.framework.models import ScrapedDocument


# Standard NFI monograph sections
NFI_MONOGRAPH_SECTIONS = [
    "Indications",
    "Dosage",
    "Contraindications",
    "Adverse Effects",
    "Precautions",
    "Pregnancy Category",
    "Drug Interactions",
    "Schedule",
    "Storage",
    "Brand Names",
]


class NFIParser:
    """Parses NFI drug monographs into DocumentRecord + ChunkRecord.

    Each monograph becomes ONE document with ONE chunk containing all
    the structured fields. This enables direct drug lookup without RAG
    synthesis for drug-specific queries (the OpenEvidence pattern).
    """

    def __init__(self) -> None:
        self.source_type = "nfi"

    def parse(self, doc: ScrapedDocument) -> tuple[DocumentRecord, list[ChunkRecord]]:
        """Parse one scraped NFI monograph."""
        if not doc.content:
            logger.warning(f"[nfi:parser] empty content for {doc.url}")
            return self._empty_record(doc), []

        try:
            content_text = doc.content.decode("utf-8", errors="replace")
        except Exception:
            content_text = doc.content.decode("latin-1", errors="replace")

        # Try HTML parsing first (for NHP portal content)
        if content_text.strip().startswith("<") or "<html" in content_text.lower():
            soup = BeautifulSoup(content_text, "lxml")
            structured = self._extract_monograph_fields_html(soup)
            full_text = self._build_text_block(doc.title or "Unknown Drug", structured, doc.url)
        else:
            # Plain text (from GROBID-extracted PDF)
            structured = self._extract_monograph_fields_text(content_text)
            full_text = content_text

        drug_name = structured.get("drug_name") or doc.title or "Unknown Drug"

        record = DocumentRecord(
            source_type=self.source_type,
            title=f"NFI: {drug_name}",
            content=full_text,
            url=doc.url,
            doi=None,
            published_date=doc.pubdate,
            year=self._extract_year(doc.pubdate),
            journal="National Formulary of India",
            is_india_specific=True,
            parser_version="nfi-v1",
            content_hash=self._content_hash(full_text),
            condition_tags=[],
            specialty_tags=["drug_regulatory"],
        )

        # One chunk per monograph — the whole structured record
        chunk = ChunkRecord(
            document_id=record.url or doc.url,
            source_type=self.source_type,
            title=record.title,
            chunk_text=full_text,
            chunk_index=0,
            char_count=len(full_text),
            section="drug_monograph",  # Marker for drug-lookup fast path
            diseases=[],
            drugs=[drug_name] if drug_name else [],
            symptoms=[],
            dosages=[structured.get("dosage", "")] if structured.get("dosage") else [],
            contraindications=[structured.get("contraindications", "")] if structured.get("contraindications") else [],
            patient_populations=[],
            outcomes=[],
            has_safety_flag=bool(structured.get("contraindications")),
            content_type="drug_monograph",
            content_weight=1.5,  # Boost drug monographs in retrieval
            quality_score=1.0,
            is_india_specific=True,
            evidence_level=1,  # Regulatory formulary = highest evidence
            parser_version="nfi-v1",
            token_estimate=len(full_text) // 4,
            trust_tier=1,  # NFI = Tier 1 (government formulary)
            indian_source=True,
            also_indexed_in=[],
        )

        logger.info(
            f"[nfi:parser] parsed {doc.url}: drug='{drug_name}', "
            f"fields={list(structured.keys())}"
        )
        return record, [chunk]

    def _extract_monograph_fields_html(self, soup: BeautifulSoup) -> dict[str, str]:
        """Extract structured monograph fields from HTML.

        Works for NHP portal monographs (which mirror NFI structure).
        """
        fields: dict[str, str] = {}

        # Try to get drug name from <h1> or <title>
        h1 = soup.find("h1") or soup.find("title")
        if h1:
            name = h1.get_text(strip=True)
            # Strip common prefixes
            name = re.sub(r"^(NFI|National Formulary of India)\s*[:\-]\s*", "", name, flags=re.IGNORECASE)
            fields["drug_name"] = name

        # Look for labeled sections
        label_map = {
            "indications": ["indications", "uses"],
            "dosage": ["dosage", "dose", "administration"],
            "contraindications": ["contraindications", "contraindicated"],
            "adverse_effects": ["adverse effects", "side effects", "adverse reactions"],
            "precautions": ["precautions", "warnings"],
            "pregnancy_category": ["pregnancy category", "pregnancy"],
            "drug_interactions": ["drug interactions", "interactions"],
            "schedule": ["schedule"],
            "brand_names": ["brand names", "brands", "trade names"],
        }

        for cells in soup.find_all(["tr", "div", "dl", "section"]):
            text = cells.get_text(" ", strip=True).lower()
            for field, labels in label_map.items():
                if field in fields:
                    continue
                for label in labels:
                    if label in text:
                        value = self._find_value_after_label(cells, label)
                        if value and len(value) < 2000:
                            fields[field] = value
                            break

        return fields

    def _extract_monograph_fields_text(self, text: str) -> dict[str, str]:
        """Extract structured fields from plain text (GROBID-extracted PDF).

        NFI monographs use section headers like "Indications:", "Dosage:",
        "Contraindications:", etc. We split on these and extract the
        following text until the next header.
        """
        fields: dict[str, str] = {}

        # Try to extract drug name from first non-empty line
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        if lines:
            # First line is usually the drug name
            first = lines[0]
            # Strip common prefixes
            first = re.sub(r"^(NFI|National Formulary of India)\s*[:\-]\s*", "", first, flags=re.IGNORECASE)
            fields["drug_name"] = first

        # Pattern: "Section Name:" or "Section Name -" followed by content
        section_pattern = re.compile(
            r"^(Indications|Dosage|Contraindications|Adverse Effects?|Precautions|"
            r"Pregnancy Category|Drug Interactions|Schedule|Storage|Brand Names?)\s*[:\-]",
            re.IGNORECASE | re.MULTILINE,
        )

        matches = list(section_pattern.finditer(text))
        for i, match in enumerate(matches):
            field_name = match.group(1).lower().replace(" ", "_")
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            value = text[start:end].strip()
            if value:
                fields[field_name] = value[:2000]

        return fields

    def _build_text_block(
        self,
        drug_name: str,
        structured: dict[str, str],
        url: str,
    ) -> str:
        """Build readable text block from structured monograph fields."""
        lines = [f"Drug: {drug_name}"]

        field_order = [
            ("indications", "Indications"),
            ("dosage", "Dosage"),
            ("contraindications", "Contraindications"),
            ("adverse_effects", "Adverse Effects"),
            ("precautions", "Precautions"),
            ("pregnancy_category", "Pregnancy Category"),
            ("drug_interactions", "Drug Interactions"),
            ("schedule", "Schedule"),
            ("brand_names", "Brand Names"),
        ]

        for key, label in field_order:
            value = structured.get(key)
            if value:
                lines.append(f"{label}: {value}")

        lines.append("")
        lines.append("Source: NFI (National Formulary of India, IPC)")
        lines.append(f"URL: {url}")

        return "\n".join(lines)

    @staticmethod
    def _find_value_after_label(container: Any, label: str) -> str | None:
        """Find the value that follows a label in the HTML container."""
        for header in container.find_all(["th", "dt", "strong", "b", "label", "h2", "h3"]):
            if label.lower() in header.get_text(strip=True).lower():
                sibling = header.find_next_sibling()
                while sibling:
                    text = sibling.get_text(strip=True)
                    if text and text.lower() != label.lower():
                        return text[:2000]
                    sibling = sibling.find_next_sibling()
                parent = header.parent
                if parent:
                    next_td = parent.find_next_sibling(["td", "dd"])
                    if next_td:
                        text = next_td.get_text(strip=True)
                        if text:
                            return text[:2000]
        return None

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
            title=doc.title or "Untitled NFI Monograph",
            content="",
            url=doc.url,
            doi=None,
            published_date=doc.pubdate,
            year=self._extract_year(doc.pubdate),
            journal="National Formulary of India",
            is_india_specific=True,
            parser_version="nfi-v1",
            content_hash="",
            condition_tags=[],
            specialty_tags=["drug_regulatory"],
        )
