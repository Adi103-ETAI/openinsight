"""
MedQuAD XML Parser

MedQuAD (Medical Question Answering Dataset) is a curated collection of
11,274 medical QA pairs sourced from 12 NIH websites (CancerGov, CDC,
NIDDK, etc.).  Each XML file contains a <Document> with one or more
<QAPair> entries under <QAPairs>.

Two XML layout variants exist:
  1. Full format  (CancerGov, NIDDK, …) — <FocusAnnotations><UMLS>…
  2. Simple format (CDC, …)             — flat <UMLS> without <FocusAnnotations>

This parser reads a single XML file (or a directory of XML files) and
yields one DocumentRecord per <QAPair>.

Usage:
    parser = MedQuADParser(path="/data/MedQuAD/1_CancerGov_QA")
    documents = parser.parse()
"""

from pathlib import Path
from typing import Optional
import xml.etree.ElementTree as ET

from loguru import logger

from src.ingestion.document_db import DocumentRecord
from src.ingestion.parsers.base import BaseParser

# Maximum characters kept from the Answer for an abstract-style preview.
_ABSTRACT_MAX_CHARS = 500


class MedQuADParser(BaseParser):
    """Parse MedQuAD XML files into DocumentRecord objects."""

    def __init__(self, path: str | Path):
        """
        Args:
            path: Path to a single MedQuAD XML file *or* a directory
                  containing many XML files.
        """
        self._path = Path(path)

    # -- BaseParser interface ------------------------------------------------

    @property
    def source_type(self) -> str:
        return "medquad"

    def parse(self) -> list[DocumentRecord]:
        xml_files = self._collect_xml_files()
        if not xml_files:
            logger.warning(f"[MedQuAD] No XML files found at {self._path}")
            return []

        documents: list[DocumentRecord] = []
        for xml_file in xml_files:
            try:
                docs = self._parse_single_file(xml_file)
                documents.extend(docs)
            except ET.ParseError as exc:
                logger.warning(f"[MedQuAD] XML parse error in {xml_file}: {exc}")
            except (RuntimeError, ValueError, TypeError, OSError) as exc:
                logger.warning(f"[MedQuAD] Error processing {xml_file}: {exc}")

        logger.info(
            f"[MedQuAD] Parsed {len(xml_files)} files → {len(documents)} documents"
        )
        return documents

    # -- Internal helpers ----------------------------------------------------

    def _collect_xml_files(self) -> list[Path]:
        """Return a sorted list of XML file paths from *self._path*."""
        if not self._path.exists():
            logger.error(f"[MedQuAD] Path does not exist: {self._path}")
            return []

        if self._path.is_file():
            return [self._path] if self._path.suffix.lower() == ".xml" else []

        # Directory — gather all .xml files recursively
        return sorted(self._path.rglob("*.xml"))

    def _parse_single_file(self, xml_file: Path) -> list[DocumentRecord]:
        """Parse one XML file and return a list of DocumentRecords."""
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # -- Document-level attributes ---------------------------------------
        doc_id_attr = root.attrib.get("id", "")
        source_attr = root.attrib.get("source", "")
        url_attr = root.attrib.get("url", "")

        # Focus text (may be missing)
        focus = self._text(root, "Focus") or ""

        # -- Build keyword list from Focus + qtype later per QAPair ----------

        # -- Process each QAPair --------------------------------------------
        qa_pairs = root.findall(".//QAPair")
        if not qa_pairs:
            logger.debug(f"[MedQuAD] No QAPairs in {xml_file.name}")
            return []

        records: list[DocumentRecord] = []
        for qa in qa_pairs:
            record = self._build_record(
                qa=qa,
                doc_id_attr=doc_id_attr,
                source_attr=source_attr,
                url_attr=url_attr,
                focus=focus,
                xml_file=xml_file,
            )
            if record is not None:
                records.append(record)

        return records

    def _build_record(
        self,
        qa: ET.Element,
        doc_id_attr: str,
        source_attr: str,
        url_attr: str,
        focus: str,
        xml_file: Path,
    ) -> Optional[DocumentRecord]:
        """Convert a single <QAPair> element into a DocumentRecord."""

        question_el = qa.find("Question")
        answer_el = qa.find("Answer")

        if question_el is None or answer_el is None:
            return None

        question_text = (question_el.text or "").strip()
        answer_text = (answer_el.text or "").strip()

        if not question_text or not answer_text:
            return None

        # qid attribute on <Question>
        qid = question_el.attrib.get("qid", "")
        # qtype attribute on <Question>
        qtype = question_el.attrib.get("qtype", "")

        # Derive a stable doc_id: "medquad_{qid}"
        # e.g. qid="0000001_1-1" → doc_id="medquad_0000001_1-1"
        stable_id = f"medquad_{qid}" if qid else None

        # Content = full Q&A
        content = f"Q: {question_text}\n\nA: {answer_text}"

        # Abstract = truncated answer (for preview / indexing hint)
        abstract_preview = answer_text[:_ABSTRACT_MAX_CHARS]

        # Keywords: [focus, qtype] — filter empty strings
        keywords: list[str] = [k for k in (focus, qtype) if k]

        # Build the DocumentRecord, mapping to available schema fields.
        # doc_id / stable_id is noted via doi for traceability since
        # DocumentRecord uses MongoDB _id as the primary key.
        return DocumentRecord(
            source_type=self.source_type,
            title=question_text,
            content=content,
            url=url_attr or None,
            doi=stable_id,              # store stable medquad id for traceability
            published_date=None,
            condition_tags=keywords,     # focus + qtype as condition tags
            specialty_tags=[],
            year=None,
            journal=source_attr or None,  # e.g. "CancerGov", "CDC"
            study_type=None,
            population=None,
            evidence_level=4,            # curated QA, moderate evidence
            is_india_specific=False,
            parser_version="v1",
        )

    # -- Utility -------------------------------------------------------------

    @staticmethod
    def _text(element: ET.Element, tag: str) -> str:
        """Safely extract stripped text from the first child matching *tag*."""
        child = element.find(tag)
        if child is not None and child.text:
            return child.text.strip()
        return ""
