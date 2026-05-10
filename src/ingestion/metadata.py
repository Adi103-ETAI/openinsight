from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional


class DocType(str, Enum):
    RCT = "rct"
    SYSTEMATIC_REVIEW = "systematic_review"
    META_ANALYSIS = "meta_analysis"
    GUIDELINE = "guideline"
    REVIEW = "review"
    CASE_REPORT = "case_report"
    EDITORIAL = "editorial"
    COHORT = "cohort"
    UNKNOWN = "unknown"


class EvidenceLevel(str, Enum):
    LEVEL_1A = "1a"
    LEVEL_1B = "1b"
    LEVEL_2A = "2a"
    LEVEL_2B = "2b"
    LEVEL_3 = "3"
    LEVEL_4 = "4"
    LEVEL_5 = "5"
    UNKNOWN = "unknown"


DOC_TYPE_TO_EVIDENCE_LEVEL: dict[DocType, EvidenceLevel] = {
    DocType.META_ANALYSIS: EvidenceLevel.LEVEL_1A,
    DocType.SYSTEMATIC_REVIEW: EvidenceLevel.LEVEL_1A,
    DocType.RCT: EvidenceLevel.LEVEL_1B,
    DocType.COHORT: EvidenceLevel.LEVEL_2B,
    DocType.GUIDELINE: EvidenceLevel.LEVEL_5,
    DocType.REVIEW: EvidenceLevel.LEVEL_5,
    DocType.CASE_REPORT: EvidenceLevel.LEVEL_4,
    DocType.EDITORIAL: EvidenceLevel.LEVEL_5,
    DocType.UNKNOWN: EvidenceLevel.UNKNOWN,
}


EVIDENCE_BOOST_SCORE: dict[EvidenceLevel, float] = {
    EvidenceLevel.LEVEL_1A: 1.35,
    EvidenceLevel.LEVEL_1B: 1.25,
    EvidenceLevel.LEVEL_2A: 1.15,
    EvidenceLevel.LEVEL_2B: 1.10,
    EvidenceLevel.LEVEL_3: 1.05,
    EvidenceLevel.LEVEL_4: 1.00,
    EvidenceLevel.LEVEL_5: 1.10,
    EvidenceLevel.UNKNOWN: 1.00,
}


@dataclass
class ChunkMetadataV2:
    doc_id: str
    chunk_id: str
    chunk_type: str

    source: str
    doc_type: DocType
    evidence_level: EvidenceLevel

    title: str
    year: int
    journal: str
    authors: list[str]
    doi: Optional[str]
    pmid: Optional[str]

    specialty: list[str]
    mesh_terms: list[str]
    keywords: list[str]

    section_title: str
    chunk_index: int
    total_chunks: int

    india_relevant: bool = False
    has_indian_data: bool = False
    indian_source: bool = False

    has_table: bool = False
    has_drug_dosing: bool = False
    has_lab_values: bool = False

    evidence_boost: float = 1.0

    def to_vector_payload(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "chunk_id": self.chunk_id,
            "chunk_type": self.chunk_type,
            "source": self.source,
            "doc_type": self.doc_type.value,
            "evidence_level": self.evidence_level.value,
            "evidence_boost": self.evidence_boost,
            "title": self.title,
            "year": self.year,
            "journal": self.journal,
            "authors": self.authors,
            "doi": self.doi,
            "pmid": self.pmid,
            "specialty": self.specialty,
            "mesh_terms": self.mesh_terms,
            "keywords": self.keywords,
            "section_title": self.section_title,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "india_relevant": self.india_relevant,
            "has_indian_data": self.has_indian_data,
            "indian_source": self.indian_source,
            "has_table": self.has_table,
            "has_drug_dosing": self.has_drug_dosing,
            "has_lab_values": self.has_lab_values,
        }


class MetadataEnricherV2:
    INDIA_KEYWORDS = [
        "india",
        "indian",
        "icmr",
        "nmc",
        "aiims",
        "pgimer",
        "jipmer",
        "rssdi",
        "cardiological society of india",
        "japi",
        "nmj",
        "ijmr",
    ]

    SPECIALTY_KEYWORDS = {
        "cardiology": [
            "cardiac",
            "heart",
            "coronary",
            "myocardial",
            "arrhythmia",
            "hypertension",
        ],
        "endocrinology": [
            "diabetes",
            "insulin",
            "thyroid",
            "hormone",
            "hba1c",
            "glycemic",
        ],
        "pulmonology": [
            "asthma",
            "copd",
            "lung",
            "respiratory",
            "pneumonia",
            "tuberculosis",
        ],
        "neurology": [
            "stroke",
            "seizure",
            "epilepsy",
            "neuro",
            "dementia",
            "parkinson",
        ],
        "infectious_disease": [
            "infection",
            "antibiotic",
            "viral",
            "bacterial",
            "dengue",
            "malaria",
        ],
        "gastroenterology": [
            "liver",
            "hepatitis",
            "gastric",
            "pancreas",
            "ibd",
            "colitis",
        ],
        "oncology": [
            "cancer",
            "tumor",
            "tumour",
            "carcinoma",
            "neoplasm",
            "chemotherapy",
        ],
        "nephrology": ["kidney", "renal", "creatinine", "dialysis", "gfr", "nephritis"],
    }

    INDIAN_JOURNALS = [
        "japi",
        "journal of association of physicians of india",
        "ijmr",
        "indian heart journal",
        "journal of indian medical association",
    ]

    RCT_TITLE_PATTERNS = [
        "randomized",
        "randomised",
        "randomization",
        "randomisation",
        "controlled trial",
        "rct",
    ]

    SYSTEMATIC_REVIEW_PATTERNS = [
        "systematic review",
        "meta-analysis",
        "meta analysis",
        "cochrane",
    ]

    CASE_REPORT_PATTERNS = [
        "case report",
        "case series",
    ]

    COHORT_PATTERNS = [
        "cohort",
        "prospective",
        "retrospective",
        "longitudinal",
        "follow-up",
    ]

    EDITORIAL_PATTERNS = [
        "editorial",
        "commentary",
        "perspective",
        "opinion",
    ]

    GUIDELINE_PATTERNS = [
        "guideline",
        "recommendation",
        "consensus",
        "protocol",
        "standard of care",
    ]

    def enrich_document(self, doc: Any, source: str) -> dict[str, Any]:
        title = self._coerce_str(self._get_field(doc, "title", ""))
        abstract = self._coerce_str(self._get_field(doc, "abstract", ""))
        content = self._coerce_str(self._get_field(doc, "content", ""))
        journal = self._coerce_str(self._get_field(doc, "journal", ""))
        safe_source = self._coerce_str(source)
        full_text = "\n".join([title, abstract, content]).strip()
        text_lower = full_text.lower()

        doc_type = self._infer_doc_type(title.lower(), text_lower)
        evidence_level = DOC_TYPE_TO_EVIDENCE_LEVEL.get(doc_type, EvidenceLevel.UNKNOWN)
        evidence_boost = EVIDENCE_BOOST_SCORE.get(evidence_level, 1.0)

        specialty = self._detect_specialties(text_lower)
        india_relevant, has_indian_data, indian_source = self._detect_india_relevance(
            text_lower=text_lower,
            source=safe_source,
            journal=journal,
        )

        has_drug_dosing = self._detect_drug_dosing(text_lower)
        has_lab_values = self._detect_lab_values(text_lower)

        year_raw = self._get_field(doc, "year", None)
        year = (
            int(year_raw)
            if isinstance(year_raw, int)
            else self._extract_year(text_lower)
        )

        mesh_terms = self._coerce_str_list(self._get_field(doc, "mesh_terms", []))
        keywords = self._coerce_str_list(self._get_field(doc, "keywords", []))
        authors = self._coerce_str_list(self._get_field(doc, "authors", []))

        return {
            "source": safe_source,
            "doc_type": doc_type,
            "evidence_level": evidence_level,
            "evidence_boost": evidence_boost,
            "specialty": specialty,
            "india_relevant": india_relevant,
            "has_indian_data": has_indian_data,
            "indian_source": indian_source,
            "has_drug_dosing": has_drug_dosing,
            "has_lab_values": has_lab_values,
            "year": year,
            "journal": journal,
            "title": title,
            "authors": authors,
            "doi": self._get_field(doc, "doi", None),
            "pmid": self._get_field(doc, "pmid", None),
            "mesh_terms": mesh_terms,
            "keywords": keywords,
        }

    def build_chunk_metadata(
        self,
        *,
        doc_id: str,
        chunk_id: str,
        chunk_type: str,
        section_title: str,
        chunk_index: int,
        total_chunks: int,
        source: str,
        doc_metadata: dict[str, Any],
        has_table: bool,
    ) -> ChunkMetadataV2:
        return ChunkMetadataV2(
            doc_id=doc_id,
            chunk_id=chunk_id,
            chunk_type=chunk_type,
            source=source,
            doc_type=doc_metadata["doc_type"],
            evidence_level=doc_metadata["evidence_level"],
            evidence_boost=doc_metadata["evidence_boost"],
            title=doc_metadata.get("title", ""),
            year=doc_metadata.get("year", 0),
            journal=doc_metadata.get("journal", ""),
            authors=doc_metadata.get("authors", []),
            doi=doc_metadata.get("doi"),
            pmid=doc_metadata.get("pmid"),
            specialty=doc_metadata.get("specialty", []),
            mesh_terms=doc_metadata.get("mesh_terms", []),
            keywords=doc_metadata.get("keywords", []),
            section_title=section_title,
            chunk_index=chunk_index,
            total_chunks=total_chunks,
            india_relevant=doc_metadata.get("india_relevant", False),
            has_indian_data=doc_metadata.get("has_indian_data", False),
            indian_source=doc_metadata.get("indian_source", False),
            has_table=has_table,
            has_drug_dosing=doc_metadata.get("has_drug_dosing", False),
            has_lab_values=doc_metadata.get("has_lab_values", False),
        )

    def _infer_doc_type(self, title_lower: str, text_lower: str) -> DocType:
        if any(p in text_lower for p in self.SYSTEMATIC_REVIEW_PATTERNS):
            if "meta" in text_lower:
                return DocType.META_ANALYSIS
            return DocType.SYSTEMATIC_REVIEW

        if any(p in text_lower for p in self.GUIDELINE_PATTERNS):
            return DocType.GUIDELINE

        if any(p in text_lower for p in self.RCT_TITLE_PATTERNS):
            return DocType.RCT

        if any(p in text_lower for p in self.COHORT_PATTERNS):
            return DocType.COHORT

        if any(p in text_lower for p in self.CASE_REPORT_PATTERNS):
            return DocType.CASE_REPORT

        if any(p in text_lower for p in self.EDITORIAL_PATTERNS):
            return DocType.EDITORIAL

        if "review" in title_lower or "review" in text_lower:
            return DocType.REVIEW

        return DocType.UNKNOWN

    def _detect_specialties(self, text_lower: str) -> list[str]:
        found: list[str] = []
        for specialty, keywords in self.SPECIALTY_KEYWORDS.items():
            if any(kw in text_lower for kw in keywords):
                found.append(specialty)
        return found[:5]

    def _detect_india_relevance(
        self, *, text_lower: str, source: str, journal: str
    ) -> tuple[bool, bool, bool]:
        source_lower = source.lower()
        journal_lower = journal.lower()

        india_relevant = any(kw in text_lower for kw in self.INDIA_KEYWORDS)
        has_indian_data = (
            "indian patients" in text_lower
            or "patients in india" in text_lower
            or "india" in text_lower
            or "new delhi" in text_lower
            or "mumbai" in text_lower
        )
        indian_source = source_lower in {
            "icmr",
            "nmc_guideline",
            "rssdi",
            "who_india",
        } or any(j in journal_lower for j in self.INDIAN_JOURNALS)

        return india_relevant, has_indian_data, indian_source

    def _detect_drug_dosing(self, text_lower: str) -> bool:
        dosing_patterns = [
            "mg/kg",
            "mg/day",
            "mg twice",
            "mg once",
            "units/kg",
            "tablet",
            "capsule",
            "dose",
            "dosage",
            "titrat",
            "po bid",
            "i.v.",
            "p.o.",
        ]
        return any(p in text_lower for p in dosing_patterns)

    def _detect_lab_values(self, text_lower: str) -> bool:
        lab_patterns = [
            "reference range",
            "normal range",
            "hba1c",
            "serum creatinine",
            "mg/dl",
            "mmol/l",
            "hemoglobin",
            "platelet",
            "wbc",
            "ldl",
            "hdl",
            "egfr",
        ]
        return any(p in text_lower for p in lab_patterns)

    def _extract_year(self, text_lower: str) -> int:
        matches = re.findall(r"\b(19\d{2}|20\d{2})\b", text_lower)
        if not matches:
            return 0
        years = [int(m) for m in matches]
        return max(years)

    def _get_field(self, obj: Any, key: str, default: Any) -> Any:
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    def _coerce_str_list(self, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(v).strip() for v in value if str(v).strip()]
        if isinstance(value, str):
            text = value.strip()
            return [text] if text else []
        return []

    def _coerce_str(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip()
        return str(value).strip()
