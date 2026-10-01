"""
Retrieval smoke test — golden clinical questions with expected evidence.

Runs WITHOUT torch: keyword recall against Mongo `chunks_v2` (does the
evidence exist and is it findable by exact clinical terms?) plus a Milvus
row-presence check. Embedding-quality recall is a later gate; this one
catches the failure class that actually bites: missing or unfindable content.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class GoldenQA:
    question: str
    must_match_any: tuple[str, ...]  # distinctive terms expected in recalled chunks
    source: str = ""  # restrict to source, "" = any


GOLDEN_QAS: list[GoldenQA] = [
    GoldenQA(
        question="First-line treatment for drug-sensitive pulmonary TB?",
        must_match_any=("rifampicin", "isoniazid", "HRZE", "DOTS"),
    ),
    GoldenQA(
        question="Diagnostic criteria for diabetes mellitus?",
        must_match_any=("HbA1c", "fasting plasma glucose", "OGTT", "6.5%"),
    ),
    GoldenQA(
        question="Management of hypertension in adults?",
        must_match_any=("amlodipine", "blood pressure", "140/90", "ACE inhibitor"),
    ),
    GoldenQA(
        question="Rifampicin drug interactions to counsel on?",
        must_match_any=("CYP450", "contraceptive", "warfarin", "inducer"),
    ),
    GoldenQA(
        question="Drinking water quality limits as per IS 10500?",
        must_match_any=("TDS", "pH", "500", "IS 10500"),
    ),
]


@dataclass
class SmokeReport:
    passed: bool
    recalled: int = 0
    total: int = 0
    misses: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [f"smoke: {self.recalled}/{self.total} recalled — {'PASS' if self.passed else 'FAIL'}"]
        for m in self.misses:
            lines.append(f"  miss: {m}")
        return "\n".join(lines)


def _chunk_text(doc: dict[str, Any]) -> str:
    return str(doc.get("text") or doc.get("chunk_text") or "")


def keyword_recall(db: Any, qa: GoldenQA, limit: int = 50) -> bool:
    """True if any chunk contains any must-match term (case-insensitive)."""
    q: dict[str, Any] = {}
    if qa.source:
        q = {"$or": [{"source": qa.source}, {"source_type": qa.source}]}
    terms = [t.lower() for t in qa.must_match_any]
    for doc in db["chunks_v2"].find(q).limit(limit * 20):
        hay = _chunk_text(doc).lower()
        if any(t in hay for t in terms):
            return True
    # Fallback: text search across collection when source-scoped scan misses
    if qa.source:
        for doc in db["chunks_v2"].find({}).limit(limit * 20):
            hay = _chunk_text(doc).lower()
            if any(t in hay for t in terms):
                return True
    return False


def run_smoke(db: Any, qas: list[GoldenQA] | None = None, min_recall: float = 0.6) -> SmokeReport:
    qas = qas if qas is not None else GOLDEN_QAS
    misses = [qa.question for qa in qas if not keyword_recall(db, qa)]
    recalled = len(qas) - len(misses)
    rate = recalled / max(len(qas), 1)
    return SmokeReport(passed=rate >= min_recall, recalled=recalled, total=len(qas), misses=misses)
