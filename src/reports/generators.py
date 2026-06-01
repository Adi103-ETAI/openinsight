"""
Report Generators — Clinical Summary and Evidence Review.
Transform search results and validation data into structured reports.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from loguru import logger

from src.reports.models import (
    Citation,
    ClinicalSummaryReport,
    ConfidenceBreakdown,
    EvidenceReviewReport,
    ReportMetadata,
    SafetyWarning,
)


def _build_metadata(
    report_type: str,
    query: str,
    confidence_score: float,
    recommendation: str,
    citations: list[Citation],
    safety_warnings: list[SafetyWarning],
) -> ReportMetadata:
    """Build standard report metadata."""
    return ReportMetadata(
        report_id=str(uuid.uuid4()),
        report_type=report_type,
        generated_at=datetime.utcnow(),
        query=query,
        confidence_score=confidence_score,
        recommendation=recommendation,
        citation_count=len(citations),
        safety_warning_count=len(safety_warnings),
        format="json",
    )


def _extract_key_findings(answer: str, citations: list[Citation]) -> list[str]:
    """Extract key findings from answer text."""
    findings = []
    sentences = [s.strip() for s in answer.split(".") if s.strip()]

    for sentence in sentences:
        if len(sentence) < 20:
            continue
        lower = sentence.lower()
        if any(
            kw in lower
            for kw in [
                "should",
                "recommend",
                "first-line",
                "effective",
                "contraindicated",
                "significant",
                "meta-analysis",
                "systematic review",
                "evidence",
                "demonstrates",
                "associated with",
            ]
        ):
            findings.append(sentence.rstrip(".") + ".")

    if not findings and sentences:
        findings = [s.rstrip(".") + "." for s in sentences[:3] if len(s) > 20]

    return findings[:8]


def _build_evidence_summary(citations: list[Citation]) -> str:
    """Build a summary of evidence quality from citations."""
    if not citations:
        return "No citations available for evidence summary."

    source_counts: dict[str, int] = {}
    level_counts: dict[int, int] = {}
    for c in citations:
        source_counts[c.source_type] = source_counts.get(c.source_type, 0) + 1
        level_counts[c.evidence_level] = level_counts.get(c.evidence_level, 0) + 1

    parts = [f"Based on {len(citations)} source(s)"]
    if source_counts:
        sources = ", ".join(f"{count} {src}" for src, count in source_counts.items())
        parts.append(f"from {sources}")

    high_quality = sum(1 for c in citations if c.evidence_level <= 2)
    if high_quality:
        parts.append(f"({high_quality} high-quality source(s) with evidence level 1-2)")

    return ". ".join(parts) + "."


def _build_confidence_assessment(
    confidence_score: float,
    breakdown: ConfidenceBreakdown | None,
) -> str:
    """Build human-readable confidence assessment."""
    if confidence_score >= 0.8:
        level = "High"
    elif confidence_score >= 0.5:
        level = "Moderate"
    else:
        level = "Low"

    text = f"Confidence: {level} ({confidence_score:.0%})"

    if breakdown:
        issues = []
        if breakdown.hallucination_score > 0.3:
            issues.append("some claims may not be fully grounded in sources")
        if breakdown.citation_score < 0.5:
            issues.append("citation coverage is limited")
        if breakdown.safety_penalty > 0:
            issues.append("safety concerns flagged")
        if issues:
            text += ". Notes: " + "; ".join(issues) + "."

    return text


def generate_clinical_summary(
    query: str,
    answer: str,
    citations: list[Citation] | None = None,
    safety_warnings: list[SafetyWarning] | None = None,
    confidence_score: float = 0.0,
    confidence_breakdown: ConfidenceBreakdown | None = None,
    recommendation: str = "NEEDS_REVIEW",
) -> ClinicalSummaryReport:
    """
    Generate a Clinical Summary Report.

    Designed for doctors who need a quick, actionable summary of clinical evidence
    for a specific query, with citations and safety warnings.
    """
    citations = citations or []
    safety_warnings = safety_warnings or []

    metadata = _build_metadata(
        report_type="clinical_summary",
        query=query,
        confidence_score=confidence_score,
        recommendation=recommendation,
        citations=citations,
        safety_warnings=safety_warnings,
    )

    key_findings = _extract_key_findings(answer, citations)
    evidence_summary = _build_evidence_summary(citations)
    confidence_assessment = _build_confidence_assessment(confidence_score, confidence_breakdown)

    return ClinicalSummaryReport(
        metadata=metadata,
        query=query,
        clinical_answer=answer,
        key_findings=key_findings,
        evidence_summary=evidence_summary,
        citations=citations,
        safety_warnings=safety_warnings,
        confidence_assessment=confidence_assessment,
        recommendation=recommendation,
    )


def generate_evidence_review(
    query: str,
    answer: str,
    citations: list[Citation] | None = None,
    safety_warnings: list[SafetyWarning] | None = None,
    confidence_score: float = 0.0,
    confidence_breakdown: ConfidenceBreakdown | None = None,
    recommendation: str = "NEEDS_REVIEW",
    source_chunks: list[dict[str, Any]] | None = None,
) -> EvidenceReviewReport:
    """
    Generate an Evidence Review Report.

    Detailed breakdown of source quality, evidence levels, and confidence
    scoring for thorough clinical review.
    """
    citations = citations or []
    safety_warnings = safety_warnings or []
    source_chunks = source_chunks or []

    metadata = _build_metadata(
        report_type="evidence_review",
        query=query,
        confidence_score=confidence_score,
        recommendation=recommendation,
        citations=citations,
        safety_warnings=safety_warnings,
    )

    # Source breakdown by type
    source_breakdown: dict[str, int] = {}
    for c in citations:
        source_breakdown[c.source_type] = source_breakdown.get(c.source_type, 0) + 1

    # Evidence level distribution
    evidence_levels: dict[str, int] = {}
    level_names = {1: "Systematic Review/Meta-Analysis", 2: "RCT", 3: "Cohort/Observational",
                   4: "Case Series/Expert Opinion", 5: "Unknown"}
    for c in citations:
        name = level_names.get(c.evidence_level, "Unknown")
        evidence_levels[name] = evidence_levels.get(name, 0) + 1

    # Hallucination flags from source chunks (simple text coverage check)
    hallucination_flags = []
    if source_chunks and answer:
        answer_words = set(answer.lower().split())
        for chunk in source_chunks[:5]:
            chunk_text = chunk.get("chunk_text", "")
            chunk_words = set(chunk_text.lower().split())
            overlap = answer_words & chunk_words
            coverage = len(overlap) / max(len(answer_words), 1)
            if coverage < 0.1:
                hallucination_flags.append({
                    "chunk_title": chunk.get("title", "Unknown"),
                    "coverage": round(coverage, 3),
                    "note": "Low word overlap with answer",
                })

    # Quality notes
    quality_notes_parts = []
    if confidence_breakdown:
        if confidence_breakdown.hallucination_score > 0.3:
            quality_notes_parts.append(
                f"Hallucination risk is elevated ({confidence_breakdown.hallucination_score:.0%}). "
                "Verify claims against primary sources."
            )
        if confidence_breakdown.evidence_score < 0.5:
            quality_notes_parts.append(
                "Evidence quality is moderate. Consider searching for higher-level evidence."
            )

    if not citations:
        quality_notes_parts.append("No citations found. Answer may not be evidence-based.")

    return EvidenceReviewReport(
        metadata=metadata,
        query=query,
        answer=answer,
        evidence_analysis=_build_evidence_summary(citations),
        source_breakdown=source_breakdown,
        evidence_levels=evidence_levels,
        citations=citations,
        hallucination_flags=hallucination_flags,
        safety_warnings=safety_warnings,
        confidence_breakdown=confidence_breakdown,
        quality_notes=" ".join(quality_notes_parts) if quality_notes_parts else "No quality concerns identified.",
    )
