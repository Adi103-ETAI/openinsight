"""
Answer Validator - Main Orchestrator
Coordinates hallucination detection, citation checking, safety checking,
and confidence scoring to produce a comprehensive validation result.
"""

from dataclasses import dataclass, field
from typing import Literal, Optional

from loguru import logger

from .citation_checker import CitationCheckResult, check_citations
from .confidence_scorer import ConfidenceBreakdown, score_confidence
from .hallucination_detector import (
    HallucinationFlag,
    HallucinationResult,
    detect_hallucinations,
)
from .medical_safety import SafetyCheckResult, SafetyWarning, check_safety


@dataclass
class UnverifiedClaim:
    """A claim that could not be verified against sources."""

    text: str
    reason: str
    confidence: float  # How confident we are this is unverified


@dataclass
class ValidationResult:
    """Complete validation result for an answer."""

    # Overall scores
    confidence_score: float  # 0-1, overall answer reliability
    recommendation: Literal["SAFE", "NEEDS_REVIEW", "UNSAFE"]

    # Detailed results
    unverified_claims: list[UnverifiedClaim] = field(default_factory=list)
    safety_warnings: list[SafetyWarning] = field(default_factory=list)
    evidence_distribution: dict = field(default_factory=dict)

    # Component results (for debugging/transparency)
    hallucination_result: Optional[HallucinationResult] = None
    citation_result: Optional[CitationCheckResult] = None
    safety_result: Optional[SafetyCheckResult] = None
    confidence_breakdown: Optional[ConfidenceBreakdown] = None

    # Metadata
    is_safe: bool = True
    needs_disclaimer: bool = False


def _determine_recommendation(
    confidence_score: float,
    hallucination_result: HallucinationResult,
    safety_result: SafetyCheckResult,
    citation_result: CitationCheckResult,
) -> Literal["SAFE", "NEEDS_REVIEW", "UNSAFE"]:
    """
    Determine overall recommendation based on all validation results.

    UNSAFE: High hallucination risk, serious safety concerns, or critical citation issues
    NEEDS_REVIEW: Moderate issues that a physician should verify
    SAFE: Answer appears reliable and safe
    """
    # UNSAFE conditions
    if hallucination_result.hallucination_score > 0.6:
        logger.warning("Recommendation: UNSAFE - high hallucination risk")
        return "UNSAFE"

    if not safety_result.is_safe:
        logger.warning("Recommendation: UNSAFE - safety concerns")
        return "UNSAFE"

    if citation_result.valid_citations == 0 and citation_result.total_citations > 0:
        logger.warning("Recommendation: UNSAFE - no valid citations")
        return "UNSAFE"

    # NEEDS_REVIEW conditions
    if hallucination_result.hallucination_score > 0.3:
        logger.info("Recommendation: NEEDS_REVIEW - moderate hallucination risk")
        return "NEEDS_REVIEW"

    if safety_result.needs_disclaimer:
        logger.info("Recommendation: NEEDS_REVIEW - contains treatment advice")
        return "NEEDS_REVIEW"

    if confidence_score < 0.5:
        logger.info("Recommendation: NEEDS_REVIEW - low confidence")
        return "NEEDS_REVIEW"

    high_severity_issues = [i for i in citation_result.issues if i.severity == "HIGH"]
    if high_severity_issues:
        logger.info("Recommendation: NEEDS_REVIEW - citation issues")
        return "NEEDS_REVIEW"

    # SAFE - no major issues found
    return "SAFE"


async def validate_answer(
    answer: str,
    citations: list[dict],
    source_chunks: list[dict],
    verify_citations_in_db: bool = True,
) -> ValidationResult:
    """
    Validate an LLM-generated answer for quality and safety.

    Args:
        answer: The LLM-generated answer text
        citations: List of citation dicts with mongo_id, source_type, title, etc.
        source_chunks: Original chunks used to generate the answer
        verify_citations_in_db: Whether to verify citations exist in MongoDB

    Returns:
        ValidationResult with confidence score, warnings, and recommendation
    """
    logger.info("Starting answer validation...")

    # Handle empty answer
    if not answer:
        return ValidationResult(
            confidence_score=0.0,
            recommendation="UNSAFE",
            unverified_claims=[],
            safety_warnings=[],
            evidence_distribution={},
            is_safe=False,
        )

    # 1. Hallucination Detection
    logger.debug("Running hallucination detection...")
    hallucination_result = detect_hallucinations(answer, source_chunks)
    logger.info(
        f"Hallucination check: score={hallucination_result.hallucination_score:.2f}, "
        f"flagged={len(hallucination_result.flagged_claims)}/{hallucination_result.total_claims_count}"
    )

    # 2. Citation Validation
    logger.debug("Running citation validation...")
    citation_result = await check_citations(
        citations, verify_in_db=verify_citations_in_db
    )
    logger.info(
        f"Citation check: valid={citation_result.valid_citations}/{citation_result.total_citations}, "
        f"issues={len(citation_result.issues)}"
    )

    # 3. Medical Safety Check
    logger.debug("Running safety check...")
    safety_result = check_safety(answer, source_chunks)
    logger.info(
        f"Safety check: is_safe={safety_result.is_safe}, "
        f"warnings={len(safety_result.warnings)}, needs_disclaimer={safety_result.needs_disclaimer}"
    )

    # 4. Confidence Scoring
    logger.debug("Calculating confidence score...")
    has_high_severity = any(w.severity == "HIGH" for w in safety_result.warnings)
    confidence_breakdown = score_confidence(
        citations=citations,
        chunks=source_chunks,
        hallucination_risk=hallucination_result.hallucination_score,
        avg_evidence_level=citation_result.avg_evidence_level,
        evidence_distribution=citation_result.evidence_distribution,
        num_safety_warnings=len(safety_result.warnings),
        has_high_severity_warning=has_high_severity,
    )
    logger.info(f"Confidence score: {confidence_breakdown.final_score:.2f}")

    # 5. Determine recommendation
    recommendation = _determine_recommendation(
        confidence_breakdown.final_score,
        hallucination_result,
        safety_result,
        citation_result,
    )

    # Convert hallucination flags to unverified claims
    unverified_claims = [
        UnverifiedClaim(
            text=flag.sentence,
            reason=flag.reason,
            confidence=flag.confidence,
        )
        for flag in hallucination_result.flagged_claims
    ]

    logger.info(f"Validation complete: recommendation={recommendation}")

    return ValidationResult(
        confidence_score=confidence_breakdown.final_score,
        recommendation=recommendation,
        unverified_claims=unverified_claims,
        safety_warnings=safety_result.warnings,
        evidence_distribution=citation_result.evidence_distribution,
        hallucination_result=hallucination_result,
        citation_result=citation_result,
        safety_result=safety_result,
        confidence_breakdown=confidence_breakdown,
        is_safe=safety_result.is_safe,
        needs_disclaimer=safety_result.needs_disclaimer,
    )


def enhance_response(original_response: dict, validation: ValidationResult) -> dict:
    """
    Enhance the original response with validation results.

    Args:
        original_response: The original response dict from standard_search
        validation: The ValidationResult from validate_answer

    Returns:
        Enhanced response dict with validation fields added
    """
    # Convert safety warnings to serializable format
    safety_warnings_serialized = [
        {
            "type": w.warning_type,
            "severity": w.severity,
            "message": w.message,
            "matched_text": w.matched_text,
        }
        for w in validation.safety_warnings
    ]

    # Convert unverified claims to serializable format
    unverified_claims_serialized = [
        {
            "text": c.text,
            "reason": c.reason,
            "confidence": c.confidence,
        }
        for c in validation.unverified_claims
    ]

    # Add validation fields to response
    enhanced = {
        **original_response,
        "confidence_score": validation.confidence_score,
        "recommendation": validation.recommendation,
        "unverified_claims": unverified_claims_serialized,
        "safety_warnings": safety_warnings_serialized,
        "evidence_distribution": validation.evidence_distribution,
        "is_safe": validation.is_safe,
        "needs_disclaimer": validation.needs_disclaimer,
    }

    # Add confidence breakdown if available (useful for debugging)
    if validation.confidence_breakdown:
        enhanced["confidence_breakdown"] = {
            "citation_score": validation.confidence_breakdown.citation_score,
            "evidence_score": validation.confidence_breakdown.evidence_score,
            "hallucination_score": validation.confidence_breakdown.hallucination_score,
            "quality_score": validation.confidence_breakdown.quality_score,
            "consistency_score": validation.confidence_breakdown.consistency_score,
            "safety_penalty": validation.confidence_breakdown.safety_penalty,
        }

    return enhanced
