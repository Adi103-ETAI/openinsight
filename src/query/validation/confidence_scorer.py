"""
Confidence Scorer
Calculates overall answer confidence based on citations, evidence levels,
source consistency, hallucination risk, and quality scores.
"""

from dataclasses import dataclass
from typing import Optional

from loguru import logger


@dataclass
class ConfidenceBreakdown:
    """Breakdown of confidence score components."""

    citation_score: float  # Based on number of citations
    evidence_score: float  # Based on evidence level of citations
    hallucination_score: float  # Based on absence of hallucinations
    quality_score: float  # Based on chunk quality scores
    consistency_score: float  # Based on source agreement
    safety_penalty: float  # Penalty for safety warnings
    final_score: float  # Combined weighted score


def _calculate_citation_score(num_citations: int, max_citations: int = 8) -> float:
    """Score based on number of supporting citations (0-1)."""
    if num_citations <= 0:
        return 0.0
    # Normalize to max_citations, cap at 1.0
    return min(num_citations / max_citations, 1.0)


def _calculate_evidence_score(
    avg_evidence_level: float, evidence_distribution: dict
) -> float:
    """
    Score based on evidence level of citations (0-1).
    Evidence levels: 1=highest (systematic review), 5=lowest (expert opinion)
    """
    if avg_evidence_level <= 0:
        return 0.5  # Default if no evidence info

    # Invert scale: level 1 → score 1.0, level 5 → score 0.2
    base_score = (6 - avg_evidence_level) / 5

    # Bonus for having Grade I (systematic review) citations
    grade_i_count = evidence_distribution.get("grade_i_count", 0)
    if grade_i_count > 0:
        base_score = min(base_score + 0.1 * grade_i_count, 1.0)

    return max(min(base_score, 1.0), 0.0)


def _calculate_hallucination_score(hallucination_risk: float) -> float:
    """
    Score based on absence of hallucinations (0-1).
    hallucination_risk 0 → score 1.0, risk 1 → score 0.0
    """
    return max(1.0 - hallucination_risk, 0.0)


def _calculate_quality_score(chunks: list[dict]) -> float:
    """Score based on average chunk quality scores (0-1)."""
    if not chunks:
        return 0.5  # Default if no chunks

    quality_scores = []
    for chunk in chunks:
        # Look for quality_score in chunk metadata
        qs = chunk.get("quality_score")
        if qs is not None:
            quality_scores.append(float(qs))
        else:
            # Check score field as fallback (retrieval score)
            score = chunk.get("score", 0.5)
            # Retrieval scores are typically 0-1 already
            quality_scores.append(min(max(float(score), 0.0), 1.0))

    if not quality_scores:
        return 0.5

    return sum(quality_scores) / len(quality_scores)


def _calculate_consistency_score(chunks: list[dict]) -> float:
    """
    Score based on source consistency/agreement (0-1).
    Higher if sources agree, lower if they contradict.

    For simplicity, we measure consistency by:
    - Diversity of sources (more diverse = higher consensus if they agree)
    - Score variance (lower variance = more consistent ranking)
    """
    if not chunks or len(chunks) < 2:
        return 0.8  # Default for single source

    # Check source diversity
    sources = set()
    for chunk in chunks:
        source = chunk.get("source_type", "unknown")
        sources.add(source)

    # More diverse sources that still rank similarly = higher confidence
    source_diversity = min(len(sources) / 3, 1.0)  # Normalize to ~3 source types

    # Check score variance (lower variance = more consistent)
    scores = [chunk.get("score", 0.5) for chunk in chunks]
    if len(scores) > 1:
        mean_score = sum(scores) / len(scores)
        variance = sum((s - mean_score) ** 2 for s in scores) / len(scores)
        # Low variance (< 0.1) is good, high variance (> 0.5) is bad
        variance_score = max(1.0 - variance * 2, 0.0)
    else:
        variance_score = 0.8

    # Combine diversity and variance
    return source_diversity * 0.4 + variance_score * 0.6


def _calculate_safety_penalty(
    num_safety_warnings: int, has_high_severity: bool
) -> float:
    """Calculate penalty based on safety warnings (0-1)."""
    if num_safety_warnings == 0:
        return 0.0

    # Base penalty per warning
    penalty = min(num_safety_warnings * 0.05, 0.3)

    # Additional penalty for high severity
    if has_high_severity:
        penalty += 0.1

    return min(penalty, 0.4)  # Cap at 0.4


def score_confidence(
    citations: list[dict],
    chunks: list[dict],
    hallucination_risk: float,
    avg_evidence_level: float = 3.0,
    evidence_distribution: dict = None,
    num_safety_warnings: int = 0,
    has_high_severity_warning: bool = False,
) -> ConfidenceBreakdown:
    """
    Calculate overall confidence score for an answer.

    Formula:
    base = 0.5
    + (num_citations / max_citations) * 0.15
    + (avg_evidence_level / 5) * 0.15
    + (1 - hallucination_ratio) * 0.20
    + (avg_chunk_quality_score) * 0.15
    + (source_consistency_score) * 0.10
    - (num_safety_warnings * 0.05)

    Args:
        citations: List of citation dicts
        chunks: List of source chunks with quality scores
        hallucination_risk: 0-1 hallucination risk score
        avg_evidence_level: Average evidence level (1-5, 1 = highest)
        evidence_distribution: Dict with grade counts
        num_safety_warnings: Number of safety warnings
        has_high_severity_warning: Whether there are HIGH severity warnings

    Returns:
        ConfidenceBreakdown with component scores and final score
    """
    if evidence_distribution is None:
        evidence_distribution = {}

    # Calculate component scores
    citation_score = _calculate_citation_score(len(citations))
    evidence_score = _calculate_evidence_score(
        avg_evidence_level, evidence_distribution
    )
    hall_score = _calculate_hallucination_score(hallucination_risk)
    quality_score = _calculate_quality_score(chunks)
    consistency_score = _calculate_consistency_score(chunks)
    safety_penalty = _calculate_safety_penalty(
        num_safety_warnings, has_high_severity_warning
    )

    # Weighted combination
    # Weights sum to 0.75, base is 0.25
    final = (
        0.25  # Base score
        + citation_score * 0.15  # Citation coverage
        + evidence_score * 0.15  # Evidence quality
        + hall_score * 0.20  # Grounding/hallucination
        + quality_score * 0.15  # Chunk quality
        + consistency_score * 0.10  # Source agreement
        - safety_penalty  # Safety penalty
    )

    # Clamp to 0-1
    final = max(min(final, 1.0), 0.0)

    return ConfidenceBreakdown(
        citation_score=round(citation_score, 3),
        evidence_score=round(evidence_score, 3),
        hallucination_score=round(hall_score, 3),
        quality_score=round(quality_score, 3),
        consistency_score=round(consistency_score, 3),
        safety_penalty=round(safety_penalty, 3),
        final_score=round(final, 3),
    )
