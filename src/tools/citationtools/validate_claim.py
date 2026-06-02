"""validate_claim — check whether a claim is supported by a source text."""
from __future__ import annotations

from typing import Any, Dict

_NEGATION_WORDS = {"not", "no", "never", "none", "without"}


def claim_supported_by_source(claim: str, source_text: str) -> Dict[str, Any]:
    """
    Score overlap between claim and source. Returns:
        {overlap_ratio, overlap_count, has_negation, supported, overlap_words}
    """
    claim_words = set(claim.lower().split())
    source_words = set(source_text.lower().split())
    overlap = claim_words & source_words
    ratio = len(overlap) / len(claim_words) if claim_words else 0.0
    has_negation = any(w in claim.lower() for w in _NEGATION_WORDS)
    return {
        "overlap_ratio": ratio,
        "overlap_count": len(overlap),
        "has_negation": has_negation,
        "supported": ratio > 0.3 or (ratio > 0.2 and len(overlap) > 2),
        "overlap_words": list(overlap),
    }


def is_supported(claim: str, source_text: str) -> bool:
    """Quick boolean check: is the claim supported by source_text?"""
    return claim_supported_by_source(claim, source_text)["supported"]
