"""find_best_source — pick the most-supporting source for a claim."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.tools.citationtools.validate_claim import claim_supported_by_source


def find_best_source(claim: str, sources: List[Dict]) -> Optional[Dict[str, Any]]:
    """
    Return the source with the highest overlap_count that still supports the claim.
    Returns None if no source supports the claim.
    """
    best = None
    best_score = 0
    for src in sources:
        text = src.get("text") or src.get("excerpt", "")
        result = claim_supported_by_source(claim, text)
        if result["supported"] and result["overlap_count"] > best_score:
            best_score = result["overlap_count"]
            best = src
    return best
