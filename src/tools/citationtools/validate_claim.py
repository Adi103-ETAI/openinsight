"""validate_claim — check whether a claim is supported by a source text.

LIMITATIONS (read before relying on this in production):
- Uses **token overlap**, not semantic similarity. Two sentences with the
  same words but different meaning (e.g. "X is safe" vs "X is not safe")
  can be misclassified.
- Negation detection is **lexical only** (`not`, `no`, `never`, `none`,
  `without`). It does not parse syntax — a sentence like "it is not the
  case that X is safe" with no negation keyword near the claim can be
  mis-flagged as supported.
- No stemming, lemmatization, or synonym expansion. "diabetes" and
  "diabetic" count as different words.
- Numeric and unit claims ("dose of 500 mg") are not parsed; overlap
  with numbers can over-credit support.
- Domain words (e.g. "patient", "study", "evidence") inflate overlap
  without indicating meaningful support.

For production clinical use, register a semantic check via
`register_semantic_check()`. The semantic check is invoked first; if it
returns a verdict, that verdict wins. Token overlap is the fallback.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

_NEGATION_WORDS = {"not", "no", "never", "none", "without"}

# Type alias for a semantic check: takes (claim, source_text), returns
# {supported: bool, score: float, note: str} or None to defer to fallback.
SemanticCheck = Callable[[str, str], Optional[Dict[str, Any]]]

_semantic_check: Optional[SemanticCheck] = None


def register_semantic_check(check: Optional[SemanticCheck]) -> None:
    """
    Register a custom semantic check (e.g. embedding-similarity based).
    Pass None to clear. The check should return None to fall back to
    token-overlap, or a dict with at least `supported: bool` and
    `score: float` to override.
    """
    global _semantic_check
    _semantic_check = check
    if check is None:
        logger.info("semantic check cleared; using token-overlap fallback")
    else:
        logger.info(f"semantic check registered: {check.__name__}")


def get_semantic_check() -> Optional[SemanticCheck]:
    """Return the currently registered semantic check, if any."""
    return _semantic_check


def _token_overlap(claim: str, source_text: str) -> Dict[str, Any]:
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
        "method": "token_overlap",
    }


def claim_supported_by_source(claim: str, source_text: str) -> Dict[str, Any]:
    """
    Score whether `claim` is supported by `source_text`.

    If a semantic check is registered and returns a non-None result, that
    result wins and the token-overlap result is included as `fallback`.
    Otherwise, returns the token-overlap result.
    """
    fallback = _token_overlap(claim, source_text)
    if _semantic_check is not None:
        try:
            result = _semantic_check(claim, source_text)
        except Exception as e:
            logger.warning(f"semantic check raised: {e}; using fallback")
            result = None
        if result is not None:
            return {
                **fallback,
                **result,
                "fallback": fallback,
                "method": result.get("method", "semantic"),
            }
    return fallback


def is_supported(claim: str, source_text: str) -> bool:
    """Quick boolean check: is the claim supported by source_text?"""
    return claim_supported_by_source(claim, source_text)["supported"]
