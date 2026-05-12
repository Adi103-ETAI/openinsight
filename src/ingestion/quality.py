"""
Chunk Quality & Relevance Scorer
Assigns a 0–1 quality score to each chunk based on:
  - Content type weight (clinical > background > noise)
  - Text length (penalise very short or very long chunks)
  - Medical entity density (more entities → higher relevance)
  - Evidence level of parent document
  - Presence of actionable clinical content (dosages, guidelines, protocols)
"""

import re
from typing import Any

# Min/max token targets from chunker_v2
_MIN_TOKENS = 50
_TARGET_TOKENS = 300


# ── Scoring weights ──────────────────────────────────────────────────────────
_CONTENT_TYPE_BASE: dict[str, float] = {
    "clinical": 0.80,
    "preclinical": 0.55,
    "background": 0.40,
    "noise": 0.05,
    "unknown": 0.35,
}

_EVIDENCE_LEVEL_BONUS: dict[int, float] = {
    1: 0.15,  # meta-analysis / RCT
    2: 0.10,  # observational / guideline
    3: 0.05,  # review
    4: 0.00,  # case report
    5: -0.05,  # unknown
}

# High-value clinical signal patterns that bump the score
_HIGH_VALUE_PATTERNS = [
    r"\b(?:dosage|dose|mg|mcg|g)\b",
    r"\b(?:treatment|therapy|management|protocol|guideline|recommendation)\b",
    r"\b(?:contraindicated?|adverse|warning|caution)\b",
    r"\b(?:RCT|randomized|randomised|systematic review|meta.analysis)\b",
    r"\b(?:first.line|second.line|alternative|empirical|prophylaxis)\b",
]

# Noise / low-value patterns that reduce the score
_LOW_VALUE_PATTERNS = [
    r"^\s*\d+\s*$",  # lone page number
    r"\b(?:et al\.|ibid\.|op cit\.)\b",  # bibliographic boilerplate
    r"^(?:figure|table|box|appendix)\s+\d+",  # standalone caption
    r"http[s]?://\S+",  # bare URL lines
]


def _get_chunk_text(chunk: Any) -> str:
    text = getattr(chunk, "chunk_text", None)
    if text:
        return text
    return getattr(chunk, "text", "") or ""


def _get_token_count(chunk: Any, text: str) -> int:
    token_count = getattr(chunk, "token_count", None)
    if token_count is None:
        token_count = getattr(chunk, "token_estimate", None)
    return token_count or max(1, len(text.split()))


def _get_content_type(chunk: Any) -> str:
    content_type = getattr(chunk, "content_type", None)
    if content_type:
        return str(content_type)

    chunk_type = getattr(chunk, "chunk_type", "")
    return {
        "doc_summary": "background",
        "paragraph": "clinical",
        "table": "clinical",
    }.get(str(chunk_type), "unknown")


def _get_list_field(chunk: Any, name: str) -> list[str]:
    value = getattr(chunk, name, None)
    if isinstance(value, list):
        return value

    metadata = getattr(chunk, "metadata", None)
    if isinstance(metadata, dict):
        meta_value = metadata.get(name)
        if isinstance(meta_value, list):
            return meta_value

    return []


def _normalize_evidence_level(value: Any) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        mapping = {
            "1a": 1,
            "1b": 1,
            "2a": 2,
            "2b": 2,
            "3": 3,
            "4": 4,
            "5": 5,
        }
        return mapping.get(normalized, 5)
    return 5


def score_chunk(chunk: Any) -> float:
    """
    Compute a quality score in [0, 1] for a chunk.

    Score components:
      base     = content_type base score
      length   = length fit bonus (0..+0.10)
      entity   = entity density bonus (0..+0.10)
      evidence = evidence level bonus (-0.05..+0.15)
      signal   = high-value pattern bonus (0..+0.10)
      penalty  = low-value pattern penalty (0..-0.20)

    Final score is clamped to [0.0, 1.0].
    """
    text = _get_chunk_text(chunk)

    # ── Base from content type ───────────────────────────────────────────────
    base = _CONTENT_TYPE_BASE.get(_get_content_type(chunk), 0.35)

    # ── Length fitness ──────────────────────────────────────────────────────
    tokens = _get_token_count(chunk, text)
    if _MIN_TOKENS <= tokens <= _TARGET_TOKENS:
        length_bonus = 0.10
    elif tokens < _MIN_TOKENS:
        length_bonus = (tokens / _MIN_TOKENS) * 0.10
    else:
        # Slight penalty for oversized chunks
        length_bonus = max(
            0.0, 0.10 - (tokens - _TARGET_TOKENS) / (_TARGET_TOKENS * 2) * 0.10
        )

    # ── Entity density ──────────────────────────────────────────────────────
    total_entities = (
        len(_get_list_field(chunk, "diseases"))
        + len(_get_list_field(chunk, "drugs"))
        + len(_get_list_field(chunk, "dosages"))
        + len(_get_list_field(chunk, "symptoms"))
    )
    entity_bonus = min(0.10, total_entities * 0.015)

    # ── Evidence level ──────────────────────────────────────────────────────
    evidence_level = getattr(chunk, "evidence_level", None)
    if evidence_level is None:
        metadata = getattr(chunk, "metadata", None)
        if isinstance(metadata, dict):
            evidence_level = metadata.get("evidence_level")
    evidence_bonus = _EVIDENCE_LEVEL_BONUS.get(
        _normalize_evidence_level(evidence_level),
        0.0,
    )

    # ── High-value signal patterns ───────────────────────────────────────────
    signal_hits = sum(
        1 for p in _HIGH_VALUE_PATTERNS if re.search(p, text, re.IGNORECASE)
    )
    signal_bonus = min(0.10, signal_hits * 0.02)

    # ── Low-value penalty ────────────────────────────────────────────────────
    noise_hits = sum(
        1
        for p in _LOW_VALUE_PATTERNS
        if re.search(p, text, re.IGNORECASE | re.MULTILINE)
    )
    penalty = min(0.20, noise_hits * 0.07)

    # ── Safety flag bonus (clinically critical content) ──────────────────────
    has_safety_flag = bool(getattr(chunk, "has_safety_flag", False))
    safety_bonus = 0.05 if has_safety_flag else 0.0

    raw = (
        base
        + length_bonus
        + entity_bonus
        + evidence_bonus
        + signal_bonus
        + safety_bonus
        - penalty
    )
    return round(max(0.0, min(1.0, raw)), 4)


def score_chunks(chunks: list[Any]) -> list[Any]:
    """
    Score all chunks in-place and return the list.
    """
    for chunk in chunks:
        chunk.quality_score = score_chunk(chunk)
    return chunks
