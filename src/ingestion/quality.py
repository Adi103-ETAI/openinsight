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
from src.ingestion.document_db import ChunkRecord

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
    1: 0.15,   # meta-analysis / RCT
    2: 0.10,   # observational / guideline
    3: 0.05,   # review
    4: 0.00,   # case report
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
    r"^\s*\d+\s*$",                             # lone page number
    r"\b(?:et al\.|ibid\.|op cit\.)\b",         # bibliographic boilerplate
    r"^(?:figure|table|box|appendix)\s+\d+",    # standalone caption
    r"http[s]?://\S+",                          # bare URL lines
]


def score_chunk(chunk: ChunkRecord) -> float:
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
    text = chunk.chunk_text

    # ── Base from content type ───────────────────────────────────────────────
    base = _CONTENT_TYPE_BASE.get(chunk.content_type, 0.35)

    # ── Length fitness ──────────────────────────────────────────────────────
    tokens = chunk.token_count or max(1, len(text.split()))
    if _MIN_TOKENS <= tokens <= _TARGET_TOKENS:
        length_bonus = 0.10
    elif tokens < _MIN_TOKENS:
        length_bonus = (tokens / _MIN_TOKENS) * 0.10
    else:
        # Slight penalty for oversized chunks
        length_bonus = max(0.0, 0.10 - (tokens - _TARGET_TOKENS) / (_TARGET_TOKENS * 2) * 0.10)

    # ── Entity density ──────────────────────────────────────────────────────
    total_entities = (
        len(chunk.diseases)
        + len(chunk.drugs)
        + len(chunk.dosages)
        + len(chunk.symptoms)
    )
    entity_bonus = min(0.10, total_entities * 0.015)

    # ── Evidence level ──────────────────────────────────────────────────────
    evidence_bonus = _EVIDENCE_LEVEL_BONUS.get(chunk.evidence_level, 0.0)

    # ── High-value signal patterns ───────────────────────────────────────────
    signal_hits = sum(
        1 for p in _HIGH_VALUE_PATTERNS if re.search(p, text, re.IGNORECASE)
    )
    signal_bonus = min(0.10, signal_hits * 0.02)

    # ── Low-value penalty ────────────────────────────────────────────────────
    noise_hits = sum(
        1 for p in _LOW_VALUE_PATTERNS if re.search(p, text, re.IGNORECASE | re.MULTILINE)
    )
    penalty = min(0.20, noise_hits * 0.07)

    # ── Safety flag bonus (clinically critical content) ──────────────────────
    safety_bonus = 0.05 if chunk.has_safety_flag else 0.0

    raw = base + length_bonus + entity_bonus + evidence_bonus + signal_bonus + safety_bonus - penalty
    return round(max(0.0, min(1.0, raw)), 4)


def score_chunks(chunks: list[ChunkRecord]) -> list[ChunkRecord]:
    """
    Score all chunks in-place and return the list.
    """
    for chunk in chunks:
        chunk.quality_score = score_chunk(chunk)
    return chunks
