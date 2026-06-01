"""
Hallucination Detector
Verifies that claims in the LLM answer are grounded in source chunks.
Uses semantic similarity and entity matching to detect invented content.
"""

import re
from dataclasses import dataclass, field
from functools import lru_cache

from loguru import logger
from sentence_transformers import util

from src.config.settings import get_settings
from src.ml.embedding.embedder import get_embedder

settings = get_settings()


@dataclass
class HallucinationFlag:
    """A flagged sentence that may not be grounded in sources."""

    sentence: str
    reason: str
    confidence: float  # 0-1, how confident we are this is a hallucination
    grounding_score: float  # 0-1, semantic similarity to best matching chunk


@dataclass
class HallucinationResult:
    """Result of hallucination detection."""

    hallucination_score: float  # 0-1, overall risk (1 = high risk)
    flagged_claims: list[HallucinationFlag] = field(default_factory=list)
    verified_claims_count: int = 0
    total_claims_count: int = 0


# Patterns for extracting numerical claims (dosages, percentages)
DOSAGE_PATTERN = re.compile(
    r"\b(\d+(?:\.\d+)?)\s*(?:mg|g|kg|mcg|µg|ml|mL|IU|units?)\b", re.IGNORECASE
)
PERCENTAGE_PATTERN = re.compile(r"\b(\d+(?:\.\d+)?)\s*%")
DURATION_PATTERN = re.compile(
    r"\b(\d+)\s*(?:days?|weeks?|months?|years?|hours?)\b", re.IGNORECASE
)


@lru_cache(maxsize=1)
def _get_embedding_model():
    """Load sentence transformer for semantic similarity."""
    logger.info(
        f"Loading embedding model for hallucination detection: {settings.embedding_model}"
    )
    return get_embedder()


def _split_into_sentences(text: str) -> list[str]:
    """Split answer into sentences for claim-level analysis."""
    # Handle common medical abbreviations
    abbrevs = [
        "Dr.",
        "Mr.",
        "Mrs.",
        "Ms.",
        "Prof.",
        "etc.",
        "vs.",
        "mg.",
        "kg.",
        "approx.",
    ]
    protected = text
    for abbrev in abbrevs:
        protected = protected.replace(abbrev, abbrev.replace(".", "<DOT>"))

    # Split on sentence boundaries
    sentences = re.split(r"(?<=[.!?])\s+", protected)
    return [
        s.replace("<DOT>", ".").strip()
        for s in sentences
        if s.strip() and len(s.strip()) > 10
    ]


def _extract_numerical_claims(text: str) -> list[str]:
    """Extract numerical claims (dosages, percentages, durations) from text."""
    claims = []
    claims.extend(DOSAGE_PATTERN.findall(text))
    claims.extend(PERCENTAGE_PATTERN.findall(text))
    claims.extend(DURATION_PATTERN.findall(text))
    return claims


def _extract_medical_entities(text: str) -> set[str]:
    """Extract drug names, diseases, and procedures from text (simple pattern-based)."""
    # Common drug suffixes and medical terms
    drug_suffixes = [
        "mycin",
        "cillin",
        "azole",
        "pril",
        "sartan",
        "statin",
        "mab",
        "nib",
        "vir",
    ]

    words = set()
    for word in re.findall(r"\b[A-Za-z][a-z]{3,}\b", text):
        word_lower = word.lower()
        # Check if it looks like a drug name
        for suffix in drug_suffixes:
            if word_lower.endswith(suffix):
                words.add(word_lower)
                break
        # Keep capitalized medical terms
        if word[0].isupper() and len(word) > 4:
            words.add(word_lower)

    return words


def detect_hallucinations(
    answer: str,
    chunks: list[dict],
    similarity_threshold: float = None,
    numerical_strict: bool = True,
) -> HallucinationResult:
    """
    Detect hallucinated content in LLM answer.

    Args:
        answer: The LLM-generated answer text
        chunks: List of source chunks with 'chunk_text' field
        similarity_threshold: Min semantic similarity to consider grounded (0-1).
            Defaults to config value (0.75 recommended for medical RAG).
        numerical_strict: If True, require exact numerical matches

    Returns:
        HallucinationResult with flagged claims and overall score
    """
    # Use config threshold if not specified
    if similarity_threshold is None:
        similarity_threshold = settings.hallucination_threshold
    if not answer or not chunks:
        return HallucinationResult(
            hallucination_score=1.0 if answer else 0.0,
            flagged_claims=[],
            verified_claims_count=0,
            total_claims_count=0,
        )

    model = _get_embedding_model()
    sentences = _split_into_sentences(answer)

    if not sentences:
        return HallucinationResult(
            hallucination_score=0.0,
            flagged_claims=[],
            verified_claims_count=0,
            total_claims_count=0,
        )

    # Combine all chunk texts for entity extraction
    all_chunk_text = " ".join(c.get("chunk_text", "") for c in chunks)
    chunk_entities = _extract_medical_entities(all_chunk_text)
    chunk_numbers = set(_extract_numerical_claims(all_chunk_text))

    # Embed all chunks using DualEmbedderV2's embed_batch method
    # Note: DualEmbedderV2 doesn't have an encode method, it uses embed_batch
    chunk_texts = [c.get("chunk_text", "") for c in chunks]
    chunk_embeddings, _ = model.embed_batch(chunk_texts)
    sentence_embeddings, _ = model.embed_batch(sentences)

    flagged_claims: list[HallucinationFlag] = []
    verified_count = 0

    for idx, sentence in enumerate(sentences):
        # Skip very short sentences or headers
        if len(sentence) < 20 or sentence.endswith(":"):
            verified_count += 1
            continue

        # Compute semantic similarity to all chunks
        similarities = util.cos_sim(sentence_embeddings[idx], chunk_embeddings)[0]
        max_similarity = float(similarities.max())

        reasons = []
        confidence = 0.0

        # Check semantic grounding
        if max_similarity < similarity_threshold:
            reasons.append(f"Low semantic similarity ({max_similarity:.2f})")
            confidence += 0.4

        # Check entity grounding
        sentence_entities = _extract_medical_entities(sentence)
        ungrounded_entities = sentence_entities - chunk_entities
        if ungrounded_entities and len(ungrounded_entities) > 1:
            reasons.append(
                f"Ungrounded entities: {', '.join(list(ungrounded_entities)[:3])}"
            )
            confidence += 0.3

        # Check numerical claims
        if numerical_strict:
            sentence_numbers = set(_extract_numerical_claims(sentence))
            ungrounded_numbers = sentence_numbers - chunk_numbers
            if ungrounded_numbers:
                reasons.append(f"Unverified numbers: {', '.join(ungrounded_numbers)}")
                confidence += 0.3

        if reasons:
            flagged_claims.append(
                HallucinationFlag(
                    sentence=sentence[:200],  # Truncate long sentences
                    reason="; ".join(reasons),
                    confidence=min(confidence, 1.0),
                    grounding_score=max_similarity,
                )
            )
        else:
            verified_count += 1

    # Calculate overall hallucination score
    total = len(sentences)
    if total == 0:
        hallucination_score = 0.0
    else:
        # Weighted by confidence of each flagged claim
        weighted_flags = sum(f.confidence for f in flagged_claims)
        hallucination_score = min(weighted_flags / total, 1.0)

    return HallucinationResult(
        hallucination_score=hallucination_score,
        flagged_claims=flagged_claims,
        verified_claims_count=verified_count,
        total_claims_count=total,
    )
