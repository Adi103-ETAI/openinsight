"""
Deduplication Engine
Detects duplicate documents by DOI, normalised title, or content hash.
Prevents re-embedding identical or near-identical documents.
"""
import hashlib
import re
from typing import Optional

from loguru import logger
from motor.motor_asyncio import AsyncIOMotorDatabase

from src.ingestion.document_db import DocumentRecord


def compute_content_hash(text: str) -> str:
    """Return SHA-256 hex digest of whitespace-normalised text."""
    normalised = re.sub(r"\s+", " ", text.strip().lower())
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()


def _normalise_title(title: str) -> str:
    """Lowercase, strip punctuation/extra spaces for fuzzy title matching."""
    title = title.lower()
    title = re.sub(r"[^\w\s]", "", title)
    title = re.sub(r"\s+", " ", title).strip()
    return title


def _title_similarity(a: str, b: str) -> float:
    """
    Simple character n-gram Jaccard similarity between two normalised titles.
    Returns value in [0, 1]; faster than full edit distance.
    """
    if not a or not b:
        return 0.0
    ngram_size = 3
    set_a = {a[i : i + ngram_size] for i in range(len(a) - ngram_size + 1)}
    set_b = {b[i : i + ngram_size] for i in range(len(b) - ngram_size + 1)}
    if not set_a or not set_b:
        return 1.0 if a == b else 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union


async def is_duplicate(
    db: AsyncIOMotorDatabase,
    document: DocumentRecord,
    title_similarity_threshold: float = 0.9,
) -> tuple[bool, Optional[str]]:
    """
    Check whether an equivalent document already exists in MongoDB.

    Checks (in priority order):
      1. Exact DOI match (most reliable)
      2. Exact content hash match
      3. High-similarity title match (Jaccard n-gram ≥ threshold)

    Returns:
        (True, existing_document_id)  — if duplicate found
        (False, None)                 — if document is new
    """
    documents_col = db["documents"]

    # 1. DOI match
    doi = (document.doi or "").strip()
    if doi:
        existing = await documents_col.find_one({"doi": doi}, {"_id": 1})
        if existing:
            eid = str(existing["_id"])
            logger.debug(f"[dedup] DOI match: {doi} → {eid}")
            return True, eid

    # 2. Content hash match
    content_hash = compute_content_hash(document.content)
    existing = await documents_col.find_one({"content_hash": content_hash}, {"_id": 1})
    if existing:
        eid = str(existing["_id"])
        logger.debug(f"[dedup] Content-hash match: {content_hash[:12]}… → {eid}")
        return True, eid

    # 3. Title similarity (only within same source_type to limit scan)
    norm_title = _normalise_title(document.title)
    if norm_title:
        cursor = documents_col.find(
            {"source_type": document.source_type},
            {"_id": 1, "title": 1},
        ).limit(500)
        async for doc in cursor:
            existing_norm = _normalise_title(doc.get("title", ""))
            sim = _title_similarity(norm_title, existing_norm)
            if sim >= title_similarity_threshold:
                eid = str(doc["_id"])
                logger.debug(
                    f"[dedup] Title similarity {sim:.2f} for '{document.title[:60]}' → {eid}"
                )
                return True, eid

    return False, None


def enrich_document_hashes(document: DocumentRecord) -> DocumentRecord:
    """
    Compute and set content_hash on a DocumentRecord before insertion.
    Call this before persisting to MongoDB.
    """
    document.content_hash = compute_content_hash(document.content)
    return document
