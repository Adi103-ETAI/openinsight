"""
Data Quality Validation Layer
Validates documents and chunks before they are stored/embedded.

Checks:
  - Document: title not empty, content of reasonable length, source_type valid
  - Chunk: text not empty, length within bounds, not obviously garbled
  - Medical plausibility: chunk is not a random symbol soup
"""
import re
from typing import Optional
from src.ingestion.document_db import ChunkRecord, DocumentRecord

# Valid source types
VALID_SOURCE_TYPES = {
    # International / generic
    "pubmed", "cochrane", "who", "cdc", "nih", "statpearls", "research",
    # Indian government + regulatory (Layer 1+3+4)
    "icmr", "nmc", "nmc_guideline", "mohfw", "state_guideline", "rssdi",
    # Phase 1 — Indian journals
    "indmed", "medknow", "pmc_india",
    # Phase 2 — Foundational open-access (will be expanded as parsers ship)
    "ncbi_bookshelf", "nmc_curriculum",
    # Phase 3 — Drug & regulatory (will be expanded as parsers ship)
    "nfi", "cdsco", "ctri", "pvpi", "ipc",
    # Phase 4 — Specialty guidelines (will be expanded as parsers ship)
    "csi", "ntep", "nvbdcp",
    # Phase 5 — Epidemiology (will be expanded as parsers ship)
    "nfhs", "ncdir",
}

# Minimum and maximum document content length (characters)
_DOC_MIN_CHARS = 200
_DOC_MAX_CHARS = 2_000_000  # 2 MB of text is the upper sanity limit

# Chunk bounds
_CHUNK_MIN_CHARS = 80
_CHUNK_MAX_CHARS = 8_000

# Ratio of non-ASCII characters that flags garbled OCR text
_GARBLE_RATIO = 0.25


def _garble_score(text: str) -> float:
    """Fraction of characters that are non-ASCII or control characters."""
    if not text:
        return 1.0
    non_ascii = sum(1 for c in text if ord(c) > 127 or (ord(c) < 32 and c not in "\n\t\r"))
    return non_ascii / len(text)


def validate_document(document: DocumentRecord) -> tuple[bool, Optional[str]]:
    """
    Validate a DocumentRecord.

    Returns:
        (True, None)           — document is valid
        (False, reason_str)    — document failed validation
    """
    if not document.title or not document.title.strip():
        return False, "title is empty"

    if len(document.title.strip()) < 5:
        return False, f"title too short ({len(document.title.strip())} chars)"

    if not document.content or not document.content.strip():
        return False, "content is empty"

    content_len = len(document.content.strip())
    if content_len < _DOC_MIN_CHARS:
        return False, f"content too short ({content_len} chars, min {_DOC_MIN_CHARS})"

    if content_len > _DOC_MAX_CHARS:
        return False, f"content too large ({content_len} chars, max {_DOC_MAX_CHARS})"

    if document.source_type not in VALID_SOURCE_TYPES:
        return False, f"unknown source_type '{document.source_type}'"

    if _garble_score(document.content) > _GARBLE_RATIO:
        return False, "content appears garbled (high non-ASCII ratio)"

    return True, None


def validate_chunk(chunk: ChunkRecord) -> tuple[bool, Optional[str]]:
    """
    Validate a ChunkRecord before embedding.

    Returns:
        (True, None)           — chunk is valid
        (False, reason_str)    — chunk failed validation
    """
    text = chunk.chunk_text

    if not text or not text.strip():
        return False, "chunk_text is empty"

    text_len = len(text.strip())
    if text_len < _CHUNK_MIN_CHARS:
        return False, f"chunk too short ({text_len} chars, min {_CHUNK_MIN_CHARS})"

    if text_len > _CHUNK_MAX_CHARS:
        return False, f"chunk too long ({text_len} chars, max {_CHUNK_MAX_CHARS})"

    if _garble_score(text) > _GARBLE_RATIO:
        return False, "chunk appears garbled (high non-ASCII ratio)"

    # Reject chunks that are mostly digits/punctuation (e.g. reference lists)
    word_chars = re.sub(r"[^a-zA-Z]", "", text)
    if len(word_chars) < text_len * 0.4:
        return False, "chunk has too few alphabetic characters (likely a table or reference list)"

    return True, None


def filter_valid_chunks(
    chunks: list[ChunkRecord],
) -> tuple[list[ChunkRecord], list[dict]]:
    """
    Filter a list of ChunkRecords, returning (valid_chunks, rejected_info).
    rejected_info is a list of dicts with {chunk_index, reason}.
    """
    valid: list[ChunkRecord] = []
    rejected: list[dict] = []
    for chunk in chunks:
        ok, reason = validate_chunk(chunk)
        if ok:
            valid.append(chunk)
        else:
            rejected.append({"chunk_index": chunk.chunk_index, "reason": reason})
    return valid, rejected
