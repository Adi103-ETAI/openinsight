"""
Text Utility Functions

Shared text processing utilities for the OpenInsight codebase.
Handles keyword extraction, text quality assessment, and common
text transformations used across ingestion parsers.
"""

from __future__ import annotations

import re
import string
from typing import Iterable


# Pre-compiled regex patterns for performance
_WORD_PATTERN = re.compile(r"[A-Za-z0-9\-]+")
_NON_ASCII_PATTERN = re.compile(r"[^\x00-\x7F]+")
_MULTI_SPACE_PATTERN = re.compile(r"\s{2,}")
_HTML_TAG_PATTERN = re.compile(r"<[^>]+>")

# Common English stop words and query words that don't make good tags
_COMMON_STOP_WORDS: set[str] = {
    # Generic medical query words
    "india", "treatment", "management", "clinical", "study",
    "research", "review", "analysis", "report", "data",
    # Common English stop words
    "the", "a", "an", "and", "or", "but", "in", "on", "at",
    "to", "for", "of", "with", "by", "from", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "will", "would", "could", "should",
    "may", "might", "must", "shall", "can", "need", "dare",
    "it", "its", "this", "that", "these", "those", "i", "you",
    "he", "she", "we", "they", "what", "which", "who", "whom",
    "whose", "where", "when", "why", "how", "all", "each",
    "every", "both", "few", "more", "most", "other", "some",
    "such", "no", "nor", "not", "only", "own", "same", "so",
    "than", "too", "very", "just", "about", "above", "below",
    "between", "into", "through", "during", "before", "after",
    "up", "down", "out", "off", "over", "under", "again",
    "further", "then", "once", "here", "there",
}


def extract_keywords_from_query(
    query: str,
    min_length: int = 4,
    exclude: set[str] | None = None,
) -> list[str]:
    """
    Extract meaningful keywords/tags from a search query string.

    Tokenizes the query, filters out short tokens and common stop words,
    and returns unique keywords in order of appearance.

    Args:
        query: The search query string.
        min_length: Minimum token length to include (default: 4).
        exclude: Additional words to exclude (merged with default stop words).

    Returns:
        List of unique keyword strings.

    Examples:
        >>> extract_keywords_from_query("malaria treatment in India")
        ['malaria']
        >>> extract_keywords_from_query("diabetes management clinical study")
        ['diabetes']
    """
    tokens = _WORD_PATTERN.findall(query.lower())
    exclude_set = _COMMON_STOP_WORDS | (exclude or set())

    seen: set[str] = set()
    keywords: list[str] = []

    for token in tokens:
        if len(token) < min_length:
            continue
        if token in exclude_set:
            continue
        if token not in seen:
            seen.add(token)
            keywords.append(token)

    return keywords


def calculate_garble_ratio(text: str) -> float:
    """
    Calculate the ratio of non-text / garbled characters in a string.

    Measures the proportion of non-ASCII characters and control characters
    that might indicate OCR errors, encoding issues, or binary content.

    Args:
        text: The text string to analyze.

    Returns:
        Float between 0.0 (clean) and 1.0 (heavily garbled).

    Examples:
        >>> calculate_garble_ratio("Hello world")
        0.0
        >>> calculate_garble_ratio("H\x00e\x01llo")
        0.18...
    """
    if not text:
        return 0.0

    total_chars = len(text)
    garbled_chars = 0

    for char in text:
        # Control characters (except common whitespace)
        if ord(char) < 32 and char not in ("\n", "\r", "\t"):
            garbled_chars += 1
        # Non-ASCII characters
        elif ord(char) > 127:
            garbled_chars += 1

    return garbled_chars / total_chars


def is_text_quality_acceptable(
    text: str,
    max_garble_ratio: float = 0.15,
    min_length: int = 10,
) -> bool:
    """
    Determine if text quality is acceptable for processing.

    Args:
        text: The text to evaluate.
        max_garble_ratio: Maximum acceptable garble ratio (0.0-1.0).
        min_length: Minimum acceptable text length.

    Returns:
        True if text passes quality checks.
    """
    if len(text) < min_length:
        return False

    garble_ratio = calculate_garble_ratio(text)
    return garble_ratio <= max_garble_ratio


def clean_text(text: str) -> str:
    """
    Clean text by removing HTML tags, normalizing whitespace, and stripping.

    Args:
        text: Raw text that may contain HTML or extra whitespace.

    Returns:
        Cleaned text string.
    """
    if not text:
        return ""

    # Remove HTML tags
    text = _HTML_TAG_PATTERN.sub("", text)

    # Normalize whitespace
    text = _MULTI_SPACE_PATTERN.sub(" ", text)

    return text.strip()


def extract_text_chunks(text: str, max_chunk_size: int = 1000) -> list[str]:
    """
    Split text into chunks of approximately max_chunk_size characters.

    Tries to split on sentence boundaries when possible.

    Args:
        text: The text to split.
        max_chunk_size: Maximum characters per chunk.

    Returns:
        List of text chunks.
    """
    if not text:
        return []

    if len(text) <= max_chunk_size:
        return [text.strip()]

    # Split on sentence boundaries
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_chunk_size:
            current_chunk += (" " if current_chunk else "") + sentence
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def truncate_text(text: str, max_length: int = 500, suffix: str = "...") -> str:
    """
    Truncate text to a maximum length, preserving word boundaries.

    Args:
        text: The text to truncate.
        max_length: Maximum length including suffix.
        suffix: Suffix to append if truncated.

    Returns:
        Truncated text string.
    """
    if not text or len(text) <= max_length:
        return text

    # Find last space before the limit
    cutoff = max_length - len(suffix)
    truncated = text[:cutoff]
    last_space = truncated.rfind(" ")

    if last_space > 0:
        truncated = truncated[:last_space]

    return truncated + suffix


def normalize_whitespace(text: str) -> str:
    """
    Normalize all whitespace in text to single spaces.

    Args:
        text: Text with potentially irregular whitespace.

    Returns:
        Text with normalized whitespace.
    """
    return _MULTI_SPACE_PATTERN.sub(" ", text).strip()


def contains_any(text: str, patterns: Iterable[str]) -> bool:
    """
    Check if text contains any of the given patterns (case-insensitive).

    Args:
        text: The text to search.
        patterns: Iterable of pattern strings to search for.

    Returns:
        True if any pattern is found in the text.
    """
    text_lower = text.lower()
    return any(p.lower() in text_lower for p in patterns)


def remove_punctuation(text: str, keep: str = "") -> str:
    """
    Remove punctuation from text, optionally keeping specified characters.

    Args:
        text: The text to process.
        keep: String of punctuation characters to preserve.

    Returns:
        Text with punctuation removed.
    """
    exclude = set(string.punctuation) - set(keep)
    return "".join(char for char in text if char not in exclude)
