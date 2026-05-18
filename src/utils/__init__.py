"""
Shared utilities for the OpenInsight codebase.
"""

from src.utils.date_utils import (
    extract_year_from_text,
    is_valid_year,
    normalize_year,
    parse_date_string,
    parse_pubmed_date,
)
from src.utils.pubmed_client import PubMedArticle, PubMedClient
from src.utils.text_utils import (
    calculate_garble_ratio,
    clean_text,
    contains_any,
    extract_keywords_from_query,
    extract_text_chunks,
    is_text_quality_acceptable,
    normalize_whitespace,
    remove_punctuation,
    truncate_text,
)

__all__ = [
    # PubMed client
    "PubMedClient",
    "PubMedArticle",
    # Date utilities
    "extract_year_from_text",
    "parse_pubmed_date",
    "parse_date_string",
    "is_valid_year",
    "normalize_year",
    # Text utilities
    "extract_keywords_from_query",
    "calculate_garble_ratio",
    "is_text_quality_acceptable",
    "clean_text",
    "extract_text_chunks",
    "truncate_text",
    "normalize_whitespace",
    "contains_any",
    "remove_punctuation",
]
