"""
Date Utility Functions

Shared date parsing and extraction utilities for the OpenInsight codebase.
Handles year extraction from various date formats found in PubMed, WHO, CDC,
and other medical literature sources.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Optional


# Pre-compiled regex patterns for performance
_YEAR_PATTERN = re.compile(r"\b(19|20)\d{2}\b")
_ISO_DATE_PATTERN = re.compile(
    r"(\d{4})-(\d{1,2})-(\d{1,2})"
)  # YYYY-MM-DD
_US_DATE_PATTERN = re.compile(
    r"(\d{1,2})/(\d{1,2})/(\d{4})"
)  # MM/DD/YYYY or DD/MM/YYYY

# Month name to number mapping
_MONTH_MAP: dict[str, int] = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}


def extract_year_from_text(text: str) -> str:
    """
    Extract a 4-digit year from arbitrary text.

    Searches for years in the range 1900-2099. Returns the first match found.

    Args:
        text: Text string that may contain a year.

    Returns:
        Year as a string (e.g., "2023"), or empty string if no year found.

    Examples:
        >>> extract_year_from_text("2023 Jan-Feb")
        '2023'
        >>> extract_year_from_text("Published in 1998")
        '1998'
        >>> extract_year_from_text("No date available")
        ''
    """
    if not text:
        return ""

    match = _YEAR_PATTERN.search(text)
    return match.group(0) if match else ""


def parse_pubmed_date(
    pub_date: dict,
) -> tuple[str, Optional[int]]:
    """
    Parse a PubMed-style PubDate dict into year string and integer.

    Handles the various PubDate structures returned by Entrez XML:
    - {'Year': '2023', 'Month': 'Jan', 'Day': '15'}
    - {'MedlineDate': '2023 Jan-Feb'}
    - {'Year': '2023', 'Season': 'Spring'}
    - {'MedlineDate': '2023'}

    Args:
        pub_date: Dict from PubMed XML PubDate element.

    Returns:
        Tuple of (year_string, year_int). Both may be empty/None.
    """
    if not isinstance(pub_date, dict):
        return "", None

    # Try direct Year field first
    year_str = str(pub_date.get("Year", "")).strip()
    if year_str and year_str.isdigit():
        return year_str, int(year_str)

    # Fall back to MedlineDate
    medline_date = str(pub_date.get("MedlineDate", "")).strip()
    if medline_date:
        year_str = extract_year_from_text(medline_date)
        if year_str:
            return year_str, int(year_str)

    return "", None


def parse_date_string(date_str: str) -> tuple[str, Optional[int]]:
    """
    Parse a date string into year string and integer.

    Handles multiple formats:
    - ISO: "2023-01-15"
    - US: "01/15/2023"
    - Free text: "January 2023", "2023 Jan-Feb"
    - Year only: "2023"

    Args:
        date_str: Date string in any common format.

    Returns:
        Tuple of (year_string, year_int). Both may be empty/None.
    """
    if not date_str:
        return "", None

    date_str = date_str.strip()

    # ISO format: YYYY-MM-DD
    match = _ISO_DATE_PATTERN.match(date_str)
    if match:
        year = match.group(1)
        return year, int(year)

    # US format: MM/DD/YYYY or DD/MM/YYYY
    match = _US_DATE_PATTERN.match(date_str)
    if match:
        year = match.group(3)
        return year, int(year)

    # Free text with year
    year = extract_year_from_text(date_str)
    if year:
        return year, int(year)

    return "", None


def is_valid_year(year: str | int) -> bool:
    """
    Check if a year value is within a reasonable range.

    Args:
        year: Year as string or integer.

    Returns:
        True if year is between 1800 and current year + 1.
    """
    try:
        y = int(year)
        current_year = datetime.now().year
        return 1800 <= y <= current_year + 1
    except (ValueError, TypeError):
        return False


def normalize_year(year: str | int | None) -> Optional[int]:
    """
    Normalize a year value to an integer, or None if invalid.

    Args:
        year: Year as string, integer, or None.

    Returns:
        Integer year if valid, None otherwise.
    """
    if year is None:
        return None

    try:
        y = int(year)
        return y if is_valid_year(y) else None
    except (ValueError, TypeError):
        return None
