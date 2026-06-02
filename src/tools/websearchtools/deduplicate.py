"""deduplicate — remove duplicate URLs from results."""
from __future__ import annotations

from typing import Dict, List


def deduplicate_by_url(results: List[Dict]) -> List[Dict]:
    """Return a list with unique URLs (trailing slash normalized). Preserves first occurrence."""
    seen: set = set()
    deduped: List[Dict] = []
    for r in results:
        url = (r.get("url", "") or "").rstrip("/")
        if url and url not in seen:
            seen.add(url)
            deduped.append(r)
    return deduped


def deduplicate_by_title(results: List[Dict]) -> List[Dict]:
    """Return a list with unique (case-insensitive) titles."""
    seen: set = set()
    deduped: List[Dict] = []
    for r in results:
        title = (r.get("title", "") or "").lower().strip()
        if title and title not in seen:
            seen.add(title)
            deduped.append(r)
    return deduped
