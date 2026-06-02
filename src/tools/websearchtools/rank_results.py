"""rank_results — relevance-based ranking of web search results."""
from __future__ import annotations

from typing import Dict, List


def rank_by_keywords(query: str, results: List[Dict]) -> List[Dict]:
    """Score each result by keyword overlap with title + snippet. Returns sorted list (highest first)."""
    words = set(query.lower().split())
    for r in results:
        title_words = set(r.get("title", "").lower().split())
        snippet_words = set(r.get("snippet", "").lower().split())
        r["relevance_score"] = len(words & title_words) + len(words & snippet_words)
    return sorted(results, key=lambda x: x.get("relevance_score", 0), reverse=True)


def top_n(results: List[Dict], n: int) -> List[Dict]:
    """Return the top-n results (assumes results are already scored/sorted)."""
    return results[:n]
