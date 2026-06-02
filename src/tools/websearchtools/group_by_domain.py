"""group_by_domain — bucket results by their domain."""
from __future__ import annotations

from typing import Dict, List

from src.tools.websearchtools.extract_domain import extract_domain


def group_by_domain(results: List[Dict]) -> Dict[str, List[Dict]]:
    """Return a dict mapping each domain to its list of results."""
    groups: Dict[str, List[Dict]] = {}
    for r in results:
        domain = extract_domain(r.get("url", ""))
        groups.setdefault(domain, []).append(r)
    return groups


def count_per_domain(results: List[Dict]) -> Dict[str, int]:
    """Return a dict mapping each domain to its result count."""
    return {d: len(rs) for d, rs in group_by_domain(results).items()}
