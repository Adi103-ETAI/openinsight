"""filter_medical — keep only results from trusted medical domains."""
from __future__ import annotations

from typing import Dict, List

from src.tools.websearchtools.extract_domain import extract_domain

_MEDICAL_DOMAINS = {
    "medlineplus.gov", "mayoclinic.org", "cdc.gov", "nih.gov", "who.int",
    "webmd.com", "uptodate.com", "pubmed.ncbi.nlm.nih.gov",
    "pubchem.ncbi.nlm.nih.gov", "cancer.gov", "fda.gov", "ema.europa.eu",
}


def is_medical_domain(url: str) -> bool:
    """Return True if URL is from a known medical authority."""
    return extract_domain(url) in _MEDICAL_DOMAINS


def filter_medical(results: List[Dict]) -> List[Dict]:
    """Tag each result with is_medical/domain and return only the medical ones."""
    for r in results:
        domain = extract_domain(r.get("url", ""))
        r["domain"] = domain
        r["is_medical"] = domain in _MEDICAL_DOMAINS
    return [r for r in results if r.get("is_medical")]


def list_medical_domains() -> List[str]:
    """Return the list of trusted medical domains (for reference)."""
    return sorted(_MEDICAL_DOMAINS)
