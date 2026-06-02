"""extract_domain — parse the domain from a URL."""
from __future__ import annotations

from urllib.parse import urlparse


def extract_domain(url: str) -> str:
    """Return the netloc of a URL, stripping 'www.' prefix. Returns 'unknown' on failure or empty netloc."""
    if not url:
        return "unknown"
    try:
        netloc = urlparse(url).netloc
        if not netloc:
            return "unknown"
        return netloc.replace("www.", "")
    except Exception:
        return "unknown"


def is_same_domain(url_a: str, url_b: str) -> bool:
    """Return True if both URLs share the same domain."""
    return extract_domain(url_a) == extract_domain(url_b)
