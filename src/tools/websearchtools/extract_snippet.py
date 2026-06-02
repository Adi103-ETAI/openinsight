"""extract_snippet — strip HTML tags and create a plain-text snippet."""
from __future__ import annotations

import re


def extract_snippet(html: str, max_len: int = 300) -> str:
    """Strip HTML tags, collapse whitespace, and truncate to max_len."""
    plain = re.sub(r"<[^>]+>", " ", html)
    plain = re.sub(r"\s+", " ", plain).strip()
    if len(plain) <= max_len:
        return plain
    cut = plain[:max_len].rsplit(" ", 1)[0]
    return cut + "..."


def extract_text_blocks(html: str) -> list[str]:
    """Return a list of plain text blocks (split on double newlines / major tags)."""
    plain = re.sub(r"<[^>]+>", " ", html)
    plain = re.sub(r"\s+", " ", plain).strip()
    return [p.strip() for p in re.split(r"\.\s+", plain) if p.strip()]
