"""format_citations — apply formatting rules to inline citation markers."""
from __future__ import annotations

import re

_CITATION_RE = re.compile(r"\[(CHUNK|WEB)_(\d+)\]")


def format_citations_inline(text: str) -> str:
    """Normalize citation markers (no-op for now; placeholder for future formatting)."""
    return _CITATION_RE.sub(lambda m: f"[{m.group(1)}_{m.group(2)}]", text)


def count_citations(text: str) -> dict:
    """Return a dict with counts of chunk vs web citations in text."""
    return {
        "chunk_count": len(re.findall(r"\[CHUNK_(\d+)\]", text)),
        "web_count": len(re.findall(r"\[WEB_(\d+)\]", text)),
    }
