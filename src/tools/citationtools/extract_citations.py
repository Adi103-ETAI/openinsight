"""extract_citations — parse inline citation markers from text."""
from __future__ import annotations

import re
from typing import Dict, List

_CHUNK_RE = re.compile(r"\[CHUNK_(\d+)\]")
_WEB_RE = re.compile(r"\[WEB_(\d+)\]")
_COMBINED_RE = re.compile(r"\[(CHUNK|WEB)_(\d+)\]")


def extract_chunk_ids(text: str) -> List[str]:
    """Return all CHUNK citation IDs found in text (zero-padded to 3 digits)."""
    return [f"CHUNK_{int(m):03d}" for m in _CHUNK_RE.findall(text)]


def extract_web_ids(text: str) -> List[str]:
    """Return all WEB citation IDs found in text (zero-padded to 3 digits)."""
    return [f"WEB_{int(m):03d}" for m in _WEB_RE.findall(text)]


def extract_all_citations(text: str) -> Dict[str, List[str]]:
    """Return a dict with chunk_citations and web_citations lists."""
    return {
        "chunk_citations": extract_chunk_ids(text),
        "web_citations": extract_web_ids(text),
    }


def extract_citation_markers(text: str) -> List[Dict[str, str]]:
    """Return ordered list of citation marker dicts: {type, id, raw}."""
    return [
        {
            "type": "corpus" if m.group(1) == "CHUNK" else "web",
            "id": f"{m.group(1)}_{int(m.group(2)):03d}",
            "raw": m.group(0),
        }
        for m in _COMBINED_RE.finditer(text)
    ]
