"""build_citation_schema — assemble machine-readable citation schema for UI."""
from __future__ import annotations

from typing import Any, Dict, List


def build_citation_schema(claims: List[Dict], sources: List[Dict]) -> List[Dict[str, Any]]:
    """
    Build a UI-friendly citation schema. Skips claims whose source_id is missing.
    Each entry: claim_id, claim_text, source_id, source_type, source_title, source_url, source_excerpt, confidence, status.
    """
    source_map = {s["id"]: s for s in sources}
    schema: List[Dict[str, Any]] = []
    for claim in claims:
        sid = claim.get("source_id", "")
        src = source_map.get(sid)
        if not src:
            continue
        text_field = src.get("text") or src.get("excerpt", "")
        schema.append({
            "claim_id": claim.get("claim_id", ""),
            "claim_text": claim.get("claim_text", ""),
            "source_id": sid,
            "source_type": "corpus" if sid.startswith("CHUNK") else "web",
            "source_title": src.get("title", ""),
            "source_url": src.get("url"),
            "source_excerpt": text_field[:200],
            "confidence": claim.get("confidence", 0.5),
            "status": claim.get("status", "verified"),
        })
    return schema
