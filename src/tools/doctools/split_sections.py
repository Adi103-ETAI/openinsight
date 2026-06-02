"""split_sections — parse a clinical answer into labeled sections."""
from __future__ import annotations

from typing import Dict, List

_SECTION_MARKERS = [
    "Diagnosis", "Treatment", "Dosage", "Monitoring",
    "Recommendation", "Summary", "Limitations", "Key Findings",
    "Protocol", "Contraindications", "Side Effects",
]


def split_sections(answer: str) -> Dict[str, str]:
    """
    Split an answer into sections based on labeled headers.
    Falls back to paragraph splitting if no labeled headers are found.
    """
    sections: Dict[str, str] = {}
    current = "summary"
    lines: List[str] = []

    for line in answer.split("\n"):
        stripped = line.strip()
        marker = None
        for m in _SECTION_MARKERS:
            if stripped.lower().startswith(m.lower() + ":"):
                marker = m
                break
        if marker:
            if lines:
                sections[current] = "\n".join(lines).strip()
            current = marker.lower().replace(" ", "_")
            lines = []
        else:
            lines.append(line)

    if lines:
        sections[current] = "\n".join(lines).strip()

    # Fallback: split on double newlines if no labels found
    if len(sections) <= 1 and "summary" in sections:
        paragraphs = [p.strip() for p in answer.split("\n\n") if p.strip()]
        if len(paragraphs) > 1:
            sections = {f"paragraph_{i+1}": p for i, p in enumerate(paragraphs[:5])}
            sections["full_answer"] = answer
        else:
            sections["full_answer"] = answer

    return sections
