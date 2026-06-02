"""generate_filename — produce a safe, timestamped filename for reports."""
from __future__ import annotations

from datetime import datetime


def generate_filename(title: str, ext: str = "pdf") -> str:
    """
    Return a filename of the form openinsight_<safe-title>_<timestamp>.<ext>.
    Strips characters that aren't alphanumeric, space, dash, or underscore.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = "".join(c for c in title if c.isalnum() or c in " -_").strip()
    safe = safe.replace(" ", "_")
    if not safe:
        safe = "report"
    return f"openinsight_{safe}_{ts}.{ext}"
