"""get_pdf_metadata — read metadata (pages, title, author, dates) from a PDF file.

The `PyPDF2` library has historically returned metadata date fields as
either `datetime.datetime` objects, `str`, or `None`, depending on the
version and the source PDF. We defensively coerce each value to a
human-readable string.
"""
from __future__ import annotations

import logging
import os
import re
from datetime import datetime
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Match common PDF date formats: D:YYYYMMDDhhmmss[+HH'mm'] or plain ISO-8601.
# Capture groups: year, month, day, hour, minute, second, tz
_PDF_DATE_RE = re.compile(
    r"^D:(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})([+\-]\d{2}'\d{2}')?$"
)


def _coerce_date(value: Any) -> str:
    """
    Convert a PDF metadata date to an ISO-like string. Falls back to "Unknown"
    if the value is None, an unsupported type, or unparseable.
    """
    if value is None:
        return "Unknown"
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return "Unknown"
        # Try PyPDF2's "D:YYYYMMDDhhmmss..." format
        m = _PDF_DATE_RE.match(s)
        if m:
            year, month, day, hour, minute, second = m.group(1, 2, 3, 4, 5, 6)
            try:
                return f"{year}-{month}-{day} {hour}:{minute}:{second}"
            except ValueError:
                return s
        # Try generic ISO-8601
        try:
            return datetime.fromisoformat(s).strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            return s
    # Some PyPDF2 versions return a "IndirectObject" — convert to str
    try:
        return str(value)
    except Exception:
        return "Unknown"


def _coerce_str(value: Any) -> str:
    if value is None:
        return "Unknown"
    if isinstance(value, str):
        return value or "Unknown"
    try:
        return str(value) or "Unknown"
    except Exception:
        return "Unknown"


def get_pdf_metadata(path: str) -> Dict[str, Any]:
    """
    Return a dict of PDF metadata. Falls back to safe defaults if PyPDF2 is
    missing or the file is unreadable.
    """
    out: Dict[str, Any] = {
        "page_count": 1,
        "title": "Unknown",
        "author": "Unknown",
        "creator": "Unknown",
        "producer": "Unknown",
        "creation_date": "Unknown",
        "modification_date": "Unknown",
        "file_size": os.path.getsize(path) if os.path.exists(path) else 0,
    }
    try:
        import PyPDF2
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            out["page_count"] = len(reader.pages)
            meta = reader.metadata
            if meta:
                out["title"] = _coerce_str(getattr(meta, "title", None))
                out["author"] = _coerce_str(getattr(meta, "author", None))
                out["creator"] = _coerce_str(getattr(meta, "creator", None))
                out["producer"] = _coerce_str(getattr(meta, "producer", None))
                out["creation_date"] = _coerce_date(getattr(meta, "creation_date", None))
                out["modification_date"] = _coerce_date(getattr(meta, "modification_date", None))
    except ImportError:
        logger.debug("PyPDF2 not installed; returning defaults")
    except Exception as e:
        logger.warning(f"failed to read PDF metadata for {path}: {e}")
    return out
