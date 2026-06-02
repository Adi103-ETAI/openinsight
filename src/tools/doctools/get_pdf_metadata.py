"""get_pdf_metadata — read metadata (pages, title, author, dates) from a PDF file."""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def get_pdf_metadata(path: str) -> Dict[str, Any]:
    """
    Return a dict of PDF metadata. Falls back to safe defaults if PyPDF2 is missing.
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
            meta = reader.metadata
            out["page_count"] = len(reader.pages)
            if meta:
                out["title"] = meta.title or "Unknown"
                out["author"] = meta.author or "Unknown"
                out["creator"] = meta.creator or "Unknown"
                out["producer"] = meta.producer or "Unknown"
                if meta.creation_date:
                    out["creation_date"] = meta.creation_date.strftime("%Y-%m-%d %H:%M:%S")
                if meta.modification_date:
                    out["modification_date"] = meta.modification_date.strftime("%Y-%m-%d %H:%M:%S")
    except ImportError:
        logger.debug("PyPDF2 not installed; returning defaults")
    except Exception as e:
        logger.warning(f"failed to read PDF metadata for {path}: {e}")
    return out
