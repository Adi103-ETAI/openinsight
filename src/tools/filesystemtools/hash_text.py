"""hash_text — compute hashes (SHA256, MD5) for strings and files."""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def hash_string(text: str, algorithm: str = "sha256") -> str:
    """Return hex digest of a string using the given algorithm."""
    h = hashlib.new(algorithm)
    h.update(text.encode("utf-8"))
    return h.hexdigest()


async def hash_file(path: str, algorithm: str = "sha256") -> Optional[str]:
    """Return hex digest of a file. Returns None if file not found."""
    p = Path(path)
    if not p.exists() or not p.is_file():
        logger.warning(f"file not found: {path}")
        return None
    h = hashlib.new(algorithm)
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def cache_key(query: str, prefix: str = "oi") -> str:
    """Generate a short cache key (16 chars) from a query string."""
    return f"{prefix}_{hash_string(query.lower())[:16]}"
