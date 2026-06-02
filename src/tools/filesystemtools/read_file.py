"""read_file — read text or JSON content from a file."""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

try:
    import aiofiles
except ImportError:
    aiofiles = None

logger = logging.getLogger(__name__)


async def read_text(path: str) -> Optional[str]:
    """Read text content from a file. Returns None if not found."""
    try:
        if aiofiles:
            async with aiofiles.open(path, "r", encoding="utf-8") as f:
                return await f.read()
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        logger.warning(f"file not found: {path}")
        return None


async def read_json(path: str) -> Optional[Dict[str, Any]]:
    """Read and parse JSON from a file. Returns None on failure."""
    raw = await read_text(path)
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        logger.warning(f"json decode failed for {path}: {e}")
        return None


async def read_bytes(path: str) -> Optional[bytes]:
    """Read raw bytes from a file."""
    try:
        if aiofiles:
            async with aiofiles.open(path, "rb") as f:
                return await f.read()
        with open(path, "rb") as f:
            return f.read()
    except FileNotFoundError:
        return None
