"""save_chunk — persist and load individual chunk dicts as JSON files."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import aiofiles
except ImportError:
    aiofiles = None

from src.tools.filesystemtools.make_directory import make_temp_dir
from src.tools.filesystemtools.read_file import read_json
from src.tools.safety import (
    ensure_safe_path,
    is_path_safe,
    sanitize_filename,
)

logger = logging.getLogger(__name__)


async def save_chunk(data: Dict[str, Any], dir_name: str) -> str:
    """
    Save a chunk dict to a JSON file in dir_name (under the temp dir).
    Returns the file path. Sanitizes both the directory and the chunk id.
    """
    chunk_dir = await make_temp_dir(dir_name)
    chunk_id = data.get("id", "unknown")
    safe_id = sanitize_filename(str(chunk_id))
    path = Path(chunk_dir) / f"{safe_id}.json"
    ensure_safe_path(path)
    blob = json.dumps(data, indent=2, ensure_ascii=False)
    if aiofiles:
        async with aiofiles.open(path, "w", encoding="utf-8") as f:
            await f.write(blob)
    else:
        with open(path, "w", encoding="utf-8") as f:
            f.write(blob)
    return str(path)


async def load_chunk(path: str) -> Optional[Dict[str, Any]]:
    """
    Load a chunk dict from a JSON file. Returns None on failure or if the
    path is outside allowed roots.
    """
    if not is_path_safe(path):
        logger.warning(f"refusing to load chunk outside allowed roots: {path}")
        return None
    return await read_json(path)
