"""list_files — list and filter files in a directory."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

from src.tools.safety import is_path_safe

logger = logging.getLogger(__name__)


async def list_files(dir_path: str, pattern: str = "*", recursive: bool = False) -> List[str]:
    """List files in a directory matching pattern. Returns sorted list of paths."""
    root = Path(dir_path)
    if not is_path_safe(root):
        logger.warning(f"refusing to list outside allowed roots: {dir_path}")
        return []
    if not root.exists() or not root.is_dir():
        return []
    if recursive:
        return sorted(str(p) for p in root.rglob(pattern))
    return sorted(str(p) for p in root.glob(pattern))


async def list_by_extension(dir_path: str, ext: str, recursive: bool = False) -> List[str]:
    """List files with a specific extension (e.g. '.pdf', '.json')."""
    if not ext.startswith("."):
        ext = "." + ext
    return await list_files(dir_path, pattern=f"*{ext}", recursive=recursive)


async def get_file_size(path: str) -> int:
    """Return file size in bytes. Returns 0 if not found or path unsafe."""
    p = Path(path)
    if not is_path_safe(p):
        logger.warning(f"refusing to stat outside allowed roots: {path}")
        return 0
    if p.exists() and p.is_file():
        return p.stat().st_size
    return 0


async def get_file_info(path: str) -> Optional[dict]:
    """Return basic file info (size, modified, created) or None."""
    p = Path(path)
    if not is_path_safe(p):
        logger.warning(f"refusing to stat outside allowed roots: {path}")
        return None
    if not p.exists():
        return None
    stat = p.stat()
    return {
        "path": str(p),
        "size": stat.st_size,
        "modified": stat.st_mtime,
        "created": stat.st_ctime,
        "is_file": p.is_file(),
        "is_dir": p.is_dir(),
    }
