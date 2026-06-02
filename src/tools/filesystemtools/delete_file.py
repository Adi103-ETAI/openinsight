"""delete_file — remove files and clean up temp directories."""
from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path

logger = logging.getLogger(__name__)

_TEMP_DIR = Path("/tmp") / "openinsight_temp"


async def delete_file(path: str) -> bool:
    """Delete a single file. Returns True if deleted, False otherwise."""
    p = Path(path)
    if not p.exists():
        return False
    if p.is_file():
        p.unlink()
        return True
    return False


async def delete_directory(path: str) -> bool:
    """Recursively delete a directory. Returns True if deleted."""
    p = Path(path)
    if not p.exists():
        return False
    if p.is_dir():
        shutil.rmtree(p)
        return True
    return False


async def cleanup_temp_files(max_age_hours: int = 24, root: Path = _TEMP_DIR) -> int:
    """Remove temp files older than max_age_hours. Returns count removed."""
    if not root.exists():
        return 0
    cutoff = time.time() - max_age_hours * 3600
    removed = 0
    for p in root.rglob("*"):
        if p.is_file() and p.stat().st_mtime < cutoff:
            try:
                p.unlink()
                removed += 1
            except OSError as e:
                logger.warning(f"failed to delete {p}: {e}")
    return removed
