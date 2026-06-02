"""delete_file — remove files and clean up temp directories.

DESTRUCTIVE. All operations are restricted to the standard allowed roots
(see src.tools.safety.ALLOWED_ROOTS). Operations on paths outside those
roots require `confirm=True` and will log a warning.
"""
from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path

from src.tools.safety import (
    ensure_safe_path,
    is_path_safe,
    require_confirm,
)

logger = logging.getLogger(__name__)

_TEMP_DIR = Path("/tmp") / "openinsight_temp"


async def delete_file(path: str, confirm: bool = False) -> bool:
    """
    Delete a single file. Confirms against allowed roots by default.
    Returns True if deleted, False otherwise.
    """
    p = Path(path)
    if not p.exists():
        return False
    if not p.is_file():
        logger.warning(f"refusing to delete non-file: {path}")
        return False
    try:
        require_confirm("delete file", p, confirm=confirm)
    except PermissionError as e:
        logger.warning(str(e))
        return False
    p.unlink()
    return True


async def delete_directory(path: str, confirm: bool = False) -> bool:
    """
    Recursively delete a directory. Restricted to allowed roots unless
    `confirm=True` is passed.
    """
    p = Path(path)
    if not p.exists():
        return False
    if not p.is_dir():
        logger.warning(f"refusing to delete non-directory: {path}")
        return False
    try:
        require_confirm("delete directory", p, confirm=confirm)
    except PermissionError as e:
        logger.warning(str(e))
        return False
    try:
        ensure_safe_path(p)
    except ValueError as e:
        logger.warning(str(e))
        return False
    shutil.rmtree(p)
    return True


async def cleanup_temp_files(
    max_age_hours: int = 24,
    root: Path = _TEMP_DIR,
    confirm: bool = False,
) -> int:
    """
    Remove temp files older than `max_age_hours` under `root`. By default
    operates only on /tmp/openinsight_temp. Returns count removed.

    Note: Even though this is bounded to `_TEMP_DIR` by default, we still
    require `confirm=True` if the caller passes a non-standard `root`.
    """
    if max_age_hours < 0:
        raise ValueError("max_age_hours must be non-negative")
    if not root.exists():
        return 0
    if not is_path_safe(root):
        try:
            require_confirm("cleanup", root, confirm=confirm)
        except PermissionError as e:
            logger.warning(str(e))
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
