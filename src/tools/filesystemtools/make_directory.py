"""make_directory — create directories (including nested)."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from src.tools.safety import (
    is_absolute_or_traversal,
    is_path_safe,
    sanitize_directory_name,
)

logger = logging.getLogger(__name__)

_TEMP_DIR = Path("/tmp") / "openinsight_temp"


async def make_dir(name: str, parent: Optional[Path] = None) -> str:
    """
    Create a directory (and parents if needed) under `parent` (default:
    the standard temp dir). Returns the path. Rejects unsafe names.
    """
    if is_absolute_or_traversal(name):
        raise ValueError(f"name must be a relative, non-traversal string, got: {name!r}")
    safe_name = sanitize_directory_name(name)
    base = parent or _TEMP_DIR
    if not is_path_safe(base):
        raise ValueError(f"parent is outside allowed roots: {base}")
    path = base / safe_name
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


async def make_temp_dir(name: str) -> str:
    """Create a subdirectory under the temp dir. Returns the path."""
    return await make_dir(name, parent=_TEMP_DIR)


async def make_reports_dir() -> str:
    """Create and return the standard reports output directory."""
    path = Path("/tmp") / "openinsight_reports"
    if not is_path_safe(path):
        raise ValueError(f"reports dir is outside allowed roots: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return str(path)
