"""make_directory — create directories (including nested)."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_TEMP_DIR = Path("/tmp") / "openinsight_temp"


async def make_dir(name: str, parent: Optional[Path] = None) -> str:
    """Create a directory (and parents if needed). Returns the path."""
    base = parent or _TEMP_DIR
    path = base / name
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


async def make_temp_dir(name: str) -> str:
    """Create a subdirectory under the temp dir. Returns the path."""
    return await make_dir(name, parent=_TEMP_DIR)


async def make_reports_dir() -> str:
    """Create and return the standard reports output directory."""
    path = Path("/tmp") / "openinsight_reports"
    path.mkdir(parents=True, exist_ok=True)
    return str(path)
