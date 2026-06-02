"""write_file — write text or JSON content to a file."""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import aiofiles
except ImportError:
    aiofiles = None

from src.tools.safety import (
    ensure_safe_path,
    is_absolute_or_traversal,
    is_path_safe,
    sanitize_filename,
)

logger = logging.getLogger(__name__)

_DEFAULT_DIR = Path("/tmp") / "openinsight_temp"


async def write_text(
    content: str,
    prefix: str = "file",
    output_dir: Optional[Path] = None,
) -> str:
    """
    Write text content to a file under the standard temp dir (or `output_dir`).
    Returns the file path. Rejects absolute paths in `prefix` and unsafe
    `output_dir` values.
    """
    if is_absolute_or_traversal(prefix):
        raise ValueError(f"prefix must be a relative name, got: {prefix!r}")
    safe_prefix = sanitize_filename(prefix)

    base_dir = Path(output_dir) if output_dir else _DEFAULT_DIR
    if not is_path_safe(base_dir):
        raise ValueError(f"output_dir is outside allowed roots: {base_dir}")
    base_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = base_dir / f"{safe_prefix}_{timestamp}.txt"
    ensure_safe_path(path)

    if aiofiles:
        async with aiofiles.open(path, "w", encoding="utf-8") as f:
            await f.write(content)
    else:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
    logger.info(f"wrote text file: {path}")
    return str(path)


async def write_json(
    data: Dict[str, Any],
    prefix: str = "data",
    output_dir: Optional[Path] = None,
) -> str:
    """Write JSON-serializable data to a file. Returns the file path."""
    blob = json.dumps(data, indent=2, ensure_ascii=False)
    return await write_text(blob, prefix=prefix, output_dir=output_dir)


async def write_bytes(
    content: bytes,
    prefix: str = "blob",
    ext: str = "bin",
    output_dir: Optional[Path] = None,
) -> str:
    """Write raw bytes to a file. Returns the file path."""
    if is_absolute_or_traversal(prefix):
        raise ValueError(f"prefix must be a relative name, got: {prefix!r}")
    if not ext.replace(".", "").isalnum():
        raise ValueError(f"ext must be alphanumeric, got: {ext!r}")
    safe_prefix = sanitize_filename(prefix)

    base_dir = Path(output_dir) if output_dir else _DEFAULT_DIR
    if not is_path_safe(base_dir):
        raise ValueError(f"output_dir is outside allowed roots: {base_dir}")
    base_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = base_dir / f"{safe_prefix}_{timestamp}.{ext}"
    ensure_safe_path(path)

    if aiofiles:
        async with aiofiles.open(path, "wb") as f:
            await f.write(content)
    else:
        with open(path, "wb") as f:
            f.write(content)
    return str(path)
