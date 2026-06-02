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

logger = logging.getLogger(__name__)

_TEMP_DIR = Path("/tmp") / "openinsight_temp"
_TEMP_DIR.mkdir(exist_ok=True)


async def write_text(content: str, prefix: str = "file", output_dir: Optional[Path] = None) -> str:
    """Write text content to a file. Returns the file path."""
    base_dir = output_dir or _TEMP_DIR
    base_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = base_dir / f"{prefix}_{timestamp}.txt"
    if aiofiles:
        async with aiofiles.open(path, "w", encoding="utf-8") as f:
            await f.write(content)
    else:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
    logger.info(f"wrote text file: {path}")
    return str(path)


async def write_json(data: Dict[str, Any], prefix: str = "data", output_dir: Optional[Path] = None) -> str:
    """Write JSON-serializable data to a file. Returns the file path."""
    blob = json.dumps(data, indent=2, ensure_ascii=False)
    return await write_text(blob, prefix=prefix, output_dir=output_dir)


async def write_bytes(content: bytes, prefix: str = "blob", ext: str = "bin", output_dir: Optional[Path] = None) -> str:
    """Write raw bytes to a file. Returns the file path."""
    base_dir = output_dir or _TEMP_DIR
    base_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = base_dir / f"{prefix}_{timestamp}.{ext}"
    if aiofiles:
        async with aiofiles.open(path, "wb") as f:
            await f.write(content)
    else:
        with open(path, "wb") as f:
            f.write(content)
    return str(path)
