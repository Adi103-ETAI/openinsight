"""edit_file — in-place editing of file content (append, replace, insert)."""
from __future__ import annotations

import logging
from typing import Optional

try:
    import aiofiles
except ImportError:
    aiofiles = None

from src.tools.safety import is_path_safe

logger = logging.getLogger(__name__)


def _validate(path: str) -> bool:
    if not is_path_safe(path):
        logger.warning(f"refusing to edit outside allowed roots: {path}")
        return False
    return True


async def append_to_file(path: str, content: str) -> bool:
    """Append text to the end of a file. Returns False if path is unsafe."""
    if not _validate(path):
        return False
    if aiofiles:
        async with aiofiles.open(path, "a", encoding="utf-8") as f:
            await f.write(content)
    else:
        with open(path, "a", encoding="utf-8") as f:
            f.write(content)
    return True


async def replace_in_file(path: str, old: str, new: str) -> int:
    """Replace all occurrences of `old` with `new` in file. Returns count replaced."""
    if not _validate(path):
        return 0
    if not old:
        return 0
    try:
        if aiofiles:
            async with aiofiles.open(path, "r", encoding="utf-8") as f:
                content = await f.read()
        else:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
    except FileNotFoundError:
        logger.warning(f"file not found: {path}")
        return 0

    count = content.count(old)
    if count == 0:
        return 0

    new_content = content.replace(old, new)
    if aiofiles:
        async with aiofiles.open(path, "w", encoding="utf-8") as f:
            await f.write(new_content)
    else:
        with open(path, "w", encoding="utf-8") as f:
            f.write(new_content)
    return count


async def insert_at_line(path: str, line_number: int, content: str) -> bool:
    """Insert content at a specific 1-indexed line number. Returns False on safety violation."""
    if not _validate(path):
        return False
    if line_number < 1:
        return False
    if aiofiles:
        async with aiofiles.open(path, "r", encoding="utf-8") as f:
            lines = await f.readlines()
    else:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()

    idx = max(0, min(line_number - 1, len(lines)))
    lines.insert(idx, content if content.endswith("\n") else content + "\n")

    if aiofiles:
        async with aiofiles.open(path, "w", encoding="utf-8") as f:
            await f.writelines(lines)
    else:
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(lines)
    return True
