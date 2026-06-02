"""safety — path validation, filename sanitization, and allowed-roots enforcement.

Centralizes defensive checks for the filesystem tools so each individual tool
file stays small. Every tool that accepts a path or a directory name should
import from here.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Iterable, List, Optional

logger = logging.getLogger(__name__)


# Standard allowed roots for the system. Tools SHOULD confine writes/deletes
# to these. Callers can pass an explicit `allowed_roots=` to extend the list.
ALLOWED_ROOTS = [
    Path("/tmp") / "openinsight_temp",
    Path("/tmp") / "openinsight_reports",
    Path("/tmp"),  # broadest fallback for any sub-dir of /tmp
]

# Filename-friendly character set. Anything outside is replaced with '_'.
_FILENAME_SAFE_RE = re.compile(r"[^A-Za-z0-9._\-]")


def sanitize_filename(name: str, max_length: int = 200) -> str:
    """
    Strip directory separators, NUL bytes, and control characters from a name.
    Replaces unsafe characters with '_'. Truncates to max_length.
    """
    if not name:
        return "unnamed"
    # Strip path separators and NUL
    name = name.replace("/", "_").replace("\\", "_").replace("\x00", "")
    # Replace other unsafe characters
    name = _FILENAME_SAFE_RE.sub("_", name)
    # Collapse repeated underscores
    name = re.sub(r"_+", "_", name).strip("._-")
    if not name:
        name = "unnamed"
    if len(name) > max_length:
        name = name[:max_length].rstrip("._-")
    return name


def sanitize_directory_name(name: str) -> str:
    """Like sanitize_filename but stricter — no dots, no dashes at boundaries."""
    s = sanitize_filename(name)
    # Strip leading dots to avoid hidden dirs
    s = s.lstrip(".")
    if not s:
        s = "unnamed"
    return s


def is_path_safe(path, allowed_roots: Optional[Iterable[Path]] = None) -> bool:
    """
    Return True if `path` resolves to a location under one of the allowed roots.
    Accepts either a `str` or a `Path`. Prevents accidental writes/deletes
    outside the project's working area.
    """
    p = path if isinstance(path, Path) else Path(path)
    try:
        resolved = p.resolve(strict=False)
    except (OSError, RuntimeError) as e:
        logger.warning(f"failed to resolve {p}: {e}")
        return False

    roots = [Path(r).resolve(strict=False) for r in (allowed_roots or ALLOWED_ROOTS)]
    for root in roots:
        try:
            resolved.relative_to(root)
            return True
        except ValueError:
            continue
    return False


def ensure_safe_path(
    path, allowed_roots: Optional[Iterable[Path]] = None
) -> Path:
    """
    Validate that `path` is under an allowed root, then return the resolved Path.
    Accepts either a `str` or a `Path`. Raises ValueError if outside allowed roots.
    """
    p = path if isinstance(path, Path) else Path(path)
    if not is_path_safe(p, allowed_roots):
        raise ValueError(
            f"refusing to operate on path outside allowed roots: {p}"
        )
    return p.resolve(strict=False)


def is_absolute_or_traversal(path_str: str) -> bool:
    """
    Return True if the path is an absolute path or attempts parent traversal.
    Used to reject user-supplied names that try to escape their sandbox.
    """
    if not path_str:
        return True
    if path_str.startswith("/") or path_str.startswith("\\"):
        return True
    if path_str.startswith("~"):
        return True
    # Reject any path that contains '..' as a path segment
    parts = path_str.replace("\\", "/").split("/")
    if any(p == ".." for p in parts):
        return True
    return False


def require_confirm(
    operation: str, path, confirm: bool
) -> None:
    """
    Guard for destructive operations on paths outside the standard allowed roots.
    Accepts either a `str` or a `Path`. Raises PermissionError unless `confirm=True`.
    """
    p = path if isinstance(path, Path) else Path(path)
    if is_path_safe(p):
        return
    if not confirm:
        raise PermissionError(
            f"refusing to {operation} {p}: outside allowed roots. "
            f"Pass confirm=True to override."
        )
    logger.warning(
        f"performing {operation} on path outside allowed roots: {p} (explicit confirm)"
    )
