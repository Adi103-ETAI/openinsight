"""Raw lake storage: scrapers collect, ingestion transforms.

Layout (per docs/RAW_LAKE.md sections 1-2):
    data/raw/{source}/{doc_id}.bin   # raw bytes exactly as fetched
    data/raw/{source}/manifest.jsonl  # ONE JSON object per line (append-only)

Manifest schema keys (exactly): doc_id, source, url, fetched_at (UTC ISO),
sha256 (of .bin bytes), content_type, license, title, error.

Stdlib only — no torch, no pandas, no settings import.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

__all__ = [
    "RAW_ROOT",
    "MANIFEST_FIELDS",
    "get_raw_root",
    "sanitize_doc_id",
    "write_raw",
    "iter_raw",
    "find_by_hash",
    "lake_stats",
]

MANIFEST_FIELDS = (
    "doc_id",
    "source",
    "url",
    "fetched_at",
    "sha256",
    "content_type",
    "license",
    "title",
    "error",
)

# Default: <repo>/data/raw  (this file is <repo>/src/ingestion/raw_lake.py)
_DEFAULT_RAW_ROOT = Path(__file__).resolve().parents[2] / "data" / "raw"

# Import-time default; per-call resolution honours OPENINSIGHT_RAW_ROOT env
# overrides AND monkeypatching of this name in tests.
RAW_ROOT: Path = Path(os.environ.get("OPENINSIGHT_RAW_ROOT", _DEFAULT_RAW_ROOT))

_ENV_VAR = "OPENINSIGHT_RAW_ROOT"

_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]")


def get_raw_root() -> Path:
    """Resolve the lake root: env override wins, else module RAW_ROOT."""
    override = os.environ.get(_ENV_VAR)
    if override:
        return Path(override)
    return Path(RAW_ROOT)


def sanitize_doc_id(doc_id: str) -> str:
    """Map an arbitrary doc_id to a safe single-path-component filename stem.

    Replaces every char outside [A-Za-z0-9._-] with '_', so URL-like ids
    (``https://...``) and traversal attempts (``..``, ``../x``) can never
    produce a path separator and escape the source dir. Reserved stems
    ('', '.', '..') and leading dots are replaced with a stable hash fallback.
    """
    raw = str(doc_id)
    safe = _SAFE_RE.sub("_", raw)
    # Strip leading dots so ".hidden"/".."-style stems can't hide or climb.
    safe = safe.lstrip(".")
    if not safe or safe in (".", ".."):
        safe = "doc_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    # Cap length to stay well under filesystem limits.
    if len(safe) > 200:
        safe = safe[:184] + "_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:15]
    return safe


def _source_dir(source: str) -> Path:
    # Sanitize source too so it stays a single path component.
    safe_source = _SAFE_RE.sub("_", str(source)).strip(".") or "unknown"
    return get_raw_root() / safe_source


def _manifest_path(source: str) -> Path:
    return _source_dir(source) / "manifest.jsonl"


def _coerce_fetched_at(value: Any) -> str:
    if isinstance(value, datetime):
        dt = value
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat()
    if isinstance(value, str) and value:
        return value
    return datetime.now(timezone.utc).isoformat()


def write_raw(source: str, doc_id: str, content: bytes, meta: dict | None = None) -> dict:
    """Persist raw bytes + append ONE manifest.jsonl line. Return the entry.

    Best-effort caller contract: raises only on real I/O failure; callers
    doing best-effort lake writes (e.g. BaseScraper.fetch_one) must try/except.
    """
    meta = dict(meta or {})
    if not isinstance(content, (bytes, bytearray)):
        raise TypeError(f"content must be bytes, got {type(content).__name__}")
    content = bytes(content)
    source = str(source)
    doc_id = str(doc_id)

    digest = hashlib.sha256(content).hexdigest()
    entry: dict[str, Any] = {
        "doc_id": doc_id,
        "source": source,
        "url": str(meta.get("url") or ""),
        "fetched_at": _coerce_fetched_at(meta.get("fetched_at")),
        "sha256": digest,
        "content_type": str(meta.get("content_type") or ""),
        "license": str(meta.get("license") or ""),
        "title": str(meta.get("title") or ""),
        "error": meta.get("error"),
    }

    source_dir = _source_dir(source)
    source_dir.mkdir(parents=True, exist_ok=True)
    bin_path = source_dir / (sanitize_doc_id(doc_id) + ".bin")
    bin_path.write_bytes(content)

    manifest = _manifest_path(source)
    line = json.dumps(entry, ensure_ascii=False)
    # Atomic-ish append: single write() call in append mode.
    with open(manifest, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    return entry


def iter_raw(source: str, limit: int = 0) -> Iterator[tuple[dict, bytes]]:
    """Yield (meta, content_bytes) pairs from a source manifest in order.

    limit=0 means no limit. Missing .bin files or corrupt manifest lines
    are skipped (audit trail stays append-only; readers stay tolerant).
    """
    manifest = _manifest_path(str(source))
    if not manifest.is_file():
        return
    count = 0
    with open(manifest, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                meta = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(meta, dict):
                continue
            try:
                bin_path = _source_dir(str(source)) / (sanitize_doc_id(str(meta.get("doc_id", ""))) + ".bin")
                content = bin_path.read_bytes()
            except OSError:
                continue
            yield meta, content
            count += 1
            if limit and count >= limit:
                return


def find_by_hash(sha256: str) -> dict | None:
    """Scan all source manifests for a sha256 hex digest. Return meta or None."""
    target = str(sha256).lower()
    root = get_raw_root()
    if not root.is_dir():
        return None
    for manifest in sorted(root.glob("*/manifest.jsonl")):
        try:
            with open(manifest, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        meta = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(meta, dict) and str(meta.get("sha256", "")).lower() == target:
                        return meta
        except OSError:
            continue
    return None


def lake_stats(source: str | None = None) -> dict:
    """Return {source: {'docs': n, 'bytes': total_bin_bytes}}.

    With source given, the dict holds just that source (possibly zeros).
    Counts manifest lines; sizes .bin files on disk (missing files count 0).
    """
    root = get_raw_root()
    sources: list[str]
    if source is not None:
        sources = [str(source)]
    elif root.is_dir():
        sources = sorted(p.name for p in root.iterdir() if p.is_dir())
    else:
        return {}
    stats: dict[str, dict[str, int]] = {}
    for src in sources:
        docs = 0
        total = 0
        manifest = _manifest_path(src)
        if manifest.is_file():
            try:
                with open(manifest, encoding="utf-8") as fh:
                    for line in fh:
                        if line.strip():
                            docs += 1
            except OSError:
                docs = 0
        src_dir = _source_dir(src)
        if src_dir.is_dir():
            for bin_path in src_dir.glob("*.bin"):
                try:
                    total += bin_path.stat().st_size
                except OSError:
                    continue
        stats[src] = {"docs": docs, "bytes": total}
    return stats
