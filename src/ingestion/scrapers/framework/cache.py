"""3-tier cache for scraper HTTP responses + raw strings.

Tier L1: in-process LRU (1000 entries, fast)
Tier L2: Redis (shared across workers, 24h TTL)
Tier L3: Filesystem (for Kaggle runs without Redis, JSON-encoded)

The cache is opt-in: callers pass `cache=...` to the HTTP client. If no cache
is provided, fetches always hit the network.

Cache keys are SHA-256 of METHOD + URL + body_hash, truncated to 16 chars.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

from loguru import logger


def _cache_key(method: str, url: str, body: bytes | None = None) -> str:
    """Stable 16-char cache key for a request."""
    h = hashlib.sha256()
    h.update(method.upper().encode())
    h.update(b"\x00")
    h.update(url.encode())
    if body:
        h.update(b"\x00")
        h.update(body)
    return h.hexdigest()[:16]


class MemoryTier:
    """L1: in-process LRU cache (1000 entries)."""

    MAX_ENTRIES = 1000

    def __init__(self) -> None:
        self._data: OrderedDict[str, tuple[float, Any]] = OrderedDict()

    def get(self, key: str) -> Any | None:
        entry = self._data.get(key)
        if entry is None:
            return None
        expires_at, value = entry
        if expires_at and time.time() > expires_at:
            self._data.pop(key, None)
            return None
        self._data.move_to_end(key)
        return value

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        expires_at = (time.time() + ttl) if ttl else 0.0
        self._data[key] = (expires_at, value)
        self._data.move_to_end(key)
        while len(self._data) > self.MAX_ENTRIES:
            self._data.popitem(last=False)

    def clear(self) -> None:
        self._data.clear()


class RedisTier:
    """L2: Redis-backed cache (shared across workers).

    Falls back to no-op if Redis is unreachable — never raises.
    """

    def __init__(self, redis_url: str | None = None) -> None:
        self._redis = None
        if not redis_url:
            return
        try:
            import redis.asyncio as aioredis  # type: ignore
            self._redis = aioredis.from_url(redis_url, decode_responses=False)
        except Exception as e:
            logger.warning(f"[cache:redis] init failed: {e} — L2 disabled")

    async def get(self, key: str) -> bytes | None:
        if self._redis is None:
            return None
        try:
            return await self._redis.get(f"scrape:{key}")
        except Exception as e:
            logger.warning(f"[cache:redis] get failed: {e}")
            return None

    async def set(self, key: str, value: bytes, ttl: int | None = None) -> None:
        if self._redis is None:
            return
        try:
            await self._redis.set(f"scrape:{key}", value, ex=ttl or 86400)
        except Exception as e:
            logger.warning(f"[cache:redis] set failed: {e}")

    async def get_string(self, key: str) -> str | None:
        raw = await self.get(key)
        return raw.decode("utf-8") if raw else None

    async def set_string(self, key: str, value: str, ttl: int | None = None) -> None:
        await self.set(key, value.encode("utf-8"), ttl)

    async def close(self) -> None:
        if self._redis is not None:
            await self._redis.close()


class FilesystemTier:
    """L3: filesystem cache (JSON-encoded, for Kaggle runs without Redis).

    Handles binary content by base64-encoding bytes fields before JSON
    serialization (json.dumps can't serialize bytes directly).
    """

    def __init__(self, root: str = "data/cache/http") -> None:
        self._root = Path(root)
        try:
            self._root.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.warning(f"[cache:fs] could not create {self._root}: {e}")
            self._root = None  # type: ignore

    def get(self, key: str) -> dict | None:
        if self._root is None:
            return None
        path = self._root / f"{key}.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
            expires_at = data.get("expires_at", 0)
            if expires_at and time.time() > expires_at:
                path.unlink(missing_ok=True)
                return None
            payload = data.get("payload", {})
            # Decode any base64-encoded bytes fields
            return self._decode_bytes(payload)
        except Exception as e:
            logger.warning(f"[cache:fs] get failed for {key}: {e}")
            return None

    def set(self, key: str, value: dict, ttl: int | None = None) -> None:
        if self._root is None:
            return
        path = self._root / f"{key}.json"
        expires_at = (time.time() + ttl) if ttl else 0.0
        try:
            encoded = self._encode_bytes(value)
            path.write_text(json.dumps({"expires_at": expires_at, "payload": encoded}))
        except Exception as e:
            logger.warning(f"[cache:fs] set failed for {key}: {e}")

    @staticmethod
    def _encode_bytes(obj: Any) -> Any:
        """Recursively replace bytes with {"__bytes__": "<base64>"} for JSON."""
        if isinstance(obj, bytes):
            import base64
            return {"__bytes__": base64.b64encode(obj).decode("ascii")}
        if isinstance(obj, dict):
            return {k: FilesystemTier._encode_bytes(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [FilesystemTier._encode_bytes(v) for v in obj]
        return obj

    @staticmethod
    def _decode_bytes(obj: Any) -> Any:
        """Reverse of _encode_bytes — restore bytes from {"__bytes__": ...}."""
        if isinstance(obj, dict):
            if "__bytes__" in obj and len(obj) == 1:
                import base64
                return base64.b64decode(obj["__bytes__"])
            return {k: FilesystemTier._decode_bytes(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [FilesystemTier._decode_bytes(v) for v in obj]
        return obj


class ScrapeCache:
    """3-tier cache orchestrator.

    Tiers checked in order: memory → redis → filesystem.
    On hit, value is backfilled to higher tiers.
    """

    def __init__(
        self,
        redis_url: str | None = None,
        fs_root: str = "data/cache/http",
    ) -> None:
        self.memory = MemoryTier()
        self.redis = RedisTier(redis_url)
        self.fs = FilesystemTier(fs_root)
        self._redis_url = redis_url

    def get_cached_response(self, method: str, url: str, body: bytes | None = None) -> dict | None:
        """Synchronous memory/filesystem check (no Redis — that's async)."""
        key = _cache_key(method, url, body)
        # L1
        cached = self.memory.get(key)
        if cached is not None:
            return {**cached, "_cache_layer": "memory"}
        # L3 (sync)
        cached = self.fs.get(key)
        if cached is not None:
            self.memory.set(key, cached)  # backfill L1
            return {**cached, "_cache_layer": "filesystem"}
        return None

    async def get_cached_response_async(self, method: str, url: str, body: bytes | None = None) -> dict | None:
        """Full async tier check: memory → redis → filesystem."""
        key = _cache_key(method, url, body)
        # L1
        cached = self.memory.get(key)
        if cached is not None:
            return {**cached, "_cache_layer": "memory"}
        # L2
        raw = await self.redis.get(key)
        if raw is not None:
            try:
                cached = json.loads(raw)
                self.memory.set(key, cached)  # backfill L1
                return {**cached, "_cache_layer": "redis"}
            except Exception:
                pass
        # L3
        cached = self.fs.get(key)
        if cached is not None:
            self.memory.set(key, cached)  # backfill L1
            return {**cached, "_cache_layer": "filesystem"}
        return None

    def store_response(
        self,
        method: str,
        url: str,
        response: dict,
        ttl: int | None = 86400,
        body: bytes | None = None,
    ) -> None:
        """Store response in L1 + L3 (sync). L2 is async-only — use store_response_async."""
        key = _cache_key(method, url, body)
        self.memory.set(key, response, ttl)
        self.fs.set(key, response, ttl)

    async def store_response_async(
        self,
        method: str,
        url: str,
        response: dict,
        ttl: int | None = 86400,
        body: bytes | None = None,
    ) -> None:
        """Store response in all 3 tiers."""
        key = _cache_key(method, url, body)
        self.memory.set(key, response, ttl)
        await self.redis.set(key, json.dumps(response).encode("utf-8"), ttl)
        self.fs.set(key, response, ttl)

    # Convenience for robots.txt cache (used by robots.py)
    async def get_string(self, key: str) -> str | None:
        return await self.redis.get_string(key)

    async def set_string(self, key: str, value: str, ttl: int | None = None) -> None:
        await self.redis.set_string(key, value, ttl)

    async def close(self) -> None:
        await self.redis.close()
