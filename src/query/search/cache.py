from __future__ import annotations

import hashlib
import json
from enum import Enum
from typing import Any

import redis.asyncio as aioredis

from src.core.config import get_settings


class SearchCache:
    CACHE_VERSION = "v2"

    def __init__(self, redis_url: str | None = None):
        settings = get_settings()
        self.cache_version = settings.cache_version or self.CACHE_VERSION
        self.ttl_search = settings.cache_ttl_search
        self.ttl_rerank = settings.cache_ttl_rerank
        self.key_prefix_length = settings.cache_key_prefix_length
        self.redis = aioredis.from_url(
            redis_url or settings.redis_url,
            encoding="utf-8",
            decode_responses=True,
        )

    def _make_key(self, operation: str, *components: Any) -> str:
        content = "|".join(str(c) for c in components)
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:self.key_prefix_length]
        return f"openinsight:{self.cache_version}:{operation}:{digest}"

    async def get_search_result(
        self, query: str, filters: Any
    ) -> dict[str, Any] | None:
        filters_json = json.dumps(self._json_safe(filters), sort_keys=True)
        key = self._make_key("search", query.lower().strip(), filters_json)
        cached = await self.redis.get(key)
        return json.loads(cached) if cached else None

    async def set_search_result(
        self,
        query: str,
        filters: Any,
        result: dict[str, Any],
        ttl: int | None = None,
    ) -> None:
        filters_json = json.dumps(self._json_safe(filters), sort_keys=True)
        key = self._make_key("search", query.lower().strip(), filters_json)
        await self.redis.setex(
            key,
            ttl if ttl is not None else self.ttl_search,
            json.dumps(self._json_safe(result)),
        )

    async def get_query_embedding(self, query: str) -> list[float] | None:
        """Cache query embedding for faster repeated queries."""
        key = self._make_key("embed", query.lower().strip())
        cached = await self.redis.get(key)
        return json.loads(cached) if cached else None

    async def set_query_embedding(
        self, query: str, embedding: list[float], ttl: int | None = None
    ) -> None:
        """Cache query embedding."""
        key = self._make_key("embed", query.lower().strip())
        await self.redis.setex(
            key,
            ttl if ttl is not None else self.ttl_search,
            json.dumps(embedding),
        )

    async def get_reranked(
        self, query: str, chunk_ids: list[str]
    ) -> list[dict[str, Any]] | None:
        key = self._make_key(
            "rerank", query.lower().strip(), "|".join(sorted(chunk_ids))
        )
        cached = await self.redis.get(key)
        return json.loads(cached) if cached else None

    async def set_reranked(
        self,
        query: str,
        chunk_ids: list[str],
        reranked: list[dict[str, Any]],
        ttl: int | None = None,
    ) -> None:
        key = self._make_key(
            "rerank", query.lower().strip(), "|".join(sorted(chunk_ids))
        )
        await self.redis.setex(
            key,
            ttl if ttl is not None else self.ttl_rerank,
            json.dumps(self._json_safe(reranked)),
        )

    async def invalidate_all(self) -> int:
        pattern = f"openinsight:{self.cache_version}:*"
        keys = [key async for key in self.redis.scan_iter(match=pattern)]
        if not keys:
            return 0
        return await self.redis.delete(*keys)

    def _json_safe(self, value: Any) -> Any:
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, dict):
            return {k: self._json_safe(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._json_safe(v) for v in value]
        if isinstance(value, tuple):
            return [self._json_safe(v) for v in value]
        if hasattr(value, "dict"):
            return self._json_safe(value.dict())
        if hasattr(value, "__dict__") and not isinstance(
            value, (str, int, float, bool, type(None))
        ):
            return self._json_safe(vars(value))
        return value
