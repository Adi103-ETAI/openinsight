"""
Rate Limiting Middleware — Token bucket rate limiter for API endpoints.
Uses in-memory storage; suitable for single-instance deployments.
"""
from __future__ import annotations

import time
from collections import defaultdict
from typing import Callable

from fastapi import Request, Response
from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse


class RateLimitBucket:
    """Token bucket for a single client."""

    __slots__ = ("capacity", "tokens", "refill_rate", "last_refill")

    def __init__(self, capacity: int, refill_rate: float) -> None:
        self.capacity = capacity
        self.tokens = float(capacity)
        self.refill_rate = refill_rate  # tokens per second
        self.last_refill = time.monotonic()

    def consume(self, now: float | None = None) -> bool:
        """Try to consume one token. Returns True if allowed."""
        now = now or time.monotonic()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now

        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return True
        return False


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting middleware using token bucket algorithm.

    Default: 60 requests per minute per client IP, burst of 10.
    Configurable per-path via the path_limits parameter.
    """

    def __init__(
        self,
        app,
        default_rate: float = 1.0,  # requests per second
        default_capacity: int = 10,  # burst size
        path_limits: dict[str, tuple[float, int]] | None = None,
        excluded_paths: set[str] | None = None,
    ) -> None:
        super().__init__(app)
        self.default_rate = default_rate
        self.default_capacity = default_capacity
        self.path_limits = path_limits or {}
        self.excluded_paths = excluded_paths or {"/health", "/health/detailed", "/health/ready", "/metrics"}
        self._buckets: dict[str, RateLimitBucket] = defaultdict(
            lambda: RateLimitBucket(self.default_capacity, self.default_rate)
        )
        self._last_cleanup = time.monotonic()
        self._cleanup_interval = 60.0  # Clean stale buckets every 60s

    def _get_client_id(self, request: Request) -> str:
        """Extract client identifier from request."""
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    def _get_limits(self, path: str) -> tuple[float, int]:
        """Get rate limits for a given path."""
        for prefix, limits in self.path_limits.items():
            if path.startswith(prefix):
                return limits
        return self.default_rate, self.default_capacity

    def _cleanup_stale_buckets(self) -> None:
        """Remove buckets that haven't been used recently."""
        now = time.monotonic()
        if now - self._last_cleanup < self._cleanup_interval:
            return
        self._last_cleanup = now
        stale_keys = [
            key for key, bucket in self._buckets.items()
            if now - bucket.last_refill > 300  # 5 minutes inactive
        ]
        for key in stale_keys:
            del self._buckets[key]

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip rate limiting for excluded paths
        if request.url.path in self.excluded_paths:
            return await call_next(request)

        # Skip for health checks and docs
        if request.url.path.startswith("/docs") or request.url.path.startswith("/openapi"):
            return await call_next(request)

        client_id = self._get_client_id(request)
        rate, capacity = self._get_limits(request.url.path)
        bucket_key = f"{client_id}:{request.url.path}"

        # Periodically clean up stale buckets
        self._cleanup_stale_buckets()

        # Get or create bucket
        bucket = self._buckets[bucket_key]
        if bucket.capacity != capacity or bucket.refill_rate != rate:
            # Path-specific limits changed; create new bucket
            bucket = RateLimitBucket(capacity, rate)
            self._buckets[bucket_key] = bucket

        if not bucket.consume():
            retry_after = 1.0 / rate
            logger.warning(
                f"Rate limit exceeded: client={client_id} path={request.url.path}"
            )
            return JSONResponse(
                status_code=429,
                content={
                    "detail": "Rate limit exceeded. Please retry after a short delay.",
                    "retry_after_seconds": round(retry_after, 1),
                },
                headers={"Retry-After": str(int(retry_after) + 1)},
            )

        return await call_next(request)
