"""Per-domain token-bucket rate limiter.

Politeness is enforced per-domain: each domain gets its own token bucket with
a configurable refill rate (default 1 token/sec). On Kaggle (no Redis), falls
back to in-process buckets.

Buckets are created on first request and persist for the process lifetime.
Optional Redis backing distributes the limit across multiple workers — but
note that distributed token buckets have inherent clock-drift issues; for
strict per-domain politeness, run a single ingestion worker.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

from loguru import logger


@dataclass
class TokenBucket:
    """In-process token bucket.

    capacity: max burst (tokens)
    refill_rate: tokens per second
    """
    capacity: float
    refill_rate: float
    tokens: float = field(default=0.0)
    last_refill: float = field(default_factory=time.monotonic)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def __post_init__(self) -> None:
        if self.tokens == 0.0:
            self.tokens = self.capacity

    async def acquire(self, tokens: float = 1.0, timeout: float = 60.0) -> bool:
        """Wait until `tokens` are available, then consume them.

        Returns True if acquired, False if timed out.
        """
        deadline = time.monotonic() + timeout
        async with self.lock:
            while True:
                now = time.monotonic()
                elapsed = now - self.last_refill
                self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
                self.last_refill = now
                if self.tokens >= tokens:
                    self.tokens -= tokens
                    return True
                if now >= deadline:
                    return False
                # Wait until we'd have enough tokens
                needed = tokens - self.tokens
                wait = needed / self.refill_rate
                await asyncio.sleep(min(wait, deadline - now))


class RateLimiter:
    """Per-domain rate limiter.

    Usage:
        limiter = RateLimiter()
        await limiter.acquire("pubmed.ncbi.nlm.nih.gov")  # blocks until allowed
        # ... do the fetch ...
    """

    def __init__(
        self,
        default_rate: float = 1.0,  # req/sec
        default_burst: float = 3.0,
        per_domain_overrides: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        """Initialize with default rate/burst and optional per-domain overrides.

        Args:
            default_rate: requests per second (default 1.0)
            default_burst: max burst (default 3.0)
            per_domain_overrides: {domain: (rate, burst)} overrides
                Example: {"indmedinfo.nic.in": (0.5, 1.0)}  # slow NIC servers
        """
        self.default_rate = default_rate
        self.default_burst = default_burst
        self.overrides = per_domain_overrides or {}
        self._buckets: dict[str, TokenBucket] = {}

    def _domain_of(self, url: str) -> str:
        return urlparse(url).netloc.lower()

    def _get_bucket(self, domain: str) -> TokenBucket:
        bucket = self._buckets.get(domain)
        if bucket is None:
            if domain in self.overrides:
                rate, burst = self.overrides[domain]
            else:
                rate, burst = self.default_rate, self.default_burst
            bucket = TokenBucket(capacity=burst, refill_rate=rate)
            self._buckets[domain] = bucket
        return bucket

    async def acquire(self, url: str, tokens: float = 1.0) -> bool:
        """Block until fetch is allowed for `url`'s domain."""
        domain = self._domain_of(url)
        bucket = self._get_bucket(domain)
        return await bucket.acquire(tokens)

    def set_domain_rate(self, domain: str, rate: float, burst: float | None = None) -> None:
        """Override rate (and optionally burst) for a specific domain.

        Creates a new bucket if none exists; replaces existing bucket's capacity.
        """
        if burst is None:
            burst = self.default_burst
        self.overrides[domain.lower()] = (rate, burst)
        # If a bucket already exists, replace it
        existing = self._buckets.get(domain.lower())
        if existing:
            existing.capacity = burst
            existing.refill_rate = rate
            existing.tokens = min(existing.tokens, burst)

    def stats(self) -> dict[str, dict[str, float]]:
        """Per-domain bucket state for observability."""
        out: dict[str, dict[str, float]] = {}
        for domain, bucket in self._buckets.items():
            out[domain] = {
                "tokens": bucket.tokens,
                "capacity": bucket.capacity,
                "refill_rate": bucket.refill_rate,
            }
        return out
