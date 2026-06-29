"""Async HTTP client with retries, backoff, caching, rate limiting, robots.txt.

This is the workhorse of the scraper framework. All source fetches go through
HttpClient — sources never call httpx directly.

Pipeline per fetch:
    1. URL normalization
    2. Robots.txt check (allow/deny)
    3. Rate limiter acquire (per-domain)
    4. Cache lookup (3-tier)
    5. Network fetch (with conditional GET headers if cache has ETag/Last-Modified)
    6. Retry on transient errors (429, 5xx, network)
    7. Cache store (with change-detection metadata)
    8. Return ScrapeResult

Retry strategy:
    - 429 Too Many Requests: honor Retry-After header, then retry up to 3x
    - 5xx: exponential backoff (1s, 2s, 4s), up to 3 retries
    - Network errors (ConnectError, ReadTimeout): exponential backoff, up to 3 retries
    - 4xx (except 429): no retry, return failed ScrapeResult
"""
from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx
from loguru import logger

from src.ingestion.scrapers.framework.cache import ScrapeCache
from src.ingestion.scrapers.framework.models import ScrapeResult
from src.ingestion.scrapers.framework.rate_limiter import RateLimiter
from src.ingestion.scrapers.framework.robots import RobotsChecker


class HttpClient:
    """Async HTTP client with all the politeness + resilience bells."""

    DEFAULT_USER_AGENT = "OpenInsight-Bot/0.1 (clinical evidence indexing; contact: hello@openinsight.in)"
    DEFAULT_TIMEOUT = 30.0
    MAX_RETRIES = 3
    BACKOFF_BASE = 1.0  # seconds
    BACKOFF_MAX = 30.0

    def __init__(
        self,
        cache: ScrapeCache | None = None,
        rate_limiter: RateLimiter | None = None,
        robots: RobotsChecker | None = None,
        user_agent: str = DEFAULT_USER_AGENT,
        timeout: float = DEFAULT_TIMEOUT,
        verify_robots: bool = True,
    ) -> None:
        self.cache = cache
        self.rate_limiter = rate_limiter or RateLimiter()
        self.robots = robots or RobotsChecker(cache=cache)
        self.user_agent = user_agent
        self.timeout = timeout
        self.verify_robots = verify_robots
        # Reuse a single AsyncClient for connection pooling
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "HttpClient":
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        await self.close()

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self.timeout,
                follow_redirects=True,
                headers={"User-Agent": self.user_agent},
                http2=True,
            )
        return self._client

    async def fetch(
        self,
        url: str,
        method: str = "GET",
        body: bytes | None = None,
        headers: dict[str, str] | None = None,
        use_cache: bool = True,
        cache_ttl: int = 86400,
        user_agent: str | None = None,
    ) -> ScrapeResult:
        """Fetch `url` with full framework pipeline (robots, rate, cache, retry).

        Args:
            url: URL to fetch
            method: HTTP method (default GET)
            body: request body (for POST)
            headers: additional headers (User-Agent overridden if user_agent set)
            use_cache: whether to consult / write to cache
            cache_ttl: cache TTL in seconds (default 24h)
            user_agent: override default UA for this request

        Returns:
            ScrapeResult (always — failures are encoded as ok=False, not raised)
        """
        start = time.monotonic()
        method = method.upper()
        ua = user_agent or self.user_agent

        # 1. Robots check
        if self.verify_robots:
            allowed = await self.robots.can_fetch(url, user_agent=ua)
            if not allowed:
                logger.info(f"[http] robots.txt disallows {url}")
                return ScrapeResult(
                    url=url,
                    ok=False,
                    error="robots.txt disallows",
                    elapsed_ms=(time.monotonic() - start) * 1000,
                )

        # 2. Cache lookup (GET only — don't cache POST)
        cached: dict | None = None
        if use_cache and method == "GET":
            cached = await self.cache.get_cached_response_async(method, url) if self.cache else None
            if cached:
                logger.debug(f"[http] cache hit ({cached.get('_cache_layer')}) for {url}")
                return ScrapeResult(
                    url=url,
                    ok=True,
                    status_code=cached.get("status_code", 200),
                    content=cached.get("content"),
                    content_type=cached.get("content_type"),
                    encoding=cached.get("encoding"),
                    headers=cached.get("headers", {}),
                    from_cache=True,
                    cache_layer=cached.get("_cache_layer"),
                    elapsed_ms=(time.monotonic() - start) * 1000,
                )

        # 3. Rate limit
        await self.rate_limiter.acquire(url)

        # 4. Network fetch with retry
        result = await self._fetch_with_retry(url, method, body, headers, ua, start)

        # 5. Cache store on success
        if result.ok and use_cache and method == "GET" and self.cache and result.content:
            try:
                await self.cache.store_response_async(
                    method,
                    url,
                    {
                        "status_code": result.status_code,
                        "content": result.content,
                        "content_type": result.content_type,
                        "encoding": result.encoding,
                        "headers": dict(result.headers),
                    },
                    ttl=cache_ttl,
                )
            except Exception as e:
                logger.warning(f"[http] cache store failed for {url}: {e}")

        return result

    async def _fetch_with_retry(
        self,
        url: str,
        method: str,
        body: bytes | None,
        headers: dict[str, str] | None,
        user_agent: str,
        start: float,
    ) -> ScrapeResult:
        """Fetch with exponential backoff on transient failures."""
        client = self._ensure_client()
        merged_headers: dict[str, str] = {"User-Agent": user_agent}
        if headers:
            merged_headers.update(headers)

        last_error: str | None = None
        for attempt in range(self.MAX_RETRIES + 1):
            try:
                req_kwargs: dict[str, Any] = {"headers": merged_headers}
                if body and method in ("POST", "PUT", "PATCH"):
                    req_kwargs["content"] = body
                resp = await client.request(method, url, **req_kwargs)

                # Success
                if 200 <= resp.status_code < 300:
                    return ScrapeResult(
                        url=url,
                        ok=True,
                        status_code=resp.status_code,
                        content=resp.content,
                        content_type=resp.headers.get("content-type", "").split(";")[0].strip(),
                        encoding=resp.encoding or "utf-8",
                        headers=dict(resp.headers),
                        elapsed_ms=(time.monotonic() - start) * 1000,
                    )

                # 429: honor Retry-After
                if resp.status_code == 429 and attempt < self.MAX_RETRIES:
                    retry_after = resp.headers.get("retry-after", "")
                    wait = self._parse_retry_after(retry_after) or self._backoff(attempt)
                    logger.warning(f"[http] 429 on {url} — retry in {wait:.1f}s (attempt {attempt+1}/{self.MAX_RETRIES})")
                    await asyncio.sleep(wait)
                    continue

                # 5xx: retry with backoff
                if 500 <= resp.status_code < 600 and attempt < self.MAX_RETRIES:
                    wait = self._backoff(attempt)
                    logger.warning(f"[http] {resp.status_code} on {url} — retry in {wait:.1f}s (attempt {attempt+1}/{self.MAX_RETRIES})")
                    await asyncio.sleep(wait)
                    continue

                # 4xx (except 429): no retry
                return ScrapeResult(
                    url=url,
                    ok=False,
                    status_code=resp.status_code,
                    error=f"HTTP {resp.status_code}",
                    headers=dict(resp.headers),
                    elapsed_ms=(time.monotonic() - start) * 1000,
                )

            except (httpx.ConnectError, httpx.ReadTimeout, httpx.WriteTimeout, httpx.PoolTimeout) as e:
                last_error = f"{type(e).__name__}: {e}"
                if attempt < self.MAX_RETRIES:
                    wait = self._backoff(attempt)
                    logger.warning(f"[http] {type(e).__name__} on {url} — retry in {wait:.1f}s (attempt {attempt+1}/{self.MAX_RETRIES})")
                    await asyncio.sleep(wait)
                    continue
                logger.error(f"[http] {url} failed after {self.MAX_RETRIES} retries: {last_error}")
            except Exception as e:
                last_error = f"{type(e).__name__}: {e}"
                logger.error(f"[http] unexpected error on {url}: {e}")
                break

        return ScrapeResult(
            url=url,
            ok=False,
            error=last_error or "unknown error",
            elapsed_ms=(time.monotonic() - start) * 1000,
        )

    def _backoff(self, attempt: int) -> float:
        """Exponential backoff: 1s, 2s, 4s, ... capped at 30s."""
        return min(self.BACKOFF_BASE * (2 ** attempt), self.BACKOFF_MAX)

    def _parse_retry_after(self, value: str) -> float | None:
        """Parse Retry-After header (seconds or HTTP date)."""
        if not value:
            return None
        value = value.strip()
        if value.isdigit():
            return float(value)
        # HTTP date format — try to parse
        from email.utils import parsedate_to_datetime
        try:
            dt = parsedate_to_datetime(value)
            if dt:
                from datetime import datetime, timezone
                now = datetime.now(timezone.utc)
                return max(0.0, (dt - now).total_seconds())
        except Exception:
            pass
        return None
