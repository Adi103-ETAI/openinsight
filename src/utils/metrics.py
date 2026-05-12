"""
Observability & Metrics Collection
Provides structured logging, request tracing, and metrics for the API.
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from loguru import logger

# Request ID context for propagation across components
request_id_var: ContextVar[str] = ContextVar("request_id", default="unknown")


def get_request_id() -> str:
    """Get current request ID from context."""
    return request_id_var.get()


@dataclass
class RequestMetrics:
    """Metrics for a single request."""
    request_id: str
    endpoint: str
    method: str
    started_at: datetime
    duration_ms: float = 0.0
    status_code: int = 0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "endpoint": self.endpoint,
            "method": self.method,
            "started_at": self.started_at.isoformat(),
            "duration_ms": round(self.duration_ms, 2),
            "status_code": self.status_code,
            "error": self.error,
        }


@dataclass
class AggregatedMetrics:
    """Aggregated metrics over a time window."""
    total_requests: int = 0
    failed_requests: int = 0
    total_duration_ms: float = 0.0
    status_counts: dict[int, int] = field(default_factory=dict)
    endpoint_counts: dict[str, int] = field(default_factory=dict)
    latency_percentiles: dict[str, float] = field(default_factory=dict)

    def add_request(self, metrics: RequestMetrics) -> None:
        self.total_requests += 1
        self.total_duration_ms += metrics.duration_ms
        self.status_counts[metrics.status_code] = self.status_counts.get(metrics.status_code, 0) + 1
        self.endpoint_counts[metrics.endpoint] = self.endpoint_counts.get(metrics.endpoint, 0) + 1
        if metrics.status_code >= 400:
            self.failed_requests += 1

    def compute_percentiles(self, latencies: list[float]) -> None:
        if not latencies:
            return
        sorted_latencies = sorted(latencies)
        n = len(sorted_latencies)
        self.latency_percentiles = {
            "p50": round(sorted_latencies[int(n * 0.5)], 2),
            "p95": round(sorted_latencies[int(n * 0.95)], 2) if n > 1 else sorted_latencies[0],
            "p99": round(sorted_latencies[min(int(n * 0.99), n - 1)], 2),
        }

    def to_dict(self) -> dict[str, Any]:
        avg_duration = self.total_duration_ms / self.total_requests if self.total_requests > 0 else 0.0
        return {
            "total_requests": self.total_requests,
            "failed_requests": self.failed_requests,
            "success_rate": round((self.total_requests - self.failed_requests) / self.total_requests * 100, 2) if self.total_requests > 0 else 0.0,
            "avg_duration_ms": round(avg_duration, 2),
            "status_counts": self.status_counts,
            "endpoint_counts": self.endpoint_counts,
            "latency_percentiles": self.latency_percentiles,
        }


class MetricsCollector:
    """
    Collects and aggregates request metrics.
    Thread-safe for concurrent access.
    """

    def __init__(self) -> None:
        self._requests: list[RequestMetrics] = []
        self._lock = asyncio.Lock()
        self._executor = ThreadPoolExecutor(max_workers=2)

    async def record_request(self, metrics: RequestMetrics) -> None:
        """Record a completed request."""
        async with self._lock:
            self._requests.append(metrics)

    def get_aggregated(self, window_minutes: int = 60) -> AggregatedMetrics:
        """Get aggregated metrics for the specified time window."""
        cutoff = datetime.utcnow().timestamp() - (window_minutes * 60)
        recent = [r for r in self._requests if r.started_at.timestamp() > cutoff]
        
        aggregated = AggregatedMetrics()
        latencies = []
        
        for r in recent:
            aggregated.add_request(r)
            latencies.append(r.duration_ms)
        
        aggregated.compute_percentiles(sorted(latencies))
        
        # Keep only recent requests to prevent memory growth
        if len(self._requests) > 10000:
            self._requests = self._requests[-5000:]
        
        return aggregated

    async def get_summary(self) -> dict[str, Any]:
        """Get a comprehensive metrics summary."""
        aggregated = self.get_aggregated(window_minutes=60)
        return aggregated.to_dict()


class DependencyHealthChecker:
    """
    Checks health of all system dependencies.
    """

    def __init__(self) -> None:
        self._settings = None

    async def check_mongodb(self) -> dict[str, Any]:
        """Check MongoDB connectivity and status."""
        try:
            from motor.motor_asyncio import AsyncIOMotorClient
            from src.config.settings import get_settings
            
            settings = get_settings()
            client = AsyncIOMotorClient(settings.mongodb_url, serverSelectionTimeoutMS=3000)
            
            # Ping the server
            await asyncio.get_event_loop().run_in_executor(
                None, lambda: client.admin.command('ping')
            )
            
            # Get server status
            try:
                server_status = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: client.admin.command('serverStatus')
                )
                version = server_status.get("version", "unknown")
            except Exception:
                version = "unknown"
            
            await client.aclose()
            
            return {
                "status": "healthy",
                "version": version,
                "latency_ms": None,  # Could add detailed timing
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)[:200],
            }

    async def check_milvus(self) -> dict[str, Any]:
        """Check Milvus connectivity and status."""
        try:
            from pymilvus import MilvusClient
            from src.config.settings import get_settings
            
            settings = get_settings()
            token = settings.vector_token if settings.vector_token else None
            
            if token:
                client = MilvusClient(uri=settings.vector_uri, token=token, db_name=settings.milvus_db_name)
            else:
                client = MilvusClient(uri=settings.vector_uri, db_name=settings.milvus_db_name)
            
            # List collections to verify connectivity
            collections = client.list_collections()
            client.close()
            
            return {
                "status": "healthy",
                "collections_count": len(collections),
                "latency_ms": None,
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)[:200],
            }

    async def check_redis(self) -> dict[str, Any]:
        """Check Redis connectivity and status."""
        try:
            import redis.asyncio as aioredis
            from src.config.settings import get_settings
            
            settings = get_settings()
            client = aioredis.from_url(settings.redis_url, encoding="utf-8", decode_responses=True)
            
            # Ping Redis
            pong = await client.ping()
            info = await client.info("server")
            
            version = info.get("redis_version", "unknown")
            await client.aclose()
            
            return {
                "status": "healthy" if pong else "unhealthy",
                "version": version,
                "latency_ms": None,
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)[:200],
            }

    async def check_all(self) -> dict[str, Any]:
        """Check all dependencies and return aggregated health status."""
        results = await asyncio.gather(
            self.check_mongodb(),
            self.check_milvus(),
            self.check_redis(),
            return_exceptions=True,
        )
        
        mongodb_status = results[0] if not isinstance(results[0], Exception) else {"status": "unhealthy", "error": str(results[0])}
        milvus_status = results[1] if not isinstance(results[1], Exception) else {"status": "unhealthy", "error": str(results[1])}
        redis_status = results[2] if not isinstance(results[2], Exception) else {"status": "unhealthy", "error": str(results[2])}
        
        all_healthy = (
            mongodb_status.get("status") == "healthy" and
            milvus_status.get("status") == "healthy" and
            redis_status.get("status") == "healthy"
        )
        
        return {
            "status": "healthy" if all_healthy else "degraded",
            "timestamp": datetime.utcnow().isoformat(),
            "dependencies": {
                "mongodb": mongodb_status,
                "milvus": milvus_status,
                "redis": redis_status,
            },
        }


# Global metrics collector instance
_metrics_collector: MetricsCollector | None = None
_health_checker: DependencyHealthChecker | None = None


def get_metrics_collector() -> MetricsCollector:
    """Get or create the global metrics collector."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


def get_health_checker() -> DependencyHealthChecker:
    """Get or create the global health checker."""
    global _health_checker
    if _health_checker is None:
        _health_checker = DependencyHealthChecker()
    return _health_checker


class TimingMiddleware:
    """
    Middleware that tracks request timing and records metrics.
    """

    def __init__(self) -> None:
        self._collector = get_metrics_collector()

    async def __call__(self, request: Any, call_next: Any) -> Any:
        request_id = request.headers.get("X-Request-ID", "unknown")
        request_id_var.set(request_id)
        
        start_time = time.perf_counter()
        started_at = datetime.utcnow()
        
        try:
            response = await call_next(request)
            duration_ms = (time.perf_counter() - start_time) * 1000
            
            metrics = RequestMetrics(
                request_id=request_id,
                endpoint=request.url.path,
                method=request.method,
                started_at=started_at,
                duration_ms=duration_ms,
                status_code=response.status_code,
            )
            await self._collector.record_request(metrics)
            
            return response
        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            
            metrics = RequestMetrics(
                request_id=request_id,
                endpoint=request.url.path,
                method=request.method,
                started_at=started_at,
                duration_ms=duration_ms,
                status_code=500,
                error=str(e)[:200],
            )
            await self._collector.record_request(metrics)
            raise


def setup_structured_logging() -> None:
    """
    Configure loguru with structured logging and request ID context.
    """
    from loguru import logger as loguru_logger

    class RequestIdFilter:
        def __init__(self) -> None:
            self.request_id = "unknown"

        def __call__(self, message: Any) -> bool:
            message.record["request_id"] = get_request_id()
            return True

    loguru_logger.configure(
        patch=True,
        extra={"request_id": "unknown"},
    )


def log_with_context(level: str, message: str, **kwargs: Any) -> None:
    """
    Log a message with request context.
    
    Usage:
        log_with_context("info", "Processing document", doc_id="abc123")
    """
    extra = {"request_id": get_request_id(), **kwargs}
    
    if level == "debug":
        logger.debug(message, **extra)
    elif level == "info":
        logger.info(message, **extra)
    elif level == "warning":
        logger.warning(message, **extra)
    elif level == "error":
        logger.error(message, **extra)
    else:
        logger.log(level, message, **extra)
