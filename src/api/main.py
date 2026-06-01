import os
import uuid
from contextvars import ContextVar
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware

# Route and component imports
from src.api.routes import search as search_router
from src.api.routes import deep_insights as deep_insights_router
from src.api.routes import vault as vault_router
from src.api.routes import reports as reports_router
from src.api.middleware.rate_limit import RateLimitMiddleware
from src.query.search.cache import SearchCache
from src.query.search.query_understanding import QueryUnderstanding
from src.query.search.reranker import get_reranker
from src.query.search.retriever import HybridRetriever

# Initialize loguru with centralized configuration before anything else
from src.config.logging_config import configure_loguru

configure_loguru()

# Request ID context variable for propagation across components
request_id_var: ContextVar[str] = ContextVar("request_id", default="unknown")


def get_request_id() -> str:
    """Get current request ID from context."""
    return request_id_var.get()


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Middleware that generates and propagates unique request IDs."""

    def __init__(self, app):
        # Required for BaseHTTPMiddleware to work with FastAPI's add_middleware
        super().__init__(app)

    async def dispatch(self, request: Request, call_next):
        # Check for existing request ID in headers (from upstream)
        request_id = request.headers.get("X-Request-ID")
        if not request_id:
            request_id = str(uuid.uuid4())

        # Set in context variable for propagation
        request_id_var.set(request_id)

        # Add to request state for access in handlers
        request.state.request_id = request_id

        # Bind request_id to logger for this request context using loguru's bind()
        request_logger = logger.bind(request_id=request_id)

        # Log request start with structured context
        request_logger.info(
            "request_started | method={method} | path={path}",
            method=request.method,
            path=request.url.path,
        )

        try:
            response = await call_next(request)
            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id
            # Log request completion
            request_logger.info(
                "request_completed | status={status} | method={method} | path={path}",
                status=response.status_code,
                method=request.method,
                path=request.url.path,
            )
            return response
        except Exception as exc:
            request_logger.error(
                "request_failed | error={error} | method={method} | path={path}",
                error=str(exc)[:200],
                method=request.method,
                path=request.url.path,
            )
            raise


class TimingMiddleware(BaseHTTPMiddleware):
    """Middleware that tracks request timing and records metrics."""

    def __init__(self, app):
        # Required for BaseHTTPMiddleware to work with FastAPI's add_middleware
        super().__init__(app)

    async def dispatch(self, request: Request, call_next):
        import time
        from datetime import datetime

        request_id = request_id_var.get()
        start_time = time.perf_counter()
        started_at = datetime.utcnow()

        # Bind request_id to logger for this request context
        request_logger = logger.bind(request_id=request_id)

        try:
            response = await call_next(request)
            duration_ms = (time.perf_counter() - start_time) * 1000

            # Log timing info
            request_logger.debug(
                "request_timing | duration_ms={duration_ms} | status={status}",
                duration_ms=round(duration_ms, 2),
                status=response.status_code,
            )

            # Add timing header
            response.headers["X-Response-Time-Ms"] = str(round(duration_ms, 2))

            return response
        except Exception as exc:
            duration_ms = (time.perf_counter() - start_time) * 1000
            request_logger.warning(
                "request_timing_error | duration_ms={duration_ms} | error={error}",
                duration_ms=round(duration_ms, 2),
                error=str(exc)[:100],
            )
            raise


@asynccontextmanager
async def lifespan(app: FastAPI):  # pylint: disable=redefined-outer-name
    app.state.search_components = {}
    degraded = []

    # Initialize components with graceful degradation
    try:
        app.state.search_components["query_understanding"] = QueryUnderstanding()
    except Exception as exc:
        logger.warning(f"QueryUnderstanding init failed (degraded mode): {exc}")
        degraded.append("query_understanding")

    try:
        app.state.search_components["retriever"] = HybridRetriever()
    except Exception as exc:
        logger.warning(f"HybridRetriever init failed (degraded mode): {exc}")
        degraded.append("retriever")

    try:
        app.state.search_components["reranker"] = get_reranker()
    except Exception as exc:
        logger.warning(f"Reranker init failed (degraded mode): {exc}")
        degraded.append("reranker")

    try:
        app.state.search_components["cache"] = SearchCache()
    except Exception as exc:
        logger.warning(f"Cache init failed (degraded mode): {exc}")
        degraded.append("cache")

    # Store degradation status
    app.state.degraded_components = degraded
    if degraded:
        logger.warning(f"API starting in degraded mode — failed components: {degraded}")
    else:
        logger.info("Search v2 singletons initialized — all components healthy")

    yield

    # Shutdown: close connections gracefully
    cache = app.state.search_components.get("cache")
    if cache is not None:
        try:
            await cache.redis.aclose()
        except (RuntimeError, ValueError, TypeError) as exc:
            logger.warning(f"Failed to close redis cache cleanly: {exc}")

    # Close NIM client if initialized
    try:
        from src.services.llm_client import _nim_client
        if _nim_client is not None:
            await _nim_client.close()
    except Exception:
        pass

    # Close all dynamic LLM providers
    try:
        from src.services.llm.registry import close_all_clients
        await close_all_clients()
    except Exception:
        pass

    # Close LLM router
    try:
        from src.services.llm.router import _router
        if _router is not None:
            await _router.close()
    except Exception:
        pass


def setup_logging_with_request_id():
    """Configure loguru to include request ID in all log messages.

    Note: Request ID binding is handled per-request in RequestIDMiddleware
    via logger.bind(request_id=...). The centralized config in logging_config.py
    already includes request_id_str in the format when present.
    """
    # Loguru is already configured by configure_loguru() above.
    # The RequestIDMiddleware binds request_id per-request via logger.bind().
    pass


app = FastAPI(
    title="OpenInsight API",
    description="AI Clinical Decision Support for Indian Physicians — SentArc Labs",
    version="0.1.0",
    redirect_slashes=False,
    lifespan=lifespan,
)

# Add request ID middleware FIRST (before other middleware)
app.add_middleware(RequestIDMiddleware)

# Setup logging with request ID support
setup_logging_with_request_id()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        origin.strip()
        for origin in os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:5173").split(",")
        if origin.strip()
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Request-ID", "X-User-ID"],
)

# Rate limiting: 60 req/min default, 10 req/min for search (LLM calls)
app.add_middleware(
    RateLimitMiddleware,
    default_rate=1.0,       # 1 request per second (60/min)
    default_capacity=10,    # burst of 10
    path_limits={
        "/search": (0.167, 5),        # ~10/min, burst 5 (LLM calls are expensive)
        "/deep-insights": (0.083, 3), # ~5/min, burst 3
        "/reports": (0.167, 5),       # ~10/min, burst 5
    },
    excluded_paths={"/health", "/health/detailed", "/health/ready", "/metrics", "/docs", "/openapi.json"},
)


@app.get("/health")
async def health():
    """Basic health check endpoint with degradation status."""
    degraded = getattr(app.state, "degraded_components", [])
    status = "ok" if not degraded else "degraded"
    result = {
        "status": status,
        "service": "openinsight-api",
    }
    if degraded:
        result["degraded_components"] = degraded
        result["message"] = f"Running with degraded components: {', '.join(degraded)}"
    return result


@app.get("/health/detailed")
async def health_detailed():
    """
    Detailed health check with dependency status.
    Verifies MongoDB, Milvus, and Redis connectivity.
    """
    from src.utils.metrics import DependencyHealthChecker

    checker = DependencyHealthChecker()
    health_status = await checker.check_all()

    return health_status


@app.get("/health/ready")
async def health_ready():
    """
    Kubernetes-style readiness probe.
    Returns 503 if any critical dependency is unhealthy.
    """
    from src.utils.metrics import DependencyHealthChecker

    checker = DependencyHealthChecker()
    health_status = await checker.check_all()

    is_ready = health_status.get("status") == "healthy"

    if is_ready:
        return JSONResponse(
            content={
                "status": "ready",
                "service": "openinsight-api",
                "timestamp": health_status.get("timestamp"),
            }
        )
    else:
        return JSONResponse(
            content=health_status,
            status_code=503,
        )


@app.get("/metrics")
async def metrics():
    """
    API metrics endpoint.
    Returns aggregated request metrics and latency percentiles.
    """
    from src.utils.metrics import MetricsCollector, get_metrics_collector

    collector = get_metrics_collector()
    summary = await collector.get_summary()

    return {
        "service": "openinsight-api",
        "window": "60 minutes",
        **summary,
    }


app.include_router(search_router.router, prefix="/search", tags=["Search"])
app.include_router(
    deep_insights_router.router, prefix="/deep-insights", tags=["DeepInsights"]
)
app.include_router(vault_router.router, prefix="/vault", tags=["Vault"])
app.include_router(reports_router.router, prefix="/reports", tags=["Reports"])
