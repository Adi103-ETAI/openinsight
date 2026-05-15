import uuid
from contextvars import ContextVar
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware

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
    try:
        app.state.search_components = {
            "query_understanding": QueryUnderstanding(),
            "retriever": HybridRetriever(),
            "reranker": get_reranker(),
            "cache": SearchCache(),
        }
        logger.info("Search v2 singletons initialized")
    except Exception as exc:
        logger.warning(f"Search v2 startup degraded; using lazy init: {exc}")

    yield

    cache = app.state.search_components.get("cache")
    if cache is not None:
        try:
            await cache.redis.aclose()
        except (RuntimeError, ValueError, TypeError) as exc:
            logger.warning(f"Failed to close redis cache cleanly: {exc}")


from src.api.routes import search as search_router
from src.api.routes import deep_insights as deep_insights_router
from src.query.search.cache import SearchCache
from src.query.search.query_understanding import QueryUnderstanding
from src.query.search.reranker import get_reranker
from src.query.search.retriever import HybridRetriever


def setup_logging_with_request_id():
    """Configure loguru to include request ID in all log messages."""
    from loguru import logger
    import sys

    # Remove default handler to reconfigure with custom format
    logger.remove()

    # Add handler with custom format that includes request_id from extra
    # Use a custom format that safely handles missing request_id
    def format_with_request_id(record):
        request_id = record["extra"].get("request_id", "unknown")
        return (
            f"{record['time'].strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} | "
            f"{record['level']:<8} | "
            f"{record['name']}:{record['function']}:{record['line']} - "
            f"{record['message']} | request_id={request_id}"
        )

    logger.add(
        sys.stderr,
        format=format_with_request_id,
        level="INFO",
        serialize=False,
    )

    # Note: Request ID is bound per-request in the middleware via request_logger.bind()
    # Static configuration at module load time is not needed as request_id defaults to "unknown"
    # The middleware will properly bind the correct request_id for each request


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
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    """Basic health check endpoint."""
    return {"status": "ok", "service": "openinsight-api"}


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
