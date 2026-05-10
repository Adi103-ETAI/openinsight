import uuid
from contextvars import ContextVar
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

# Request ID context variable for propagation across components
request_id_var: ContextVar[str] = ContextVar("request_id", default="unknown")


def get_request_id() -> str:
    """Get current request ID from context."""
    return request_id_var.get()


class RequestIDMiddleware:
    """Middleware that generates and propagates unique request IDs."""

    async def __call__(self, request: Request, call_next):
        # Check for existing request ID in headers (from upstream)
        request_id = request.headers.get("X-Request-ID")
        if not request_id:
            request_id = str(uuid.uuid4())

        # Set in context variable for propagation
        request_id_var.set(request_id)

        # Add to request state for access in handlers
        request.state.request_id = request_id

        # Log request start with structured context
        logger.info(
            "request_started | request_id={request_id} | method={method} | path={path}",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
        )

        try:
            response = await call_next(request)
            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id
            # Log request completion
            logger.info(
                "request_completed | request_id={request_id} | status={status} | method={method} | path={path}",
                request_id=request_id,
                status=response.status_code,
                method=request.method,
                path=request.url.path,
            )
            return response
        except Exception as exc:
            logger.error(
                "request_failed | request_id={request_id} | error={error} | method={method} | path={path}",
                request_id=request_id,
                error=str(exc)[:200],
                method=request.method,
                path=request.url.path,
            )
            raise


class TimingMiddleware:
    """Middleware that tracks request timing and records metrics."""

    async def __call__(self, request: Request, call_next):
        import time
        from datetime import datetime
        
        request_id = request_id_var.get()
        start_time = time.perf_counter()
        started_at = datetime.utcnow()

        try:
            response = await call_next(request)
            duration_ms = (time.perf_counter() - start_time) * 1000

            # Log timing info
            logger.debug(
                "request_timing | request_id={request_id} | duration_ms={duration_ms} | status={status}",
                request_id=request_id,
                duration_ms=round(duration_ms, 2),
                status=response.status_code,
            )

            # Add timing header
            response.headers["X-Response-Time-Ms"] = str(round(duration_ms, 2))

            return response
        except Exception as exc:
            duration_ms = (time.perf_counter() - start_time) * 1000
            logger.warning(
                "request_timing_error | request_id={request_id} | duration_ms={duration_ms} | error={error}",
                request_id=request_id,
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
            "reranker": CrossEncoderReranker(),
            "cache": SearchCache(),
        }
        logger.info("Search v2 singletons initialized")
    except (RuntimeError, ValueError, TypeError) as exc:
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
from src.query.search.reranker import CrossEncoderReranker
from src.query.search.retriever import HybridRetriever


def setup_logging_with_request_id():
    """Configure loguru to include request ID in all log messages."""
    from loguru import logger

    class RequestIdFilter:
        def __init__(self):
            self.request_id = "unknown"

        def __call__(self, message):
            message.record["request_id"] = get_request_id()
            return True

    logger.configure(
        patch=True,
        extra={"request_id": "unknown"}
    )


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
        return {
            "status": "ready",
            "service": "openinsight-api",
            "timestamp": health_status.get("timestamp"),
        }
    else:
        return Response(
            content=health_status.json() if hasattr(health_status, 'json') else str(health_status),
            status_code=503,
            media_type="application/json",
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
app.include_router(deep_insights_router.router, prefix="/deep-insights", tags=["DeepInsights"])
