from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from src.api.routes import search as search_router
from src.api.routes import deep_insights as deep_insights_router
from src.query.search.cache import SearchCache
from src.query.search.query_understanding import QueryUnderstanding
from src.query.search.reranker import CrossEncoderReranker
from src.query.search.retriever import HybridRetriever


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


app = FastAPI(
    title="OpenInsight API",
    description="AI Clinical Decision Support for Indian Physicians — SentArc Labs",
    version="0.1.0",
    redirect_slashes=False,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok", "service": "openinsight-api"}


app.include_router(search_router.router, prefix="/search", tags=["Search"])
app.include_router(deep_insights_router.router, prefix="/deep-insights", tags=["DeepInsights"])
