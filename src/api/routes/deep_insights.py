from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, Response
from loguru import logger
from pydantic import BaseModel
from typing import Any, Optional

from src.query.agents.deep_insights import DeepInsightsOrchestrator
from src.query.agents.intent_router import IntentRouter, QueryComplexity

router = APIRouter()

# Singleton orchestrator
_orchestrator: DeepInsightsOrchestrator | None = None


def get_orchestrator() -> DeepInsightsOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = DeepInsightsOrchestrator()
    return _orchestrator


class DeepInsightsRequest(BaseModel):
    query: str
    top_k: int = 8
    force_deep: bool = False  # Force deep mode even for simple queries


class DeepInsightsResponse(BaseModel):
    answer: str
    sections: dict[str, str]
    citations: list[dict]
    sub_queries: list[dict]
    contradictions: list[dict]
    confidence: float
    complexity_detected: str
    processing_time_ms: float
    mode: str = "deep_insights"


class RoutingCheckResponse(BaseModel):
    complexity: str
    reason: str
    confidence: float
    detected_intent: str
    entities: dict[str, list[str]]
    sub_query_types: list[str]
    recommended_mode: str  # "standard" or "deep_insights"


@router.post("", response_model=DeepInsightsResponse)
async def deep_insights_endpoint(
    payload: DeepInsightsRequest,
    request: Request,
    response: Response,
) -> DeepInsightsResponse:
    """DeepInsights - Multi-agent complex query processing."""
    logger.info(f"[DeepInsights] Query: {payload.query}")

    if not payload.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    orchestrator = get_orchestrator()
    result = await orchestrator.process(
        query=payload.query,
        top_k=payload.top_k,
        force_deep=payload.force_deep,
    )

    return DeepInsightsResponse(
        answer=result.answer,
        sections=result.sections,
        citations=result.citations,
        sub_queries=[
            {
                "id": sr.sub_query.id,
                "query": sr.sub_query.query,
                "focus": sr.sub_query.focus,
                "chunks_retrieved": len(sr.chunks),
                "error": sr.error,
            }
            for sr in result.sub_query_results
        ],
        contradictions=result.contradictions,
        confidence=result.confidence,
        complexity_detected=result.complexity_detected,
        processing_time_ms=result.processing_time_ms,
    )


@router.get("/route-check", response_model=RoutingCheckResponse)
async def route_check(query: str) -> RoutingCheckResponse:
    """Check how a query would be routed (for debugging)."""
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    router = IntentRouter()
    routing = router.route(query)

    recommended = (
        "deep_insights"
        if routing.complexity in {QueryComplexity.COMPLEX, QueryComplexity.MEDIUM}
        else "standard"
    )

    return RoutingCheckResponse(
        complexity=routing.complexity.value,
        reason=routing.reason,
        confidence=routing.confidence,
        detected_intent=routing.detected_intent,
        entities=routing.entities,
        sub_query_types=routing.sub_query_types,
        recommended_mode=recommended,
    )