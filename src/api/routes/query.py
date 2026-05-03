from typing import Any, Literal, Optional, cast

from fastapi import APIRouter, HTTPException, Request, Response
from loguru import logger
from pydantic import BaseModel

from src.api.routes.search import SearchRequest, search_endpoint
from src.core.config import get_settings
from src.query.standard import standard_search

router = APIRouter()
settings = get_settings()


class QueryRequest(BaseModel):
    query: str
    top_k: int = 8
    mode: str = "auto"


class SafetyWarningModel(BaseModel):
    type: str
    severity: str
    message: str
    matched_text: Optional[str] = None


class UnverifiedClaimModel(BaseModel):
    text: str
    reason: str
    confidence: float


class ConfidenceBreakdownModel(BaseModel):
    citation_score: float
    evidence_score: float
    hallucination_score: float
    quality_score: float
    consistency_score: float
    safety_penalty: float


class QueryResponse(BaseModel):
    answer: str
    citations: list[dict]
    query: str
    rewritten_query: str = ""
    model: str
    chunks_retrieved: int
    mode: str
    # Validation fields
    confidence_score: float = 0.0
    recommendation: Literal["SAFE", "NEEDS_REVIEW", "UNSAFE"] = "NEEDS_REVIEW"
    unverified_claims: list[dict] = []
    safety_warnings: list[dict] = []
    evidence_distribution: dict = {}
    is_safe: bool = True
    needs_disclaimer: bool = False
    confidence_breakdown: Optional[dict] = None


def _normalize_mode(mode: str) -> str:
    normalized = (mode or "auto").strip().lower()
    return normalized or "auto"


def _legacy_recommendation(value: Any) -> Literal["SAFE", "NEEDS_REVIEW", "UNSAFE"]:
    text = str(value or "NEEDS_REVIEW").upper()
    if text not in {"SAFE", "NEEDS_REVIEW", "UNSAFE"}:
        return "NEEDS_REVIEW"
    return cast(Literal["SAFE", "NEEDS_REVIEW", "UNSAFE"], text)


def _mark_legacy_deprecation(response: Response) -> None:
    if not settings.enable_query_deprecation_headers:
        return
    response.headers["Deprecation"] = "true"
    response.headers["Warning"] = (
        '299 - "Legacy /query pipeline is deprecated; switch to mode=v2 or /search."'
    )


async def _run_legacy(query: str, top_k: int) -> QueryResponse:
    result = await standard_search(query, top_k)

    return QueryResponse(
        answer=result.get("answer", ""),
        citations=result.get("citations", []),
        query=result.get("query", query),
        rewritten_query=result.get("rewritten_query", ""),
        model=result.get("model", settings.nim_model),
        chunks_retrieved=int(result.get("chunks_retrieved", 0) or 0),
        mode="legacy",
        confidence_score=float(result.get("confidence_score", 0.0) or 0.0),
        recommendation=_legacy_recommendation(result.get("recommendation")),
        unverified_claims=result.get("unverified_claims", []),
        safety_warnings=result.get("safety_warnings", []),
        evidence_distribution=result.get("evidence_distribution", {}),
        is_safe=bool(result.get("is_safe", True)),
        needs_disclaimer=bool(result.get("needs_disclaimer", False)),
        confidence_breakdown=result.get("confidence_breakdown"),
    )


async def _run_v2(query: str, top_k: int, request: Request) -> QueryResponse:
    search_result = await search_endpoint(
        SearchRequest(query=query, top_k=top_k),
        request,
    )
    data = search_result.model_dump()

    return QueryResponse(
        answer=data.get("answer", ""),
        citations=data.get("citations", []),
        query=query,
        rewritten_query="",
        model=settings.nim_model,
        chunks_retrieved=int(data.get("chunks_retrieved", 0) or 0),
        mode="v2",
        confidence_score=float(data.get("confidence_score", 0.0) or 0.0),
        recommendation=_legacy_recommendation(data.get("recommendation")),
        unverified_claims=data.get("unverified_claims", []),
        safety_warnings=data.get("safety_warnings", []),
        evidence_distribution=data.get("evidence_distribution", {}),
        is_safe=bool(data.get("is_safe", True)),
        needs_disclaimer=bool(data.get("needs_disclaimer", False)),
        confidence_breakdown=data.get("confidence_breakdown"),
    )


@router.post("", response_model=QueryResponse)
async def query_endpoint(
    payload: QueryRequest,
    request: Request,
    response: Response,
) -> QueryResponse:
    logger.info(f"Incoming query: {payload.query}")
    if not payload.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    mode = _normalize_mode(payload.mode)
    if mode not in {"auto", "v2", "search", "legacy", "standard"}:
        raise HTTPException(
            status_code=400,
            detail="mode must be one of: auto, v2, search, legacy, standard",
        )

    default_pipeline = settings.query_default_pipeline.strip().lower()
    default_to_v2 = default_pipeline != "legacy"

    if mode == "auto":
        use_v2 = default_to_v2
    else:
        use_v2 = mode in {"v2", "search"}

    if not use_v2:
        if not settings.enable_legacy_query:
            raise HTTPException(
                status_code=410,
                detail="Legacy /query pipeline is disabled. Use mode=v2 or /search.",
            )
        _mark_legacy_deprecation(response)
        return await _run_legacy(payload.query, payload.top_k)

    try:
        return await _run_v2(payload.query, payload.top_k, request)
    except HTTPException as exc:
        should_fallback = (
            mode == "auto"
            and settings.query_auto_fallback_to_legacy
            and settings.enable_legacy_query
            and exc.status_code >= 500
        )
        if not should_fallback:
            raise

        logger.warning(
            "V2 query pipeline failed (status={}); falling back to legacy",
            exc.status_code,
        )
        response.headers["X-OpenInsight-Query-Fallback"] = "legacy"
        _mark_legacy_deprecation(response)
        return await _run_legacy(payload.query, payload.top_k)
