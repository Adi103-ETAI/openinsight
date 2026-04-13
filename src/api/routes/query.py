from typing import Any, Literal, Optional

from fastapi import APIRouter, HTTPException
from loguru import logger
from pydantic import BaseModel

from src.query.standard import standard_search

router = APIRouter()


class QueryRequest(BaseModel):
    query: str
    top_k: int = 8
    mode: str = "standard"


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


@router.post("", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest) -> QueryResponse:
    logger.info(f"Incoming query: {request.query}")
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    result = await standard_search(request.query, request.top_k)
    return QueryResponse(**result, mode="standard")
