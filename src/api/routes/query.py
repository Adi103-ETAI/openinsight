from fastapi import APIRouter, HTTPException
from loguru import logger
from pydantic import BaseModel

from src.query.standard import standard_search

router = APIRouter()


class QueryRequest(BaseModel):
    query: str
    top_k: int = 8
    mode: str = "standard"


class QueryResponse(BaseModel):
    answer: str
    citations: list[dict]
    query: str
    rewritten_query: str = ""
    model: str
    chunks_retrieved: int
    mode: str


@router.post("", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest) -> QueryResponse:
    logger.info(f"Incoming query: {request.query}")
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    result = await standard_search(request.query, request.top_k)
    return QueryResponse(**result, mode="standard")
