from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, HTTPException, Request, Response
from loguru import logger
from pydantic import BaseModel, Field, field_validator

from src.config.settings import get_settings
from src.query.prompts import SYSTEM_PROMPT
from src.query.validation.validator import enhance_response, validate_answer
from src.query.search.cache import SearchCache
from src.query.search.context_builder import assemble_context, build_citation_list
from src.query.search.fusion import reciprocal_rank_fusion
from src.query.search.mmr import maximal_marginal_relevance
from src.query.search.query_understanding import QueryUnderstanding
from src.query.search.reranker import BaseReranker, get_reranker
from src.query.search.retriever import HybridRetriever
import re

from src.services.llm_client import get_nim_client
from src.tools import generate_pdf, generate_docx, build_doc_sections
from pathlib import Path

router = APIRouter()
settings = get_settings()

# Component initialization lock to prevent race conditions
_component_locks: dict[str, asyncio.Lock] = {}

# Query sanitization patterns
_DANGEROUS_PATTERNS = re.compile(
    r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]|'  # Control characters
    r'<script|script>|javascript:|on\w+=|'  # XSS patterns
    r'\$\{|__|SELECT|UNION|INSERT|UPDATE|DELETE|DROP',  # Injection patterns (case insensitive)
    re.IGNORECASE
)
_WHITESPACE_PATTERN = re.compile(r'\s+')


def _sanitize_query(query: str) -> str:
    """Sanitize query by removing dangerous patterns and normalizing whitespace."""
    # Remove control characters
    sanitized = ''.join(char for char in query if ord(char) >= 32 or char in '\n\t')
    # Normalize whitespace
    sanitized = _WHITESPACE_PATTERN.sub(' ', sanitized).strip()
    return sanitized


class SearchRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Search query string (1-500 characters)"
    )
    top_k: int = Field(
        default=6,
        ge=1,
        le=50,
        description="Number of results to return (1-50)"
    )
    save_to_vault: bool = Field(
        default=False,
        description="Auto-save search results to vault"
    )
    vault_tags: list[str] = Field(
        default_factory=list,
        description="Tags to apply when saving to vault"
    )

    @field_validator('query')
    @classmethod
    def validate_query(cls, v: str) -> str:
        # Check for dangerous patterns
        if _DANGEROUS_PATTERNS.search(v):
            raise ValueError("Query contains potentially dangerous characters or patterns")
        # Sanitize and normalize
        return _sanitize_query(v)

    @field_validator('top_k')
    @classmethod
    def validate_top_k(cls, v: int) -> int:
        return max(1, min(50, v))


class SearchResponse(BaseModel):
    answer: str
    citations: list[dict[str, Any]]
    query_intent: str
    chunks_retrieved: int
    cache_hit: bool
    confidence_score: float = 0.0
    recommendation: str = "NEEDS_REVIEW"
    unverified_claims: list[dict[str, Any]] = Field(default_factory=list)
    safety_warnings: list[dict[str, Any]] = Field(default_factory=list)
    evidence_distribution: dict[str, Any] = Field(default_factory=dict)
    is_safe: bool = True
    needs_disclaimer: bool = False
    confidence_breakdown: dict[str, Any] | None = None


async def _get_or_create_component(request: Request, name: str) -> Any:
    """Thread-safe component creation with async lock to prevent race conditions."""
    components = getattr(request.app.state, "search_components", None)
    if components is None:
        request.app.state.search_components = {}
        components = request.app.state.search_components

    # Fast path: component already exists
    component = components.get(name)
    if component is not None:
        return component

    # Slow path: need to create component - use lock to prevent race condition
    if name not in _component_locks:
        _component_locks[name] = asyncio.Lock()

    async with _component_locks[name]:
        # Double-check after acquiring lock (another request may have created it)
        component = components.get(name)
        if component is not None:
            return component

        if name == "query_understanding":
            component = QueryUnderstanding()
        elif name == "retriever":
            component = HybridRetriever()
        elif name == "reranker":
            component = get_reranker()
        elif name == "cache":
            component = SearchCache()
        else:
            raise RuntimeError(f"Unknown component requested: {name}")

        components[name] = component
        return component


def _evidence_level_to_numeric(level: Any) -> int:
    if isinstance(level, int):
        return min(max(level, 1), 5)

    level_text = str(level or "").strip().lower()
    if not level_text or level_text == "unknown":
        return 3

    if level_text.startswith("1"):
        return 1
    if level_text.startswith("2"):
        return 2
    if level_text.startswith("3"):
        return 3
    if level_text.startswith("4"):
        return 4
    if level_text.startswith("5"):
        return 5
    return 3


async def _generate_answer(query: str, context: str) -> str:
    client = get_nim_client()

    prompt = (
        "Clinical context passages:\n\n"
        f"{context}\n\n"
        f"Doctor's question:\n{query}\n\n"
        "Provide a concise, clinically actionable answer with numbered citations."
    )

    content = await client.chat_completions(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=settings.nim_temperature,
        max_tokens=settings.nim_max_tokens,
    )
    return content or ""


@router.post("", response_model=SearchResponse)
async def search_endpoint(payload: SearchRequest, request: Request) -> SearchResponse:
    # Query is already sanitized and validated by SearchRequest validator
    query = payload.query
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    query_understanding: QueryUnderstanding = await _get_or_create_component(
        request, "query_understanding"
    )
    retriever: HybridRetriever = await _get_or_create_component(request, "retriever")
    reranker: BaseReranker = await _get_or_create_component(request, "reranker")
    cache: SearchCache = await _get_or_create_component(request, "cache")

    analysis = query_understanding.analyze(query)

    # LLM-based query rewriting for improved retrieval
    if settings.llm_query_rewrite:
        try:
            rewritten = await query_understanding.rewrite_query(query)
            if rewritten:
                analysis.rewritten_query = rewritten
        except Exception as exc:
            logger.warning(f"Query rewrite failed (using original): {exc}")

    # Convert FilterExpression to dict for cache serialization
    filters = analysis.metadata_filters
    filters_dict = None
    if filters is not None and hasattr(filters, "model_dump"):
        filters_dict = filters.model_dump()
    elif filters is not None and hasattr(filters, "__dict__"):
        filters_dict = dict(filters.__dict__) if not isinstance(filters, (str, int, float, bool, type(None))) else None

    # Cache lookup — failure should not break search
    cached = None
    try:
        cached = await cache.get_search_result(query, filters_dict if filters_dict is not None else filters)
    except Exception as exc:
        logger.warning(f"Cache read failed (proceeding without cache): {exc}")

    if cached:
        cached["cache_hit"] = True
        return SearchResponse(**cached)

    retrieval_k = max(settings.top_k_retrieval, payload.top_k)

    try:
        dense_results, sparse_results = await retriever.retrieve(
            query,
            analysis,
            top_k=retrieval_k,
        )
    except Exception as exc:
        logger.error(f"Search retrieval failed for query='{query}': {exc}")
        raise HTTPException(
            status_code=503, detail="Vector search unavailable"
        ) from exc

    fused = reciprocal_rank_fusion(
        dense_results,
        sparse_results,
        top_n=settings.top_k_after_fusion,
    )

    # Reranking — failure should fall back to fused results
    try:
        reranked = reranker.rerank(
            query,
            fused,
            top_k=min(settings.top_k_after_rerank, len(fused)) if fused else 0,
        )
    except Exception as exc:
        logger.warning(f"Reranking failed (using fusion results): {exc}")
        reranked = fused

    final_k = max(1, payload.top_k)
    final_chunks = maximal_marginal_relevance(
        reranked,
        retriever.embedder,
        lambda_param=settings.mmr_lambda,
        top_k=min(final_k, len(reranked)) if reranked else 0,
    )

    if not final_chunks:
        # Safely handle potential None intent
        query_intent = analysis.intent.value if analysis.intent is not None else "general"
        empty_response = {
            "answer": "No relevant clinical information found in the knowledge base for this query.",
            "citations": [],
            "query_intent": query_intent,
            "chunks_retrieved": 0,
            "cache_hit": False,
            "confidence_score": 0.0,
            "recommendation": "NEEDS_REVIEW",
            "unverified_claims": [],
            "safety_warnings": [],
            "evidence_distribution": {},
            "is_safe": True,
            "needs_disclaimer": False,
            "confidence_breakdown": None,
        }
        await cache.set_search_result(query, filters_dict if filters_dict is not None else filters, empty_response)
        return SearchResponse(**empty_response)

    context = assemble_context(final_chunks)

    try:
        answer = await _generate_answer(query, context)
    except Exception as exc:
        logger.error(f"Search answer generation failed for query='{query}': {exc}")
        raise HTTPException(status_code=503, detail="LLM service unavailable") from exc

    citations = build_citation_list(final_chunks)

    validator_citations: list[dict[str, Any]] = []
    validator_chunks: list[dict[str, Any]] = []

    for idx, chunk in enumerate(final_chunks, start=1):
        metadata = chunk.metadata or {}
        source_type = str(metadata.get("source_type", "unknown"))

        validator_citations.append(
            {
                "index": idx,
                "title": metadata.get("title", ""),
                "source_type": source_type,
                "mongo_id": str(metadata.get("mongo_id", "")),
                "evidence_level": _evidence_level_to_numeric(
                    metadata.get("evidence_level")
                ),
            }
        )

        validator_chunks.append(
            {
                "chunk_text": chunk.text,
                "title": metadata.get("title", ""),
                "source_type": source_type,
                "score": float(chunk.score),
                "quality_score": metadata.get("quality_score"),
            }
        )

    validation = await validate_answer(
        answer=answer,
        citations=validator_citations,
        source_chunks=validator_chunks,
        verify_citations_in_db=False,
    )

    base_response = {
        "answer": answer,
        "citations": citations,
        "query_intent": analysis.intent.value if analysis.intent is not None else "general",
        "chunks_retrieved": len(final_chunks),
        "cache_hit": False,
    }
    response = enhance_response(base_response, validation)

    # Save to vault if requested
    if payload.save_to_vault:
        try:
            from src.data.mongo.vault_store import VaultStore

            vault_store = getattr(request.app.state, "vault_store", None)
            if vault_store is None:
                vault_store = VaultStore(
                    mongo_url=settings.mongodb_url,
                    db_name=settings.mongodb_db,
                )
                request.app.state.vault_store = vault_store

            user_id = request.headers.get("X-User-ID", "default_user")
            vault_item = await vault_store.create_item(
                user_id=user_id,
                item_type="search_result",
                title=f"Search: {query[:100]}",
                content=answer,
                metadata={
                    "query": query,
                    "query_intent": response.get("query_intent", "general"),
                    "chunks_retrieved": response.get("chunks_retrieved", 0),
                    "confidence_score": response.get("confidence_score", 0),
                },
                tags=payload.vault_tags,
            )
            response["vault_item_id"] = vault_item["_id"]
        except Exception as e:
            logger.warning(f"Failed to save search result to vault: {e}")

    # Cache write — failure should not break the response
    try:
        await cache.set_search_result(query, filters_dict if filters_dict is not None else filters, response)
    except Exception as exc:
        logger.warning(f"Cache write failed (response still returned): {exc}")

    return SearchResponse(**response)


class SearchDocRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    format: str = Field(default="pdf", pattern="^(pdf|docx)$")
    title: str = Field(default="")


@router.post("/document")
async def search_document_endpoint(payload: SearchDocRequest, request: Request):
    """
    Search and return results as a downloadable PDF/DOCX document.
    """
    # Run search first
    search_payload = SearchRequest(query=payload.query, top_k=6)
    result = await search_endpoint(search_payload, request)

    if not result.answer or result.answer == "No relevant clinical information found in the knowledge base for this query.":
        raise HTTPException(status_code=404, detail="No results found to generate document")

    # Build document sections
    sections = build_doc_sections(
        answer=result.answer,
        citations=result.citations,
        title=payload.title or f"Search: {payload.query[:100]}",
    )

    # Generate document
    title = payload.title or f"Search_{payload.query[:50]}"
    try:
        if payload.format == "pdf":
            path = generate_pdf(sections, title)
            if not path:
                raise HTTPException(status_code=500, detail="PDF generation failed (reportlab not installed)")
            with open(path, "rb") as f:
                content = f.read()
            return Response(
                content=content,
                media_type="application/pdf",
                headers={"Content-Disposition": f'attachment; filename="{Path(path).name}"'},
            )
        elif payload.format == "docx":
            path = generate_docx(sections, title)
            if not path:
                raise HTTPException(status_code=500, detail="DOCX generation failed (python-docx not installed)")
            with open(path, "rb") as f:
                content = f.read()
            return Response(
                content=content,
                media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                headers={"Content-Disposition": f'attachment; filename="{Path(path).name}"'},
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Document generation failed: {str(e)}")

    raise HTTPException(status_code=400, detail=f"Unsupported format: {payload.format}")
