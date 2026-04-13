from fastapi import HTTPException
from loguru import logger
from openai import AsyncOpenAI
from time import perf_counter

from src.core.config import get_settings
from src.ingestion.embeddings import embed_query
from src.ingestion.vector_db import hybrid_search
from src.query.prompts import SYSTEM_PROMPT, build_prompt
from src.query.reranker import rerank_chunks
from src.query.rewriter import rewrite_query
from src.query.validator import enhance_response, validate_answer

settings = get_settings()


async def standard_search(query: str, top_k: int = 8) -> dict:
    logger.info(f"Query received: {query}")
    t0 = perf_counter()

    try:
        t_rewrite_start = perf_counter()
        rewritten_query = await rewrite_query(query)
        t_rewrite = perf_counter() - t_rewrite_start

        t_embed_start = perf_counter()
        query_vector = embed_query(rewritten_query)
        t_embed = perf_counter() - t_embed_start

        logger.info(f"Using rewritten query for embedding: {rewritten_query}")
        retrieval_k = min(
            max(top_k * settings.retrieval_multiplier, settings.retrieval_min_k),
            settings.retrieval_max_k,
        )
        t_retrieve_start = perf_counter()
        results = hybrid_search(
            query_text=rewritten_query,
            query_vector=query_vector,
            top_k=retrieval_k,
        )
        t_retrieve = perf_counter() - t_retrieve_start
    except Exception as exc:
        logger.error(f"Vector retrieval failed for query='{query}': {exc}")
        raise HTTPException(
            status_code=503, detail="Vector search unavailable"
        ) from exc

    chunks: list[dict] = []
    for result in results:
        payload = result.payload or {}
        chunks.append(
            {
                "chunk_text": payload.get("chunk_text", ""),
                "title": payload.get("title", ""),
                "source_type": payload.get("source_type", ""),
                "score": float(result.score or 0.0),
                "mongo_id": str(payload.get("mongo_id", "")),
            }
        )

    logger.info(f"Chunks retrieved: {len(chunks)}")
    # Rerank retrieved chunks by true relevance to query
    t_rerank_start = perf_counter()
    chunks = rerank_chunks(rewritten_query, chunks, top_n=settings.reranker_top_n)
    t_rerank = perf_counter() - t_rerank_start

    if not chunks:
        return {
            "answer": "No relevant clinical information found in the knowledge base for this query.",
            "citations": [],
            "query": query,
            "rewritten_query": rewritten_query,
            "model": settings.nim_model,
            "chunks_retrieved": 0,
        }

    prompt = build_prompt(rewritten_query, chunks)
    client = AsyncOpenAI(
        api_key=settings.nvidia_nim_api_key,
        base_url=settings.nvidia_nim_base_url,
    )

    try:
        t_llm_start = perf_counter()
        response = await client.chat.completions.create(
            model=settings.nim_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=1024,
        )
        t_llm = perf_counter() - t_llm_start
    except Exception as exc:
        logger.error(f"NIM completion failed for query='{query}': {exc}")
        raise HTTPException(status_code=503, detail="LLM service unavailable") from exc

    answer = response.choices[0].message.content or ""
    logger.info(f"Model used: {settings.nim_model}; response length: {len(answer)}")

    citations = [
        {
            "index": idx,
            "title": chunk["title"],
            "source_type": chunk["source_type"],
            "chunk_text": chunk["chunk_text"],
            "score": chunk["score"],
            "mongo_id": chunk["mongo_id"],
        }
        for idx, chunk in enumerate(chunks, start=1)
    ]

    # Validate answer for quality and safety
    validation = await validate_answer(
        answer=answer,
        citations=citations,
        source_chunks=chunks,
        verify_citations_in_db=False,  # Skip DB verification for performance
    )
    t_total = perf_counter() - t0
    logger.info(
        "Query timing seconds | rewrite={:.2f} embed={:.2f} retrieve={:.2f} rerank={:.2f} llm={:.2f} total={:.2f}".format(
            t_rewrite,
            t_embed,
            t_retrieve,
            t_rerank,
            t_llm,
            t_total,
        )
    )

    base_response = {
        "answer": answer,
        "citations": citations,
        "query": query,
        "rewritten_query": rewritten_query,
        "model": settings.nim_model,
        "chunks_retrieved": len(chunks),
    }

    # Enhance response with validation results
    return enhance_response(base_response, validation)
