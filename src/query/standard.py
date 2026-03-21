from fastapi import HTTPException
from loguru import logger
from openai import AsyncOpenAI

from src.core.config import get_settings
from src.ingestion.embeddings import embed_query
from src.ingestion.vector_db import search
from src.query.prompts import SYSTEM_PROMPT, build_prompt

settings = get_settings()


async def standard_search(query: str, top_k: int = 8) -> dict:
    logger.info(f"Query received: {query}")

    try:
        query_vector = embed_query(query)
        results = search(query_vector, top_k=top_k)
    except Exception as exc:
        logger.error(f"Vector retrieval failed for query='{query}': {exc}")
        raise HTTPException(status_code=503, detail="Vector search unavailable")

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

    if not chunks:
        return {
            "answer": "No relevant clinical information found in the knowledge base for this query.",
            "citations": [],
            "query": query,
            "model": settings.nim_model,
            "chunks_retrieved": 0,
        }

    prompt = build_prompt(query, chunks)
    client = AsyncOpenAI(
        api_key=settings.nvidia_nim_api_key,
        base_url=settings.nvidia_nim_base_url,
    )

    try:
        response = await client.chat.completions.create(
            model=settings.nim_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=1024,
        )
    except Exception as exc:
        logger.error(f"NIM completion failed for query='{query}': {exc}")
        raise HTTPException(status_code=503, detail="LLM service unavailable")

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

    return {
        "answer": answer,
        "citations": citations,
        "query": query,
        "model": settings.nim_model,
        "chunks_retrieved": len(chunks),
    }
