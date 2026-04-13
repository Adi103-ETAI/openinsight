"""
Cross-Encoder Reranker
Reranks retrieved chunks by true relevance to the query.
Uses BAAI/bge-reranker-base — free, runs on CPU, good medical text performance.
"""

from functools import lru_cache
from loguru import logger
from sentence_transformers import CrossEncoder

from src.core.config import get_settings

settings = get_settings()

RERANKER_MODEL = "BAAI/bge-reranker-base"


def _truncate_for_rerank(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    logger.info(f"Loading reranker model: {RERANKER_MODEL}")
    model = CrossEncoder(RERANKER_MODEL, max_length=512)
    logger.info("Reranker model loaded.")
    return model


def rerank_chunks(query: str, chunks: list[dict], top_n: int = 8) -> list[dict]:
    """
    Rerank chunks by relevance to query using cross-encoder.

    Args:
        query: the (rewritten) doctor query
        chunks: list of chunk dicts with 'chunk_text' key
        top_n: how many top chunks to return after reranking

    Returns:
        top_n chunks sorted by reranker score descending
    """
    if not chunks:
        return chunks

    if len(chunks) <= top_n:
        return chunks

    try:
        model = get_reranker()
        pairs = [
            (
                query,
                _truncate_for_rerank(chunk["chunk_text"], settings.reranker_max_chars),
            )
            for chunk in chunks
        ]
        scores = model.predict(
            pairs,
            show_progress_bar=False,
            batch_size=settings.reranker_batch_size,
        )

        scored_chunks = [
            {**chunk, "reranker_score": float(score)}
            for chunk, score in zip(chunks, scores)
        ]

        scored_chunks.sort(key=lambda x: x["reranker_score"], reverse=True)
        top_chunks = scored_chunks[:top_n]

        logger.info(
            f"Reranked {len(chunks)} chunks → kept top {len(top_chunks)}. "
            f"Top score: {top_chunks[0]['reranker_score']:.3f}, "
            f"Bottom score: {top_chunks[-1]['reranker_score']:.3f}"
        )
        return top_chunks

    except (RuntimeError, ValueError, TypeError) as exc:
        logger.error(f"Reranking failed, returning original chunks: {exc}")
        return chunks[:top_n]
