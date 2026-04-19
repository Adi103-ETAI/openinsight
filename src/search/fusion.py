from __future__ import annotations

from collections import defaultdict

from src.search.retriever import RetrievedChunk


EVIDENCE_BOOST_SCORE = {
    "1a": 1.35,
    "1b": 1.25,
    "2a": 1.15,
    "2b": 1.10,
    "3": 1.05,
    "4": 1.00,
    "5": 1.10,
    "unknown": 1.00,
}

RECENCY_BOOST = {
    2025: 1.10,
    2024: 1.08,
    2023: 1.05,
    2022: 1.03,
}


def reciprocal_rank_fusion(
    dense_results: list[RetrievedChunk],
    sparse_results: list[RetrievedChunk],
    k: int = 60,
    top_n: int = 20,
) -> list[RetrievedChunk]:
    all_chunks: dict[str, RetrievedChunk] = {}

    for result in dense_results + sparse_results:
        existing = all_chunks.get(result.chunk_id)
        if existing is None or result.score > existing.score:
            all_chunks[result.chunk_id] = result

    rrf_scores: dict[str, float] = defaultdict(float)

    for rank, chunk in enumerate(dense_results):
        rrf_scores[chunk.chunk_id] += 1.0 / (k + rank + 1)

    for rank, chunk in enumerate(sparse_results):
        rrf_scores[chunk.chunk_id] += 1.0 / (k + rank + 1)

    for chunk_id, chunk in all_chunks.items():
        evidence_level = str(chunk.metadata.get("evidence_level", "unknown")).lower()
        rrf_scores[chunk_id] *= EVIDENCE_BOOST_SCORE.get(evidence_level, 1.0)

        year = chunk.metadata.get("year", 0)
        try:
            year_value = int(year)
        except (TypeError, ValueError):
            year_value = 0
        rrf_scores[chunk_id] *= RECENCY_BOOST.get(year_value, 1.0)

    sorted_ids = sorted(
        rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True
    )

    fused: list[RetrievedChunk] = []
    for chunk_id in sorted_ids[:top_n]:
        chunk = all_chunks[chunk_id]
        chunk.score = float(rrf_scores[chunk_id])
        if chunk.retrieval_source == "dense" and any(
            s.chunk_id == chunk_id for s in sparse_results
        ):
            chunk.retrieval_source = "both"
        elif chunk.retrieval_source == "sparse" and any(
            d.chunk_id == chunk_id for d in dense_results
        ):
            chunk.retrieval_source = "both"
        fused.append(chunk)

    return fused
