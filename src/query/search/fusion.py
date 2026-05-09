from __future__ import annotations

from collections import defaultdict

from .retriever import RetrievedChunk
from src.core.constants import EvidenceBoost, RecencyBoost, RRF_K


def reciprocal_rank_fusion(
    dense_results: list[RetrievedChunk],
    sparse_results: list[RetrievedChunk],
    k: int = RRF_K,
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

    # Create sets for O(1) lookup instead of O(n) any()
    dense_ids = {c.chunk_id for c in dense_results}
    sparse_ids = {c.chunk_id for c in sparse_results}

    for chunk_id, chunk in all_chunks.items():
        evidence_level = str(chunk.metadata.get("evidence_level", "unknown")).lower()
        rrf_scores[chunk_id] *= EvidenceBoost.get_boost(evidence_level)

        year = chunk.metadata.get("year", 0)
        try:
            year_value = int(year)
        except (TypeError, ValueError):
            year_value = 0
        rrf_scores[chunk_id] *= RecencyBoost.get_boost(year_value)

    sorted_ids = sorted(
        rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True
    )

    fused: list[RetrievedChunk] = []
    for chunk_id in sorted_ids[:top_n]:
        chunk = all_chunks[chunk_id]
        chunk.score = float(rrf_scores[chunk_id])
        if chunk_id in dense_ids and chunk_id in sparse_ids:
            chunk.retrieval_source = "both"
        elif chunk_id in sparse_ids:
            chunk.retrieval_source = "sparse"
        else:
            chunk.retrieval_source = "dense"
        fused.append(chunk)

    return fused
