"""
Vector DB compatibility helpers (legacy query + ingestion pipelines).
Internally this module uses the backend-agnostic VectorStore abstraction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.core.config import get_settings
from src.vectorstore.filters import FilterCondition, FilterExpression, FilterOperator
from src.vectorstore.registry import get_vector_store
from src.vectorstore.types import ScoredPoint, SparseVector, VectorPoint

settings = get_settings()


@dataclass
class LegacySearchResult:
    id: str
    score: float
    payload: dict[str, Any]


def ensure_collection(
    *, recreate: bool = False, collection_name: str | None = None
) -> None:
    store = get_vector_store()
    store.ensure_collection(
        recreate=recreate,
        collection_name=collection_name or settings.vector_collection,
    )


def drop_collection(*, collection_name: str | None = None) -> None:
    store = get_vector_store()
    store.drop_collection(collection_name=collection_name or settings.vector_collection)


def upsert_chunks(
    points: list[Any],
    *,
    collection_name: str | None = None,
    batch_size: int = 100,
) -> int:
    if not points:
        return 0

    vector_points = [_coerce_vector_point(point) for point in points]
    store = get_vector_store()
    return store.upsert_points(
        vector_points,
        collection_name=collection_name or settings.vector_collection,
        batch_size=batch_size,
    )


def search(
    query_vector: list[float],
    top_k: int = 8,
    source_type: str | None = None,
) -> list[LegacySearchResult]:
    store = get_vector_store()
    filters = _source_filter(source_type)
    dense_results = store.search_dense(
        query_vector,
        top_k=max(1, top_k),
        filters=filters,
        collection_name=settings.vector_collection,
    )
    return [_to_legacy(hit) for hit in dense_results]


def build_sparse_vector(text: str) -> dict[int, float]:
    """
    Build a simple sparse vector from text using term frequencies.
    Maps term hash → frequency. Used for BM25-style keyword search.
    """
    import re
    from collections import Counter

    tokens = re.findall(r"\b[a-zA-Z0-9]{2,}\b", text.lower())
    tf = Counter(tokens)
    return {abs(hash(term)) % 2_000_000: float(freq) for term, freq in tf.items()}


def hybrid_search(
    query_text: str,
    query_vector: list[float],
    top_k: int = 50,
    source_type: str | None = None,
) -> list[LegacySearchResult]:
    store = get_vector_store()
    filters = _source_filter(source_type)

    dense_results = store.search_dense(
        query_vector,
        top_k=max(1, top_k),
        filters=filters,
        collection_name=settings.vector_collection,
    )

    sparse_mapping = build_sparse_vector(query_text)
    sparse_results = store.search_sparse(
        SparseVector.from_mapping(sparse_mapping),
        top_k=max(1, top_k),
        filters=filters,
        collection_name=settings.vector_collection,
    )

    combined = _rrf_merge(dense_results, sparse_results, top_k=max(1, top_k))
    return [_to_legacy(hit) for hit in combined]


def _source_filter(source_type: str | None) -> FilterExpression | None:
    if not source_type:
        return None
    return FilterExpression.from_conditions(
        [
            FilterCondition(
                field="source_type", operator=FilterOperator.EQ, value=source_type
            )
        ]
    )


def _rrf_merge(
    dense_results: list[ScoredPoint],
    sparse_results: list[ScoredPoint],
    *,
    top_k: int,
    k: int = 60,
) -> list[ScoredPoint]:
    by_id: dict[str, ScoredPoint] = {}
    scores: dict[str, float] = {}

    for rank, hit in enumerate(dense_results):
        by_id[hit.point_id] = hit
        scores[hit.point_id] = scores.get(hit.point_id, 0.0) + 1.0 / (k + rank + 1)

    for rank, hit in enumerate(sparse_results):
        existing = by_id.get(hit.point_id)
        if existing is None or hit.score > existing.score:
            by_id[hit.point_id] = hit
        scores[hit.point_id] = scores.get(hit.point_id, 0.0) + 1.0 / (k + rank + 1)

    ranked_ids = sorted(scores.keys(), key=lambda pid: scores[pid], reverse=True)
    fused: list[ScoredPoint] = []
    for pid in ranked_ids[:top_k]:
        best_hit = by_id[pid]
        fused.append(
            ScoredPoint(
                point_id=best_hit.point_id,
                score=float(scores[pid]),
                payload=best_hit.payload,
                retrieval_source=best_hit.retrieval_source,
            )
        )
    return fused


def _to_legacy(hit: ScoredPoint) -> LegacySearchResult:
    return LegacySearchResult(
        id=hit.point_id,
        score=float(hit.score),
        payload=hit.payload,
    )


def _coerce_vector_point(point: Any) -> VectorPoint:
    if isinstance(point, VectorPoint):
        return point

    # Backward-compatibility with legacy PointStruct-like objects.
    if hasattr(point, "id") and hasattr(point, "vector") and hasattr(point, "payload"):
        point_id = str(getattr(point, "id"))
        payload = dict(getattr(point, "payload") or {})
        vector_obj = getattr(point, "vector")

        dense_vector: list[float]
        sparse_vector: SparseVector | None
        if isinstance(vector_obj, dict):
            dense_raw = vector_obj.get("dense", [])
            dense_vector = [
                float(v) for v in (dense_raw.tolist() if hasattr(dense_raw, "tolist") else dense_raw)
            ]
            sparse_raw = vector_obj.get("sparse", {})
            if isinstance(sparse_raw, dict) and "indices" in sparse_raw and "values" in sparse_raw:
                sparse_vector = SparseVector.from_index_values(
                    indices=[int(i) for i in sparse_raw.get("indices", [])],
                    values=[float(v) for v in sparse_raw.get("values", [])],
                )
            elif hasattr(sparse_raw, "indices") and hasattr(sparse_raw, "values"):
                sparse_vector = SparseVector.from_index_values(
                    indices=[int(i) for i in getattr(sparse_raw, "indices", [])],
                    values=[float(v) for v in getattr(sparse_raw, "values", [])],
                )
            elif isinstance(sparse_raw, dict):
                sparse_vector = SparseVector.from_mapping(
                    {int(k): float(v) for k, v in sparse_raw.items()}
                )
            else:
                sparse_vector = SparseVector()
        else:
            dense_vector = [
                float(v)
                for v in (
                    vector_obj.tolist() if hasattr(vector_obj, "tolist") else list(vector_obj)
                )
            ]
            sparse_vector = SparseVector()

        return VectorPoint(
            point_id=point_id,
            dense_vector=dense_vector,
            sparse_vector=sparse_vector,
            payload=payload,
        )

    raise TypeError(f"Unsupported point type for upsert: {type(point)}")

