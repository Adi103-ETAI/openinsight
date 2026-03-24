"""
Vector DB — Qdrant
Stores embeddings of document chunks for semantic search.
Each point in the collection maps to one ChunkRecord in MongoDB.
"""
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue
)
from typing import Optional
from loguru import logger
from src.core.config import get_settings

settings = get_settings()

_client: Optional[QdrantClient] = None


def get_qdrant() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(url=settings.qdrant_url)
    return _client


def ensure_collection():
    """Create the Qdrant collection with dense + sparse vectors if it doesn't exist."""
    from qdrant_client.models import (
        VectorParams, SparseVectorParams, Distance,
        SparseIndexParams
    )
    client = get_qdrant()
    existing = [c.name for c in client.get_collections().collections]
    if settings.qdrant_collection not in existing:
        client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config={
                "dense": VectorParams(
                    size=settings.embedding_dim,
                    distance=Distance.COSINE,
                )
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(
                    index=SparseIndexParams(on_disk=False)
                )
            }
        )
        logger.info(f"Created Qdrant collection with hybrid search: {settings.qdrant_collection}")
    else:
        logger.info(f"Collection already exists: {settings.qdrant_collection}")


def upsert_chunks(points: list[PointStruct]):
    """Batch upsert embedded chunks into Qdrant."""
    client = get_qdrant()
    client.upsert(
        collection_name=settings.qdrant_collection,
        points=points,
    )


def search(
    query_vector: list[float],
    top_k: int = 8,
    source_type: Optional[str] = None,   # filter by "icmr", "pubmed", etc.
) -> list:
    """Semantic search with optional source filter."""
    client = get_qdrant()
    query_filter = None
    if source_type:
        query_filter = Filter(
            must=[FieldCondition(
                key="source_type",
                match=MatchValue(value=source_type)
            )]
        )
    return client.search(
        collection_name=settings.qdrant_collection,
        query_vector=query_vector,
        limit=top_k,
        query_filter=query_filter,
        with_payload=True,
    )


def build_sparse_vector(text: str) -> dict:
    """
    Build a simple sparse vector from text using term frequencies.
    Maps term hash → frequency. Used for BM25-style keyword search.
    """
    import re
    from collections import Counter
    tokens = re.findall(r"\b[a-zA-Z0-9]{2,}\b", text.lower())
    tf = Counter(tokens)
    # Use positive integer indices (hash mod large prime)
    sparse = {abs(hash(term)) % 2_000_000: float(freq) for term, freq in tf.items()}
    return sparse


def hybrid_search(
    query_text: str,
    query_vector: list[float],
    top_k: int = 50,
    source_type: Optional[str] = None,
) -> list:
    """
    Hybrid search — combine dense semantic search with sparse keyword search.
    Falls back to dense-only search if hybrid fails.
    """
    from qdrant_client.models import (
        SparseVector, Filter, FieldCondition, MatchValue,
        Prefetch, FusionQuery, Fusion
    )
    client = get_qdrant()

    query_filter = None
    if source_type:
        query_filter = Filter(
            must=[FieldCondition(key="source_type", match=MatchValue(value=source_type))]
        )

    try:
        sparse_vec = build_sparse_vector(query_text)

        results = client.query_points(
            collection_name=settings.qdrant_collection,
            prefetch=[
                Prefetch(
                    query=query_vector,
                    using="dense",
                    limit=top_k,
                ),
                Prefetch(
                    query=SparseVector(
                        indices=list(sparse_vec.keys()),
                        values=list(sparse_vec.values())
                    ),
                    using="sparse",
                    limit=top_k,
                ),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=top_k,
            with_payload=True,
            query_filter=query_filter,
        )
        return results.points if hasattr(results, "points") else results
    except Exception as exc:
        logger.warning(f"Hybrid search failed, falling back to dense: {exc}")
        return search(query_vector, top_k=top_k, source_type=source_type)
