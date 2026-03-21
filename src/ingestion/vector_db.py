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
from src.core.config import get_settings

settings = get_settings()

_client: Optional[QdrantClient] = None


def get_qdrant() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(url=settings.qdrant_url)
    return _client


def ensure_collection():
    """Create the Qdrant collection if it doesn't exist yet."""
    client = get_qdrant()
    existing = [c.name for c in client.get_collections().collections]
    if settings.qdrant_collection not in existing:
        client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=VectorParams(
                size=settings.embedding_dim,   # 768 for PubMedBERT
                distance=Distance.COSINE,
            ),
        )
        print(f"Created Qdrant collection: {settings.qdrant_collection}")
    else:
        print(f"Collection already exists: {settings.qdrant_collection}")


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
