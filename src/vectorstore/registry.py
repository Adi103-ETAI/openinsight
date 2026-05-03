from __future__ import annotations

from src.core.config import get_settings
from src.vectorstore.base import VectorStore

_VECTOR_STORE: VectorStore | None = None


def get_vector_store() -> VectorStore:
    global _VECTOR_STORE
    if _VECTOR_STORE is not None:
        return _VECTOR_STORE

    settings = get_settings()
    backend = settings.vector_backend.strip().lower()

    if backend == "milvus":
        from src.vectorstore.backends.milvus_store import MilvusVectorStore

        _VECTOR_STORE = MilvusVectorStore(
            uri=settings.vector_uri,
            token=settings.vector_token,
            db_name=settings.milvus_db_name,
            default_collection=settings.vector_collection,
            dense_dim=settings.vector_dim,
            id_field=settings.vector_id_field,
            dense_field=settings.vector_dense_field,
            sparse_field=settings.vector_sparse_field,
            dense_metric=settings.vector_dense_metric,
            sparse_metric=settings.vector_sparse_metric,
        )
        return _VECTOR_STORE

    raise ValueError(f"Unsupported vector backend: {settings.vector_backend}")


def reset_vector_store() -> None:
    global _VECTOR_STORE
    _VECTOR_STORE = None

