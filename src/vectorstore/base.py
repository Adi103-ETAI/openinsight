from __future__ import annotations

from abc import ABC, abstractmethod

from src.vectorstore.filters import FilterExpression
from src.vectorstore.types import ScoredPoint, SparseVector, VectorPoint


class VectorStore(ABC):
    @abstractmethod
    def ensure_collection(
        self, *, recreate: bool = False, collection_name: str | None = None
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def drop_collection(self, *, collection_name: str | None = None) -> None:
        raise NotImplementedError

    @abstractmethod
    def upsert_points(
        self,
        points: list[VectorPoint],
        *,
        collection_name: str | None = None,
        batch_size: int = 100,
    ) -> int:
        raise NotImplementedError

    @abstractmethod
    def search_dense(
        self,
        dense_vector: list[float],
        *,
        top_k: int,
        filters: FilterExpression | None = None,
        collection_name: str | None = None,
    ) -> list[ScoredPoint]:
        raise NotImplementedError

    @abstractmethod
    def search_sparse(
        self,
        sparse_vector: SparseVector,
        *,
        top_k: int,
        filters: FilterExpression | None = None,
        collection_name: str | None = None,
    ) -> list[ScoredPoint]:
        raise NotImplementedError

    @abstractmethod
    def health_check(self) -> bool:
        raise NotImplementedError

