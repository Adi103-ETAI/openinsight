from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SparseVector:
    indices: list[int] = field(default_factory=list)
    values: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if len(self.indices) != len(self.values):
            raise ValueError(
                f"SparseVector indices and values must have the same length, "
                f"got {len(self.indices)} indices and {len(self.values)} values"
            )

    def is_empty(self) -> bool:
        return len(self.indices) == 0

    def to_mapping(self) -> dict[int, float]:
        return {int(i): float(v) for i, v in zip(self.indices, self.values, strict=True)}

    @classmethod
    def from_mapping(cls, mapping: dict[int, float]) -> "SparseVector":
        if not mapping:
            return cls()
        sorted_items = sorted((int(k), float(v)) for k, v in mapping.items())
        return cls(
            indices=[item[0] for item in sorted_items],
            values=[item[1] for item in sorted_items],
        )

    @classmethod
    def from_index_values(
        cls, indices: list[int], values: list[float]
    ) -> "SparseVector":
        if len(indices) != len(values):
            raise ValueError("SparseVector indices and values must have same length")
        return cls(indices=[int(i) for i in indices], values=[float(v) for v in values])


@dataclass
class VectorPoint:
    point_id: str
    dense_vector: list[float]
    sparse_vector: SparseVector | None
    payload: dict[str, Any]


@dataclass
class ScoredPoint:
    point_id: str
    score: float
    payload: dict[str, Any]
    retrieval_source: str

