from __future__ import annotations

import re
from enum import Enum
from typing import Any

from pymilvus import DataType, MilvusClient

from src.vectorstore.base import VectorStore
from src.vectorstore.filters import FilterCondition, FilterExpression, FilterOperator
from src.vectorstore.types import ScoredPoint, SparseVector, VectorPoint


# Allowed field names for filtering (whitelist approach)
ALLOWED_FILTER_FIELDS = frozenset({
    "year",
    "doc_type",
    "source",
    "source_type",
    "evidence_level",
    "india_relevant",
    "has_drug_dosing",
    "chunk_type",
    "pmid",
})

# Pattern to detect potentially dangerous characters in filter values
DANGEROUS_CHARS_PATTERN = re.compile(r"[\x00-\x1f\x7f-\x9f'\";\\]")


def validate_filter_field_name(field_name: str) -> bool:
    """
    Validate that the filter field name is in the allowed list.

    This prevents injection attacks through unknown field names.
    """
    return field_name in ALLOWED_FILTER_FIELDS


def sanitize_filter_value(value: Any) -> Any:
    """
    Sanitize filter values to prevent injection attacks.

    Removes control characters and dangerous symbols from string values.
    """
    if isinstance(value, str):
        # Remove control characters and dangerous symbols
        return DANGEROUS_CHARS_PATTERN.sub("", value)
    if isinstance(value, (list, tuple, set)):
        return type(value)(sanitize_filter_value(v) for v in value)
    return value


class MilvusVectorStore(VectorStore):
    def __init__(
        self,
        *,
        uri: str,
        token: str,
        db_name: str,
        default_collection: str,
        dense_dim: int,
        id_field: str,
        dense_field: str,
        sparse_field: str,
        dense_metric: str,
        sparse_metric: str,
        is_cloud: bool = False,
    ) -> None:
        self.default_collection = default_collection
        self.dense_dim = dense_dim
        self.id_field = id_field
        self.dense_field = dense_field
        self.sparse_field = sparse_field
        self.dense_metric = dense_metric
        self.sparse_metric = sparse_metric
        self.is_cloud = is_cloud  # Milvus Cloud doesn't need load_collection on each search

        # Handle None or empty token properly to avoid AttributeError
        if token and token.strip():
            self.client = MilvusClient(uri=uri, token=token, db_name=db_name)
        else:
            self.client = MilvusClient(uri=uri, db_name=db_name)

    def ensure_collection(
        self, *, recreate: bool = False, collection_name: str | None = None
    ) -> None:
        target = self._resolve_collection_name(collection_name)

        if recreate and self.client.has_collection(target):
            self.client.drop_collection(target)

        if self.client.has_collection(target):
            self.client.load_collection(target)
            return

        schema = MilvusClient.create_schema(
            auto_id=False,
            enable_dynamic_field=True,
        )
        schema.add_field(
            field_name=self.id_field,
            datatype=DataType.VARCHAR,
            is_primary=True,
            max_length=128,
        )
        schema.add_field(
            field_name=self.dense_field,
            datatype=DataType.FLOAT_VECTOR,
            dim=self.dense_dim,
        )
        schema.add_field(
            field_name=self.sparse_field,
            datatype=DataType.SPARSE_FLOAT_VECTOR,
        )
        schema.add_field(field_name="year", datatype=DataType.INT64)
        schema.add_field(field_name="doc_type", datatype=DataType.VARCHAR, max_length=64)
        schema.add_field(field_name="source", datatype=DataType.VARCHAR, max_length=64)
        schema.add_field(
            field_name="source_type", datatype=DataType.VARCHAR, max_length=64
        )
        schema.add_field(
            field_name="evidence_level", datatype=DataType.VARCHAR, max_length=16
        )
        schema.add_field(field_name="india_relevant", datatype=DataType.BOOL)
        schema.add_field(field_name="has_drug_dosing", datatype=DataType.BOOL)
        schema.add_field(
            field_name="chunk_type", datatype=DataType.VARCHAR, max_length=64
        )
        schema.add_field(field_name="pmid", datatype=DataType.VARCHAR, max_length=64)

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name=self.dense_field,
            metric_type=self.dense_metric,
        )
        index_params.add_index(
            field_name=self.sparse_field,
            index_type="SPARSE_INVERTED_INDEX",
            metric_type=self.sparse_metric,
        )

        self.client.create_collection(
            collection_name=target,
            schema=schema,
            index_params=index_params,
        )
        self.client.load_collection(target)

    def drop_collection(self, *, collection_name: str | None = None) -> None:
        target = self._resolve_collection_name(collection_name)
        if self.client.has_collection(target):
            self.client.drop_collection(target)

    def upsert_points(
        self,
        points: list[VectorPoint],
        *,
        collection_name: str | None = None,
        batch_size: int = 100,
    ) -> int:
        if not points:
            return 0

        target = self._resolve_collection_name(collection_name)
        rows = [self._serialize_point(point) for point in points]

        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            self.client.upsert(collection_name=target, data=batch)

        return len(rows)

    def search_dense(
        self,
        dense_vector: list[float],
        *,
        top_k: int,
        filters: FilterExpression | None = None,
        collection_name: str | None = None,
    ) -> list[ScoredPoint]:
        if not dense_vector:
            return []

        target = self._resolve_collection_name(collection_name)
        # Milvus Cloud handles collection loading automatically - skip explicit load
        if not self.is_cloud:
            self.client.load_collection(target)
        results = self.client.search(
            collection_name=target,
            data=[[float(v) for v in dense_vector]],
            anns_field=self.dense_field,
            limit=max(1, top_k),
            filter=self._to_milvus_filter(filters),
            output_fields=["*"],
            search_params={"metric_type": self.dense_metric, "params": {"level": 2}},
        )
        hits = results[0] if results else []
        return [self._to_scored_point(hit, "dense") for hit in hits]

    def search_sparse(
        self,
        sparse_vector: SparseVector,
        *,
        top_k: int,
        filters: FilterExpression | None = None,
        collection_name: str | None = None,
    ) -> list[ScoredPoint]:
        if sparse_vector.is_empty():
            return []

        target = self._resolve_collection_name(collection_name)
        # Milvus Cloud handles collection loading automatically - skip explicit load
        if not self.is_cloud:
            self.client.load_collection(target)
        results = self.client.search(
            collection_name=target,
            data=[sparse_vector.to_mapping()],
            anns_field=self.sparse_field,
            limit=max(1, top_k),
            filter=self._to_milvus_filter(filters),
            output_fields=["*"],
            search_params={"metric_type": self.sparse_metric, "params": {}},
        )
        hits = results[0] if results else []
        return [self._to_scored_point(hit, "sparse") for hit in hits]

    def health_check(self) -> bool:
        try:
            self.client.list_collections()
            return True
        except (RuntimeError, ValueError, TypeError):
            return False

    def _resolve_collection_name(self, collection_name: str | None) -> str:
        return collection_name or self.default_collection

    def _serialize_point(self, point: VectorPoint) -> dict[str, Any]:
        payload = self._json_safe(point.payload)
        row = dict(payload)

        row[self.id_field] = str(point.point_id)
        row[self.dense_field] = [float(v) for v in point.dense_vector]
        row[self.sparse_field] = (
            point.sparse_vector.to_mapping() if point.sparse_vector is not None else {}
        )

        row["year"] = self._coerce_int(payload.get("year", 0))
        row["doc_type"] = str(payload.get("doc_type", "unknown"))
        row["source"] = str(payload.get("source", payload.get("source_type", "unknown")))
        row["source_type"] = str(payload.get("source_type", payload.get("source", "unknown")))
        row["evidence_level"] = str(payload.get("evidence_level", "unknown"))
        row["india_relevant"] = bool(payload.get("india_relevant", False))
        row["has_drug_dosing"] = bool(payload.get("has_drug_dosing", False))
        row["chunk_type"] = str(payload.get("chunk_type", "unknown"))
        row["pmid"] = str(payload.get("pmid") or "")

        return row

    def _to_milvus_filter(self, filters: FilterExpression | None) -> str:
        if filters is None or filters.is_empty():
            return ""
        return " and ".join(self._condition_to_expr(cond) for cond in filters.must)

    def _condition_to_expr(self, condition: FilterCondition) -> str:
        field_name = condition.field

        # Validate field name against whitelist
        if not validate_filter_field_name(field_name):
            raise ValueError(
                f"Invalid filter field: '{field_name}'. "
                f"Allowed fields: {', '.join(sorted(ALLOWED_FILTER_FIELDS))}"
            )

        # Sanitize the value to prevent injection
        value = sanitize_filter_value(condition.value)

        if condition.operator == FilterOperator.EQ:
            return f"{field_name} == {self._format_value(value)}"
        if condition.operator == FilterOperator.IN:
            if not isinstance(value, (list, tuple, set)):
                raise ValueError(f"IN filter expects list-like value for '{field_name}'")
            rendered = ", ".join(self._format_value(v) for v in value)
            return f"{field_name} in [{rendered}]"
        if condition.operator == FilterOperator.GT:
            return f"{field_name} > {self._format_value(value)}"
        if condition.operator == FilterOperator.GTE:
            return f"{field_name} >= {self._format_value(value)}"
        if condition.operator == FilterOperator.LT:
            return f"{field_name} < {self._format_value(value)}"
        if condition.operator == FilterOperator.LTE:
            return f"{field_name} <= {self._format_value(value)}"

        raise ValueError(f"Unsupported filter operator: {condition.operator}")

    def _format_value(self, value: Any) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)):
            return str(value)
        if isinstance(value, Enum):
            return self._format_value(value.value)
        text = str(value).replace("\\", "\\\\").replace('"', '\\"')
        return f'"{text}"'

    def _to_scored_point(self, hit: Any, source: str) -> ScoredPoint:
        if isinstance(hit, dict):
            point_id = str(hit.get("id", ""))
            score = float(hit.get("distance", hit.get("score", 0.0)) or 0.0)
            entity = hit.get("entity") or {}
        else:
            point_id = str(getattr(hit, "id", ""))
            score = float(getattr(hit, "distance", getattr(hit, "score", 0.0)) or 0.0)
            entity = getattr(hit, "entity", {}) or {}

        payload = dict(entity) if isinstance(entity, dict) else {}
        payload.pop(self.dense_field, None)
        payload.pop(self.sparse_field, None)
        payload.setdefault("chunk_id", point_id)

        return ScoredPoint(
            point_id=point_id,
            score=score,
            payload=payload,
            retrieval_source=source,
        )

    def _coerce_int(self, value: Any) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    def _json_safe(self, value: Any) -> Any:
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, dict):
            return {k: self._json_safe(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._json_safe(v) for v in value]
        return value

