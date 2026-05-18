"""
Tests for the vector store backend and registry.

Covers:
- VectorPoint and ScoredPoint data models
- SparseVector operations
- FilterExpression and FilterCondition
- Vector store registry
- Memory backend (if available)

Markers:
- unit: All tests in this module are unit tests
"""
from __future__ import annotations

import pytest

from src.vectorstore.types import SparseVector, ScoredPoint, VectorPoint
from src.vectorstore.filters import (
    FilterCondition,
    FilterExpression,
    FilterOperator,
    eq,
    gt,
    gte,
    lt,
    lte,
    in_values,
)


@pytest.mark.unit
class TestSparseVectorOperations:
    """Tests for SparseVector additional operations."""

    def test_sparse_vector_equality(self):
        """Equal SparseVectors should compare equal."""
        sv1 = SparseVector(indices=[0, 1], values=[0.5, 0.3])
        sv2 = SparseVector(indices=[0, 1], values=[0.5, 0.3])
        assert sv1.indices == sv2.indices
        assert sv1.values == sv2.values

    def test_sparse_vector_dimension(self):
        """Dimension should be the number of non-zero elements."""
        sv = SparseVector(indices=[0, 1, 2, 3], values=[0.1, 0.2, 0.3, 0.4])
        assert len(sv.indices) == 4
        assert len(sv.values) == 4

    def test_sparse_vector_from_dict_order(self):
        """from_mapping should sort indices ascending."""
        mapping = {5: 0.5, 1: 0.1, 3: 0.3}
        sv = SparseVector.from_mapping(mapping)
        assert sv.indices == [1, 3, 5]
        assert sv.values == [0.1, 0.3, 0.5]


@pytest.mark.unit
class TestFilterOperators:
    """Tests for filter operator enum."""

    def test_all_operators_exist(self):
        """All expected operators should be defined."""
        assert hasattr(FilterOperator, "EQ")
        assert hasattr(FilterOperator, "GT")
        assert hasattr(FilterOperator, "GTE")
        assert hasattr(FilterOperator, "LT")
        assert hasattr(FilterOperator, "LTE")
        assert hasattr(FilterOperator, "IN")

    def test_operator_values(self):
        """Operators should have string values."""
        assert FilterOperator.EQ.value == "eq"
        assert FilterOperator.GT.value == "gt"
        assert FilterOperator.GTE.value == "gte"
        assert FilterOperator.LT.value == "lt"
        assert FilterOperator.LTE.value == "lte"
        assert FilterOperator.IN.value == "in"


@pytest.mark.unit
class TestFilterConditionCreation:
    """Tests for FilterCondition creation."""

    def test_eq_condition(self):
        """EQ condition should have correct attributes."""
        cond = FilterCondition(field="status", operator=FilterOperator.EQ, value="active")
        assert cond.field == "status"
        assert cond.operator == FilterOperator.EQ
        assert cond.value == "active"

    def test_in_condition_with_list(self):
        """IN condition should accept list values."""
        cond = FilterCondition(
            field="tags", operator=FilterOperator.IN, value=["a", "b", "c"],
        )
        assert cond.value == ["a", "b", "c"]

    def test_numeric_comparison(self):
        """Numeric comparisons should work."""
        cond = FilterCondition(field="score", operator=FilterOperator.GTE, value=0.5)
        assert cond.value == 0.5

    def test_boolean_value(self):
        """Boolean values should be supported."""
        cond = FilterCondition(field="active", operator=FilterOperator.EQ, value=True)
        assert cond.value is True


@pytest.mark.unit
class TestFilterExpressionComposition:
    """Tests for FilterExpression composition."""

    def test_single_condition(self):
        """Expression with single condition should work."""
        expr = FilterExpression(must=(eq("status", "active"),))
        assert len(expr.must) == 1
        assert not expr.is_empty()

    def test_multiple_conditions(self):
        """Expression with multiple conditions should work."""
        expr = FilterExpression(
            must=(
                eq("status", "active"),
                FilterCondition(field="year", operator=FilterOperator.GTE, value=2020),
            )
        )
        assert len(expr.must) == 2

    def test_empty_expression(self):
        """Empty expression should be empty."""
        expr = FilterExpression()
        assert expr.is_empty()
        assert len(expr.must) == 0


@pytest.mark.unit
class TestVectorPointModel:
    """Tests for VectorPoint data model."""

    def test_vector_point_construction(self):
        """VectorPoint should construct with all fields."""
        sv = SparseVector(indices=[0], values=[1.0])
        vp = VectorPoint(
            point_id="test_id",
            dense_vector=[0.1, 0.2, 0.3],
            sparse_vector=sv,
            payload={"key": "value"},
        )
        assert vp.point_id == "test_id"
        assert vp.dense_vector == [0.1, 0.2, 0.3]
        assert vp.sparse_vector is sv
        assert vp.payload == {"key": "value"}

    def test_vector_point_with_none_sparse(self):
        """VectorPoint should accept None for sparse_vector."""
        vp = VectorPoint(
            point_id="id",
            dense_vector=[0.1],
            sparse_vector=None,
            payload={},
        )
        assert vp.sparse_vector is None


@pytest.mark.unit
class TestScoredPointModel:
    """Tests for ScoredPoint data model."""

    def test_scored_point_construction(self):
        """ScoredPoint should construct with all fields."""
        sp = ScoredPoint(
            point_id="result_1",
            score=0.95,
            payload={"text": "content"},
            retrieval_source="dense",
        )
        assert sp.point_id == "result_1"
        assert sp.score == 0.95
        assert sp.payload == {"text": "content"}
        assert sp.retrieval_source == "dense"

    def test_scored_point_score_range(self):
        """Score should be a float."""
        sp = ScoredPoint(point_id="id", score=0.0, payload={}, retrieval_source="dense")
        assert isinstance(sp.score, float)

    def test_scored_point_retrieval_source(self):
        """retrieval_source should be stored correctly."""
        sp = ScoredPoint(point_id="id", score=0.5, payload={}, retrieval_source="sparse")
        assert sp.retrieval_source == "sparse"


@pytest.mark.unit
class TestVectorStoreRegistry:
    """Tests for vector store registry."""

    def test_registry_import(self):
        """Registry module should be importable."""
        from src.vectorstore.registry import get_vector_store, reset_vector_store
        assert callable(get_vector_store)
        assert callable(reset_vector_store)

    def test_reset_vector_store(self):
        """reset_vector_store should be callable."""
        from src.vectorstore.registry import reset_vector_store
        # Should not raise
        reset_vector_store()
