"""
Unit tests for src/vectorstore/types.py and src/vectorstore/filters.py.

Covers:
- SparseVector construction validation (__post_init__)
- SparseVector round-tripping (from_mapping / to_mapping)
- SparseVector.from_index_values length guard
- SparseVector.is_empty behaviour
- FilterExpression / FilterCondition helpers

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
    in_values,
    lt,
    lte,
)


# ---------------------------------------------------------------------------
# SparseVector – construction & validation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSparseVectorConstruction:
    """Tests for SparseVector construction and validation."""

    def test_valid_construction(self):
        """Valid indices and values should construct successfully."""
        sv = SparseVector(indices=[0, 2, 5], values=[0.1, 0.3, 0.5])
        assert sv.indices == [0, 2, 5]
        assert sv.values == [0.1, 0.3, 0.5]

    def test_empty_construction(self):
        """Empty SparseVector should construct with empty lists."""
        sv = SparseVector()
        assert sv.indices == []
        assert sv.values == []

    def test_mismatched_lengths_raise(self):
        """Mismatched indices and values should raise ValueError."""
        with pytest.raises(ValueError, match="same length"):
            SparseVector(indices=[0, 1, 2], values=[0.1, 0.2])

    def test_mismatched_lengths_error_message_contains_counts(self):
        """Error message should include the mismatched counts."""
        with pytest.raises(ValueError, match="3 indices and 1 values"):
            SparseVector(indices=[0, 1, 2], values=[0.5])

    def test_single_element(self):
        """Single element SparseVector should construct correctly."""
        sv = SparseVector(indices=[7], values=[0.9])
        assert sv.indices == [7]
        assert sv.values == [0.9]


# ---------------------------------------------------------------------------
# SparseVector – is_empty
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSparseVectorIsEmpty:
    """Tests for SparseVector.is_empty method."""

    def test_empty_vector_is_empty(self):
        """Empty SparseVector should report is_empty=True."""
        assert SparseVector().is_empty() is True

    def test_non_empty_vector_not_empty(self):
        """Non-empty SparseVector should report is_empty=False."""
        sv = SparseVector(indices=[0], values=[1.0])
        assert sv.is_empty() is False

    def test_multi_element_not_empty(self):
        """Multi-element SparseVector should report is_empty=False."""
        sv = SparseVector(indices=[0, 1, 2], values=[0.1, 0.2, 0.3])
        assert sv.is_empty() is False


# ---------------------------------------------------------------------------
# SparseVector – to_mapping / from_mapping round-trip
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSparseVectorRoundTrip:
    """Tests for SparseVector mapping round-trip operations."""

    def test_to_mapping_basic(self):
        """to_mapping should return correct index-value dictionary."""
        sv = SparseVector(indices=[0, 2, 5], values=[0.1, 0.3, 0.5])
        assert sv.to_mapping() == {0: 0.1, 2: 0.3, 5: 0.5}

    def test_to_mapping_empty(self):
        """to_mapping on empty vector should return empty dict."""
        assert SparseVector().to_mapping() == {}

    def test_from_mapping_basic(self):
        """from_mapping should construct SparseVector with sorted indices."""
        mapping = {2: 0.3, 0: 0.1, 5: 0.5}
        sv = SparseVector.from_mapping(mapping)
        assert sv.indices == [0, 2, 5]
        assert sv.values == [0.1, 0.3, 0.5]

    def test_from_mapping_empty(self):
        """from_mapping with empty dict should return empty SparseVector."""
        sv = SparseVector.from_mapping({})
        assert sv.is_empty()
        assert sv.indices == []
        assert sv.values == []

    def test_round_trip_mapping(self):
        """Round-trip through from_mapping/to_mapping should preserve data."""
        original = {10: 0.9, 3: 0.2, 7: 0.5}
        sv = SparseVector.from_mapping(original)
        recovered = sv.to_mapping()
        assert recovered == original

    def test_from_mapping_type_coercion(self):
        """from_mapping should coerce string keys to int."""
        sv = SparseVector.from_mapping({"1": 0.5, "3": 0.7})  # type: ignore[arg-type]
        assert sv.indices == [1, 3]
        assert sv.values == [0.5, 0.7]

    def test_to_mapping_type_coercion(self):
        """to_mapping should coerce values to float."""
        sv = SparseVector(indices=[0, 1], values=[1, 2])
        mapping = sv.to_mapping()
        assert all(isinstance(k, int) for k in mapping)
        assert all(isinstance(v, float) for v in mapping.values())


# ---------------------------------------------------------------------------
# SparseVector – from_index_values
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSparseVectorFromIndexValues:
    """Tests for SparseVector.from_index_values factory method."""

    def test_basic(self):
        """from_index_values should construct correctly."""
        sv = SparseVector.from_index_values([0, 3, 7], [0.1, 0.4, 0.8])
        assert sv.indices == [0, 3, 7]
        assert sv.values == [0.1, 0.4, 0.8]

    def test_type_coercion(self):
        """from_index_values should coerce types appropriately."""
        sv = SparseVector.from_index_values([0, 1], [1, 2])
        assert all(isinstance(i, int) for i in sv.indices)
        assert all(isinstance(v, float) for v in sv.values)

    def test_mismatched_lengths_raise(self):
        """Mismatched lengths should raise ValueError."""
        with pytest.raises(ValueError, match="same length"):
            SparseVector.from_index_values([0, 1, 2], [0.1, 0.2])

    def test_empty(self):
        """Empty lists should return empty SparseVector."""
        sv = SparseVector.from_index_values([], [])
        assert sv.is_empty()


# ---------------------------------------------------------------------------
# FilterCondition and helpers
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFilterConditionHelpers:
    """Tests for FilterCondition helper functions."""

    def test_eq(self):
        """eq helper should create EQ condition."""
        cond = eq("status", "active")
        assert cond.field == "status"
        assert cond.operator == FilterOperator.EQ
        assert cond.value == "active"

    def test_in_values(self):
        """in_values helper should create IN condition."""
        cond = in_values("tag", ["a", "b"])
        assert cond.operator == FilterOperator.IN
        assert cond.value == ["a", "b"]

    def test_gt(self):
        """gt helper should create GT condition."""
        cond = gt("score", 0.5)
        assert cond.operator == FilterOperator.GT

    def test_gte(self):
        """gte helper should create GTE condition."""
        cond = gte("score", 0.5)
        assert cond.operator == FilterOperator.GTE

    def test_lt(self):
        """lt helper should create LT condition."""
        cond = lt("score", 0.5)
        assert cond.operator == FilterOperator.LT

    def test_lte(self):
        """lte helper should create LTE condition."""
        cond = lte("score", 0.5)
        assert cond.operator == FilterOperator.LTE

    def test_frozen(self):
        """FilterCondition should be immutable (frozen dataclass)."""
        cond = eq("field", "val")
        with pytest.raises((AttributeError, TypeError)):
            cond.field = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# FilterExpression
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFilterExpression:
    """Tests for FilterExpression."""

    def test_empty_expression_is_empty(self):
        """Empty FilterExpression should report is_empty=True."""
        expr = FilterExpression()
        assert expr.is_empty() is True

    def test_non_empty_expression_not_empty(self):
        """Non-empty FilterExpression should report is_empty=False."""
        expr = FilterExpression(must=(eq("a", 1),))
        assert expr.is_empty() is False

    def test_from_conditions(self):
        """from_conditions should create FilterExpression with conditions."""
        conditions = [eq("x", 1), gt("y", 2.0)]
        expr = FilterExpression.from_conditions(conditions)
        assert len(expr.must) == 2
        assert expr.must[0] == eq("x", 1)
        assert expr.must[1] == gt("y", 2.0)

    def test_from_conditions_empty(self):
        """from_conditions with empty list should return empty expression."""
        expr = FilterExpression.from_conditions([])
        assert expr.is_empty()

    def test_frozen(self):
        """FilterExpression should be immutable (frozen dataclass)."""
        expr = FilterExpression()
        with pytest.raises((AttributeError, TypeError)):
            expr.must = (eq("a", 1),)  # type: ignore[misc]


# ---------------------------------------------------------------------------
# VectorPoint and ScoredPoint – basic construction
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestVectorPoint:
    """Tests for VectorPoint data model."""

    def test_construction(self):
        """VectorPoint should construct with all fields."""
        sv = SparseVector(indices=[0], values=[1.0])
        vp = VectorPoint(
            point_id="abc",
            dense_vector=[0.1, 0.2, 0.3],
            sparse_vector=sv,
            payload={"key": "val"},
        )
        assert vp.point_id == "abc"
        assert vp.sparse_vector is sv

    def test_none_sparse_vector(self):
        """VectorPoint should accept None for sparse_vector."""
        vp = VectorPoint(
            point_id="xyz",
            dense_vector=[0.5],
            sparse_vector=None,
            payload={},
        )
        assert vp.sparse_vector is None


@pytest.mark.unit
class TestScoredPoint:
    """Tests for ScoredPoint data model."""

    def test_construction(self):
        """ScoredPoint should construct with all fields."""
        sp = ScoredPoint(
            point_id="id1",
            score=0.95,
            payload={"doc": "text"},
            retrieval_source="dense",
        )
        assert sp.point_id == "id1"
        assert sp.score == 0.95
        assert sp.retrieval_source == "dense"

    def test_retrieval_source_stored(self):
        """retrieval_source should be stored correctly."""
        sp = ScoredPoint(
            point_id="id",
            score=0.5,
            payload={},
            retrieval_source="sparse",
        )
        assert sp.retrieval_source == "sparse"
