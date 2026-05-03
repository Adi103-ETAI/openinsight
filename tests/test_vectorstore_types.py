"""
Unit tests for src/vectorstore/types.py and src/vectorstore/filters.py.

Covers:
- SparseVector construction validation (__post_init__)
- SparseVector round-tripping (from_mapping / to_mapping)
- SparseVector.from_index_values length guard
- SparseVector.is_empty behaviour
- FilterExpression / FilterCondition helpers
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


class TestSparseVectorConstruction:
    def test_valid_construction(self):
        sv = SparseVector(indices=[0, 2, 5], values=[0.1, 0.3, 0.5])
        assert sv.indices == [0, 2, 5]
        assert sv.values == [0.1, 0.3, 0.5]

    def test_empty_construction(self):
        sv = SparseVector()
        assert sv.indices == []
        assert sv.values == []

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError, match="same length"):
            SparseVector(indices=[0, 1, 2], values=[0.1, 0.2])

    def test_mismatched_lengths_error_message_contains_counts(self):
        with pytest.raises(ValueError, match="3 indices and 1 values"):
            SparseVector(indices=[0, 1, 2], values=[0.5])

    def test_single_element(self):
        sv = SparseVector(indices=[7], values=[0.9])
        assert sv.indices == [7]
        assert sv.values == [0.9]


# ---------------------------------------------------------------------------
# SparseVector – is_empty
# ---------------------------------------------------------------------------


class TestSparseVectorIsEmpty:
    def test_empty_vector_is_empty(self):
        assert SparseVector().is_empty() is True

    def test_non_empty_vector_not_empty(self):
        sv = SparseVector(indices=[0], values=[1.0])
        assert sv.is_empty() is False

    def test_multi_element_not_empty(self):
        sv = SparseVector(indices=[0, 1, 2], values=[0.1, 0.2, 0.3])
        assert sv.is_empty() is False


# ---------------------------------------------------------------------------
# SparseVector – to_mapping / from_mapping round-trip
# ---------------------------------------------------------------------------


class TestSparseVectorRoundTrip:
    def test_to_mapping_basic(self):
        sv = SparseVector(indices=[0, 2, 5], values=[0.1, 0.3, 0.5])
        assert sv.to_mapping() == {0: 0.1, 2: 0.3, 5: 0.5}

    def test_to_mapping_empty(self):
        assert SparseVector().to_mapping() == {}

    def test_from_mapping_basic(self):
        mapping = {2: 0.3, 0: 0.1, 5: 0.5}
        sv = SparseVector.from_mapping(mapping)
        # indices must be sorted ascending
        assert sv.indices == [0, 2, 5]
        assert sv.values == [0.1, 0.3, 0.5]

    def test_from_mapping_empty(self):
        sv = SparseVector.from_mapping({})
        assert sv.is_empty()
        assert sv.indices == []
        assert sv.values == []

    def test_round_trip_mapping(self):
        original = {10: 0.9, 3: 0.2, 7: 0.5}
        sv = SparseVector.from_mapping(original)
        recovered = sv.to_mapping()
        assert recovered == original

    def test_from_mapping_type_coercion(self):
        # keys/values are coerced to int/float
        sv = SparseVector.from_mapping({"1": 0.5, "3": 0.7})  # type: ignore[arg-type]
        assert sv.indices == [1, 3]
        assert sv.values == [0.5, 0.7]

    def test_to_mapping_type_coercion(self):
        sv = SparseVector(indices=[0, 1], values=[1, 2])  # ints, not floats
        mapping = sv.to_mapping()
        assert all(isinstance(k, int) for k in mapping)
        assert all(isinstance(v, float) for v in mapping.values())


# ---------------------------------------------------------------------------
# SparseVector – from_index_values
# ---------------------------------------------------------------------------


class TestSparseVectorFromIndexValues:
    def test_basic(self):
        sv = SparseVector.from_index_values([0, 3, 7], [0.1, 0.4, 0.8])
        assert sv.indices == [0, 3, 7]
        assert sv.values == [0.1, 0.4, 0.8]

    def test_type_coercion(self):
        sv = SparseVector.from_index_values([0, 1], [1, 2])
        assert all(isinstance(i, int) for i in sv.indices)
        assert all(isinstance(v, float) for v in sv.values)

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError, match="same length"):
            SparseVector.from_index_values([0, 1, 2], [0.1, 0.2])

    def test_empty(self):
        sv = SparseVector.from_index_values([], [])
        assert sv.is_empty()


# ---------------------------------------------------------------------------
# FilterCondition and helpers
# ---------------------------------------------------------------------------


class TestFilterConditionHelpers:
    def test_eq(self):
        cond = eq("status", "active")
        assert cond.field == "status"
        assert cond.operator == FilterOperator.EQ
        assert cond.value == "active"

    def test_in_values(self):
        cond = in_values("tag", ["a", "b"])
        assert cond.operator == FilterOperator.IN
        assert cond.value == ["a", "b"]

    def test_gt(self):
        cond = gt("score", 0.5)
        assert cond.operator == FilterOperator.GT

    def test_gte(self):
        cond = gte("score", 0.5)
        assert cond.operator == FilterOperator.GTE

    def test_lt(self):
        cond = lt("score", 0.5)
        assert cond.operator == FilterOperator.LT

    def test_lte(self):
        cond = lte("score", 0.5)
        assert cond.operator == FilterOperator.LTE

    def test_frozen(self):
        cond = eq("field", "val")
        with pytest.raises((AttributeError, TypeError)):
            cond.field = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# FilterExpression
# ---------------------------------------------------------------------------


class TestFilterExpression:
    def test_empty_expression_is_empty(self):
        expr = FilterExpression()
        assert expr.is_empty() is True

    def test_non_empty_expression_not_empty(self):
        expr = FilterExpression(must=(eq("a", 1),))
        assert expr.is_empty() is False

    def test_from_conditions(self):
        conditions = [eq("x", 1), gt("y", 2.0)]
        expr = FilterExpression.from_conditions(conditions)
        assert len(expr.must) == 2
        assert expr.must[0] == eq("x", 1)
        assert expr.must[1] == gt("y", 2.0)

    def test_from_conditions_empty(self):
        expr = FilterExpression.from_conditions([])
        assert expr.is_empty()

    def test_frozen(self):
        expr = FilterExpression()
        with pytest.raises((AttributeError, TypeError)):
            expr.must = (eq("a", 1),)  # type: ignore[misc]


# ---------------------------------------------------------------------------
# VectorPoint and ScoredPoint – basic construction
# ---------------------------------------------------------------------------


class TestVectorPoint:
    def test_construction(self):
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
        vp = VectorPoint(
            point_id="xyz",
            dense_vector=[0.5],
            sparse_vector=None,
            payload={},
        )
        assert vp.sparse_vector is None


class TestScoredPoint:
    def test_construction(self):
        sp = ScoredPoint(
            point_id="id1",
            score=0.95,
            payload={"doc": "text"},
            retrieval_source="dense",
        )
        assert sp.point_id == "id1"
        assert sp.score == 0.95
        assert sp.retrieval_source == "dense"
