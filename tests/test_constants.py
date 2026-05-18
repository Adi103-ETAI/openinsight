"""
Tests for the constants module.

Covers:
- EvidenceBoost values and get_boost method
- RecencyBoost values and get_boost method
- RRF_K constant value

Markers:
- unit: All tests in this module are unit tests
"""
from __future__ import annotations

import pytest

from src.constants import EvidenceBoost, RecencyBoost, RRF_K


@pytest.mark.unit
class TestEvidenceBoost:
    """Tests for evidence level boost values."""

    def test_evidence_boost_returns_numeric(self):
        """Evidence boost should return a numeric value."""
        for level in ["1", "2", "3", "4", "5", "unknown"]:
            boost = EvidenceBoost.get_boost(level)
            assert isinstance(boost, (int, float))
            assert boost > 0

    def test_unknown_evidence_level(self):
        """Unknown evidence level should return a valid boost."""
        boost = EvidenceBoost.get_boost("unknown")
        assert boost > 0

    def test_empty_evidence_level(self):
        """Empty evidence level should return a valid boost."""
        boost = EvidenceBoost.get_boost("")
        assert boost > 0

    def test_numeric_string_levels(self):
        """Numeric string levels should return valid boosts."""
        for level in ["1", "2", "3", "4", "5"]:
            boost = EvidenceBoost.get_boost(level)
            assert boost > 0, f"Boost for level {level} should be positive"


@pytest.mark.unit
class TestRecencyBoost:
    """Tests for recency boost values."""

    def test_recency_boost_returns_numeric(self):
        """Recency boost should return a numeric value."""
        for year in [2024, 2020, 2010, 2000, 0, -1]:
            boost = RecencyBoost.get_boost(year)
            assert isinstance(boost, (int, float))
            assert boost > 0

    def test_current_year_boost(self):
        """Current year should have a valid boost."""
        boost = RecencyBoost.get_boost(2024)
        assert boost > 0

    def test_zero_year(self):
        """Year 0 should return a valid boost."""
        boost = RecencyBoost.get_boost(0)
        assert boost > 0

    def test_negative_year(self):
        """Negative year should return a valid boost."""
        boost = RecencyBoost.get_boost(-1)
        assert boost > 0


@pytest.mark.unit
class TestRRFK:
    """Tests for RRF_K constant."""

    def test_rrf_k_is_positive(self):
        """RRF_K should be positive."""
        assert RRF_K > 0

    def test_rrf_k_is_integer(self):
        """RRF_K should be an integer."""
        assert isinstance(RRF_K, int)

    def test_rrf_k_typical_value(self):
        """RRF_K is typically 60 in literature."""
        assert RRF_K == 60
