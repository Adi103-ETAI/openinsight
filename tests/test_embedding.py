"""
Tests for the embedding module.

Covers:
- Sparse vector computation (TF-IDF based)
- Medical tokenization
- IDF weight calculation
- Term-to-index mapping
- BaseEmbedder abstract interface

Markers:
- unit: All tests in this module are unit tests
"""
from __future__ import annotations

import pytest

# The embedder module imports torch at module level, so we need to skip
# if torch is not available. However, the pure functions we test don't
# actually need torch - they're CPU-only TF-IDF computations.
# We use a workaround: import the pure functions directly from the module source.
torch = pytest.importorskip("torch", reason="torch required for embedding module")

from src.ml.embedding.embedder import (
    _compute_sparse_vector,
    _medical_tokenize,
    _term_to_index,
    _get_idf_weight,
    MEDICAL_COMPOUNDS,
    STOPWORDS,
)


@pytest.mark.unit
class TestMedicalTokenize:
    """Tests for medical text tokenization."""

    def test_basic_tokenization(self):
        """Basic text should be tokenized into words."""
        tokens = _medical_tokenize("patient has diabetes")
        assert "patient" in tokens
        assert "diabetes" in tokens

    def test_medical_compound_detection(self):
        """Medical compound terms should be detected and tokenized."""
        tokens = _medical_tokenize("The patient has type 2 diabetes.")
        assert "type_2_diabetes" in tokens

    def test_stopwords_removed(self):
        """Common stopwords should be filtered out."""
        tokens = _medical_tokenize("the patient has a condition")
        assert "the" not in tokens
        assert "a" not in tokens
        assert "has" not in tokens

    def test_empty_text(self):
        """Empty text should return empty token list."""
        tokens = _medical_tokenize("")
        assert tokens == []

    def test_short_words_filtered(self):
        """Words shorter than 3 characters should be filtered."""
        tokens = _medical_tokenize("an ox is in the barn")
        # "ox" is only 2 chars, should be filtered
        assert "ox" not in tokens

    @pytest.mark.parametrize(
        "compound",
        MEDICAL_COMPOUNDS[:5],  # Test first 5 compounds
    )
    def test_medical_compounds_recognized(self, compound: str):
        """All medical compounds should be recognized."""
        text = f"The study focuses on {compound}."
        tokens = _medical_tokenize(text)
        compound_token = compound.replace(" ", "_")
        assert compound_token in tokens, f"Compound '{compound}' not tokenized"

    def test_mixed_case_handling(self):
        """Tokenization should be case-insensitive."""
        tokens = _medical_tokenize("Type 2 Diabetes and TYPE 1 DIABETES")
        assert "type_2_diabetes" in tokens
        assert "type_1_diabetes" in tokens


@pytest.mark.unit
class TestGetIdfWeight:
    """Tests for IDF weight calculation."""

    @pytest.mark.parametrize(
        "term, expected_weight",
        [
            pytest.param("myocardial_infarction", 3.5, id="compound_term"),
            pytest.param("electrocardiogram", 3.0, id="long_term"),
            pytest.param("treatment", 2.0, id="medium_term"),
            pytest.param("drug", 1.0, id="short_term"),
        ],
    )
    def test_idf_weight_by_term_length(self, term: str, expected_weight: float):
        """IDF weight should vary by term characteristics."""
        assert _get_idf_weight(term) == expected_weight

    def test_compound_terms_have_highest_weight(self):
        """Compound terms (with underscores) should have highest weight."""
        assert _get_idf_weight("heart_failure") == 3.5
        assert _get_idf_weight("heart_failure") > _get_idf_weight("heart")


@pytest.mark.unit
class TestTermToIndex:
    """Tests for term-to-index hashing."""

    def test_deterministic(self):
        """Same term should always map to same index."""
        idx1 = _term_to_index("diabetes", 50000)
        idx2 = _term_to_index("diabetes", 50000)
        assert idx1 == idx2

    def test_different_terms_different_indices(self):
        """Different terms should (usually) map to different indices."""
        idx1 = _term_to_index("diabetes", 50000)
        idx2 = _term_to_index("hypertension", 50000)
        # With 50000 vocab size, collision is unlikely
        assert idx1 != idx2

    def test_index_within_vocab_range(self):
        """Index should be within vocab size range."""
        vocab_size = 1000
        for term in ["diabetes", "hypertension", "treatment"]:
            idx = _term_to_index(term, vocab_size)
            assert 0 <= idx < vocab_size

    def test_unicode_term_handling(self):
        """Unicode terms should produce valid indices."""
        idx = _term_to_index("tuberculose", 50000)
        assert 0 <= idx < 50000


@pytest.mark.unit
class TestComputeSparseVector:
    """Tests for sparse vector computation."""

    def test_basic_sparse_vector(self):
        """Basic text should produce non-empty sparse vector."""
        result = _compute_sparse_vector("patient has diabetes", 50000)
        assert len(result["indices"]) > 0
        assert len(result["values"]) > 0
        assert len(result["indices"]) == len(result["values"])

    def test_empty_text(self):
        """Empty text should produce empty sparse vector."""
        result = _compute_sparse_vector("", 50000)
        assert result["indices"] == []
        assert result["values"] == []

    def test_indices_sorted(self):
        """Indices should be sorted in ascending order."""
        result = _compute_sparse_vector("diabetes treatment hypertension", 50000)
        indices = result["indices"]
        assert indices == sorted(indices)

    def test_values_positive(self):
        """All values should be positive."""
        result = _compute_sparse_vector("patient has diabetes mellitus", 50000)
        assert all(v > 0 for v in result["values"])

    def test_medical_compounds_weighted_higher(self):
        """Medical compound terms should contribute higher weights."""
        # Text with compound term
        result_compound = _compute_sparse_vector("type 2 diabetes", 50000)
        # Text without compound
        result_simple = _compute_sparse_vector("diabetes", 50000)
        # Compound text should have higher total weight
        assert sum(result_compound["values"]) >= sum(result_simple["values"])

    def test_vocabulary_size_affects_index_range(self):
        """Vocabulary size should constrain index range."""
        small_vocab = 100
        result = _compute_sparse_vector("test content here", small_vocab)
        assert all(0 <= idx < small_vocab for idx in result["indices"])

    def test_stopwords_excluded(self):
        """Stopwords should not contribute to sparse vector."""
        # Text with only stopwords
        result = _compute_sparse_vector("the a an is are", 50000)
        assert result["indices"] == []
        assert result["values"] == []
