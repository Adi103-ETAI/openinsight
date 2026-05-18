"""
Tests for the MMR (Maximal Marginal Relevance) module.

Covers:
- Cosine similarity computation
- MMR selection algorithm
- Lambda parameter effects
- Edge cases (empty input, single chunk)

Markers:
- unit: All tests in this module are unit tests
"""
from __future__ import annotations

import numpy as np
import pytest

# MMR imports from retriever which imports from embedder (requires torch)
torch = pytest.importorskip("torch", reason="torch required for embedding module")

from src.query.search.mmr import cosine_similarity, maximal_marginal_relevance
from src.query.search.retriever import RetrievedChunk


@pytest.mark.unit
class TestCosineSimilarity:
    """Tests for cosine similarity computation."""

    def test_identical_vectors(self):
        """Identical vectors should have similarity of 1.0."""
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([1.0, 0.0, 0.0])
        assert cosine_similarity(a, b) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        """Orthogonal vectors should have similarity of 0.0."""
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        assert cosine_similarity(a, b) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        """Opposite vectors should have similarity of -1.0."""
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([-1.0, 0.0, 0.0])
        assert cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_zero_vector(self):
        """Zero vector should result in 0.0 similarity."""
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([1.0, 0.0, 0.0])
        assert cosine_similarity(a, b) == 0.0

    def test_both_zero_vectors(self):
        """Both zero vectors should result in 0.0 similarity."""
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([0.0, 0.0, 0.0])
        assert cosine_similarity(a, b) == 0.0

    def test_partial_similarity(self):
        """Partially similar vectors should have intermediate similarity."""
        a = np.array([1.0, 1.0, 0.0])
        b = np.array([1.0, 0.0, 0.0])
        sim = cosine_similarity(a, b)
        assert 0.0 < sim < 1.0


@pytest.mark.unit
class TestMaximalMarginalRelevance:
    """Tests for MMR selection algorithm."""

    @pytest.fixture
    def sample_chunks(self):
        """Create sample chunks with varying scores."""
        return [
            RetrievedChunk(
                chunk_id=f"chunk_{i}",
                doc_id="doc_1",
                score=0.9 - i * 0.1,
                text=f"Medical content {i}",
                contextual_text=f"Context {i}",
                metadata={"evidence_level": 1, "year": 2024},
                retrieval_source="dense",
            )
            for i in range(5)
        ]

    def test_empty_chunks(self, mock_embedder):
        """Empty chunk list should return empty result."""
        result = maximal_marginal_relevance(mock_embedder, [], n_select=3)
        assert result == []

    def test_single_chunk(self, mock_embedder, sample_chunks):
        """Single chunk should be returned as-is."""
        result = maximal_marginal_relevance(mock_embedder, sample_chunks[:1], n_select=3)
        assert len(result) == 1

    def test_fewer_chunks_than_n_select(self, mock_embedder, sample_chunks):
        """Fewer chunks than n_select should return all chunks."""
        result = maximal_marginal_relevance(mock_embedder, sample_chunks[:2], n_select=5)
        assert len(result) == 2

    def test_selects_top_k_chunks(self, mock_embedder, sample_chunks):
        """MMR should select exactly n_select chunks."""
        result = maximal_marginal_relevance(mock_embedder, sample_chunks, n_select=3)
        assert len(result) == 3

    def test_first_chunk_is_highest_scoring(self, mock_embedder, sample_chunks):
        """First selected chunk should be the highest scoring one."""
        result = maximal_marginal_relevance(mock_embedder, sample_chunks, n_select=1)
        assert result[0].chunk_id == "chunk_0"  # Highest score

    def test_lambda_affects_diversity(self, mock_embedder, sample_chunks):
        """Higher lambda should favor relevance; lower lambda should favor diversity."""
        # With lambda=1.0, only relevance matters
        result_high_lambda = maximal_marginal_relevance(
            mock_embedder, sample_chunks, lambda_param=1.0, n_select=3,
        )
        # With lambda=0.0, only diversity matters
        result_low_lambda = maximal_marginal_relevance(
            mock_embedder, sample_chunks, lambda_param=0.0, n_select=3,
        )
        # Results should differ (different chunk orderings)
        high_ids = [c.chunk_id for c in result_high_lambda]
        low_ids = [c.chunk_id for c in result_low_lambda]
        assert high_ids != low_ids or len(high_ids) != len(low_ids)

    def test_top_k_parameter_alias(self, mock_embedder, sample_chunks):
        """top_k parameter should work as alias for n_select."""
        result = maximal_marginal_relevance(
            mock_embedder, sample_chunks, top_k=2,
        )
        assert len(result) == 2

    def test_preserves_chunk_data(self, mock_embedder, sample_chunks):
        """Selected chunks should preserve original data."""
        result = maximal_marginal_relevance(mock_embedder, sample_chunks, n_select=2)
        for chunk in result:
            assert chunk.doc_id == "doc_1"
            assert "evidence_level" in chunk.metadata
