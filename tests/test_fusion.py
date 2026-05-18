"""
Tests for the reciprocal rank fusion module.

Covers:
- Basic RRF scoring
- Evidence level boosting
- Recency boosting
- Retrieval source assignment
- Top-N limiting

Markers:
- unit: All tests in this module are unit tests
"""
from __future__ import annotations

import pytest

# Fusion imports from retriever which imports from embedder (requires torch)
torch = pytest.importorskip("torch", reason="torch required for embedding module")

from src.query.search.fusion import reciprocal_rank_fusion
from src.query.search.retriever import RetrievedChunk
from src.constants import RRF_K


def _make_retrieved_chunk(
    chunk_id: str,
    score: float = 0.5,
    evidence_level: int = 3,
    year: int = 2024,
    metadata: dict | None = None,
) -> RetrievedChunk:
    """Factory for creating RetrievedChunk test instances."""
    meta = metadata or {}
    meta["evidence_level"] = evidence_level
    meta["year"] = year
    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_id="doc_1",
        score=score,
        text=f"Content for {chunk_id}",
        contextual_text=f"Context for {chunk_id}",
        metadata=meta,
        retrieval_source="unknown",
    )


@pytest.mark.unit
class TestReciprocalRankFusion:
    """Tests for RRF fusion algorithm."""

    def test_basic_fusion(self):
        """Basic RRF should combine dense and sparse results."""
        dense = [_make_retrieved_chunk("chunk_a", score=0.9)]
        sparse = [_make_retrieved_chunk("chunk_b", score=0.8)]

        result = reciprocal_rank_fusion(dense, sparse, top_n=10)

        assert len(result) == 2
        chunk_ids = {c.chunk_id for c in result}
        assert "chunk_a" in chunk_ids
        assert "chunk_b" in chunk_ids

    def test_duplicate_chunks_merged(self):
        """Same chunk in both lists should appear once."""
        chunk = _make_retrieved_chunk("shared", score=0.7)
        dense = [chunk]
        sparse = [chunk]

        result = reciprocal_rank_fusion(dense, sparse, top_n=10)

        assert len(result) == 1
        assert result[0].chunk_id == "shared"

    def test_rrf_scoring(self):
        """RRF scores should reflect rank positions."""
        dense = [
            _make_retrieved_chunk("first", score=0.9),
            _make_retrieved_chunk("second", score=0.8),
        ]
        sparse = []

        result = reciprocal_rank_fusion(dense, sparse, top_n=10)

        # First-ranked should have higher RRF score
        assert result[0].chunk_id == "first"
        assert result[0].score > result[1].score

    def test_top_n_limit(self):
        """Result should be limited to top_n."""
        chunks = [_make_retrieved_chunk(f"chunk_{i}") for i in range(10)]
        dense = chunks[:5]
        sparse = chunks[5:]

        result = reciprocal_rank_fusion(dense, sparse, top_n=3)

        assert len(result) == 3

    def test_empty_inputs(self):
        """Empty inputs should return empty result."""
        result = reciprocal_rank_fusion([], [], top_n=10)
        assert result == []

    def test_dense_only(self):
        """Dense-only input should work."""
        dense = [_make_retrieved_chunk("only_dense")]
        result = reciprocal_rank_fusion(dense, [], top_n=10)
        assert len(result) == 1
        assert result[0].chunk_id == "only_dense"

    def test_sparse_only(self):
        """Sparse-only input should work."""
        sparse = [_make_retrieved_chunk("only_sparse")]
        result = reciprocal_rank_fusion([], sparse, top_n=10)
        assert len(result) == 1
        assert result[0].chunk_id == "only_sparse"


@pytest.mark.unit
class TestEvidenceBoosting:
    """Tests for evidence level boosting in RRF."""

    def test_high_evidence_boosted(self):
        """High evidence level should boost RRF score."""
        high_evidence = _make_retrieved_chunk("high_ev", evidence_level=1)
        low_evidence = _make_retrieved_chunk("low_ev", evidence_level=5)

        result = reciprocal_rank_fusion([high_evidence], [low_evidence], top_n=10)

        # High evidence chunk should rank higher
        assert result[0].chunk_id == "high_ev"

    def test_evidence_level_in_metadata(self):
        """Evidence level should be read from metadata."""
        chunk = _make_retrieved_chunk("test", evidence_level=2)
        result = reciprocal_rank_fusion([chunk], [], top_n=10)
        assert result[0].metadata["evidence_level"] == 2


@pytest.mark.unit
class TestRecencyBoosting:
    """Tests for recency boosting in RRF."""

    def test_recent_content_boosted(self):
        """Recent content should have higher score."""
        recent = _make_retrieved_chunk("recent", year=2024)
        old = _make_retrieved_chunk("old", year=2010)

        result = reciprocal_rank_fusion([recent], [old], top_n=10)

        # Recent chunk should rank higher
        assert result[0].chunk_id == "recent"


@pytest.mark.unit
class TestRetrievalSourceAssignment:
    """Tests for retrieval source assignment after fusion."""

    def test_dense_only_source(self):
        """Chunk only in dense should have source 'dense'."""
        chunk = _make_retrieved_chunk("dense_only")
        result = reciprocal_rank_fusion([chunk], [], top_n=10)
        assert result[0].retrieval_source == "dense"

    def test_sparse_only_source(self):
        """Chunk only in sparse should have source 'sparse'."""
        chunk = _make_retrieved_chunk("sparse_only")
        result = reciprocal_rank_fusion([], [chunk], top_n=10)
        assert result[0].retrieval_source == "sparse"

    def test_both_source(self):
        """Chunk in both should have source 'both'."""
        chunk = _make_retrieved_chunk("both")
        result = reciprocal_rank_fusion([chunk], [chunk], top_n=10)
        assert result[0].retrieval_source == "both"


@pytest.mark.unit
class TestRRFConstants:
    """Tests for RRF constant values."""

    def test_rrf_k_is_positive(self):
        """RRF_K constant should be positive."""
        assert RRF_K > 0

    def test_custom_k_parameter(self):
        """Custom k parameter should affect scoring."""
        chunk = _make_retrieved_chunk("test")
        result_k60 = reciprocal_rank_fusion([chunk], [], k=60, top_n=10)
        result_k1 = reciprocal_rank_fusion([chunk], [], k=1, top_n=10)
        # Different k values should produce different scores
        assert result_k60[0].score != result_k1[0].score
