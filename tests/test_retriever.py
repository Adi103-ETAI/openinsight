"""
Tests for the search retriever module.

Covers:
- RetrievedChunk and RetrievedParentChunk data models
- HybridRetriever initialization
- Parent-child chunk inference
- HYDE generation (mocked)

Markers:
- unit: All tests in this module are unit tests
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Retriever imports from embedder (requires torch)
torch = pytest.importorskip("torch", reason="torch required for embedding module")

from src.query.search.retriever import (
    RetrievedChunk,
    RetrievedParentChunk,
    HybridRetriever,
)
from src.vectorstore.types import ScoredPoint


@pytest.mark.unit
class TestRetrievedChunk:
    """Tests for RetrievedChunk data model."""

    def test_construction(self):
        """RetrievedChunk should construct with all fields."""
        chunk = RetrievedChunk(
            chunk_id="chunk_1",
            doc_id="doc_1",
            score=0.95,
            text="Medical content",
            contextual_text="Section\nMedical content",
            metadata={"evidence_level": 1, "year": 2024},
            retrieval_source="dense",
        )
        assert chunk.chunk_id == "chunk_1"
        assert chunk.doc_id == "doc_1"
        assert chunk.score == 0.95
        assert chunk.text == "Medical content"
        assert chunk.retrieval_source == "dense"


@pytest.mark.unit
class TestRetrievedParentChunk:
    """Tests for RetrievedParentChunk data model."""

    def test_construction(self):
        """RetrievedParentChunk should construct with all fields."""
        child = RetrievedChunk(
            chunk_id="child_1",
            doc_id="doc_1",
            score=0.8,
            text="Child content",
            contextual_text="Child context",
            metadata={},
            retrieval_source="dense",
        )
        parent = RetrievedParentChunk(
            chunk_id="parent_1",
            doc_id="doc_1",
            score=0.9,
            text="Parent content",
            contextual_text="Parent context",
            metadata={},
            retrieval_source="parent",
            child_chunks=[child],
            parent_chunk_id=None,
        )
        assert parent.chunk_id == "parent_1"
        assert len(parent.child_chunks) == 1
        assert parent.child_chunks[0].chunk_id == "child_1"

    def test_default_child_chunks(self):
        """Default child_chunks should be empty list."""
        parent = RetrievedParentChunk(
            chunk_id="p1",
            doc_id="d1",
            score=0.5,
            text="text",
            contextual_text="context",
            metadata={},
            retrieval_source="parent",
        )
        assert parent.child_chunks == []


@pytest.mark.unit
class TestHybridRetrieverParentInference:
    """Tests for parent chunk ID inference."""

    @pytest.fixture
    def retriever(self, mock_embedder, mock_vector_store):
        """Create HybridRetriever with mocked dependencies."""
        with patch("src.query.search.retriever.get_settings") as mock_settings, \
             patch("src.query.search.retriever.get_embedder", return_value=mock_embedder), \
             patch("src.query.search.retriever.get_vector_store", return_value=mock_vector_store):

            mock_settings.return_value = MagicMock(
                vector_collection_v2="test_collection",
                vector_dim=768,
                hyde_enabled=False,
            )

            return HybridRetriever(embedder=mock_embedder)

    def test_infer_parent_ids_basic(self, retriever):
        """Parent IDs should be inferred from child IDs."""
        child_ids = [
            "doc_123_child_parent_0_1",
            "doc_456_child_parent_2_3",
        ]
        parent_ids = retriever._infer_parent_ids(child_ids)

        assert "doc_123_parent_0" in parent_ids
        assert "doc_456_parent_2" in parent_ids

    def test_infer_parent_ids_empty(self, retriever):
        """Empty child IDs should return empty set."""
        parent_ids = retriever._infer_parent_ids([])
        assert parent_ids == set()

    def test_infer_parent_ids_no_child_pattern(self, retriever):
        """IDs without child pattern should be skipped."""
        child_ids = ["simple_id", "another_id"]
        parent_ids = retriever._infer_parent_ids(child_ids)
        assert parent_ids == set()

    def test_get_parent_id_from_child(self, retriever):
        """Parent ID should be extracted from child ID."""
        child_id = "doc_abc_child_parent_1_5"
        parent_id = retriever._get_parent_id_from_child(child_id)
        assert parent_id == "doc_abc_parent_1"

    def test_get_parent_id_from_child_no_pattern(self, retriever):
        """Non-matching ID should return None."""
        parent_id = retriever._get_parent_id_from_child("simple_id")
        assert parent_id is None


@pytest.mark.unit
class TestHybridRetrieverToChunk:
    """Tests for ScoredPoint to RetrievedChunk conversion."""

    @pytest.fixture
    def retriever(self, mock_embedder, mock_vector_store):
        """Create HybridRetriever with mocked dependencies."""
        with patch("src.query.search.retriever.get_settings") as mock_settings, \
             patch("src.query.search.retriever.get_embedder", return_value=mock_embedder), \
             patch("src.query.search.retriever.get_vector_store", return_value=mock_vector_store):

            mock_settings.return_value = MagicMock(
                vector_collection_v2="test_collection",
                vector_dim=768,
                hyde_enabled=False,
            )

            return HybridRetriever(embedder=mock_embedder)

    def test_to_chunk_basic(self, retriever):
        """ScoredPoint should convert to RetrievedChunk."""
        point = ScoredPoint(
            point_id="point_1",
            score=0.9,
            payload={
                "chunk_id": "chunk_1",
                "doc_id": "doc_1",
                "chunk_text": "Medical text",
                "contextual_text": "Context",
                "evidence_level": 1,
            },
        )
        chunk = retriever._to_chunk(point, "dense")

        assert chunk.chunk_id == "chunk_1"
        assert chunk.doc_id == "doc_1"
        assert chunk.score == 0.9
        assert chunk.text == "Medical text"
        assert chunk.retrieval_source == "dense"

    def test_to_chunk_fallback_text(self, retriever):
        """Should fallback to chunk_text if raw_text missing."""
        point = ScoredPoint(
            point_id="point_1",
            score=0.8,
            payload={
                "chunk_id": "chunk_1",
                "doc_id": "doc_1",
                "chunk_text": "Fallback text",
            },
        )
        chunk = retriever._to_chunk(point, "sparse")
        assert chunk.text == "Fallback text"

    def test_to_chunk_empty_payload(self, retriever):
        """Empty payload should use defaults."""
        point = ScoredPoint(
            point_id="point_1",
            score=0.5,
            payload={},
        )
        chunk = retriever._to_chunk(point, "dense")
        assert chunk.chunk_id == "point_1"
        assert chunk.doc_id == ""
        assert chunk.text == ""


@pytest.mark.unit
class TestHybridRetrieverToParentChunk:
    """Tests for ScoredPoint to RetrievedParentChunk conversion."""

    @pytest.fixture
    def retriever(self, mock_embedder, mock_vector_store):
        """Create HybridRetriever with mocked dependencies."""
        with patch("src.query.search.retriever.get_settings") as mock_settings, \
             patch("src.query.search.retriever.get_embedder", return_value=mock_embedder), \
             patch("src.query.search.retriever.get_vector_store", return_value=mock_vector_store):

            mock_settings.return_value = MagicMock(
                vector_collection_v2="test_collection",
                vector_dim=768,
                hyde_enabled=False,
            )

            return HybridRetriever(embedder=mock_embedder)

    def test_to_parent_chunk(self, retriever):
        """ScoredPoint should convert to RetrievedParentChunk."""
        point = ScoredPoint(
            point_id="parent_1",
            score=0.85,
            payload={
                "chunk_id": "parent_1",
                "doc_id": "doc_1",
                "chunk_text": "Parent text",
                "contextual_text": "Parent context",
            },
        )
        parent = retriever._to_parent_chunk(point)

        assert parent.chunk_id == "parent_1"
        assert parent.doc_id == "doc_1"
        assert parent.text == "Parent text"
        assert parent.retrieval_source == "parent"
        assert parent.child_chunks == []
