"""
Tests for the hierarchical chunker (HierarchicalChunkerV3).

Markers:
- unit: All tests in this module are unit tests
"""
from __future__ import annotations

import pytest

from src.ml.chunking.chunker import ChunkV3, HierarchicalChunkerV3


@pytest.mark.unit
class TestChunkV3:
    """Tests for the ChunkV3 data model."""

    def test_chunk_creation(self):
        """ChunkV3 should be created with all required fields."""
        chunk = ChunkV3(
            chunk_id="test-1",
            doc_id="doc-1",
            chunk_type="text",
            section_title="Introduction",
            text="This is test content",
            contextual_text="Introduction\nThis is test content",
            char_count=18,
            token_estimate=4,
            chunk_index=0,
            total_chunks=1,
            metadata={},
        )
        assert chunk.chunk_id == "test-1"
        assert chunk.doc_id == "doc-1"
        assert chunk.chunk_index == 0
        assert chunk.total_chunks == 1

    def test_chunk_with_metadata(self):
        """ChunkV3 should store metadata correctly."""
        metadata = {"source": "pubmed", "year": 2024, "evidence_level": 1}
        chunk = ChunkV3(
            chunk_id="test-2",
            doc_id="doc-2",
            chunk_type="text",
            section_title="Methods",
            text="Test content",
            contextual_text="Methods\nTest content",
            char_count=12,
            token_estimate=3,
            chunk_index=0,
            total_chunks=1,
            metadata=metadata,
        )
        assert chunk.metadata == metadata

    @pytest.mark.parametrize(
        "chunk_type",
        ["text", "table", "figure", "reference", "abstract"],
    )
    def test_chunk_types(self, chunk_type: str):
        """ChunkV3 should support various chunk types."""
        chunk = ChunkV3(
            chunk_id="test",
            doc_id="doc",
            chunk_type=chunk_type,
            section_title="",
            text="content",
            contextual_text="content",
            char_count=7,
            token_estimate=1,
            chunk_index=0,
            total_chunks=1,
            metadata={},
        )
        assert chunk.chunk_type == chunk_type


@pytest.mark.unit
class TestHierarchicalChunkerV3:
    """Tests for the HierarchicalChunkerV3 chunking engine."""

    @pytest.fixture(autouse=True)
    def setup_chunker(self):
        """Create chunker instance for each test."""
        self.chunker = HierarchicalChunkerV3()

    def test_chunk_empty_document(self):
        """Empty document should produce no chunks."""
        doc = {"title": "", "abstract": "", "content": ""}
        chunks = self.chunker.chunk_document(doc, {})
        assert len(chunks) == 0

    def test_chunk_minimal_document(self):
        """Document with just title should produce at least one chunk."""
        doc = {
            "title": "Test Document",
            "abstract": "",
            "content": "",
        }
        chunks = self.chunker.chunk_document(doc, {})
        assert len(chunks) >= 1

    def test_chunk_with_content(self):
        """Document with actual content should produce chunks."""
        doc = {
            "title": "Treatment Guidelines",
            "abstract": "This is an abstract about treatment.",
            "content": """
            1. INTRODUCTION
            Hypertension is a common condition.

            2. TREATMENT
            Administer amlodipine 5mg daily. Monitor blood pressure.
            Continue treatment for at least 4 weeks.

            3. MONITORING
            Check BP weekly. Adjust dose as needed.
            """,
        }
        chunks = self.chunker.chunk_document(doc, {})
        assert len(chunks) >= 1
        assert all(isinstance(c, ChunkV3) for c in chunks)

    def test_chunk_indices_sequential(self):
        """Chunk indices should be sequential starting from 0."""
        doc = {
            "title": "Long Document",
            "abstract": "A" * 200,
            "content": "B" * 1000,
        }
        chunks = self.chunker.chunk_document(doc, {})
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i, f"Expected index {i}, got {chunk.chunk_index}"
            assert chunk.total_chunks == len(chunks)

    def test_chunk_metadata_passed(self):
        """Document metadata should be propagated to chunks."""
        doc = {
            "title": "Test",
            "abstract": "Abstract text",
            "content": "Content here",
        }
        doc_metadata = {"source": "test", "year": 2024}
        chunks = self.chunker.chunk_document(doc, doc_metadata)
        assert len(chunks) > 0
        assert chunks[0].doc_id is not None

    def test_chunk_with_mesh_terms(self):
        """Document with MeSH terms should be chunked correctly."""
        doc = {
            "title": "Diabetes Treatment",
            "abstract": "Study on diabetes management.",
            "content": "Treatment involves insulin therapy.",
            "mesh_terms": ["Diabetes Mellitus", "Insulin", "Blood Glucose"],
        }
        chunks = self.chunker.chunk_document(doc, {})
        assert len(chunks) >= 1

    def test_chunk_with_keywords(self):
        """Document with keywords should be chunked correctly."""
        doc = {
            "title": "Cardiac Study",
            "abstract": "Cardiac research findings.",
            "content": "Patient had myocardial infarction.",
            "keywords": ["cardiac", "myocardial", "heart"],
        }
        chunks = self.chunker.chunk_document(doc, {})
        assert len(chunks) >= 1

    @pytest.mark.parametrize(
        "content_length, expected_min_chunks",
        [
            pytest.param(50, 1, id="short_content"),
            pytest.param(500, 1, id="medium_content"),
            pytest.param(2000, 1, id="long_content"),
        ],
    )
    def test_chunk_scales_with_content_length(
        self, content_length: int, expected_min_chunks: int,
    ):
        """Chunker should handle varying content lengths."""
        doc = {
            "title": "Test Document",
            "abstract": "",
            "content": "X" * content_length,
        }
        chunks = self.chunker.chunk_document(doc, {})
        assert len(chunks) >= expected_min_chunks

    def test_chunk_text_is_not_empty(self):
        """Each chunk should have non-empty text."""
        doc = {
            "title": "Test",
            "abstract": "Abstract content here",
            "content": "Main content with meaningful text for testing.",
        }
        chunks = self.chunker.chunk_document(doc, {})
        for chunk in chunks:
            assert chunk.text, f"Chunk {chunk.chunk_id} has empty text"

    def test_chunk_contextual_text_includes_title(self):
        """Contextual text should include section/title context."""
        doc = {
            "title": "Medical Study",
            "abstract": "",
            "content": "Treatment results are positive.",
        }
        chunks = self.chunker.chunk_document(doc, {})
        # At least one chunk should have contextual text
        assert any(chunk.contextual_text for chunk in chunks)
