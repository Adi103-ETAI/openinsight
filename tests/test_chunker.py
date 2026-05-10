"""
Tests for the hierarchical chunker (HierarchicalChunkerV3).
"""
import pytest
from src.ml.chunking.chunker import ChunkV3, HierarchicalChunkerV3


class TestChunkV3:
    def test_chunk_creation(self):
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


class TestHierarchicalChunkerV3:
    def setup_method(self):
        self.chunker = HierarchicalChunkerV3()

    def test_chunk_empty_document(self):
        """Test that empty document produces empty chunks."""
        doc = {"title": "", "abstract": "", "content": ""}
        chunks = self.chunker.chunk_document(doc, {})
        assert len(chunks) == 0

    def test_chunk_minimal_document(self):
        """Test minimal document with just title."""
        doc = {
            "title": "Test Document",
            "abstract": "",
            "content": "",
        }
        chunks = self.chunker.chunk_document(doc, {})
        # Should have at least the title/abstract as one chunk
        assert len(chunks) >= 1

    def test_chunk_with_content(self):
        """Test document with actual content."""
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
        """Test that chunk indices are sequential."""
        doc = {
            "title": "Long Document",
            "abstract": "A" * 200,
            "content": "B" * 1000,
        }
        chunks = self.chunker.chunk_document(doc, {})
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i
            assert chunk.total_chunks == len(chunks)

    def test_chunk_metadata_passed(self):
        """Test that doc metadata is used."""
        doc = {
            "title": "Test",
            "abstract": "Abstract text",
            "content": "Content here",
        }
        doc_metadata = {"source": "test", "year": 2024}
        chunks = self.chunker.chunk_document(doc, doc_metadata)
        # Chunks should have some metadata
        assert len(chunks) > 0
        assert chunks[0].doc_id is not None

    def test_chunk_with_mesh_terms(self):
        """Test document with MeSH terms."""
        doc = {
            "title": "Diabetes Treatment",
            "abstract": "Study on diabetes management.",
            "content": "Treatment involves insulin therapy.",
            "mesh_terms": ["Diabetes Mellitus", "Insulin", "Blood Glucose"],
        }
        chunks = self.chunker.chunk_document(doc, {})
        assert len(chunks) >= 1

    def test_chunk_with_keywords(self):
        """Test document with keywords."""
        doc = {
            "title": "Cardiac Study",
            "abstract": "Cardiac research findings.",
            "content": "Patient had myocardial infarction.",
            "keywords": ["cardiac", "myocardial", "heart"],
        }
        chunks = self.chunker.chunk_document(doc, {})
        assert len(chunks) >= 1