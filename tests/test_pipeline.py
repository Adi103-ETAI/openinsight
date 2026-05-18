"""
Tests for the ingestion pipeline module.

Covers:
- Pipeline initialization
- Document normalization
- XML parsing (PubMed articles and book articles)
- Dead letter queue
- Batch processing
- Checkpoint management

Markers:
- unit: Unit tests (external services mocked)
- integration: Integration tests (require real services)
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Pipeline imports from embedder (requires torch)
torch = pytest.importorskip("torch", reason="torch required for pipeline module")


@pytest.mark.unit
class TestPipelineDocumentNormalization:
    """Tests for document normalization logic."""

    @pytest.fixture
    def pipeline(self):
        """Create a pipeline instance with mocked dependencies."""
        with patch("src.ingestion.pipeline.get_settings") as mock_settings, \
             patch("src.ingestion.pipeline.HierarchicalChunkerV3"), \
             patch("src.ingestion.pipeline.MetadataEnricherV2"), \
             patch("src.ingestion.pipeline.create_embedder"), \
             patch("src.ingestion.pipeline.VectorIndexer"), \
             patch("src.ingestion.pipeline.MongoDocStoreV2"), \
             patch("src.ingestion.pipeline.DocumentDeduplicator"), \
             patch("src.ingestion.pipeline.IngestionMonitor"), \
             patch("src.ingestion.pipeline.CheckpointManager"):

            mock_settings.return_value = MagicMock(
                mongodb_url="mongodb://localhost:27017/test",
                mongodb_db="test_db",
                parsing_thread_workers=2,
                ingestion_thread_workers=2,
                dead_letter_enabled=False,
                dead_letter_collection="dead_letter",
                quality_score_threshold=0.0,
                vector_collection_v2="test_collection",
            )

            from src.ingestion.pipeline import IngestionPipeline
            return IngestionPipeline()

    def test_normalize_dict_document(self, pipeline):
        """Dict document should be normalized correctly."""
        doc = {
            "title": "Test Article",
            "abstract": "Test abstract",
            "content": "Test content",
            "source_type": "pubmed",
        }
        result = pipeline._normalize_document(doc, "pubmed")

        assert result["title"] == "Test Article"
        assert result["abstract"] == "Test abstract"
        assert result["content"] == "Test content"
        assert result["source_type"] == "pubmed"

    def test_normalize_sets_defaults(self, pipeline):
        """Missing fields should get defaults."""
        doc = {"title": "Test"}
        result = pipeline._normalize_document(doc, "pubmed")

        assert result["abstract"] == ""
        assert result["content"] == ""
        assert result["authors"] == []
        assert result["mesh_terms"] == []
        assert result["keywords"] == []
        assert result["sections"] == []

    def test_normalize_year_conversion(self, pipeline):
        """Year should be converted to int."""
        doc = {"title": "Test", "year": "2024"}
        result = pipeline._normalize_document(doc, "pubmed")
        assert result["year"] == 2024

    def test_normalize_invalid_year(self, pipeline):
        """Invalid year should default to 0."""
        doc = {"title": "Test", "year": "invalid"}
        result = pipeline._normalize_document(doc, "pubmed")
        assert result["year"] == 0

    def test_normalize_generates_doc_id(self, pipeline):
        """Document without doc_id should get generated ID."""
        doc = {"title": "Test", "content": "Content"}
        result = pipeline._normalize_document(doc, "pubmed")
        assert "doc_id" in result
        assert result["doc_id"].startswith("doc_")

    def test_normalize_preserves_pmid_doc_id(self, pipeline):
        """PMID-based doc_id should be preserved."""
        doc = {"title": "Test", "pmid": "12345"}
        result = pipeline._normalize_document(doc, "pubmed")
        assert result["doc_id"] == "pmid_12345"

    def test_normalize_generates_content_hash(self, pipeline):
        """Content hash should be generated."""
        doc = {"title": "Test", "abstract": "Abstract", "content": "Content"}
        result = pipeline._normalize_document(doc, "pubmed")
        assert "content_hash" in result
        assert len(result["content_hash"]) == 64  # SHA-256 hex

    def test_normalize_source_type_fallback(self, pipeline):
        """Source type should fallback to provided source."""
        doc = {"title": "Test"}
        result = pipeline._normalize_document(doc, "icmr")
        assert result["source_type"] == "icmr"


@pytest.mark.unit
class TestPipelineHashDocId:
    """Tests for document ID hashing."""

    @pytest.fixture
    def pipeline(self):
        """Create a pipeline instance with mocked dependencies."""
        with patch("src.ingestion.pipeline.get_settings") as mock_settings, \
             patch("src.ingestion.pipeline.HierarchicalChunkerV3"), \
             patch("src.ingestion.pipeline.MetadataEnricherV2"), \
             patch("src.ingestion.pipeline.create_embedder"), \
             patch("src.ingestion.pipeline.VectorIndexer"), \
             patch("src.ingestion.pipeline.MongoDocStoreV2"), \
             patch("src.ingestion.pipeline.DocumentDeduplicator"), \
             patch("src.ingestion.pipeline.IngestionMonitor"), \
             patch("src.ingestion.pipeline.CheckpointManager"):

            mock_settings.return_value = MagicMock(
                mongodb_url="mongodb://localhost:27017/test",
                mongodb_db="test_db",
                parsing_thread_workers=2,
                ingestion_thread_workers=2,
                dead_letter_enabled=False,
                dead_letter_collection="dead_letter",
                quality_score_threshold=0.0,
                vector_collection_v2="test_collection",
            )

            from src.ingestion.pipeline import IngestionPipeline
            return IngestionPipeline()

    def test_hash_is_deterministic(self, pipeline):
        """Same inputs should produce same hash."""
        hash1 = pipeline._hash_doc_id("url", "title", "content")
        hash2 = pipeline._hash_doc_id("url", "title", "content")
        assert hash1 == hash2

    def test_hash_starts_with_doc_prefix(self, pipeline):
        """Hash should start with 'doc_' prefix."""
        doc_id = pipeline._hash_doc_id("url", "title", "content")
        assert doc_id.startswith("doc_")

    def test_hash_length(self, pipeline):
        """Hash should be 16 characters (doc_ + 12 hex chars)."""
        doc_id = pipeline._hash_doc_id("url", "title", "content")
        assert len(doc_id) == 16  # "doc_" (4) + 12 hex chars

    def test_different_inputs_different_hash(self, pipeline):
        """Different inputs should produce different hashes."""
        hash1 = pipeline._hash_doc_id("url1", "title1", "content1")
        hash2 = pipeline._hash_doc_id("url2", "title2", "content2")
        assert hash1 != hash2


@pytest.mark.unit
class TestPipelinePubmedXmlParsing:
    """Tests for PubMed XML parsing."""

    @pytest.fixture
    def pipeline(self):
        """Create a pipeline instance with mocked dependencies."""
        with patch("src.ingestion.pipeline.get_settings") as mock_settings, \
             patch("src.ingestion.pipeline.HierarchicalChunkerV3"), \
             patch("src.ingestion.pipeline.MetadataEnricherV2"), \
             patch("src.ingestion.pipeline.create_embedder"), \
             patch("src.ingestion.pipeline.VectorIndexer"), \
             patch("src.ingestion.pipeline.MongoDocStoreV2"), \
             patch("src.ingestion.pipeline.DocumentDeduplicator"), \
             patch("src.ingestion.pipeline.IngestionMonitor"), \
             patch("src.ingestion.pipeline.CheckpointManager"):

            mock_settings.return_value = MagicMock(
                mongodb_url="mongodb://localhost:27017/test",
                mongodb_db="test_db",
                parsing_thread_workers=2,
                ingestion_thread_workers=2,
                dead_letter_enabled=False,
                dead_letter_collection="dead_letter",
                quality_score_threshold=0.0,
                vector_collection_v2="test_collection",
            )

            from src.ingestion.pipeline import IngestionPipeline
            return IngestionPipeline()

    def test_parse_valid_pubmed_article(self, pipeline, temp_xml_file):
        """Valid PubMed XML should be parsed correctly."""
        docs = pipeline._parse_pubmed_xml_file(temp_xml_file)
        assert len(docs) == 1
        assert docs[0]["title"] == "Test Article on MDR-TB Treatment"
        assert docs[0]["pmid"] == "12345"
        assert docs[0]["source_type"] == "pubmed"

    def test_parse_invalid_xml(self, pipeline, temp_invalid_xml_file):
        """Invalid XML should return empty list."""
        docs = pipeline._parse_pubmed_xml_file(temp_invalid_xml_file)
        assert docs == []

    def test_parse_pubmed_book_article(self, pipeline, temp_pubmed_book_xml_file):
        """PubmedBookArticle XML should be parsed correctly."""
        docs = pipeline._parse_pubmed_xml_file(temp_pubmed_book_xml_file)
        assert len(docs) == 1
        assert docs[0]["title"] == "StatPearls: Tuberculosis Management"
        assert docs[0]["pmid"] == "67890"

    def test_parse_article_with_authors(self, pipeline, temp_xml_file):
        """Authors should be extracted from XML."""
        docs = pipeline._parse_pubmed_xml_file(temp_xml_file)
        # Authors list should exist (may be empty if not in XML)
        assert "authors" in docs[0]

    def test_parse_article_with_mesh_terms(self, pipeline, temp_xml_file):
        """MeSH terms should be extracted from XML."""
        docs = pipeline._parse_pubmed_xml_file(temp_xml_file)
        assert len(docs[0]["mesh_terms"]) > 0


@pytest.mark.unit
class TestPipelineDeadLetterQueue:
    """Tests for dead letter queue functionality."""

    @pytest.fixture
    def pipeline(self):
        """Create a pipeline instance with mocked dependencies."""
        with patch("src.ingestion.pipeline.get_settings") as mock_settings, \
             patch("src.ingestion.pipeline.HierarchicalChunkerV3"), \
             patch("src.ingestion.pipeline.MetadataEnricherV2"), \
             patch("src.ingestion.pipeline.create_embedder"), \
             patch("src.ingestion.pipeline.VectorIndexer"), \
             patch("src.ingestion.pipeline.MongoDocStoreV2"), \
             patch("src.ingestion.pipeline.DocumentDeduplicator"), \
             patch("src.ingestion.pipeline.IngestionMonitor"), \
             patch("src.ingestion.pipeline.CheckpointManager"):

            mock_settings.return_value = MagicMock(
                mongodb_url="mongodb://localhost:27017/test",
                mongodb_db="test_db",
                parsing_thread_workers=2,
                ingestion_thread_workers=2,
                dead_letter_enabled=True,
                dead_letter_collection="dead_letter",
                quality_score_threshold=0.0,
                vector_collection_v2="test_collection",
            )

            from src.ingestion.pipeline import IngestionPipeline
            return IngestionPipeline()

    @pytest.mark.asyncio
    async def test_store_to_dead_letter(self, pipeline, tmp_path):
        """Failed document should be stored to dead letter queue."""
        pipeline._dead_letter_db = AsyncMock()

        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"test")

        await pipeline._store_to_dead_letter(
            test_file,
            "parse_error",
            "Test error message",
            retry_count=1,
        )

        pipeline._dead_letter_db.insert_one.assert_called_once()
        call_args = pipeline._dead_letter_db.insert_one.call_args[0][0]
        assert call_args["error_type"] == "parse_error"
        assert "Test error message" in call_args["error_message"]

    @pytest.mark.asyncio
    async def test_dead_letter_disabled(self, pipeline, tmp_path):
        """Dead letter should be skipped when disabled."""
        pipeline._dead_letter_enabled = False
        pipeline._dead_letter_db = AsyncMock()

        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"test")

        await pipeline._store_to_dead_letter(
            test_file,
            "parse_error",
            "Test error",
        )

        pipeline._dead_letter_db.insert_one.assert_not_called()

    @pytest.mark.asyncio
    async def test_reprocess_dead_letter_empty(self, pipeline):
        """Reprocessing with no failed docs should return zero counts."""
        pipeline._dead_letter_db = MagicMock()
        pipeline._dead_letter_db.find.return_value = []

        result = await pipeline.reprocess_dead_letter()

        assert result["total"] == 0
        assert result["reprocessed"] == 0
        assert result["succeeded"] == 0
        assert result["failed"] == 0

    @pytest.mark.asyncio
    async def test_reprocess_dead_letter_disabled(self, pipeline):
        """Reprocessing when disabled should return zero counts."""
        pipeline._dead_letter_enabled = False

        result = await pipeline.reprocess_dead_letter()

        assert result["total"] == 0
