"""Tests for IngestionPipeline.ingest_scraped_documents() — the scraper framework integration.

These tests verify that the new method correctly:
1. Accepts (DocumentRecord, list[ChunkRecord]) tuples from scraper parsers
2. Validates + quality-scores chunks
3. Embeds chunk_text (dense + sparse)
4. Builds VectorPoints with correct payload (trust_tier, indian_source, etc.)
5. Upserts to Milvus via VectorIndexer.store.upsert_points()
6. Stores to MongoDB via MongoDocStoreV2
7. Handles failures gracefully (dead letter queue)
8. Returns correct summary counts

All external deps (Milvus, MongoDB, embedder) are mocked — no network or
GPU required.
"""
from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.ingestion.document_db import ChunkRecord, DocumentRecord


# --- Fixtures -------------------------------------------------------------

def make_document_record(
    title: str = "Test Article on Metformin in Indian Patients",
    source_type: str = "indmed",
    url: str = "https://example.com/article/1",
    doi: str | None = "10.1/test",
    pmid: str | None = None,
    content: str = "This is the full text of the article. " * 50,
) -> DocumentRecord:
    return DocumentRecord(
        source_type=source_type,
        title=title,
        content=content,
        url=url,
        doi=doi,
        pmid=pmid,
        published_date="2024-01-15",
        year=2024,
        journal="Indian Journal of Pharmacology",
        is_india_specific=True,
        parser_version="test-v1",
        content_hash="abc123def456abc7",
        condition_tags=[],
        specialty_tags=["ijp"],
    )


def make_chunk_record(
    document_id: str = "https://example.com/article/1",
    chunk_index: int = 0,
    text: str = "Metformin is the first-line treatment for type 2 diabetes in Indian adults. "
                "The standard starting dose is 500mg twice daily, titrated to a maximum of 2g daily. "
                "Common side effects include GI upset, which can be minimized by taking with food.",
    source_type: str = "indmed",
    trust_tier: int = 3,
    indian_source: bool = True,
    section: str = "body",
) -> ChunkRecord:
    return ChunkRecord(
        document_id=document_id,
        source_type=source_type,
        title="Test Article",
        chunk_text=text,
        chunk_index=chunk_index,
        char_count=len(text),
        section=section,
        diseases=[],
        drugs=[],
        symptoms=[],
        dosages=[],
        contraindications=[],
        patient_populations=[],
        outcomes=[],
        has_safety_flag=False,
        content_type="text",
        content_weight=1.0,
        quality_score=1.0,
        is_india_specific=True,
        evidence_level=5,
        parser_version="test-v1",
        token_estimate=len(text) // 4,
        trust_tier=trust_tier,
        indian_source=indian_source,
        also_indexed_in=[],
    )


def make_pipeline_with_mocks() -> Any:
    """Create an IngestionPipeline with all external deps mocked."""
    # We can't import IngestionPipeline directly because it imports torch
    # via the embedder. Mock the heavy imports first.
    import sys
    from unittest.mock import MagicMock

    # Mock torch + sentence_transformers if not installed
    if "torch" not in sys.modules:
        sys.modules["torch"] = MagicMock()
    if "sentence_transformers" not in sys.modules:
        sys.modules["sentence_transformers"] = MagicMock()
    if "transformers" not in sys.modules:
        sys.modules["transformers"] = MagicMock()

    from src.ingestion.pipeline import IngestionPipeline

    pipeline = IngestionPipeline.__new__(IngestionPipeline)

    # Mock settings
    from src.config.settings import get_settings
    pipeline.settings = get_settings()
    pipeline.settings.vector_collection_v2 = "test_collection"
    pipeline.settings.quality_score_threshold = 0.3

    # Mock embedder — embed_batch returns (embeddings, failed_indices) tuple
    # _run_cpu wraps it in a thread executor, so we mock _run_cpu instead
    pipeline.embedder = MagicMock()
    pipeline.embedder.compute_sparse_vector = MagicMock(return_value={
        "indices": [1, 2, 3],
        "values": [0.5, 0.3, 0.2],
    })

    # Mock _run_cpu to directly call the function (bypass thread executor)
    async def mock_run_cpu(func, *args):
        result = func(*args)
        # If result is a coroutine (AsyncMock), await it
        if asyncio.iscoroutine(result):
            result = await result
        return result
    pipeline._run_cpu = mock_run_cpu

    # Make embed_batch a sync function that returns the tuple
    # (individual tests can override this)
    pipeline.embedder.embed_batch = lambda texts, batch_size: (
        [[0.1] * 768 for _ in texts],  # one embedding per text
        [],  # no failed indices
    )

    # Mock indexer + store
    pipeline.indexer = MagicMock()
    pipeline.indexer.create_collection = MagicMock()
    pipeline.indexer.store = MagicMock()
    pipeline.indexer.store.upsert_points = MagicMock(return_value=1)

    # Mock mongo store
    pipeline.mongo = MagicMock()
    pipeline.mongo.store_document = AsyncMock()
    pipeline.mongo.store_chunks = AsyncMock()

    # Mock metadata enricher
    pipeline.metadata = MagicMock()
    pipeline.metadata.enrich_document = MagicMock(side_effect=lambda doc, source: doc)

    # Mock deduplicator (not used in ingest_scraped_documents but needed for init)
    pipeline.deduplicator = MagicMock()

    # Mock monitor
    pipeline.monitor = MagicMock()
    pipeline.monitor.record_run = AsyncMock()

    # Mock dead letter
    pipeline._dead_letter_enabled = True
    pipeline._dead_letter_collection = "failed_documents"
    pipeline._dead_letter_db = MagicMock()
    pipeline._store_to_dead_letter = AsyncMock()

    return pipeline


# --- Tests ----------------------------------------------------------------

class TestIngestScrapedDocumentsSignature:
    """Verify the method exists with the correct signature."""

    def test_method_exists(self) -> None:
        import sys
        from unittest.mock import MagicMock
        if "torch" not in sys.modules:
            sys.modules["torch"] = MagicMock()
        if "sentence_transformers" not in sys.modules:
            sys.modules["sentence_transformers"] = MagicMock()
        if "transformers" not in sys.modules:
            sys.modules["transformers"] = MagicMock()
        from src.ingestion.pipeline import IngestionPipeline
        assert hasattr(IngestionPipeline, "ingest_scraped_documents")

    def test_method_signature(self) -> None:
        import inspect
        import sys
        from unittest.mock import MagicMock
        if "torch" not in sys.modules:
            sys.modules["torch"] = MagicMock()
        if "sentence_transformers" not in sys.modules:
            sys.modules["sentence_transformers"] = MagicMock()
        if "transformers" not in sys.modules:
            sys.modules["transformers"] = MagicMock()
        from src.ingestion.pipeline import IngestionPipeline
        sig = inspect.signature(IngestionPipeline.ingest_scraped_documents)
        params = list(sig.parameters.keys())
        assert params == ["self", "documents", "source", "batch_size", "recreate_index"]
        assert sig.parameters["batch_size"].default == 10
        assert sig.parameters["recreate_index"].default is False


class TestIngestScrapedDocumentsExecution:
    """End-to-end execution tests with mocked deps."""

    @pytest.mark.asyncio
    async def test_ingest_single_document(self) -> None:
        """A single (DocumentRecord, [ChunkRecord]) should be ingested successfully."""
        pipeline = make_pipeline_with_mocks()
        doc = make_document_record()
        chunk = make_chunk_record()

        result = await pipeline.ingest_scraped_documents(
            documents=[(doc, [chunk])],
            source="indmed",
        )

        assert result["documents_total"] == 1
        assert result["documents_stored"] == 1
        assert result["chunks_created"] == 1
        assert result["chunks_indexed"] == 1
        assert result["files_failed"] == 0

    @pytest.mark.asyncio
    async def test_ingest_multiple_documents(self) -> None:
        """Multiple documents in one batch should all be ingested."""
        pipeline = make_pipeline_with_mocks()
        # Mock embedder to return 3 embeddings (3 chunks total)
        pipeline.embedder.embed_batch = lambda texts, batch_size: (
            [[0.1] * 768 for _ in texts],
            [],
        )
        pipeline.indexer.store.upsert_points = MagicMock(return_value=3)

        docs = [
            (make_document_record(title=f"Article {i}", url=f"https://example.com/{i}"),
             [make_chunk_record(document_id=f"https://example.com/{i}", chunk_index=0)])
            for i in range(3)
        ]

        result = await pipeline.ingest_scraped_documents(
            documents=docs,
            source="indmed",
            batch_size=10,
        )

        assert result["documents_stored"] == 3
        assert result["chunks_indexed"] == 3

    @pytest.mark.asyncio
    async def test_empty_documents_list(self) -> None:
        """Empty input should return zero counts without errors."""
        pipeline = make_pipeline_with_mocks()
        result = await pipeline.ingest_scraped_documents(
            documents=[],
            source="indmed",
        )
        assert result["documents_total"] == 0
        assert result["documents_stored"] == 0
        assert result["chunks_indexed"] == 0

    @pytest.mark.asyncio
    async def test_document_with_no_chunks_is_skipped(self) -> None:
        """A document with empty chunk list should be skipped (not failed)."""
        pipeline = make_pipeline_with_mocks()
        doc = make_document_record()

        result = await pipeline.ingest_scraped_documents(
            documents=[(doc, [])],
            source="indmed",
        )

        assert result["documents_total"] == 1
        assert result["documents_stored"] == 0  # skipped
        assert result["files_failed"] == 0  # not a failure, just no chunks

    @pytest.mark.asyncio
    async def test_embedding_failure_goes_to_dead_letter(self) -> None:
        """If embedding fails, the doc should go to dead letter queue."""
        pipeline = make_pipeline_with_mocks()
        pipeline.embedder.embed_batch = lambda texts, batch_size: (_ for _ in ()).throw(Exception("GPU OOM"))

        doc = make_document_record()
        chunk = make_chunk_record()

        result = await pipeline.ingest_scraped_documents(
            documents=[(doc, [chunk])],
            source="indmed",
        )

        assert result["files_failed"] == 1
        assert result["documents_stored"] == 0
        assert result["chunks_indexed"] == 0
        pipeline._store_to_dead_letter.assert_called()

    @pytest.mark.asyncio
    async def test_milvus_upsert_failure_goes_to_dead_letter(self) -> None:
        """If Milvus upsert fails, docs go to dead letter."""
        pipeline = make_pipeline_with_mocks()
        pipeline.indexer.store.upsert_points = MagicMock(
            side_effect=Exception("Milvus connection refused")
        )

        doc = make_document_record()
        chunk = make_chunk_record()

        result = await pipeline.ingest_scraped_documents(
            documents=[(doc, [chunk])],
            source="indmed",
        )

        assert result["files_failed"] == 1
        assert result["chunks_indexed"] == 0
        pipeline._store_to_dead_letter.assert_called()

    @pytest.mark.asyncio
    async def test_partial_embedding_failure_filters_chunks(self) -> None:
        """If some embeddings fail, only valid chunks are indexed."""
        pipeline = make_pipeline_with_mocks()
        # 2 chunks, 1 embedding fails (index 1)
        pipeline.embedder.embed_batch = lambda texts, batch_size: (
            [[0.1] * 768],  # only 1 valid embedding
            [1],  # index 1 failed
        )
        pipeline.indexer.store.upsert_points = MagicMock(return_value=1)

        doc = make_document_record()
        chunk0 = make_chunk_record(chunk_index=0)
        chunk1 = make_chunk_record(chunk_index=1)

        result = await pipeline.ingest_scraped_documents(
            documents=[(doc, [chunk0, chunk1])],
            source="indmed",
        )

        assert result["chunks_indexed"] == 1  # only the valid one


class TestIngestScrapedDocumentsBatching:
    """Verify batching behavior."""

    @pytest.mark.asyncio
    async def test_batch_size_respected(self) -> None:
        """Documents should be processed in batches of batch_size."""
        pipeline = make_pipeline_with_mocks()
        # 5 docs, batch_size=2 → 3 batches (2+2+1)
        pipeline.embedder.embed_batch = lambda texts, batch_size: (
            [[0.1] * 768 for _ in texts],  # 1 chunk per doc
            [],
        )
        # Mock returns the actual number of points upserted
        pipeline.indexer.store.upsert_points = MagicMock(side_effect=lambda points, **kw: len(points))

        docs = [
            (make_document_record(title=f"Article {i}", url=f"https://example.com/{i}"),
             [make_chunk_record(document_id=f"https://example.com/{i}")])
            for i in range(5)
        ]

        result = await pipeline.ingest_scraped_documents(
            documents=docs,
            source="indmed",
            batch_size=2,
        )

        assert result["documents_stored"] == 5
        assert result["chunks_indexed"] == 5
        # 3 batches → 3 calls to upsert_points (1 per batch)
        assert pipeline.indexer.store.upsert_points.call_count == 3


class TestIngestScrapedDocumentsPayload:
    """Verify the VectorPoint payload contains the right provenance fields."""

    @pytest.mark.asyncio
    async def test_payload_contains_trust_tier(self) -> None:
        """The Milvus payload should include trust_tier from ChunkRecord."""
        pipeline = make_pipeline_with_mocks()
        doc = make_document_record()
        chunk = make_chunk_record(trust_tier=1)  # IJMR-tier

        await pipeline.ingest_scraped_documents(
            documents=[(doc, [chunk])],
            source="indmed",
        )

        # Inspect the VectorPoint passed to upsert_points
        call_args = pipeline.indexer.store.upsert_points.call_args
        points = call_args[0][0]
        assert len(points) == 1
        payload = points[0].payload
        assert payload["trust_tier"] == 1

    @pytest.mark.asyncio
    async def test_payload_contains_indian_source(self) -> None:
        """The Milvus payload should include indian_source from ChunkRecord."""
        pipeline = make_pipeline_with_mocks()
        doc = make_document_record()
        chunk = make_chunk_record(indian_source=True)

        await pipeline.ingest_scraped_documents(
            documents=[(doc, [chunk])],
            source="indmed",
        )

        points = pipeline.indexer.store.upsert_points.call_args[0][0]
        payload = points[0].payload
        assert payload["indian_source"] is True
        assert payload["india_relevant"] is True

    @pytest.mark.asyncio
    async def test_payload_contains_source_type(self) -> None:
        """The Milvus payload should include source_type for retrieval filtering."""
        pipeline = make_pipeline_with_mocks()
        doc = make_document_record()
        chunk = make_chunk_record(source_type="pmc_india")

        await pipeline.ingest_scraped_documents(
            documents=[(doc, [chunk])],
            source="pmc_india",
        )

        points = pipeline.indexer.store.upsert_points.call_args[0][0]
        payload = points[0].payload
        assert payload["source_type"] == "pmc_india"
        assert payload["source"] == "pmc_india"  # alias for retrieval filter compat

    @pytest.mark.asyncio
    async def test_payload_contains_raw_text(self) -> None:
        """The Milvus payload should include raw_text for RAG context retrieval."""
        pipeline = make_pipeline_with_mocks()
        doc = make_document_record()
        # Use a chunk text longer than _CHUNK_MIN_CHARS (80)
        long_text = "Metformin 500mg twice daily is the standard starting dose for type 2 diabetes in Indian adults."
        chunk = make_chunk_record(text=long_text)

        await pipeline.ingest_scraped_documents(
            documents=[(doc, [chunk])],
            source="indmed",
        )

        points = pipeline.indexer.store.upsert_points.call_args[0][0]
        payload = points[0].payload
        assert payload["raw_text"] == long_text

    @pytest.mark.asyncio
    async def test_payload_contains_chunk_id_and_doc_id(self) -> None:
        """Payload should have chunk_id and doc_id for citation linking."""
        pipeline = make_pipeline_with_mocks()
        doc = make_document_record()
        chunk = make_chunk_record(document_id="https://example.com/doc/123", chunk_index=2)

        await pipeline.ingest_scraped_documents(
            documents=[(doc, [chunk])],
            source="indmed",
        )

        points = pipeline.indexer.store.upsert_points.call_args[0][0]
        payload = points[0].payload
        assert payload["doc_id"] == "https://example.com/doc/123"
        assert "c002" in payload["chunk_id"]  # chunk_index 2 → "...-c002"
        assert payload["chunk_index"] == 2


class TestIngestScrapedDocumentsMongoStorage:
    """Verify MongoDB storage calls."""

    @pytest.mark.asyncio
    async def test_store_document_called(self) -> None:
        """store_document should be called for each successfully ingested doc."""
        pipeline = make_pipeline_with_mocks()
        doc = make_document_record()
        chunk = make_chunk_record()

        await pipeline.ingest_scraped_documents(
            documents=[(doc, [chunk])],
            source="indmed",
        )

        pipeline.mongo.store_document.assert_called_once()

    @pytest.mark.asyncio
    async def test_store_chunks_called(self) -> None:
        """store_chunks should be called with the chunk proxy objects."""
        pipeline = make_pipeline_with_mocks()
        doc = make_document_record()
        chunk = make_chunk_record()

        await pipeline.ingest_scraped_documents(
            documents=[(doc, [chunk])],
            source="indmed",
        )

        pipeline.mongo.store_chunks.assert_called_once()
        stored_chunks = pipeline.mongo.store_chunks.call_args[0][0]
        assert len(stored_chunks) == 1
        # Verify the proxy has the expected fields
        proxy = stored_chunks[0]
        assert proxy.text == chunk.chunk_text
        assert proxy.doc_id == chunk.document_id
        assert proxy.chunk_index == chunk.chunk_index
