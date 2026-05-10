from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

from src.ingestion.celery_app import celery_app
from src.ml.chunking.chunker import HierarchicalChunkerV3
from src.ingestion.dedupe import DocumentDeduplicator
from src.ml.embedding.embedder import DualEmbedderV2
from src.ingestion.metadata import MetadataEnricherV2
from src.data.mongo.doc_store import MongoDocStoreV2
from src.ingestion.vector_indexer import VectorIndexer

logger = logging.getLogger(__name__)


@dataclass
class IngestTaskResult:
    doc_id: str
    status: str
    chunks_created: int
    error: str | None = None


class IngestionWorker:
    """Worker class for processing single documents."""

    def __init__(self):
        from src.config.settings import get_settings
        self.settings = get_settings()
        
        self.chunker = HierarchicalChunkerV3()
        self.metadata = MetadataEnricherV2()
        self.embedder = DualEmbedderV2(self.settings.dense_model_name)
        self.indexer = VectorIndexer()
        self.mongo = MongoDocStoreV2(
            mongo_url=self.settings.mongodb_url,
            db_name=self.settings.mongodb_db,
        )
        self.deduplicator = DocumentDeduplicator(self.mongo)

    async def process_document(
        self,
        doc: dict[str, Any],
        source: str,
        force_reindex: bool = False,
    ) -> IngestTaskResult:
        """Process a single document through the pipeline."""
        doc_id = doc.get("doc_id", "unknown")

        try:
            # Check deduplication
            should_skip, reason = await self.deduplicator.check_document(
                doc, force_reindex
            )
            if should_skip and not force_reindex:
                return IngestTaskResult(
                    doc_id=doc_id,
                    status="skipped",
                    chunks_created=0,
                    error=reason,
                )

            # Process document
            normalized = self._normalize_document(doc, source)
            enriched = self.metadata.enrich_document(normalized, source)

            # Chunk
            chunks = self.chunker.chunk_document(normalized, enriched)
            if not chunks:
                return IngestTaskResult(
                    doc_id=doc_id,
                    status="no_chunks",
                    chunks_created=0,
                )

            # Embed
            contextual_texts = [c.contextual_text for c in chunks]
            dense_embeddings = await self._run_embed(self.embedder.embed_batch, contextual_texts)
            sparse_vectors = [
                self.embedder.compute_sparse_vector(text) for text in contextual_texts
            ]

            # Index to Milvus
            self.indexer.upsert_chunks(
                chunks=chunks,
                dense_embeddings=dense_embeddings,
                sparse_vectors=sparse_vectors,
                collection_name=self.settings.vector_collection_v2,
            )

            # Store in MongoDB
            await self.mongo.store_document(normalized, enriched)
            await self.mongo.store_chunks(chunks)

            return IngestTaskResult(
                doc_id=doc_id,
                status="success",
                chunks_created=len(chunks),
            )

        except Exception as e:
            logger.error(f"[worker] Failed to process {doc_id}: {e}")
            return IngestTaskResult(
                doc_id=doc_id,
                status="error",
                chunks_created=0,
                error=str(e),
            )

    def _normalize_document(self, doc: dict[str, Any], source: str) -> dict[str, Any]:
        out = dict(doc)
        out.setdefault("abstract", "")
        out.setdefault("content", "")
        out.setdefault("authors", [])
        out.setdefault("mesh_terms", [])
        out.setdefault("keywords", [])
        out.setdefault("sections", [])
        out.setdefault("source_type", source)
        
        year = out.get("year", 0)
        try:
            out["year"] = int(year) if year is not None else 0
        except (TypeError, ValueError):
            out["year"] = 0
            
        return out

    async def _run_embed(self, func, texts: list[str]) -> list[Any]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: func(texts, 32))


# Global worker instance
_worker: IngestionWorker | None = None


def get_worker() -> IngestionWorker:
    global _worker
    if _worker is None:
        _worker = IngestionWorker()
    return _worker


@celery_app.task(bind=True, max_retries=3)
def ingest_document_task(
    self,
    doc: dict[str, Any],
    source: str,
    force_reindex: bool = False,
) -> dict[str, Any]:
    """Celery task to ingest a single document."""
    worker = get_worker()
    result = asyncio.run(
        worker.process_document(doc, source, force_reindex)
    )
    return {
        "doc_id": result.doc_id,
        "status": result.status,
        "chunks_created": result.chunks_created,
        "error": result.error,
    }


@celery_app.task
def ingest_batch_task(
    docs: list[dict[str, Any]],
    source: str,
    force_reindex: bool = False,
) -> list[dict[str, Any]]:
    """Celery task to ingest a batch of documents."""
    worker = get_worker()
    results = asyncio.run(
        asyncio.gather(*[
            worker.process_document(doc, source, force_reindex)
            for doc in docs
        ])
    )
    return [
        {
            "doc_id": r.doc_id,
            "status": r.status,
            "chunks_created": r.chunks_created,
            "error": r.error,
        }
        for r in results
    ]