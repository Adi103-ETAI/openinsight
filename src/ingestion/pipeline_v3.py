"""
Ingestion Pipeline v3 — Orchestration
Full pipeline: parse → dedup → chunk → NER → quality score → validate → embed → store → monitor.

Supports batch processing from multiple sources with error handling, retry
logic, progress tracking, and cost-optimised rate limiting.
"""

import asyncio
from datetime import datetime
import logging
from uuid import uuid4

try:
    from tenacity import retry, stop_after_attempt, wait_exponential
except ImportError:

    def retry(*_args, **_kwargs):
        def _decorator(fn):
            return fn

        return _decorator

    def stop_after_attempt(_attempts):
        return None

    def wait_exponential(**_kwargs):
        return None


from src.core.config import get_settings
from src.ingestion.deduplication import enrich_document_hashes, is_duplicate
from src.ingestion.document_db import ChunkRecord, DocumentRecord, get_db
from src.ingestion.embeddings import embed_texts
from src.ingestion.monitoring import IngestionMonitor, RunMetrics
from src.ingestion.ner import classify_content_type, extract_entities, infer_study_type
from src.ingestion.quality import score_chunks
from src.ingestion.validation import filter_valid_chunks, validate_document
from src.ingestion.vector_db import (
    build_sparse_vector,
    ensure_collection,
    upsert_chunks,
)
from src.utils.chunker_v2 import chunk_text_v2
from src.vectorstore.types import SparseVector, VectorPoint

logger = logging.getLogger(__name__)

settings = get_settings()


@retry(
    stop=stop_after_attempt(settings.ingestion_max_retries),
    wait=wait_exponential(multiplier=settings.ingestion_retry_delay, min=2, max=30),
    reraise=True,
)
async def _embed_and_store(
    chunk_records: list[ChunkRecord],
    chunk_mongo_ids: list[str],
    _metrics: RunMetrics,
) -> int:
    """Embed chunk texts and upsert to vector DB. Returns number of points stored."""
    texts = [c.chunk_text for c in chunk_records]
    embeddings = embed_texts(texts)

    ensure_collection()

    points: list[VectorPoint] = []
    for chunk, vector, mongo_id in zip(chunk_records, embeddings, chunk_mongo_ids):
        sparse_vec = build_sparse_vector(chunk.chunk_text)
        points.append(
            VectorPoint(
                point_id=f"{mongo_id}_{chunk.chunk_index}",
                dense_vector=[float(v) for v in vector],
                sparse_vector=SparseVector.from_mapping(sparse_vec),
                payload={
                    "mongo_id": mongo_id,
                    "source_type": chunk.source_type,
                    "title": chunk.title,
                    "condition_tags": chunk.condition_tags,
                    "chunk_text": chunk.chunk_text,
                    "section": chunk.section,
                    "diseases": chunk.diseases,
                    "drugs": chunk.drugs,
                    "dosages": chunk.dosages,
                    "contraindications": chunk.contraindications,
                    "patient_populations": chunk.patient_populations,
                    "outcomes": chunk.outcomes,
                    "has_safety_flag": chunk.has_safety_flag,
                    "content_type": chunk.content_type,
                    "content_weight": chunk.content_weight,
                    "quality_score": chunk.quality_score,
                    "evidence_level": chunk.evidence_level,
                    "parser_version": "v3",
                },
            )
        )

    # Batch upsert in configurable batch sizes
    batch_size = settings.ingestion_batch_size
    for start in range(0, len(points), batch_size):
        batch = points[start : start + batch_size]
        upsert_chunks(batch)

    return len(points)


async def _process_document(
    document: DocumentRecord,
    db,
    metrics: RunMetrics,
) -> None:
    """Process a single document through the full pipeline."""
    documents_col = db["documents"]
    chunks_col = db["chunks"]

    # ── Document validation ──────────────────────────────────────────────────
    ok, reason = validate_document(document)
    if not ok:
        logger.warning(
            "[v3] Document validation failed: %s | '%s'",
            reason,
            document.title[:60],
        )
        metrics.documents_failed_validation += 1
        return

    # ── Deduplication ────────────────────────────────────────────────────────
    document = enrich_document_hashes(document)
    dup, existing_id = await is_duplicate(
        db, document, title_similarity_threshold=settings.dedup_title_similarity
    )
    if dup:
        logger.info(
            "[v3] Duplicate skipped: '%s' (matches %s)",
            document.title[:60],
            existing_id,
        )
        document.is_duplicate = True
        document.duplicate_of = existing_id
        metrics.documents_skipped_duplicate += 1
        return

    # ── Infer study type & evidence level (if not already set by parser) ─────
    if not document.study_type or document.evidence_level == 5:
        study_type, ev_level = infer_study_type(document.content, document.title)
        document.study_type = document.study_type or study_type
        if document.evidence_level == 5:
            document.evidence_level = ev_level

    # ── Store document in MongoDB ────────────────────────────────────────────
    doc_dict = document.model_dump()
    insert_result = await documents_col.insert_one(doc_dict)
    document_id = str(insert_result.inserted_id)
    metrics.documents_stored += 1

    # ── Hierarchical chunking ────────────────────────────────────────────────
    text_chunks = chunk_text_v2(document.content)
    if not text_chunks:
        logger.warning("[v3] No chunks produced for: '%s'", document.title[:60])
        return

    # ── Build ChunkRecord objects ────────────────────────────────────────────
    chunk_records: list[ChunkRecord] = []
    for text_chunk in text_chunks:
        entities = extract_entities(text_chunk.text)
        content_type, weight = classify_content_type(
            text_chunk.text, text_chunk.section
        )

        # Skip noise at chunking stage
        if content_type == "noise":
            metrics.chunks_skipped_noise += 1
            continue

        chunk_record = ChunkRecord(
            document_id=document_id,
            source_type=document.source_type,
            title=document.title,
            chunk_text=text_chunk.text,
            chunk_index=text_chunk.chunk_index,
            condition_tags=document.condition_tags,
            specialty_tags=document.specialty_tags,
            char_count=text_chunk.char_count,
            section=text_chunk.section,
            diseases=entities["diseases"],
            drugs=entities["drugs"],
            symptoms=entities["symptoms"],
            dosages=entities["dosages"],
            contraindications=entities["contraindications"],
            patient_populations=entities["patient_populations"],
            outcomes=entities["outcomes"],
            has_safety_flag=entities["has_safety_flag"],
            content_type=content_type,
            content_weight=weight,
            evidence_level=document.evidence_level,
            token_count=text_chunk.token_count,
            parser_version="v3",
        )
        chunk_records.append(chunk_record)

    # ── Quality scoring ──────────────────────────────────────────────────────
    score_chunks(chunk_records)

    # Filter out chunks below quality threshold
    before = len(chunk_records)
    chunk_records = [
        c for c in chunk_records if c.quality_score >= settings.quality_score_threshold
    ]
    skipped_quality = before - len(chunk_records)
    metrics.chunks_skipped_quality += skipped_quality

    # ── Chunk validation ─────────────────────────────────────────────────────
    chunk_records, rejected = filter_valid_chunks(chunk_records)
    metrics.chunks_failed_validation += len(rejected)
    if rejected:
        logger.debug(
            "[v3] %s chunks rejected by validator for '%s'",
            len(rejected),
            document.title[:60],
        )

    if not chunk_records:
        return

    # ── Store chunks in MongoDB ──────────────────────────────────────────────
    chunk_payloads = [c.model_dump() for c in chunk_records]
    insert_many_result = await chunks_col.insert_many(chunk_payloads)
    chunk_mongo_ids = [str(cid) for cid in insert_many_result.inserted_ids]
    metrics.chunks_created += len(chunk_mongo_ids)

    # ── Embed and store in vector DB ─────────────────────────────────────────
    try:
        n_embedded = await _embed_and_store(chunk_records, chunk_mongo_ids, metrics)

        embedded_at = datetime.utcnow()
        await chunks_col.update_many(
            {"_id": {"$in": insert_many_result.inserted_ids}},
            {"$set": {"embedded": True, "embedded_at": embedded_at}},
        )
        await documents_col.update_one(
            {"_id": insert_result.inserted_id},
            {"$set": {"total_chunks": n_embedded}},
        )

        metrics.chunks_embedded += n_embedded
        logger.info(
            "[v3] Stored %s chunks for '%s' (quality_dropped=%s)",
            n_embedded,
            document.title[:60],
            skipped_quality,
        )

    except (ValueError, TypeError, RuntimeError, OSError) as exc:
        logger.error("[v3] Embedding failed for '%s': %s", document.title[:60], exc)
        metrics.embedding_errors += 1


async def run_pipeline_v3(
    documents: list[DocumentRecord],
    source_type: str = "unknown",
    concurrency: int = 3,
) -> dict:
    """
    v3 ingestion pipeline with full orchestration.

    Args:
        documents:    List of parsed DocumentRecords ready for ingestion.
        source_type:  Label for monitoring (e.g. "pubmed", "cochrane").
        concurrency:  Max concurrent document processing tasks.

    Returns:
        Summary dict with ingestion metrics.
    """
    run_id = str(uuid4())
    metrics = RunMetrics(run_id=run_id, source_type=source_type)
    metrics.documents_fetched = len(documents)

    if not documents:
        logger.info("[v3] No documents provided (source=%s)", source_type)
        metrics.finish(status="completed")
        return metrics.summary()

    db = get_db()
    monitor = IngestionMonitor(db)

    logger.info(
        "[v3] Starting pipeline run_id=%s source=%s docs=%s",
        run_id,
        source_type,
        len(documents),
    )

    # Save initial run state
    await monitor.save_run(metrics)

    # Process documents with limited concurrency
    semaphore = asyncio.Semaphore(concurrency)

    async def _bounded(doc: DocumentRecord) -> None:
        async with semaphore:
            try:
                await _process_document(doc, db, metrics)
            except (ValueError, TypeError, RuntimeError, OSError) as exc:
                logger.error("[v3] Unhandled error for '%s': %s", doc.title[:60], exc)
                metrics.embedding_errors += 1

    tasks = [_bounded(doc) for doc in documents]
    await asyncio.gather(*tasks)

    metrics.finish(status="completed")
    await monitor.save_run(metrics)
    await monitor.alert_on_failure(metrics)

    logger.info("[v3] Pipeline complete: %s", metrics.summary())
    return metrics.summary()


async def run_pipeline_v3_from_parser(
    parser_cls,
    query: str,
    max_results: int = 100,
    concurrency: int = 3,
    **parser_kwargs,
) -> dict:
    """
    Convenience wrapper: instantiate a parser, fetch documents, run pipeline.

    Example:
        await run_pipeline_v3_from_parser(PubMedParser, "malaria treatment", max_results=200)
    """
    parser = parser_cls(query=query, max_results=max_results, **parser_kwargs)
    source_type = parser.source_type

    logger.info("[v3] Fetching from %s: '%s'", source_type, query)
    documents = parser.parse()
    logger.info("[v3] %s returned %s documents", source_type, len(documents))

    return await run_pipeline_v3(
        documents, source_type=source_type, concurrency=concurrency
    )
