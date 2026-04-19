"""
Ingestion Pipeline v2
Uses hierarchical chunking, metadata extraction, NER, content classification.
Replaces pipeline.py for new ingestion. Old pipeline.py still works for compatibility.
"""

from datetime import datetime
import importlib
import logging
from uuid import uuid4

from src.ingestion.document_db import ChunkRecord, DocumentRecord, get_db
from src.ingestion.embeddings import embed_texts
from src.ingestion.ner import extract_entities, classify_content_type
from src.ingestion.vector_db import ensure_collection, upsert_chunks
from src.utils.chunker_v2 import chunk_text_v2

logger = logging.getLogger(__name__)


async def run_pipeline_v2(documents: list[DocumentRecord]) -> dict:
    """
    v2 ingestion pipeline with hierarchical chunking, NER, content classification.
    """
    summary = {
        "documents_stored": 0,
        "chunks_created": 0,
        "chunks_embedded": 0,
        "chunks_skipped_noise": 0,
    }

    if not documents:
        logger.info("No documents provided to v2 pipeline")
        return summary

    db = get_db()
    documents_col = db["documents"]
    chunks_col = db["chunks"]
    qdrant_ready = False

    for document in documents:
        logger.info(
            "[v2] Processing: %s (%s)", document.title[:60], document.source_type
        )

        # Store document
        doc_dict = document.model_dump()
        insert_result = await documents_col.insert_one(doc_dict)
        document_id = str(insert_result.inserted_id)
        summary["documents_stored"] += 1

        # Hierarchical chunking
        text_chunks = chunk_text_v2(document.content)
        logger.info(
            "[v2] Created %s chunks for: %s", len(text_chunks), document.title[:60]
        )

        if not text_chunks:
            continue

        # Build ChunkRecord objects with v2 metadata
        chunk_records = []
        for text_chunk in text_chunks:
            # NER extraction
            entities = extract_entities(text_chunk.text)

            # Content classification
            content_type, weight = classify_content_type(
                text_chunk.text, text_chunk.section
            )

            # Skip noise chunks
            if content_type == "noise":
                summary["chunks_skipped_noise"] += 1
                continue

            # India-specific detection
            is_india = (
                document.is_india_specific
                or document.source_type in ("icmr", "nmc", "mohfw", "state_guideline")
                or any(
                    w in text_chunk.text.lower()
                    for w in ["india", "indian", "icmr", "nmc", "aiims"]
                )
            )

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
                content_type=content_type,
                content_weight=weight,
                is_india_specific=is_india,
                evidence_level=document.evidence_level,
                token_count=text_chunk.token_count,
                parser_version="v2",
            )
            chunk_records.append(chunk_record)

        if not chunk_records:
            continue

        # Store chunks in MongoDB
        chunk_payloads = [c.model_dump() for c in chunk_records]
        insert_many_result = await chunks_col.insert_many(chunk_payloads)
        chunk_mongo_ids = [str(cid) for cid in insert_many_result.inserted_ids]
        summary["chunks_created"] += len(chunk_mongo_ids)

        # Embed and store in Qdrant
        try:
            texts = [c.chunk_text for c in chunk_records]
            embeddings = embed_texts(texts)

            if not qdrant_ready:
                ensure_collection()
                qdrant_ready = True

            try:
                point_struct_cls = getattr(
                    importlib.import_module("qdrant_client.models"), "PointStruct"
                )
            except (ImportError, AttributeError) as exc:
                logger.error("[v2] Qdrant client unavailable: %s", exc)
                continue

            points = []
            for chunk, vector, mongo_id in zip(
                chunk_records, embeddings, chunk_mongo_ids
            ):
                sparse_vec = {}
                try:
                    from src.ingestion.vector_db import build_sparse_vector

                    sparse_vec = build_sparse_vector(chunk.chunk_text)
                except ImportError:
                    sparse_vec = {}
                points.append(
                    point_struct_cls(
                        id=str(uuid4()),
                        vector={
                            "dense": vector,
                            "sparse": {
                                "indices": list(sparse_vec.keys()),
                                "values": list(sparse_vec.values()),
                            },
                        },
                        payload={
                            "mongo_id": mongo_id,
                            "source_type": chunk.source_type,
                            "title": chunk.title,
                            "condition_tags": chunk.condition_tags,
                            "chunk_text": chunk.chunk_text,
                            "section": chunk.section,
                            "diseases": chunk.diseases,
                            "drugs": chunk.drugs,
                            "content_type": chunk.content_type,
                            "content_weight": chunk.content_weight,
                            "is_india_specific": chunk.is_india_specific,
                            "evidence_level": chunk.evidence_level,
                            "parser_version": "v2",
                        },
                    )
                )

            for start in range(0, len(points), 100):
                batch = points[start : start + 100]
                upsert_chunks(batch)

            embedded_at = datetime.utcnow()
            await chunks_col.update_many(
                {"_id": {"$in": insert_many_result.inserted_ids}},
                {"$set": {"embedded": True, "embedded_at": embedded_at}},
            )

            summary["chunks_embedded"] += len(points)
            logger.info(
                "[v2] Embedded %s chunks for: %s", len(points), document.title[:60]
            )

        except (ValueError, TypeError, RuntimeError, OSError) as e:
            logger.error("[v2] Embedding failed for '%s': %s", document.title, e)

    # Update document total_chunks count
    logger.info("[v2] Pipeline complete: %s", summary)
    return summary
