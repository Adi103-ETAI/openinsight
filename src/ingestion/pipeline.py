from datetime import datetime
from uuid import uuid4

from loguru import logger
from qdrant_client.models import PointStruct

from src.ingestion.document_db import ChunkRecord, DocumentRecord, get_db
from src.ingestion.embeddings import embed_texts
from src.ingestion.vector_db import ensure_collection, upsert_chunks
from src.utils.chunker import chunk_text


async def run_pipeline(documents: list[DocumentRecord]) -> dict:
    summary = {"documents_stored": 0, "chunks_created": 0, "chunks_embedded": 0}
    if not documents:
        logger.info("No documents provided to pipeline")
        return summary

    db = get_db()
    documents_collection = db["documents"]
    chunks_collection = db["chunks"]

    qdrant_ready = False

    for document in documents:
        logger.info(f"Storing document: {document.title} ({document.source_type})")
        insert_result = await documents_collection.insert_one(document.model_dump())
        document_id = str(insert_result.inserted_id)
        summary["documents_stored"] += 1

        text_chunks = chunk_text(document.content)
        logger.info(f"Created {len(text_chunks)} chunks for document: {document.title}")

        chunk_records: list[ChunkRecord] = []
        for text_chunk in text_chunks:
            chunk_records.append(
                ChunkRecord(
                    document_id=document_id,
                    source_type=document.source_type,
                    title=document.title,
                    chunk_text=text_chunk.text,
                    chunk_index=text_chunk.chunk_index,
                    condition_tags=document.condition_tags,
                    specialty_tags=document.specialty_tags,
                    char_count=text_chunk.char_count,
                )
            )

        if not chunk_records:
            continue

        chunk_payloads = [chunk.model_dump() for chunk in chunk_records]
        insert_many_result = await chunks_collection.insert_many(chunk_payloads)
        chunk_mongo_ids = [str(chunk_id) for chunk_id in insert_many_result.inserted_ids]
        summary["chunks_created"] += len(chunk_mongo_ids)

        try:
            texts = [chunk.chunk_text for chunk in chunk_records]
            embeddings = embed_texts(texts)

            if not qdrant_ready:
                ensure_collection()
                qdrant_ready = True

            points: list[PointStruct] = []
            for idx, (chunk, vector, mongo_id) in enumerate(
                zip(chunk_records, embeddings, chunk_mongo_ids, strict=False)
            ):
                points.append(
                    PointStruct(
                        id=str(uuid4()),
                        vector=vector,
                        payload={
                            "mongo_id": mongo_id,
                            "source_type": chunk.source_type,
                            "title": chunk.title,
                            "condition_tags": chunk.condition_tags,
                            "chunk_text": chunk.chunk_text,
                        },
                    )
                )

            for start in range(0, len(points), 100):
                batch = points[start : start + 100]
                upsert_chunks(batch)

            embedded_at = datetime.utcnow()
            await chunks_collection.update_many(
                {"_id": {"$in": insert_many_result.inserted_ids}},
                {"$set": {"embedded": True, "embedded_at": embedded_at}},
            )

            summary["chunks_embedded"] += len(points)
            logger.info(f"Embedded and indexed {len(points)} chunks for document: {document.title}")
        except Exception as exc:
            logger.error(
                f"Embedding/indexing failed for document '{document.title}' (doc_id={document_id}): {exc}"
            )

    logger.info(f"Pipeline summary: {summary}")
    return summary
