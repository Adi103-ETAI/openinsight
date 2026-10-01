"""Document lifecycle — supersede + delete.

Mongo (`documents_v2` / `chunks_v2`) is closer to truth; the Milvus index
is derived. Every lifecycle change must propagate to all three legs:

- Mongo document row
- Mongo chunk rows
- Milvus vectors (deleted by deterministic point id — NEVER drop_collection)

Chunk point ids follow the pipeline convention
(`src/ingestion/pipeline.py`)::

    chunk_id = f"{doc_id}-c{chunk_index:03d}"
    point_id = str(uuid5(NAMESPACE_DNS, chunk_id))

Dependency-light: stdlib + loguru only. No torch. Works with both the
real async stores (motor collections, ``MilvusVectorStore``) and plain
sync stub objects in tests.
"""
from __future__ import annotations

import inspect
from datetime import datetime, timezone
from typing import Any, Callable, Optional
from uuid import NAMESPACE_DNS, uuid5

from loguru import logger


def chunk_id_for(doc_id: str, chunk_index: int) -> str:
    """Canonical chunk id for a chunk position within a document."""
    return f"{doc_id}-c{int(chunk_index):03d}"


def chunk_point_id(doc_id: str, chunk_index: int) -> str:
    """Deterministic Milvus point id for a chunk (matches pipeline)."""
    return str(uuid5(NAMESPACE_DNS, chunk_id_for(doc_id, chunk_index)))


def point_id_for_chunk_id(chunk_id: str) -> str:
    """Deterministic Milvus point id straight from a stored chunk_id."""
    return str(uuid5(NAMESPACE_DNS, str(chunk_id)))


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


async def _await_if_needed(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def _resolve_collections(db: Any) -> tuple[Any, Any]:
    """Return (documents_coll, chunks_coll) from a store or raw db mapping."""
    docs = getattr(db, "documents", None)
    chunks = getattr(db, "chunks", None)
    if docs is not None and chunks is not None:
        return docs, chunks
    # Raw db mapping (pymongo Database / dict-like fakes).
    try:
        return db["documents_v2"], db["chunks_v2"]
    except Exception:
        pass
    try:
        return db.__getitem__("documents_v2"), db.__getitem__("chunks_v2")
    except Exception as exc:
        raise TypeError(
            "db must expose .documents/.chunks or db['documents_v2']/db['chunks_v2']"
        ) from exc


async def _collect_find(collection: Any, filt: dict[str, Any]) -> list[dict[str, Any]]:
    cursor = await _await_if_needed(collection.find(filt))
    if cursor is None:
        return []
    to_list = getattr(cursor, "to_list", None)
    if callable(to_list):
        try:
            return list(await to_list(length=1_000_000))
        except TypeError:
            return list(await to_list(1_000_000))
    if hasattr(cursor, "__aiter__"):
        return [doc async for doc in cursor]
    return list(cursor)


def _modified_count(result: Any, fallback: int = 0) -> int:
    # Update results carry modified_count/matched_count; delete results carry
    # deleted_count (stub result objects may define all three, with the
    # inapplicable ones at 0) — so take the max present count.
    counts = [
        getattr(result, attr, None)
        for attr in ("modified_count", "matched_count", "deleted_count")
    ]
    ints = [v for v in counts if isinstance(v, int)]
    if ints and any(v != 0 for v in ints):
        return max(ints)
    if isinstance(result, bool):
        return int(result)
    if isinstance(result, int):
        return result
    return fallback


def _point_ids_for_chunks(doc_id: str, chunks: list[dict[str, Any]]) -> list[str]:
    ids: list[str] = []
    for row in chunks:
        if not isinstance(row, dict):
            continue
        chunk_id = row.get("chunk_id")
        if chunk_id:
            ids.append(point_id_for_chunk_id(str(chunk_id)))
            continue
        if row.get("chunk_index") is not None:
            try:
                ids.append(chunk_point_id(doc_id, int(row["chunk_index"])))
            except (TypeError, ValueError):
                continue
    # Deduplicate, preserve order.
    return list(dict.fromkeys(ids))


async def _delete_vectors(
    store: Any, point_ids: list[str], collection_name: Optional[str] = None
) -> int:
    if not point_ids:
        return 0
    delete_fn = getattr(store, "delete_points_by_ids", None)
    if delete_fn is None:
        raise AttributeError(
            "vector store has no delete_points_by_ids(); "
            "add it to MilvusVectorStore (do NOT use drop_collection)"
        )
    kwargs: dict[str, Any] = {}
    try:
        result = delete_fn(point_ids, collection_name=collection_name)
    except TypeError:
        # Older/alternate signature without collection_name.
        result = delete_fn(point_ids)
    result = await _await_if_needed(result)
    if isinstance(result, int):
        return result
    if isinstance(result, dict):
        try:
            return int(result.get("delete_count", len(point_ids)))
        except (TypeError, ValueError):
            return len(point_ids)
    return len(point_ids)


def _zero_report(doc_id: str, status: str) -> dict[str, Any]:
    return {
        "doc_id": doc_id,
        "mongo_docs": 0,
        "mongo_chunks": 0,
        "milvus_deleted": 0,
        "status": status,
    }


async def supersede_document(
    db: Any,
    store: Any,
    doc_id: str,
    reason: str,
    *,
    collection_name: Optional[str] = None,
    superseded_by: Optional[str] = None,
) -> dict[str, Any]:
    """Mark a document version superseded on every leg.

    - Mongo ``documents_v2`` row → ``status="superseded"`` (+ ``superseded_by``,
      ``superseded_at``, ``supersede_reason``).
    - Its Milvus vectors are deleted by deterministic point id
      (collection is never dropped).
    - Its ``chunks_v2`` rows → ``status="superseded"``.

    The replacement version ingests normally afterwards. Idempotent:
    a repeat call (missing or already-superseded row) succeeds with zeros.
    """
    docs_coll, chunks_coll = _resolve_collections(db)
    existing = await _await_if_needed(docs_coll.find_one({"doc_id": doc_id}))
    if existing is None:
        logger.warning("[lifecycle] supersede: doc_id={} not found", doc_id)
        return _zero_report(doc_id, "not_found")
    if isinstance(existing, dict) and existing.get("status") == "superseded":
        logger.info("[lifecycle] supersede: doc_id={} already superseded", doc_id)
        return _zero_report(doc_id, "superseded")

    chunk_rows = await _collect_find(chunks_coll, {"doc_id": doc_id})
    point_ids = _point_ids_for_chunks(doc_id, chunk_rows)
    milvus_deleted = await _delete_vectors(store, point_ids, collection_name)

    now = _utcnow_iso()
    doc_update = {
        "status": "superseded",
        "superseded_by": superseded_by,
        "superseded_at": now,
        "supersede_reason": reason,
    }
    doc_res = await _await_if_needed(
        docs_coll.update_one({"doc_id": doc_id}, {"$set": doc_update})
    )
    chunk_res = await _await_if_needed(
        chunks_coll.update_many(
            {"doc_id": doc_id},
            {"$set": {"status": "superseded", "superseded_at": now}},
        )
    )
    mongo_docs = _modified_count(doc_res, fallback=1)
    mongo_chunks = _modified_count(chunk_res, fallback=len(chunk_rows))
    logger.info(
        "[lifecycle] supersede: doc_id={} docs={} chunks={} vectors={}",
        doc_id, mongo_docs, mongo_chunks, milvus_deleted,
    )
    return {
        "doc_id": doc_id,
        "mongo_docs": mongo_docs,
        "mongo_chunks": mongo_chunks,
        "milvus_deleted": milvus_deleted,
        "status": "superseded",
    }


async def delete_document(
    db: Any,
    store: Any,
    doc_id: str,
    purge_cache: Optional[Callable[[str], Any]] = None,
    *,
    collection_name: Optional[str] = None,
) -> dict[str, Any]:
    """Delete a document from every leg.

    - Mongo ``documents_v2`` row + ``chunks_v2`` rows are removed
      (falls back to a ``status="deleted"`` tombstone when the collection
      object has no delete methods, e.g. minimal stubs).
    - Its Milvus vectors are deleted by deterministic point id
      (collection is never dropped).
    - ``purge_cache(doc_id)`` is invoked on a best-effort basis when given:
      cache failures are logged and never fail the delete.

    Idempotent: a repeat call on a missing row succeeds with zeros.
    """
    docs_coll, chunks_coll = _resolve_collections(db)
    existing = await _await_if_needed(docs_coll.find_one({"doc_id": doc_id}))
    chunk_rows = await _collect_find(chunks_coll, {"doc_id": doc_id})
    if existing is None and not chunk_rows:
        logger.info("[lifecycle] delete: doc_id={} not found (noop)", doc_id)
        return _zero_report(doc_id, "not_found")

    point_ids = _point_ids_for_chunks(doc_id, chunk_rows)
    milvus_deleted = await _delete_vectors(store, point_ids, collection_name)

    if hasattr(docs_coll, "delete_one") and hasattr(chunks_coll, "delete_many"):
        doc_res = await _await_if_needed(docs_coll.delete_one({"doc_id": doc_id}))
        chunk_res = await _await_if_needed(chunks_coll.delete_many({"doc_id": doc_id}))
        mongo_docs = _modified_count(doc_res, fallback=1 if existing is not None else 0)
        mongo_chunks = _modified_count(chunk_res, fallback=len(chunk_rows))
    else:  # stub collections without delete support → tombstone
        now = _utcnow_iso()
        doc_res = await _await_if_needed(
            docs_coll.update_one(
                {"doc_id": doc_id}, {"$set": {"status": "deleted", "deleted_at": now}}
            )
        )
        chunk_res = await _await_if_needed(
            chunks_coll.update_many(
                {"doc_id": doc_id},
                {"$set": {"status": "deleted", "deleted_at": now}},
            )
        )
        mongo_docs = _modified_count(doc_res, fallback=1)
        mongo_chunks = _modified_count(chunk_res, fallback=len(chunk_rows))

    if purge_cache is not None:
        try:
            await _await_if_needed(purge_cache(doc_id))
        except Exception as exc:  # never fail the delete if cache is down
            logger.warning(
                "[lifecycle] delete: cache purge failed for doc_id={}: {}",
                doc_id, exc,
            )

    logger.info(
        "[lifecycle] delete: doc_id={} docs={} chunks={} vectors={}",
        doc_id, mongo_docs, mongo_chunks, milvus_deleted,
    )
    return {
        "doc_id": doc_id,
        "mongo_docs": mongo_docs,
        "mongo_chunks": mongo_chunks,
        "milvus_deleted": milvus_deleted,
        "status": "deleted",
    }
