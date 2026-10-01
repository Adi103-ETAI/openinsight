"""Supersede + delete lifecycle tests — pure logic, no DB/torch needed."""
from __future__ import annotations

from uuid import NAMESPACE_DNS, uuid5

import pytest

from src.ingestion.lifecycle import (
    chunk_point_id,
    delete_document,
    point_id_for_chunk_id,
    supersede_document,
)


def _match(doc: dict, filt: dict) -> bool:
    return all(doc.get(k) == v for k, v in filt.items())


class FakeCollection:
    """Minimal sync mongo-like collection (find/find_one/update/delete)."""

    def __init__(self, docs: list[dict] | None = None):
        self._docs: list[dict] = [dict(d) for d in (docs or [])]

    def find(self, filt):
        return [d for d in self._docs if _match(d, filt)]

    def find_one(self, filt):
        for d in self._docs:
            if _match(d, filt):
                return d
        return None

    def update_one(self, filt, update):
        for d in self._docs:
            if _match(d, filt):
                d.update(update.get("$set", {}))
                return _Result(modified_count=1, matched_count=1)
        return _Result(modified_count=0, matched_count=0)

    def update_many(self, filt, update):
        n = 0
        for d in self._docs:
            if _match(d, filt):
                d.update(update.get("$set", {}))
                n += 1
        return _Result(modified_count=n, matched_count=n)

    def delete_one(self, filt):
        for i, d in enumerate(self._docs):
            if _match(d, filt):
                del self._docs[i]
                return _Result(deleted_count=1)
        return _Result(deleted_count=0)

    def delete_many(self, filt):
        kept = [d for d in self._docs if not _match(d, filt)]
        n = len(self._docs) - len(kept)
        self._docs[:] = kept
        return _Result(deleted_count=n)


class _Result:
    def __init__(self, modified_count=0, matched_count=0, deleted_count=0):
        self.modified_count = modified_count
        self.matched_count = matched_count
        self.deleted_count = deleted_count


class FakeStore:
    """Sync stub vector store with delete_points_by_ids."""

    def __init__(self):
        self.deleted: list[str] = []
        self.collections: list[str | None] = []

    def delete_points_by_ids(self, ids, collection_name=None):
        self.deleted.extend(str(i) for i in ids)
        self.collections.append(collection_name)
        return len(ids)


class FakeDB:
    def __init__(self, docs, chunks):
        self.documents = FakeCollection(docs)
        self.chunks = FakeCollection(chunks)


def _seed_db() -> FakeDB:
    doc_id = "doc-1"
    return FakeDB(
        docs=[{"doc_id": doc_id, "title": "T", "status": "ready"}],
        chunks=[
            {"chunk_id": f"{doc_id}-c{i:03d}", "doc_id": doc_id, "chunk_index": i}
            for i in range(3)
        ],
    )


class TestPointIds:
    def test_matches_pipeline_convention(self):
        assert chunk_point_id("doc-1", 2) == str(
            uuid5(NAMESPACE_DNS, "doc-1-c002")
        )

    def test_chunk_id_helper_agrees(self):
        assert point_id_for_chunk_id("doc-1-c007") == chunk_point_id("doc-1", 7)


@pytest.mark.unit
@pytest.mark.asyncio
class TestSupersede:
    async def test_marks_all_legs(self):
        db, store = _seed_db(), FakeStore()
        report = await supersede_document(db, store, "doc-1", "newer version")

        assert report["status"] == "superseded"
        assert report == {
            "doc_id": "doc-1",
            "mongo_docs": 1,
            "mongo_chunks": 3,
            "milvus_deleted": 3,
            "status": "superseded",
        }
        doc = db.documents.find_one({"doc_id": "doc-1"})
        assert doc["status"] == "superseded"
        assert doc["supersede_reason"] == "newer version"
        assert "superseded_at" in doc and "superseded_by" in doc
        assert all(
            c.get("status") == "superseded" for c in db.chunks.find({"doc_id": "doc-1"})
        )
        assert store.deleted == [chunk_point_id("doc-1", i) for i in range(3)]

    async def test_idempotent_second_call(self):
        db, store = _seed_db(), FakeStore()
        await supersede_document(db, store, "doc-1", "r")
        report = await supersede_document(db, store, "doc-1", "r")
        assert report["milvus_deleted"] == 0
        assert report["mongo_docs"] == 0 and report["mongo_chunks"] == 0
        assert report["status"] == "superseded"

    async def test_missing_doc_is_noop(self):
        db, store = _seed_db(), FakeStore()
        report = await supersede_document(db, store, "nope", "r")
        assert report["status"] == "not_found"
        assert store.deleted == []


@pytest.mark.unit
@pytest.mark.asyncio
class TestDelete:
    async def test_removes_all_legs(self):
        db, store = _seed_db(), FakeStore()
        report = await delete_document(db, store, "doc-1")

        assert report["status"] == "deleted"
        assert report["mongo_docs"] == 1
        assert report["mongo_chunks"] == 3
        assert report["milvus_deleted"] == 3
        assert db.documents.find_one({"doc_id": "doc-1"}) is None
        assert db.chunks.find({"doc_id": "doc-1"}) == []
        assert store.deleted == [chunk_point_id("doc-1", i) for i in range(3)]

    async def test_idempotent_second_call(self):
        db, store = _seed_db(), FakeStore()
        await delete_document(db, store, "doc-1")
        report = await delete_document(db, store, "doc-1")
        assert report == {
            "doc_id": "doc-1",
            "mongo_docs": 0,
            "mongo_chunks": 0,
            "milvus_deleted": 0,
            "status": "not_found",
        }

    async def test_cache_failure_never_fails_delete(self):
        db, store = _seed_db(), FakeStore()

        def bad_purge(doc_id: str):
            raise ConnectionError("redis down")

        report = await delete_document(db, store, "doc-1", purge_cache=bad_purge)
        assert report["status"] == "deleted"
        assert db.documents.find_one({"doc_id": "doc-1"}) is None

    async def test_cache_purge_called(self):
        db, store = _seed_db(), FakeStore()
        seen: list[str] = []
        await delete_document(db, store, "doc-1", purge_cache=seen.append)
        assert seen == ["doc-1"]

    async def test_raw_mapping_db_supported(self):
        db, store = _seed_db(), FakeStore()
        raw = {"documents_v2": db.documents, "chunks_v2": db.chunks}
        report = await delete_document(raw, store, "doc-1")
        assert report["status"] == "deleted"
