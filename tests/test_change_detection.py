"""Change-detection (ingestion idempotency) unit tests — pure logic, no torch/network/DB."""
from __future__ import annotations

import pytest

from src.ingestion.deduplication import (
    compute_content_hash,
    find_existing_status_by_hash,
    new_scraped_summary,
    should_skip_unchanged,
)


class FakeCollection:
    """Dict-backed stub mimicking motor's `find_one` for `documents_v2`."""

    def __init__(self, by_hash: dict):
        self._by_hash = dict(by_hash)

    async def find_one(self, filt, projection=None):
        return self._by_hash.get((filt or {}).get("content_hash"))


class TestShouldSkipUnchanged:
    def test_ready_skips(self):
        assert should_skip_unchanged("ready") is True

    def test_failed_reprocesses(self):
        assert should_skip_unchanged("failed") is False

    def test_indexed_reprocesses(self):
        assert should_skip_unchanged("indexed") is False

    def test_validated_reprocesses(self):
        assert should_skip_unchanged("validated") is False

    def test_no_match_reprocesses(self):
        assert should_skip_unchanged(None) is False

    def test_empty_status_reprocesses(self):
        assert should_skip_unchanged("") is False

    def test_status_is_case_sensitive(self):
        assert should_skip_unchanged("Ready") is False
        assert should_skip_unchanged("READY") is False


class TestFindExistingStatusByHash:
    @pytest.mark.asyncio
    async def test_returns_ready_status(self):
        col = FakeCollection({"abc": {"_id": "1", "status": "ready"}})
        assert await find_existing_status_by_hash(col, "abc") == "ready"

    @pytest.mark.asyncio
    async def test_returns_failed_status(self):
        col = FakeCollection({"abc": {"_id": "1", "status": "failed"}})
        assert await find_existing_status_by_hash(col, "abc") == "failed"

    @pytest.mark.asyncio
    async def test_missing_hash_returns_none(self):
        col = FakeCollection({})
        assert await find_existing_status_by_hash(col, "nope") is None

    @pytest.mark.asyncio
    async def test_empty_hash_returns_none_without_query(self):
        class ExplodingCollection:
            async def find_one(self, *a, **k):
                raise AssertionError("must not query on empty hash")

        assert await find_existing_status_by_hash(ExplodingCollection(), "") is None

    @pytest.mark.asyncio
    async def test_lookup_error_returns_none(self):
        class BrokenCollection:
            async def find_one(self, *a, **k):
                raise ConnectionError("db down")

        assert await find_existing_status_by_hash(BrokenCollection(), "abc") is None

    @pytest.mark.asyncio
    async def test_doc_without_status_returns_none(self):
        col = FakeCollection({"abc": {"_id": "1"}})
        assert await find_existing_status_by_hash(col, "abc") is None

    @pytest.mark.asyncio
    async def test_non_string_status_returns_none(self):
        col = FakeCollection({"abc": {"_id": "1", "status": {"code": "ready"}}})
        assert await find_existing_status_by_hash(col, "abc") is None

    @pytest.mark.asyncio
    async def test_projection_requests_only_id_and_status(self):
        seen = {}

        class RecordingCollection:
            async def find_one(self, filt, projection=None):
                seen["filt"] = filt
                seen["projection"] = projection
                return None

        await find_existing_status_by_hash(RecordingCollection(), "abc")
        assert seen["filt"] == {"content_hash": "abc"}
        assert set(seen["projection"]) == {"_id", "status"}


class TestNewScrapedSummary:
    def test_documents_skipped_present_and_zero(self):
        assert new_scraped_summary(5)["documents_skipped"] == 0

    def test_existing_keys_preserved(self):
        summary = new_scraped_summary(3)
        assert summary["documents_total"] == 3
        assert summary["documents_stored"] == 0
        assert summary["chunks_created"] == 0
        assert summary["chunks_indexed"] == 0
        assert summary["chunks_filtered"] == 0
        assert summary["files_failed"] == 0

    def test_zero_documents(self):
        summary = new_scraped_summary(0)
        assert summary["documents_total"] == 0
        assert summary["documents_skipped"] == 0


class TestChangeDetectionEndToEnd:
    """Simulate the pipeline decision: hash content → look up → skip or not."""

    @pytest.mark.asyncio
    async def test_unchanged_ready_doc_is_skipped(self):
        content = "Metformin 500mg twice daily for type 2 diabetes."
        content_hash = compute_content_hash(content)
        col = FakeCollection({content_hash: {"_id": "doc1", "status": "ready"}})
        status = await find_existing_status_by_hash(col, content_hash)
        assert should_skip_unchanged(status) is True

    @pytest.mark.asyncio
    async def test_failed_doc_is_reprocessed(self):
        content = "Metformin 500mg twice daily for type 2 diabetes."
        content_hash = compute_content_hash(content)
        col = FakeCollection({content_hash: {"_id": "doc1", "status": "failed"}})
        status = await find_existing_status_by_hash(col, content_hash)
        assert should_skip_unchanged(status) is False

    @pytest.mark.asyncio
    async def test_new_content_is_processed(self):
        col = FakeCollection({})
        status = await find_existing_status_by_hash(
            col, compute_content_hash("brand new content")
        )
        assert should_skip_unchanged(status) is False

    @pytest.mark.asyncio
    async def test_edited_content_is_processed(self):
        old_hash = compute_content_hash("original text")
        col = FakeCollection({old_hash: {"_id": "doc1", "status": "ready"}})
        status = await find_existing_status_by_hash(
            col, compute_content_hash("original text with an edit")
        )
        assert should_skip_unchanged(status) is False
