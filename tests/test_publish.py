"""Publish gate + smoke unit tests — pure logic, no DB/torch needed."""
from __future__ import annotations

import pytest

from src.ingestion.publish import (
    check_chunk_texts,
    check_counts,
    check_dimension,
    check_metadata,
    run_gate,
)
from src.ingestion.smoke import GOLDEN_QAS, keyword_recall, run_smoke


class TestCounts:
    def test_match_passes(self):
        assert check_counts(42, 42, 42).passed

    def test_mismatch_fails(self):
        assert not check_counts(42, 40, 42).passed


class TestDimension:
    def test_match_passes(self):
        assert check_dimension(768, 768).passed

    def test_cohere_1024_fails_768(self):
        assert not check_dimension(1024, 768).passed


class TestChunkTexts:
    def test_empties_fail(self):
        assert not check_chunk_texts(["real text", "   "]).passed

    def test_empty_sample_fails(self):
        assert not check_chunk_texts([]).passed


class TestMetadata:
    def test_missing_title_fails(self):
        items = [{"title": "", "source_type": "pubmed"}]
        assert not check_metadata(items, ("title", "source_type"), "chunks").passed


class TestGate:
    def test_full_pass(self):
        metas = [{"title": "T", "source_type": "pubmed"}]
        r = run_gate(
            source="pubmed", produced_chunks=2, mongo_chunks=2, milvus_rows=2,
            collection_dim=768, expected_dim=768,
            sample_texts=["a", "b"], sample_metas=metas,
        )
        assert r.passed


class FakeCursor(list):
    def limit(self, *a, **k):
        return self


class FakeCollection:
    def __init__(self, docs):
        self._docs = docs

    def find(self, *a, **k):
        return FakeCursor(self._docs)


class FakeDB(dict):
    def __getitem__(self, name):
        return super().__getitem__(name)


class TestSmoke:
    def _db(self, texts):
        return FakeDB({"chunks_v2": FakeCollection([{"text": t} for t in texts])})

    def test_recall_hit(self):
        from src.ingestion.smoke import GoldenQA
        db = self._db(["rifampicin induces CYP450 enzymes"])
        assert keyword_recall(db, GoldenQA(question="q", must_match_any=("rifampicin",)))

    def test_recall_miss(self):
        from src.ingestion.smoke import GoldenQA
        db = self._db(["unrelated cardiology text"])
        assert not keyword_recall(db, GoldenQA(question="q", must_match_any=("rifampicin",)))

    def test_golden_set_has_five(self):
        assert len(GOLDEN_QAS) == 5
