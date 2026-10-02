"""Tests for the raw lake storage layer (docs/RAW_LAKE.md sections 1-2).

tmp_path-backed, no network/DB/torch.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.ingestion import raw_lake
from src.ingestion.raw_lake import (
    find_by_hash,
    iter_raw,
    lake_stats,
    sanitize_doc_id,
    write_raw,
)


@pytest.fixture
def lake_root(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENINSIGHT_RAW_ROOT", str(tmp_path / "raw"))
    # RAW_ROOT may have been snapshotted at import; env override wins anyway.
    monkeypatch.setattr(raw_lake, "RAW_ROOT", tmp_path / "raw")
    return tmp_path / "raw"


class TestWriteReadRoundtrip:
    def test_roundtrip_byte_identical(self, lake_root):
        content = b"<xml>hello \x00\xff binary</xml>"
        entry = write_raw("pubmed", "pmid_123", content, {"url": "https://x/y", "content_type": "application/xml"})
        assert (lake_root / "pubmed" / "pmid_123.bin").read_bytes() == content
        items = list(iter_raw("pubmed"))
        assert len(items) == 1
        meta, data = items[0]
        assert data == content
        assert meta["sha256"] == hashlib.sha256(content).hexdigest() == entry["sha256"]

    def test_limit(self, lake_root):
        for i in range(3):
            write_raw("pubmed", f"doc_{i}", f"bytes-{i}".encode(), {"url": f"https://x/{i}"})
        assert len(list(iter_raw("pubmed", limit=2))) == 2
        assert len(list(iter_raw("pubmed", limit=0))) == 3

    def test_empty_source_yields_nothing(self, lake_root):
        assert list(iter_raw("nosuchsource")) == []


class TestManifestSchema:
    def test_exact_fields(self, lake_root):
        entry = write_raw(
            "statpearls",
            "NBK123",
            b"<html/>",
            {"url": "https://s/NBK123", "content_type": "text/html", "license": "open", "title": "T"},
        )
        assert set(entry.keys()) == {
            "doc_id",
            "source",
            "url",
            "fetched_at",
            "sha256",
            "content_type",
            "license",
            "title",
            "error",
        }
        lines = (lake_root / "statpearls" / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        on_disk = json.loads(lines[0])
        assert on_disk == entry
        assert on_disk["doc_id"] == "NBK123"
        assert on_disk["source"] == "statpearls"

    def test_defaults_present(self, lake_root):
        entry = write_raw("pubmed", "pmid_9", b"data", {})
        for key in ("doc_id", "source", "url", "fetched_at", "sha256", "content_type", "license", "title", "error"):
            assert key in entry
        assert entry["fetched_at"]  # UTC ISO auto-filled


class TestFindByHash:
    def test_hit(self, lake_root):
        content = b"predictable-bytes"
        entry = write_raw("pubmed", "pmid_1", content, {"url": "https://x/1"})
        found = find_by_hash(hashlib.sha256(content).hexdigest())
        assert found is not None
        assert found["doc_id"] == entry["doc_id"]
        assert found["sha256"] == entry["sha256"]

    def test_miss(self, lake_root):
        write_raw("pubmed", "pmid_1", b"abc", {"url": "https://x/1"})
        assert find_by_hash("0" * 64) is None
        assert find_by_hash(hashlib.sha256(b"never-written").hexdigest()) is None


class TestLakeStats:
    def test_counts_and_bytes(self, lake_root):
        write_raw("pubmed", "a", b"12345", {"url": "https://x/a"})
        write_raw("pubmed", "b", b"123", {"url": "https://x/b"})
        write_raw("statpearls", "c", b"12", {"url": "https://x/c"})
        stats = lake_stats()
        assert stats["pubmed"]["docs"] == 2
        assert stats["pubmed"]["bytes"] == 8
        assert stats["statpearls"]["docs"] == 1
        assert stats["statpearls"]["bytes"] == 2

    def test_source_filter(self, lake_root):
        write_raw("pubmed", "a", b"12345", {"url": "https://x/a"})
        stats = lake_stats("pubmed")
        assert stats["pubmed"]["docs"] == 1
        assert stats["pubmed"]["bytes"] == 5


class TestSanitization:
    @pytest.mark.parametrize(
        "doc_id",
        ["https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?id=123", "..", ".", "../evil", "a/b\\c:d*e?f"],
    )
    def test_no_escape(self, lake_root, doc_id):
        content = b"payload"
        write_raw("pubmed", doc_id, content, {"url": "https://x"})
        stem = sanitize_doc_id(doc_id)
        assert "/" not in stem and "\\" not in stem
        assert stem not in ("", ".", "..")
        target = (lake_root / "pubmed" / (stem + ".bin")).resolve()
        assert str(target).startswith(str((lake_root / "pubmed").resolve()))
        assert target.read_bytes() == content
        # Manifest keeps the ORIGINAL doc_id for joining raw to index.
        metas = [m for m, _ in iter_raw("pubmed")]
        assert any(m["doc_id"] == doc_id for m in metas)

    def test_dotdot_stays_inside(self, lake_root):
        write_raw("pubmed", "..", b"x", {"url": "https://x"})
        assert ((lake_root / "pubmed").resolve() / (sanitize_doc_id("..") + ".bin")).is_file()
        assert not (lake_root / "x.bin").exists()
        assert Path(str(lake_root) + ".bin").exists() is False
