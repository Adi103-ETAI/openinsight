"""
Tests for the data quality validation layer.
"""
import pytest
from src.ingestion.validation import (
    validate_document,
    validate_chunk,
    filter_valid_chunks,
    VALID_SOURCE_TYPES,
    _garble_score,
)
from src.ingestion.document_db import ChunkRecord, DocumentRecord


def _make_doc(**kwargs) -> DocumentRecord:
    defaults = dict(
        source_type="pubmed",
        title="Efficacy of Doxycycline in Treating Malaria: A Systematic Review",
        content="A" * 300,
    )
    defaults.update(kwargs)
    return DocumentRecord(**defaults)


def _make_chunk(**kwargs) -> ChunkRecord:
    defaults = dict(
        document_id="doc1",
        source_type="pubmed",
        title="Test",
        chunk_text="The patient received doxycycline 100 mg twice daily for 7 days with good response.",
        chunk_index=0,
        char_count=90,
        token_count=20,
    )
    defaults.update(kwargs)
    return ChunkRecord(**defaults)


class TestValidateDocument:
    def test_valid_document_passes(self):
        ok, reason = validate_document(_make_doc())
        assert ok is True
        assert reason is None

    def test_empty_title_fails(self):
        ok, reason = validate_document(_make_doc(title=""))
        assert ok is False
        assert "title" in reason

    def test_short_title_fails(self):
        ok, reason = validate_document(_make_doc(title="AB"))
        assert ok is False

    def test_empty_content_fails(self):
        ok, reason = validate_document(_make_doc(content=""))
        assert ok is False

    def test_too_short_content_fails(self):
        ok, reason = validate_document(_make_doc(content="Short."))
        assert ok is False
        assert "short" in reason

    def test_invalid_source_type_fails(self):
        ok, reason = validate_document(_make_doc(source_type="twitter"))
        assert ok is False
        assert "source_type" in reason

    def test_all_valid_source_types_pass(self):
        for st in VALID_SOURCE_TYPES:
            doc = _make_doc(source_type=st)
            ok, reason = validate_document(doc)
            assert ok is True, f"source_type {st!r} should be valid but got: {reason}"

    def test_garbled_content_fails(self):
        # High non-ASCII ratio content
        garbled = "\x80\x81\x82\x83" * 100
        ok, reason = validate_document(_make_doc(content=garbled))
        assert ok is False


class TestValidateChunk:
    def test_valid_chunk_passes(self):
        ok, reason = validate_chunk(_make_chunk())
        assert ok is True
        assert reason is None

    def test_empty_chunk_fails(self):
        ok, reason = validate_chunk(_make_chunk(chunk_text=""))
        assert ok is False

    def test_too_short_chunk_fails(self):
        ok, reason = validate_chunk(_make_chunk(chunk_text="Short."))
        assert ok is False
        assert "short" in reason

    def test_too_long_chunk_fails(self):
        ok, reason = validate_chunk(_make_chunk(chunk_text="word " * 2000))
        assert ok is False
        assert "long" in reason

    def test_mostly_digits_fails(self):
        # A chunk that's mostly digits (like a reference list)
        ok, reason = validate_chunk(_make_chunk(chunk_text="1234567890 " * 30))
        assert ok is False

    def test_garbled_chunk_fails(self):
        garbled = "\x80\x81" * 100
        ok, reason = validate_chunk(_make_chunk(chunk_text=garbled))
        assert ok is False


class TestFilterValidChunks:
    def test_all_valid_returns_all(self):
        chunks = [_make_chunk(chunk_index=i) for i in range(3)]
        valid, rejected = filter_valid_chunks(chunks)
        assert len(valid) == 3
        assert len(rejected) == 0

    def test_filters_invalid(self):
        chunks = [
            _make_chunk(chunk_index=0),
            _make_chunk(chunk_index=1, chunk_text=""),   # invalid
            _make_chunk(chunk_index=2),
        ]
        valid, rejected = filter_valid_chunks(chunks)
        assert len(valid) == 2
        assert len(rejected) == 1
        assert rejected[0]["chunk_index"] == 1


class TestGarbleScore:
    def test_ascii_text_low_score(self):
        assert _garble_score("Hello world, this is normal text.") < 0.1

    def test_high_non_ascii_score(self):
        text = "\x80\x90\xa0" * 50
        assert _garble_score(text) > 0.5

    def test_empty_text(self):
        assert _garble_score("") == 1.0
