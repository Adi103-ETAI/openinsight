"""
Tests for the data quality validation layer.

Markers:
- unit: All tests in this module are unit tests
"""
from __future__ import annotations

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
    """Factory for creating test DocumentRecord instances."""
    defaults = dict(
        source_type="pubmed",
        title="Efficacy of Doxycycline in Treating Malaria: A Systematic Review",
        content="A" * 300,
    )
    defaults.update(kwargs)
    return DocumentRecord(**defaults)


def _make_chunk(**kwargs) -> ChunkRecord:
    """Factory for creating test ChunkRecord instances."""
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


@pytest.mark.unit
class TestValidateDocument:
    """Tests for document-level validation."""

    def test_valid_document_passes(self):
        """Valid document should pass validation."""
        ok, reason = validate_document(_make_doc())
        assert ok is True
        assert reason is None

    def test_empty_title_fails(self):
        """Empty title should fail validation."""
        ok, reason = validate_document(_make_doc(title=""))
        assert ok is False
        assert "title" in reason

    def test_short_title_fails(self):
        """Too-short title should fail validation."""
        ok, reason = validate_document(_make_doc(title="AB"))
        assert ok is False

    def test_empty_content_fails(self):
        """Empty content should fail validation."""
        ok, reason = validate_document(_make_doc(content=""))
        assert ok is False

    def test_too_short_content_fails(self):
        """Too-short content should fail validation."""
        ok, reason = validate_document(_make_doc(content="Short."))
        assert ok is False
        assert "short" in reason

    def test_invalid_source_type_fails(self):
        """Invalid source type should fail validation."""
        ok, reason = validate_document(_make_doc(source_type="twitter"))
        assert ok is False
        assert "source_type" in reason

    def test_all_valid_source_types_pass(self):
        """All valid source types should pass validation."""
        for st in VALID_SOURCE_TYPES:
            doc = _make_doc(source_type=st)
            ok, reason = validate_document(doc)
            assert ok is True, f"source_type {st!r} should be valid but got: {reason}"

    def test_garbled_content_fails(self):
        """Garbled (high non-ASCII) content should fail validation."""
        garbled = "\x80\x81\x82\x83" * 100
        ok, reason = validate_document(_make_doc(content=garbled))
        assert ok is False

    @pytest.mark.parametrize(
        "title, should_pass",
        [
            pytest.param("A" * 10, True, id="minimum_length_title"),
            pytest.param("AB", False, id="too_short_title"),
            pytest.param("Valid Medical Research Title", True, id="normal_title"),
        ],
    )
    def test_title_length_validation(self, title: str, should_pass: bool):
        """Title length should be validated."""
        ok, _ = validate_document(_make_doc(title=title))
        assert ok is should_pass


@pytest.mark.unit
class TestValidateChunk:
    """Tests for chunk-level validation."""

    def test_valid_chunk_passes(self):
        """Valid chunk should pass validation."""
        ok, reason = validate_chunk(_make_chunk())
        assert ok is True
        assert reason is None

    def test_empty_chunk_fails(self):
        """Empty chunk should fail validation."""
        ok, reason = validate_chunk(_make_chunk(chunk_text=""))
        assert ok is False

    def test_too_short_chunk_fails(self):
        """Too-short chunk should fail validation."""
        ok, reason = validate_chunk(_make_chunk(chunk_text="Short."))
        assert ok is False
        assert "short" in reason

    def test_too_long_chunk_fails(self):
        """Too-long chunk should fail validation."""
        ok, reason = validate_chunk(_make_chunk(chunk_text="word " * 2000))
        assert ok is False
        assert "long" in reason

    def test_mostly_digits_fails(self):
        """Chunk that's mostly digits should fail validation."""
        ok, reason = validate_chunk(_make_chunk(chunk_text="1234567890 " * 30))
        assert ok is False

    def test_garbled_chunk_fails(self):
        """Garbled chunk should fail validation."""
        garbled = "\x80\x81" * 100
        ok, reason = validate_chunk(_make_chunk(chunk_text=garbled))
        assert ok is False

    @pytest.mark.parametrize(
        "chunk_text, should_pass",
        [
            pytest.param("Normal medical text about treatment protocols and patient care guidelines for doctors.", True, id="normal_text"),
            pytest.param("", False, id="empty"),
            pytest.param("Hi", False, id="too_short"),
        ],
    )
    def test_chunk_text_validation(self, chunk_text: str, should_pass: bool):
        """Chunk text should be validated correctly."""
        ok, _ = validate_chunk(_make_chunk(chunk_text=chunk_text))
        assert ok is should_pass


@pytest.mark.unit
class TestFilterValidChunks:
    """Tests for chunk filtering."""

    def test_all_valid_returns_all(self):
        """All valid chunks should be returned."""
        chunks = [_make_chunk(chunk_index=i) for i in range(3)]
        valid, rejected = filter_valid_chunks(chunks)
        assert len(valid) == 3
        assert len(rejected) == 0

    def test_filters_invalid(self):
        """Invalid chunks should be filtered out."""
        chunks = [
            _make_chunk(chunk_index=0),
            _make_chunk(chunk_index=1, chunk_text=""),  # invalid
            _make_chunk(chunk_index=2),
        ]
        valid, rejected = filter_valid_chunks(chunks)
        assert len(valid) == 2
        assert len(rejected) == 1
        assert rejected[0]["chunk_index"] == 1

    def test_empty_list(self):
        """Empty list should return empty results."""
        valid, rejected = filter_valid_chunks([])
        assert valid == []
        assert rejected == []

    def test_all_invalid(self):
        """All invalid chunks should result in empty valid list."""
        chunks = [
            _make_chunk(chunk_index=i, chunk_text="")
            for i in range(3)
        ]
        valid, rejected = filter_valid_chunks(chunks)
        assert len(valid) == 0
        assert len(rejected) == 3


@pytest.mark.unit
class TestGarbleScore:
    """Tests for garble score computation."""

    def test_ascii_text_low_score(self):
        """Normal ASCII text should have low garble score."""
        assert _garble_score("Hello world, this is normal text.") < 0.1

    def test_high_non_ascii_score(self):
        """High non-ASCII content should have high garble score."""
        text = "\x80\x90\xa0" * 50
        assert _garble_score(text) > 0.5

    def test_empty_text(self):
        """Empty text should return maximum garble score."""
        assert _garble_score("") == 1.0

    @pytest.mark.parametrize(
        "text, expected_below",
        [
            pytest.param("Normal text", 0.1, id="normal"),
            pytest.param("Medical terminology: tuberculosis", 0.1, id="medical"),
            pytest.param("12345", 0.5, id="digits"),
        ],
    )
    def test_garble_score_thresholds(self, text: str, expected_below: float):
        """Garble score should be below threshold for valid text."""
        score = _garble_score(text)
        assert score < expected_below, f"Score {score} >= {expected_below} for '{text}'"
