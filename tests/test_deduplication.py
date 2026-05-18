"""
Tests for the deduplication engine.

Markers:
- unit: All tests in this module are unit tests
"""
from __future__ import annotations

import pytest

from src.ingestion.deduplication import (
    compute_content_hash,
    enrich_document_hashes,
    _normalise_title,
    _title_similarity,
)
from src.ingestion.document_db import DocumentRecord


@pytest.mark.unit
class TestComputeContentHash:
    """Tests for content hash computation."""

    def test_stable_hash(self):
        """Same text should always produce the same hash."""
        text = "Hello World"
        assert compute_content_hash(text) == compute_content_hash(text)

    def test_normalisation_insensitive(self):
        """Different whitespace and casing should produce the same hash."""
        a = compute_content_hash("  Hello   World  ")
        b = compute_content_hash("hello world")
        assert a == b, "Hash should be insensitive to whitespace and casing"

    def test_different_texts_differ(self):
        """Different texts should produce different hashes."""
        assert compute_content_hash("foo") != compute_content_hash("bar")

    def test_returns_hex_string(self):
        """Hash should be a 64-character hex string (SHA-256)."""
        h = compute_content_hash("test")
        assert isinstance(h, str)
        assert len(h) == 64, f"Expected 64-char hex string, got {len(h)}"

    def test_empty_string_hash(self):
        """Empty string should produce a valid hash."""
        h = compute_content_hash("")
        assert isinstance(h, str)
        assert len(h) == 64

    def test_unicode_content_hash(self):
        """Unicode content should produce a valid hash."""
        h = compute_content_hash("Tuberculosis \u00e0 la fran\u00e7aise")
        assert isinstance(h, str)
        assert len(h) == 64


@pytest.mark.unit
class TestNormaliseTitle:
    """Tests for title normalisation."""

    def test_lowercases(self):
        """Title should be lowercased."""
        assert _normalise_title("Hello World") == "hello world"

    def test_strips_punctuation(self):
        """Punctuation should be stripped."""
        assert _normalise_title("COVID-19: treatment.") == "covid19 treatment"

    def test_collapses_whitespace(self):
        """Multiple whitespace should be collapsed."""
        assert _normalise_title("  hello   world  ") == "hello world"

    @pytest.mark.parametrize(
        "input_title, expected",
        [
            pytest.param("", "", id="empty"),
            pytest.param("  ", "", id="whitespace_only"),
            pytest.param("HELLO WORLD", "hello world", id="all_caps"),
            pytest.param("a-b-c", "abc", id="hyphens"),
        ],
    )
    def test_normalise_title_parametrized(self, input_title: str, expected: str):
        """Title normalisation should handle various inputs."""
        assert _normalise_title(input_title) == expected


@pytest.mark.unit
class TestTitleSimilarity:
    """Tests for title similarity computation."""

    def test_identical_titles(self):
        """Identical titles should have similarity of 1.0."""
        score = _title_similarity("treatment of malaria", "treatment of malaria")
        assert score == 1.0

    def test_very_different_titles(self):
        """Very different titles should have low similarity."""
        score = _title_similarity("malaria treatment guidelines", "quantum physics research")
        assert score < 0.3

    def test_similar_titles_above_threshold(self):
        """Similar titles should have high similarity."""
        a = "management of hypertension in adults"
        b = "management of hypertension in adult patients"
        score = _title_similarity(a, b)
        assert score >= 0.6

    def test_empty_title(self):
        """Empty title should result in 0.0 similarity."""
        assert _title_similarity("", "something") == 0.0
        assert _title_similarity("something", "") == 0.0

    def test_both_empty(self):
        """Both empty titles should return 0.0."""
        assert _title_similarity("", "") == 0.0


@pytest.mark.unit
class TestEnrichDocumentHashes:
    """Tests for document hash enrichment."""

    @staticmethod
    def _make_doc(content: str = "Sample medical content for testing purposes.") -> DocumentRecord:
        """Create a test DocumentRecord."""
        return DocumentRecord(
            source_type="pubmed",
            title="Test Document",
            content=content,
        )

    def test_hash_is_set(self):
        """Enriched document should have a content hash."""
        doc = self._make_doc()
        result = enrich_document_hashes(doc)
        assert result.content_hash is not None
        assert len(result.content_hash) == 64

    def test_hash_is_deterministic(self):
        """Same content should produce the same hash."""
        doc1 = self._make_doc("identical content")
        doc2 = self._make_doc("identical content")
        assert enrich_document_hashes(doc1).content_hash == enrich_document_hashes(doc2).content_hash

    def test_different_content_different_hash(self):
        """Different content should produce different hashes."""
        doc1 = self._make_doc("content alpha")
        doc2 = self._make_doc("content beta")
        assert enrich_document_hashes(doc1).content_hash != enrich_document_hashes(doc2).content_hash

    def test_returns_same_document_instance(self):
        """enrich_document_hashes should return the same document."""
        doc = self._make_doc()
        result = enrich_document_hashes(doc)
        assert result is doc
