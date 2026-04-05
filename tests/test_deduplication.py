"""
Tests for the deduplication engine.
"""
import pytest
from src.ingestion.deduplication import (
    compute_content_hash,
    enrich_document_hashes,
    _normalise_title,
    _title_similarity,
)
from src.ingestion.document_db import DocumentRecord


class TestComputeContentHash:
    def test_stable_hash(self):
        text = "Hello World"
        assert compute_content_hash(text) == compute_content_hash(text)

    def test_normalisation_insensitive(self):
        # Different whitespace / casing should produce same hash
        a = compute_content_hash("  Hello   World  ")
        b = compute_content_hash("hello world")
        assert a == b

    def test_different_texts_differ(self):
        assert compute_content_hash("foo") != compute_content_hash("bar")

    def test_returns_hex_string(self):
        h = compute_content_hash("test")
        assert isinstance(h, str)
        assert len(h) == 64  # SHA-256 hex = 64 chars


class TestNormaliseTitle:
    def test_lowercases(self):
        assert _normalise_title("Hello World") == "hello world"

    def test_strips_punctuation(self):
        assert _normalise_title("COVID-19: treatment.") == "covid19 treatment"

    def test_collapses_whitespace(self):
        assert _normalise_title("  hello   world  ") == "hello world"


class TestTitleSimilarity:
    def test_identical_titles(self):
        score = _title_similarity("treatment of malaria", "treatment of malaria")
        assert score == 1.0

    def test_very_different_titles(self):
        score = _title_similarity("malaria treatment guidelines", "quantum physics research")
        assert score < 0.3

    def test_similar_titles_above_threshold(self):
        a = "management of hypertension in adults"
        b = "management of hypertension in adult patients"
        score = _title_similarity(a, b)
        assert score >= 0.6

    def test_empty_title(self):
        assert _title_similarity("", "something") == 0.0
        assert _title_similarity("something", "") == 0.0


class TestEnrichDocumentHashes:
    def _make_doc(self, content: str = "Sample medical content for testing purposes.") -> DocumentRecord:
        return DocumentRecord(
            source_type="pubmed",
            title="Test Document",
            content=content,
        )

    def test_hash_is_set(self):
        doc = self._make_doc()
        result = enrich_document_hashes(doc)
        assert result.content_hash is not None
        assert len(result.content_hash) == 64

    def test_hash_is_deterministic(self):
        doc1 = self._make_doc("identical content")
        doc2 = self._make_doc("identical content")
        assert enrich_document_hashes(doc1).content_hash == enrich_document_hashes(doc2).content_hash

    def test_different_content_different_hash(self):
        doc1 = self._make_doc("content alpha")
        doc2 = self._make_doc("content beta")
        assert enrich_document_hashes(doc1).content_hash != enrich_document_hashes(doc2).content_hash
