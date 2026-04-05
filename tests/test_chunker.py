"""
Tests for the hierarchical chunker (chunker_v2).
"""
import pytest
from src.utils.chunker_v2 import (
    TextChunk,
    chunk_text_v2,
    _detect_section_header,
    _estimate_tokens,
    _split_into_sections,
    TARGET_TOKENS,
    MAX_TOKENS,
    MIN_CHUNK_CHARS,
)


class TestEstimateTokens:
    def test_empty_string(self):
        assert _estimate_tokens("") == 0

    def test_single_word(self):
        assert _estimate_tokens("hello") == 1  # int(1 * 1.3) == 1

    def test_approximation(self):
        text = "word " * 100
        tokens = _estimate_tokens(text)
        assert 100 <= tokens <= 150


class TestDetectSectionHeader:
    def test_treatment_header_uppercase(self):
        # All-caps headers are detected
        assert _detect_section_header("TREATMENT") == "TREATMENT"

    def test_numbered_treatment_header(self):
        assert _detect_section_header("3.1 Treatment") == "3.1 Treatment"

    def test_long_line_not_header(self):
        long = "This is a very long line that clearly is not a section header " * 3
        assert _detect_section_header(long) is None

    def test_empty_line(self):
        assert _detect_section_header("") is None

    def test_numbered_management_header(self):
        assert _detect_section_header("2. Management") == "2. Management"


class TestSplitIntoSections:
    def test_single_section(self):
        text = "Introduction\nSome content here."
        sections = _split_into_sections(text)
        assert len(sections) >= 1

    def test_multiple_sections(self):
        # Use numbered headers which are reliably detected
        text = (
            "1. Treatment\n"
            "Give drug A.\n\n"
            "2. Management\n"
            "Continue therapy.\n"
        )
        sections = _split_into_sections(text)
        # Should detect at least one section header
        headers = [s[0] for s in sections if s[0]]
        assert len(headers) >= 1


class TestChunkTextV2:
    def test_empty_returns_no_chunks(self):
        assert chunk_text_v2("") == []

    def test_whitespace_only_returns_no_chunks(self):
        assert chunk_text_v2("   \n  \t  ") == []

    def test_short_text_produces_one_chunk(self):
        # Text must be at least MIN_CHUNK_CHARS (100 chars)
        text = (
            "Administer doxycycline 100 mg twice daily for 7 days "
            "in adults with uncomplicated malaria fever symptoms."
        )
        assert len(text) >= MIN_CHUNK_CHARS
        chunks = chunk_text_v2(text)
        assert len(chunks) >= 1
        assert all(isinstance(c, TextChunk) for c in chunks)

    def test_chunk_indices_are_sequential(self):
        text = ("Patients should receive " + "treatment with antibiotics. " * 50)
        chunks = chunk_text_v2(text)
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i

    def test_no_chunk_below_min_chars(self):
        text = "A" * (MIN_CHUNK_CHARS + 10) + "\n\n" + "B" * (MIN_CHUNK_CHARS + 10)
        chunks = chunk_text_v2(text)
        for c in chunks:
            assert c.char_count >= MIN_CHUNK_CHARS

    def test_token_count_within_bounds(self):
        # Build a text long enough to produce multiple chunks
        text = "The patient presents with fever. " * 200
        chunks = chunk_text_v2(text)
        for c in chunks:
            assert c.token_count <= MAX_TOKENS + 50  # allow small overshoot

    def test_section_assigned(self):
        # Use numbered headers + content long enough to exceed MIN_CHUNK_CHARS
        text = (
            "1. TREATMENT\n"
            "Give antibiotics twice daily for 7 days to infected patients with confirmed diagnosis. "
            "Intravenous therapy is preferred in severe cases requiring hospitalisation.\n"
            "2. MANAGEMENT\n"
            "Monitor patient vitals including blood pressure, temperature, and oxygen saturation. "
            "Reassess every 6 hours and adjust therapy based on clinical response."
        )
        chunks = chunk_text_v2(text)
        sections = [c.section for c in chunks]
        # At least one chunk should carry a section label
        assert any(s is not None for s in sections)

    def test_long_medical_document(self):
        text = (
            "ABSTRACT\n"
            "This study examines the treatment of hypertension. " * 5
            + "\n\n1. TREATMENT\n"
            "Amlodipine 5 mg once daily is recommended as first-line therapy. "
            "Dose may be increased to 10 mg if inadequate response. " * 10
            + "\n\n2. RESULTS\n"
            "Systolic BP was reduced by 15 mmHg on average. " * 10
        )
        chunks = chunk_text_v2(text)
        assert len(chunks) >= 2
