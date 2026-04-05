"""
Tests for the quality scoring module.
"""
import pytest
from src.ingestion.quality import score_chunk, score_chunks
from src.ingestion.document_db import ChunkRecord


def _make_chunk(**kwargs) -> ChunkRecord:
    defaults = dict(
        document_id="doc1",
        source_type="pubmed",
        title="Test Document",
        chunk_text="The patient should receive doxycycline 100 mg twice daily.",
        chunk_index=0,
        content_type="clinical",
        content_weight=1.5,
        evidence_level=1,
        token_count=50,
        diseases=["tuberculosis"],
        drugs=["doxycycline"],
        dosages=["100 mg"],
        symptoms=[],
        contraindications=[],
        patient_populations=[],
        outcomes=[],
        has_safety_flag=False,
        quality_score=0.0,
    )
    defaults.update(kwargs)
    return ChunkRecord(**defaults)


class TestScoreChunk:
    def test_clinical_high_evidence_scores_high(self):
        chunk = _make_chunk(
            content_type="clinical",
            evidence_level=1,
            token_count=150,
            diseases=["hypertension"],
            drugs=["amlodipine"],
            dosages=["5 mg"],
        )
        score = score_chunk(chunk)
        assert score >= 0.70

    def test_noise_scores_very_low(self):
        chunk = _make_chunk(
            content_type="noise",
            content_weight=0.1,
            evidence_level=5,
            token_count=20,
            diseases=[],
            drugs=[],
            dosages=[],
        )
        score = score_chunk(chunk)
        assert score <= 0.20

    def test_background_scores_medium(self):
        chunk = _make_chunk(
            content_type="background",
            content_weight=0.7,
            evidence_level=3,
            token_count=100,
            diseases=[],
            drugs=[],
            dosages=[],
        )
        score = score_chunk(chunk)
        assert 0.20 <= score <= 0.70

    def test_safety_flag_bumps_score(self):
        base = _make_chunk(has_safety_flag=False, content_type="clinical", evidence_level=1)
        flagged = _make_chunk(has_safety_flag=True, content_type="clinical", evidence_level=1)
        assert score_chunk(flagged) >= score_chunk(base)

    def test_score_bounded_zero_to_one(self):
        for content_type in ("clinical", "preclinical", "background", "noise", "unknown"):
            chunk = _make_chunk(content_type=content_type)
            s = score_chunk(chunk)
            assert 0.0 <= s <= 1.0, f"Score {s} out of bounds for content_type={content_type}"

    def test_short_chunk_lower_than_target_length(self):
        short = _make_chunk(token_count=10)
        target = _make_chunk(token_count=150)
        # Target-length chunk should score at least as high
        assert score_chunk(target) >= score_chunk(short) - 0.05


class TestScoreChunks:
    def test_scores_all_chunks(self):
        chunks = [_make_chunk(chunk_index=i) for i in range(5)]
        result = score_chunks(chunks)
        assert len(result) == 5
        for c in result:
            assert 0.0 <= c.quality_score <= 1.0

    def test_returns_same_list(self):
        chunks = [_make_chunk()]
        result = score_chunks(chunks)
        assert result is chunks  # mutated in-place
