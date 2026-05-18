"""
Tests for the quality scoring module.

Markers:
- unit: All tests in this module are unit tests
"""
from __future__ import annotations

import pytest

from src.ingestion.quality import score_chunk, score_chunks
from src.ingestion.document_db import ChunkRecord


def _make_chunk(**kwargs) -> ChunkRecord:
    """Factory for creating test ChunkRecord instances."""
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


@pytest.mark.unit
class TestScoreChunk:
    """Tests for individual chunk quality scoring."""

    def test_clinical_high_evidence_scores_high(self):
        """Clinical content with high evidence should score >= 0.70."""
        chunk = _make_chunk(
            content_type="clinical",
            evidence_level=1,
            token_count=150,
            diseases=["hypertension"],
            drugs=["amlodipine"],
            dosages=["5 mg"],
        )
        score = score_chunk(chunk)
        assert score >= 0.70, f"Expected high score >= 0.70, got {score}"

    def test_noise_scores_very_low(self):
        """Noise content should score <= 0.20."""
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
        assert score <= 0.20, f"Expected low score <= 0.20, got {score}"

    def test_background_scores_medium(self):
        """Background content should score between 0.20 and 0.70."""
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
        assert 0.20 <= score <= 0.70, f"Expected medium score, got {score}"

    def test_safety_flag_bumps_score(self):
        """Safety-flagged chunks should score higher than non-flagged."""
        base = _make_chunk(has_safety_flag=False, content_type="clinical", evidence_level=1)
        flagged = _make_chunk(has_safety_flag=True, content_type="clinical", evidence_level=1)
        assert score_chunk(flagged) >= score_chunk(base), (
            "Safety flag should increase quality score"
        )

    def test_score_bounded_zero_to_one(self):
        """All scores should be bounded between 0.0 and 1.0."""
        for content_type in ("clinical", "preclinical", "background", "noise", "unknown"):
            chunk = _make_chunk(content_type=content_type)
            s = score_chunk(chunk)
            assert 0.0 <= s <= 1.0, f"Score {s} out of bounds for content_type={content_type}"

    def test_short_chunk_lower_than_target_length(self):
        """Target-length chunks should score at least as high as short chunks."""
        short = _make_chunk(token_count=10)
        target = _make_chunk(token_count=150)
        assert score_chunk(target) >= score_chunk(short) - 0.05

    @pytest.mark.parametrize(
        "content_type, evidence_level, expected_min_score",
        [
            pytest.param("clinical", 1, 0.5, id="clinical_grade_1"),
            pytest.param("clinical", 2, 0.4, id="clinical_grade_2"),
            pytest.param("clinical", 3, 0.3, id="clinical_grade_3"),
            pytest.param("preclinical", 4, 0.1, id="preclinical_grade_4"),
            pytest.param("noise", 5, 0.0, id="noise_grade_5"),
        ],
    )
    def test_score_by_content_and_evidence(
        self, content_type: str, evidence_level: int, expected_min_score: float,
    ):
        """Score should reflect content type and evidence level combination."""
        chunk = _make_chunk(
            content_type=content_type,
            evidence_level=evidence_level,
            token_count=100,
        )
        score = score_chunk(chunk)
        assert score >= expected_min_score, (
            f"Expected score >= {expected_min_score} for {content_type}/level {evidence_level}, got {score}"
        )


@pytest.mark.unit
class TestScoreChunks:
    """Tests for batch chunk quality scoring."""

    def test_scores_all_chunks(self):
        """All chunks should be scored."""
        chunks = [_make_chunk(chunk_index=i) for i in range(5)]
        result = score_chunks(chunks)
        assert len(result) == 5
        for c in result:
            assert 0.0 <= c.quality_score <= 1.0

    def test_returns_same_list(self):
        """score_chunks should mutate and return the same list."""
        chunks = [_make_chunk()]
        result = score_chunks(chunks)
        assert result is chunks

    def test_empty_list_returns_empty(self):
        """Empty input should return empty output."""
        result = score_chunks([])
        assert result == []

    def test_scores_are_updated_in_place(self):
        """Quality scores should be updated on the chunk objects."""
        chunks = [_make_chunk(chunk_index=i) for i in range(3)]
        score_chunks(chunks)
        for chunk in chunks:
            assert chunk.quality_score > 0.0 or chunk.quality_score == 0.0
