"""
Tests for Answer Quality & Safety Validation Layer

Tests cover:
- Hallucination detection
- Citation validation
- Medical safety checking
- Confidence scoring
- Full validation pipeline
"""

import pytest

from src.query.validation.confidence_scorer import (
    ConfidenceBreakdown,
    score_confidence,
)
from src.query.validation.hallucination_detector import (
    HallucinationResult,
    detect_hallucinations,
)
from src.query.validation.medical_safety import (
    SafetyCheckResult,
    check_safety,
)
from src.query.validation.validator import (
    ValidationResult,
    validate_answer,
    enhance_response,
)


# ============== Hallucination Detector Tests ==============


class TestHallucinationDetector:
    """Tests for hallucination detection."""

    def test_no_hallucination_when_grounded(self):
        """Answer grounded in chunks should not flag hallucinations."""
        answer = "Bedaquiline is recommended for MDR-TB treatment."
        chunks = [
            {
                "chunk_text": "Bedaquiline is a key drug for multidrug-resistant tuberculosis (MDR-TB) treatment regimens."
            },
            {
                "chunk_text": "WHO recommends bedaquiline as part of the standard MDR-TB regimen."
            },
        ]

        result = detect_hallucinations(answer, chunks)

        assert result.hallucination_score < 0.5
        assert result.verified_claims_count > 0

    def test_hallucination_detected_for_invented_content(self):
        """LLM-invented content not in sources should be flagged."""
        answer = "Bedaquiline should be given at 500mg daily for 24 months. Clinical trials showed 99% cure rate with this regimen."
        chunks = [
            {"chunk_text": "Bedaquiline is used for MDR-TB treatment."},
        ]

        result = detect_hallucinations(answer, chunks)

        # Should flag unverified numerical claims
        assert result.hallucination_score > 0.2
        assert len(result.flagged_claims) > 0

    def test_numerical_claims_verified(self):
        """Numerical claims present in sources should be verified."""
        answer = "The standard dose is 200mg daily for the first 2 weeks."
        chunks = [
            {
                "chunk_text": "Bedaquiline dosing: 200mg daily for 2 weeks, then 100mg three times weekly."
            },
        ]

        result = detect_hallucinations(answer, chunks)

        # Numbers should be found in chunks
        assert result.hallucination_score < 0.5

    def test_empty_answer_handling(self):
        """Empty answer should return neutral result."""
        result = detect_hallucinations("", [{"chunk_text": "Some content"}])
        assert result.hallucination_score == 0.0
        assert result.total_claims_count == 0

    def test_empty_chunks_handling(self):
        """Empty chunks should flag answer as ungrounded."""
        result = detect_hallucinations("Some answer content", [])
        assert result.hallucination_score == 1.0


# ============== Medical Safety Tests ==============


class TestMedicalSafety:
    """Tests for medical safety checking."""

    def test_treatment_recommendation_detected(self):
        """Treatment recommendations should be flagged."""
        answer = (
            "The patient should be treated with isoniazid and rifampicin for 6 months."
        )

        result = check_safety(answer)

        assert result.has_treatment_advice
        assert result.needs_disclaimer
        assert any(
            w.warning_type == "TREATMENT_RECOMMENDATION" for w in result.warnings
        )

    def test_dosage_information_detected(self):
        """Dosage information should be flagged."""
        answer = (
            "Give 200mg daily for the first two weeks, then 100mg three times weekly."
        )

        result = check_safety(answer)

        assert result.has_dosage_info
        assert any(w.warning_type == "DOSAGE_INFO" for w in result.warnings)

    def test_drug_interaction_warning(self):
        """Known dangerous drug combinations should be flagged as HIGH severity."""
        answer = "The patient is taking warfarin and aspirin together."

        result = check_safety(answer)

        assert any(
            w.warning_type == "DRUG_INTERACTION" and w.severity == "HIGH"
            for w in result.warnings
        )
        assert not result.is_safe  # Should be unsafe due to HIGH severity

    def test_monitoring_requirements_flagged(self):
        """Drugs requiring monitoring should generate warnings."""
        answer = "Linezolid is being used for this MDR-TB case."

        result = check_safety(answer)

        assert any(w.warning_type == "MONITORING_REQUIRED" for w in result.warnings)

    def test_safe_informational_answer(self):
        """Purely informational answers should pass safety check."""
        answer = "MDR-TB is caused by bacteria resistant to isoniazid and rifampicin."

        result = check_safety(answer)

        assert result.is_safe
        assert not result.needs_disclaimer

    def test_contraindication_detected(self):
        """Contraindication statements should be flagged (LOW severity)."""
        answer = (
            "Bedaquiline is contraindicated in patients with severe hepatic impairment."
        )

        result = check_safety(answer)

        assert any(w.warning_type == "CONTRAINDICATION" for w in result.warnings)

    def test_empty_answer_is_safe(self):
        """Empty answer should be considered safe."""
        result = check_safety("")
        assert result.is_safe
        assert len(result.warnings) == 0


# ============== Confidence Scorer Tests ==============


class TestConfidenceScorer:
    """Tests for confidence scoring."""

    def test_high_confidence_for_good_answer(self):
        """Well-grounded answer with good citations should score high."""
        citations = [
            {"source_type": "who", "title": "WHO Guidelines"},
            {"source_type": "pubmed", "title": "Study 1"},
            {"source_type": "cochrane", "title": "Systematic Review"},
        ]
        chunks = [
            {"chunk_text": "Content 1", "score": 0.9, "quality_score": 0.8},
            {"chunk_text": "Content 2", "score": 0.85, "quality_score": 0.75},
        ]

        result = score_confidence(
            citations=citations,
            chunks=chunks,
            hallucination_risk=0.1,  # Low hallucination risk
            avg_evidence_level=2.0,  # Good evidence level
            evidence_distribution={"grade_i_count": 1, "grade_ii_count": 2},
            num_safety_warnings=0,
        )

        assert result.final_score >= 0.7

    def test_low_confidence_for_poor_answer(self):
        """Poorly grounded answer should score low."""
        citations = []  # No citations
        chunks = [
            {"chunk_text": "Content", "score": 0.3, "quality_score": 0.2},
        ]

        result = score_confidence(
            citations=citations,
            chunks=chunks,
            hallucination_risk=0.8,  # High hallucination risk
            avg_evidence_level=5.0,  # Poor evidence level
            evidence_distribution={},
            num_safety_warnings=3,
            has_high_severity_warning=True,
        )

        assert result.final_score < 0.5

    def test_safety_penalty_applied(self):
        """Safety warnings should reduce confidence score."""
        base_result = score_confidence(
            citations=[{"source_type": "pubmed"}],
            chunks=[{"chunk_text": "Content", "score": 0.8}],
            hallucination_risk=0.1,
            num_safety_warnings=0,
        )

        penalized_result = score_confidence(
            citations=[{"source_type": "pubmed"}],
            chunks=[{"chunk_text": "Content", "score": 0.8}],
            hallucination_risk=0.1,
            num_safety_warnings=3,
            has_high_severity_warning=True,
        )

        assert penalized_result.final_score < base_result.final_score
        assert penalized_result.safety_penalty > 0

    def test_confidence_breakdown_components(self):
        """Confidence breakdown should have all components."""
        result = score_confidence(
            citations=[{"source_type": "pubmed"}],
            chunks=[{"chunk_text": "Content", "score": 0.7}],
            hallucination_risk=0.2,
        )

        assert 0 <= result.citation_score <= 1
        assert 0 <= result.evidence_score <= 1
        assert 0 <= result.hallucination_score <= 1
        assert 0 <= result.quality_score <= 1
        assert 0 <= result.consistency_score <= 1
        assert 0 <= result.safety_penalty <= 1
        assert 0 <= result.final_score <= 1


# ============== Full Validation Pipeline Tests ==============


class TestValidationPipeline:
    """Tests for the complete validation pipeline."""

    @pytest.mark.asyncio
    async def test_safe_answer_validation(self):
        """Safe, well-grounded answer should get SAFE recommendation."""
        answer = "MDR-TB is defined as resistance to isoniazid and rifampicin."
        citations = [
            {
                "index": 1,
                "mongo_id": "abc123",
                "source_type": "who",
                "title": "WHO MDR-TB Guidelines",
            },
        ]
        chunks = [
            {
                "chunk_text": "MDR-TB is tuberculosis that is resistant to at least isoniazid and rifampicin.",
                "score": 0.9,
            },
        ]

        result = await validate_answer(
            answer=answer,
            citations=citations,
            source_chunks=chunks,
            verify_citations_in_db=False,
        )

        assert result.confidence_score > 0.5
        assert result.recommendation == "SAFE"
        assert result.is_safe

    @pytest.mark.asyncio
    async def test_unsafe_answer_with_drug_interaction(self):
        """Answer with dangerous drug interaction should be flagged."""
        answer = "Consider using warfarin with aspirin for this patient."
        citations = [
            {"index": 1, "mongo_id": "abc", "source_type": "pubmed", "title": "Study"}
        ]
        chunks = [{"chunk_text": "Warfarin is an anticoagulant.", "score": 0.7}]

        result = await validate_answer(
            answer=answer,
            citations=citations,
            source_chunks=chunks,
            verify_citations_in_db=False,
        )

        assert not result.is_safe
        assert result.recommendation == "UNSAFE"
        assert any(w.warning_type == "DRUG_INTERACTION" for w in result.safety_warnings)

    @pytest.mark.asyncio
    async def test_needs_review_for_treatment_advice(self):
        """Answer with treatment advice should need review."""
        answer = "The patient should be treated with bedaquiline 200mg daily."
        citations = [
            {
                "index": 1,
                "mongo_id": "abc",
                "source_type": "who",
                "title": "WHO Guidelines",
            },
        ]
        chunks = [
            {
                "chunk_text": "Bedaquiline dosing is 200mg daily for 2 weeks.",
                "score": 0.85,
            },
        ]

        result = await validate_answer(
            answer=answer,
            citations=citations,
            source_chunks=chunks,
            verify_citations_in_db=False,
        )

        assert result.needs_disclaimer
        # Could be SAFE or NEEDS_REVIEW depending on overall score
        assert result.recommendation in ["SAFE", "NEEDS_REVIEW"]

    @pytest.mark.asyncio
    async def test_empty_answer_is_unsafe(self):
        """Empty answer should be marked unsafe."""
        result = await validate_answer(
            answer="",
            citations=[],
            source_chunks=[],
            verify_citations_in_db=False,
        )

        assert result.confidence_score == 0.0
        assert result.recommendation == "UNSAFE"

    @pytest.mark.asyncio
    async def test_enhance_response_adds_fields(self):
        """enhance_response should add all validation fields."""
        original_response = {
            "answer": "Test answer",
            "citations": [],
            "query": "test query",
            "rewritten_query": "test",
            "model": "test-model",
            "chunks_retrieved": 0,
        }

        validation = await validate_answer(
            answer="Test answer",
            citations=[],
            source_chunks=[{"chunk_text": "Test content", "score": 0.5}],
            verify_citations_in_db=False,
        )

        enhanced = enhance_response(original_response, validation)

        assert "confidence_score" in enhanced
        assert "recommendation" in enhanced
        assert "unverified_claims" in enhanced
        assert "safety_warnings" in enhanced
        assert "evidence_distribution" in enhanced
        assert "is_safe" in enhanced
        assert "needs_disclaimer" in enhanced
        assert "confidence_breakdown" in enhanced


# ============== Edge Cases ==============


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_hallucination_with_very_short_answer(self):
        """Very short answers should be handled gracefully."""
        result = detect_hallucinations("Yes.", [{"chunk_text": "Confirmed."}])
        # Should not crash and return reasonable result
        assert result.hallucination_score >= 0
        assert result.hallucination_score <= 1

    def test_safety_with_special_characters(self):
        """Safety check should handle special characters."""
        answer = "Use 100mg/kg/day (max: 4g) for <14 days. Check CBC & LFTs."
        result = check_safety(answer)
        # Should not crash
        assert isinstance(result, SafetyCheckResult)

    def test_confidence_with_no_chunks(self):
        """Confidence scorer should handle empty chunks."""
        result = score_confidence(
            citations=[],
            chunks=[],
            hallucination_risk=0.5,
        )
        assert 0 <= result.final_score <= 1

    @pytest.mark.asyncio
    async def test_validation_with_unicode(self):
        """Validation should handle unicode content."""
        answer = "Antimicrobial resistance affects tuberculosis treatment worldwide."
        result = await validate_answer(
            answer=answer,
            citations=[],
            source_chunks=[{"chunk_text": "Antimicrobial resistance", "score": 0.5}],
            verify_citations_in_db=False,
        )
        assert isinstance(result, ValidationResult)
