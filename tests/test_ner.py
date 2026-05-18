"""
Tests for the NER (Named Entity Recognition) module.

Markers:
- unit: All tests in this module are unit tests
"""
from __future__ import annotations

import pytest

from src.ml.ner import (
    extract_entities,
    classify_content_type,
    infer_study_type,
)


@pytest.mark.unit
class TestExtractEntities:
    """Tests for entity extraction from medical text."""

    def test_drug_extraction(self):
        """Drug names should be extracted from text."""
        text = "Patient was prescribed doxycycline 100 mg twice daily."
        entities = extract_entities(text)
        assert "doxycycline" in entities["drugs"]

    def test_disease_extraction(self):
        """Disease names should be extracted from text."""
        text = "The patient has tuberculosis and diabetes."
        entities = extract_entities(text)
        assert any("tuberculosis" in d for d in entities["diseases"])
        assert any("diabetes" in d for d in entities["diseases"])

    def test_dosage_extraction(self):
        """Dosage patterns should be extracted."""
        text = "Administer 500 mg three times daily for 7 days."
        entities = extract_entities(text)
        assert len(entities["dosages"]) >= 1

    def test_safety_flag_warning(self):
        """Safety warning keywords should trigger safety flag."""
        text = "WARNING: This drug has serious hepatotoxic effects and may cause liver failure."
        entities = extract_entities(text)
        assert entities["has_safety_flag"] is True

    def test_safety_flag_absent(self):
        """Normal text should not trigger safety flag."""
        text = "The patient was given paracetamol for pain relief."
        entities = extract_entities(text)
        assert entities["has_safety_flag"] is False

    def test_contraindication_extraction(self):
        """Contraindication statements should be extracted."""
        text = "Ciprofloxacin is contraindicated in patients under 18 years of age."
        entities = extract_entities(text)
        assert len(entities["contraindications"]) >= 1

    def test_patient_population(self):
        """Patient population descriptors should be extracted."""
        text = "In pediatric patients the dose should be reduced."
        entities = extract_entities(text)
        assert any("pediatric" in p or "paediatric" in p for p in entities["patient_populations"])

    def test_outcomes(self):
        """Outcome terms should be extracted."""
        text = "Primary endpoint was 30-day mortality. Secondary endpoints included morbidity."
        entities = extract_entities(text)
        assert any("mortality" in o for o in entities["outcomes"])

    def test_empty_text(self):
        """Empty text should return empty entity lists."""
        entities = extract_entities("")
        assert entities["diseases"] == []
        assert entities["drugs"] == []
        assert entities["dosages"] == []
        assert entities["has_safety_flag"] is False

    def test_deduplication(self):
        """Duplicate entities should be deduplicated."""
        text = "Aspirin aspirin ASPIRIN is used in aspirin therapy."
        entities = extract_entities(text)
        drugs_lower = [d.lower() for d in entities["drugs"]]
        assert drugs_lower.count("aspirin") == 1

    def test_list_size_limit(self):
        """Entity lists should be capped at maximum size."""
        drugs = [
            "doxycycline", "azithromycin", "rifampicin", "isoniazid",
            "ciprofloxacin", "metronidazole", "amoxicillin", "aspirin",
            "metformin", "insulin", "warfarin", "heparin",
        ]
        text = " ".join(drugs)
        entities = extract_entities(text)
        assert len(entities["drugs"]) <= 10
        assert len(entities["diseases"]) <= 10

    @pytest.mark.parametrize(
        "text, expected_drug",
        [
            pytest.param("Patient takes metformin daily.", "metformin", id="metformin"),
            pytest.param("Warfarin was prescribed.", "warfarin", id="warfarin"),
            pytest.param("Insulin therapy was started.", "insulin", id="insulin"),
        ],
    )
    def test_specific_drug_extraction(self, text: str, expected_drug: str):
        """Specific drug names should be extracted."""
        entities = extract_entities(text)
        assert expected_drug in entities["drugs"]


@pytest.mark.unit
class TestClassifyContentType:
    """Tests for content type classification."""

    def test_clinical_content(self):
        """Clinical content should be classified as clinical with weight 1.5."""
        text = (
            "Patient should receive doxycycline 100 mg oral twice daily. "
            "Management includes IV fluids and monitoring of vital signs. "
            "Guideline recommends hospitalization."
        )
        content_type, weight = classify_content_type(text)
        assert content_type == "clinical"
        assert weight == 1.5

    def test_noise_references(self):
        """Reference sections should be classified as noise."""
        text = "References\nAcknowledgements\nCopyright 2023 All rights reserved"
        content_type, weight = classify_content_type(text)
        assert content_type == "noise"
        assert weight == 0.1

    def test_preclinical_animal_study(self):
        """Preclinical animal study text should be classified as preclinical."""
        text = (
            "Mouse models were used. In vitro cell line assay showed the mechanism pathway. "
            "Rat studies confirmed the molecular target interaction."
        )
        content_type, weight = classify_content_type(text)
        assert content_type == "preclinical"

    def test_clinical_via_section(self):
        """Clinical section header should classify as clinical."""
        text = "Standard dosing applies."
        content_type, weight = classify_content_type(text, section="Treatment Protocol")
        assert content_type == "clinical"

    def test_noise_via_section(self):
        """References section should classify as noise."""
        text = "Some text here."
        content_type, weight = classify_content_type(text, section="References")
        assert content_type == "noise"

    @pytest.mark.parametrize(
        "text, expected_type",
        [
            pytest.param("Patient should receive doxycycline 100 mg.", "clinical", id="clinical"),
            pytest.param("References\nCopyright 2023", "noise", id="noise"),
        ],
    )
    def test_content_type_classification(self, text: str, expected_type: str):
        """Content should be classified into correct type."""
        content_type, _ = classify_content_type(text)
        assert content_type == expected_type


@pytest.mark.unit
class TestInferStudyType:
    """Tests for study type inference."""

    @pytest.mark.parametrize(
        "text, expected_type, expected_level",
        [
            pytest.param(
                "A meta-analysis of 50 RCTs on hypertension.",
                "meta_analysis", 1,
                id="meta_analysis",
            ),
            pytest.param(
                "A randomized controlled trial of 200 patients.",
                "rct", 1,
                id="rct",
            ),
            pytest.param(
                "WHO guideline on antimicrobial resistance.",
                "guideline", 2,
                id="who_guideline",
            ),
            pytest.param(
                "CDC recommendation for disease prevention.",
                "guideline", 2,
                id="cdc_guideline",
            ),
            pytest.param(
                "We present a case report of a 45-year-old patient.",
                "case_report", 4,
                id="case_report",
            ),
            pytest.param(
                "Miscellaneous text without study indicators.",
                "unknown", 5,
                id="unknown",
            ),
        ],
    )
    def test_study_type_inference(self, text: str, expected_type: str, expected_level: int):
        """Study type and evidence level should be inferred correctly."""
        study_type, level = infer_study_type(text)
        assert study_type == expected_type
        assert level == expected_level

    def test_empty_text_study_type(self):
        """Empty text should return unknown study type."""
        study_type, level = infer_study_type("")
        assert study_type == "unknown"
        assert level == 5
