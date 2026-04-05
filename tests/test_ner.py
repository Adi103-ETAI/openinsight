"""
Tests for the NER module.
"""
import pytest
from src.ingestion.ner import (
    extract_entities,
    classify_content_type,
    infer_study_type,
)


class TestExtractEntities:
    def test_drug_extraction(self):
        text = "Patient was prescribed doxycycline 100 mg twice daily."
        entities = extract_entities(text)
        assert "doxycycline" in entities["drugs"]

    def test_disease_extraction(self):
        text = "The patient has tuberculosis and diabetes."
        entities = extract_entities(text)
        # Both conditions should be detected
        assert any("tuberculosis" in d for d in entities["diseases"])
        assert any("diabetes" in d for d in entities["diseases"])

    def test_dosage_extraction(self):
        text = "Administer 500 mg three times daily for 7 days."
        entities = extract_entities(text)
        # At least one dosage pattern should match
        assert len(entities["dosages"]) >= 1

    def test_safety_flag_warning(self):
        text = "WARNING: This drug has serious hepatotoxic effects and may cause liver failure."
        entities = extract_entities(text)
        assert entities["has_safety_flag"] is True

    def test_safety_flag_absent(self):
        text = "The patient was given paracetamol for pain relief."
        entities = extract_entities(text)
        # This text has no safety warning keywords
        assert entities["has_safety_flag"] is False

    def test_contraindication_extraction(self):
        text = "Ciprofloxacin is contraindicated in patients under 18 years of age."
        entities = extract_entities(text)
        assert len(entities["contraindications"]) >= 1

    def test_patient_population(self):
        text = "In pediatric patients the dose should be reduced."
        entities = extract_entities(text)
        assert any("pediatric" in p or "paediatric" in p for p in entities["patient_populations"])

    def test_outcomes(self):
        text = "Primary endpoint was 30-day mortality. Secondary endpoints included morbidity."
        entities = extract_entities(text)
        assert any("mortality" in o for o in entities["outcomes"])

    def test_empty_text(self):
        entities = extract_entities("")
        assert entities["diseases"] == []
        assert entities["drugs"] == []
        assert entities["dosages"] == []
        assert entities["has_safety_flag"] is False

    def test_deduplication(self):
        text = "Aspirin aspirin ASPIRIN is used in aspirin therapy."
        entities = extract_entities(text)
        drugs_lower = [d.lower() for d in entities["drugs"]]
        assert drugs_lower.count("aspirin") == 1

    def test_list_size_limit(self):
        # Construct a text with many drugs
        drugs = [
            "doxycycline", "azithromycin", "rifampicin", "isoniazid",
            "ciprofloxacin", "metronidazole", "amoxicillin", "aspirin",
            "metformin", "insulin", "warfarin", "heparin",
        ]
        text = " ".join(drugs)
        entities = extract_entities(text)
        assert len(entities["drugs"]) <= 10
        assert len(entities["diseases"]) <= 10


class TestClassifyContentType:
    def test_clinical_content(self):
        text = (
            "Patient should receive doxycycline 100 mg oral twice daily. "
            "Management includes IV fluids and monitoring of vital signs. "
            "Guideline recommends hospitalization."
        )
        content_type, weight = classify_content_type(text)
        assert content_type == "clinical"
        assert weight == 1.5

    def test_noise_references(self):
        text = "References\nAcknowledgements\nCopyright 2023 All rights reserved"
        content_type, weight = classify_content_type(text)
        assert content_type == "noise"
        assert weight == 0.1

    def test_preclinical_animal_study(self):
        # Must hit at least 2 preclinical pattern groups
        text = (
            "Mouse models were used. In vitro cell line assay showed the mechanism pathway. "
            "Rat studies confirmed the molecular target interaction."
        )
        content_type, weight = classify_content_type(text)
        assert content_type == "preclinical"

    def test_clinical_via_section(self):
        text = "Standard dosing applies."
        content_type, weight = classify_content_type(text, section="Treatment Protocol")
        assert content_type == "clinical"

    def test_noise_via_section(self):
        text = "Some text here."
        content_type, weight = classify_content_type(text, section="References")
        assert content_type == "noise"


class TestInferStudyType:
    def test_meta_analysis(self):
        study_type, level = infer_study_type("A meta-analysis of 50 RCTs on hypertension.")
        assert study_type == "meta_analysis"
        assert level == 1

    def test_rct(self):
        study_type, level = infer_study_type("A randomized controlled trial of 200 patients.")
        assert study_type == "rct"
        assert level == 1

    def test_guideline(self):
        study_type, level = infer_study_type("WHO guideline on antimicrobial resistance.")
        assert study_type == "guideline"
        assert level == 2

    def test_cdc_guideline(self):
        study_type, level = infer_study_type("CDC recommendation for disease prevention.")
        assert study_type == "guideline"
        assert level == 2

    def test_case_report(self):
        study_type, level = infer_study_type("We present a case report of a 45-year-old patient.")
        assert study_type == "case_report"
        assert level == 4

    def test_unknown(self):
        study_type, level = infer_study_type("Miscellaneous text without study indicators.")
        assert study_type == "unknown"
        assert level == 5
