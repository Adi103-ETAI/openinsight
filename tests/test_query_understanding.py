"""
Tests for the query understanding module.

Covers:
- Query intent classification
- Entity extraction (fallback mode)
- Metadata filter inference
- Query expansion with medical synonyms

Markers:
- unit: All tests in this module are unit tests
"""
from __future__ import annotations

import pytest

from src.query.search.query_understanding import (
    QueryAnalysis,
    QueryIntent,
    QueryUnderstanding,
)


@pytest.fixture
def query_understanding():
    """Create QueryUnderstanding instance (spacy may not be available)."""
    return QueryUnderstanding()


@pytest.mark.unit
class TestQueryIntentClassification:
    """Tests for query intent classification."""

    @pytest.mark.parametrize(
        "query, expected_intent",
        [
            pytest.param("what causes tuberculosis", QueryIntent.DIAGNOSTIC, id="causes"),
            pytest.param("cause of malaria", QueryIntent.DIAGNOSTIC, id="cause_of"),
            pytest.param("differential diagnosis", QueryIntent.DIAGNOSTIC, id="differential"),
            pytest.param("how to diagnose pneumonia", QueryIntent.DIAGNOSTIC, id="how_to_diagnose"),
            pytest.param("symptoms of dengue", QueryIntent.DIAGNOSTIC, id="symptoms"),
            pytest.param("what is diabetes", QueryIntent.DIAGNOSTIC, id="what_is"),
            pytest.param("treatment for hypertension", QueryIntent.THERAPEUTIC, id="treatment"),
            pytest.param("how to treat malaria", QueryIntent.THERAPEUTIC, id="treat"),
            pytest.param("therapy options", QueryIntent.THERAPEUTIC, id="therapy"),
            pytest.param("management of diabetes", QueryIntent.THERAPEUTIC, id="management"),
            pytest.param("drug of choice for TB", QueryIntent.THERAPEUTIC, id="drug_of_choice"),
            pytest.param("dose of amoxicillin", QueryIntent.THERAPEUTIC, id="dose"),
            pytest.param("prognosis of cancer", QueryIntent.PROGNOSTIC, id="prognosis"),
            pytest.param("survival rate", QueryIntent.PROGNOSTIC, id="survival"),
            pytest.param("mortality of MDR-TB", QueryIntent.PROGNOSTIC, id="mortality"),
            pytest.param("side effects of metformin", QueryIntent.DRUG_INFO, id="side_effects"),
            pytest.param("adverse effects of aspirin", QueryIntent.DRUG_INFO, id="adverse_effects"),
            pytest.param("drug interactions with warfarin", QueryIntent.DRUG_INFO, id="interactions"),
            pytest.param("contraindications of bedaquiline", QueryIntent.DRUG_INFO, id="contraindications"),
            pytest.param("WHO guideline for TB", QueryIntent.GUIDELINE, id="guideline"),
            pytest.param("ICMR recommendation", QueryIntent.GUIDELINE, id="icmr"),
            pytest.param("standard of care", QueryIntent.GUIDELINE, id="standard_of_care"),
            pytest.param("general medical question", QueryIntent.GENERAL, id="general"),
        ],
    )
    def test_intent_classification(
        self, query_understanding: QueryUnderstanding, query: str, expected_intent: QueryIntent,
    ):
        """Query intent should be classified correctly."""
        analysis = query_understanding.analyze(query)
        assert analysis.intent == expected_intent, (
            f"Query '{query}' expected intent {expected_intent}, got {analysis.intent}"
        )


@pytest.mark.unit
class TestEntityExtraction:
    """Tests for entity extraction from queries."""

    def test_fallback_disease_extraction(self, query_understanding: QueryUnderstanding):
        """Fallback should extract known disease names."""
        analysis = query_understanding.analyze("what is diabetes")
        assert "diabetes" in analysis.entities["disease"]

    def test_fallback_drug_extraction(self, query_understanding: QueryUnderstanding):
        """Fallback should extract known drug names."""
        analysis = query_understanding.analyze("metformin dosage")
        assert "metformin" in analysis.entities["drug"]

    def test_fallback_hypertension_extraction(self, query_understanding: QueryUnderstanding):
        """Fallback should extract hypertension."""
        analysis = query_understanding.analyze("hypertension treatment")
        assert "hypertension" in analysis.entities["disease"]

    def test_empty_entities_for_unknown_terms(self, query_understanding: QueryUnderstanding):
        """Unknown terms should not produce entities in fallback mode."""
        analysis = query_understanding.analyze("xyz123 abc456")
        assert not any(analysis.entities.values())


@pytest.mark.unit
class TestMetadataFilterInference:
    """Tests for metadata filter inference from queries."""

    def test_recent_query_adds_year_filter(self, query_understanding: QueryUnderstanding):
        """Query with 'recent' should add year >= 2020 filter."""
        analysis = query_understanding.analyze("recent treatment guidelines")
        assert analysis.metadata_filters is not None
        # Check that a year filter exists
        conditions = analysis.metadata_filters.must
        year_conditions = [c for c in conditions if c.field == "year"]
        assert len(year_conditions) > 0

    @pytest.mark.parametrize(
        "query_word",
        ["recent", "latest", "current", "2024", "2025"],
    )
    def test_recency_triggers_year_filter(
        self, query_understanding: QueryUnderstanding, query_word: str,
    ):
        """Various recency indicators should trigger year filter."""
        analysis = query_understanding.analyze(f"{query_word} diabetes treatment")
        assert analysis.metadata_filters is not None

    def test_guideline_query_adds_doc_type_filter(self, query_understanding: QueryUnderstanding):
        """Query with 'guideline' should add doc_type filter."""
        analysis = query_understanding.analyze("guideline for hypertension")
        assert analysis.metadata_filters is not None
        conditions = analysis.metadata_filters.must
        doc_type_conditions = [c for c in conditions if c.field == "doc_type"]
        assert len(doc_type_conditions) > 0

    def test_india_query_adds_india_filter(self, query_understanding: QueryUnderstanding):
        """Query mentioning India should add india_relevant filter."""
        analysis = query_understanding.analyze("diabetes treatment in India")
        assert analysis.metadata_filters is not None
        conditions = analysis.metadata_filters.must
        india_conditions = [c for c in conditions if c.field == "india_relevant"]
        assert len(india_conditions) > 0

    def test_no_filters_for_general_query(self, query_understanding: QueryUnderstanding):
        """General query should not add metadata filters."""
        analysis = query_understanding.analyze("what is tuberculosis")
        assert analysis.metadata_filters is None

    def test_dosage_query_adds_drug_dosing_filter(self, query_understanding: QueryUnderstanding):
        """Query with dosage intent should add has_drug_dosing filter."""
        analysis = query_understanding.analyze("dose of metformin for diabetes")
        assert analysis.metadata_filters is not None
        conditions = analysis.metadata_filters.must
        dosing_conditions = [c for c in conditions if c.field == "has_drug_dosing"]
        assert len(dosing_conditions) > 0


@pytest.mark.unit
class TestQueryExpansion:
    """Tests for query expansion with medical synonyms."""

    def test_heart_attack_expansion(self, query_understanding: QueryUnderstanding):
        """'heart attack' should expand to medical synonyms."""
        analysis = query_understanding.analyze("heart attack treatment")
        assert "myocardial infarction" in analysis.expanded_terms

    def test_diabetes_expansion(self, query_understanding: QueryUnderstanding):
        """'diabetes' should expand to medical synonyms."""
        analysis = query_understanding.analyze("diabetes management")
        assert "diabetes mellitus" in analysis.expanded_terms

    def test_tb_expansion(self, query_understanding: QueryUnderstanding):
        """'tb' should expand to tuberculosis."""
        analysis = query_understanding.analyze("tb treatment")
        assert "tuberculosis" in analysis.expanded_terms

    def test_no_expansion_for_unknown_terms(self, query_understanding: QueryUnderstanding):
        """Unknown terms should not be expanded."""
        analysis = query_understanding.analyze("xyz123 treatment")
        assert analysis.expanded_terms == []

    def test_no_duplicate_expanded_terms(self, query_understanding: QueryUnderstanding):
        """Expanded terms should not contain duplicates."""
        analysis = query_understanding.analyze("heart attack and myocardial infarction")
        assert len(analysis.expanded_terms) == len(set(analysis.expanded_terms))


@pytest.mark.unit
class TestQueryAnalysisResult:
    """Tests for QueryAnalysis dataclass."""

    def test_analysis_has_all_fields(self, query_understanding: QueryUnderstanding):
        """QueryAnalysis should have all expected fields."""
        analysis = query_understanding.analyze("test query")
        assert analysis.original_query == "test query"
        assert analysis.intent is not None
        assert analysis.entities is not None
        assert analysis.metadata_filters is None or hasattr(analysis.metadata_filters, "must")
        assert isinstance(analysis.use_hyde, bool)
        assert isinstance(analysis.expanded_terms, list)

    def test_original_query_preserved(self, query_understanding: QueryUnderstanding):
        """Original query should be preserved as-is."""
        analysis = query_understanding.analyze("  Test Query  ")
        assert analysis.original_query == "Test Query"
