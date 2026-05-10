from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from src.config.settings import get_settings
from src.vectorstore.filters import FilterCondition, FilterExpression, FilterOperator


class QueryIntent(str, Enum):
    DIAGNOSTIC = "diagnostic"
    THERAPEUTIC = "therapeutic"
    PROGNOSTIC = "prognostic"
    DRUG_INFO = "drug_info"
    GUIDELINE = "guideline"
    GENERAL = "general"


@dataclass
class QueryAnalysis:
    original_query: str
    intent: QueryIntent
    entities: dict[str, list[str]]
    rewritten_query: str | None
    metadata_filters: FilterExpression | None
    use_hyde: bool
    expanded_terms: list[str]


class QueryUnderstanding:
    DIAGNOSTIC_PATTERNS = [
        "what causes",
        "cause of",
        "differential",
        "how to diagnose",
        "symptoms of",
        "signs of",
        "what is",
        "aetiology",
    ]

    THERAPEUTIC_PATTERNS = [
        "treatment",
        "treat",
        "therapy",
        "management",
        "drug of choice",
        "dose",
        "dosage",
        "first line",
        "medication for",
    ]

    PROGNOSTIC_PATTERNS = [
        "prognosis",
        "outcome",
        "survival",
        "mortality",
        "risk of",
    ]

    DRUG_INFO_PATTERNS = [
        "side effects",
        "adverse effects",
        "interactions",
        "contraindications",
        "mechanism of",
    ]

    GUIDELINE_PATTERNS = [
        "guideline",
        "recommendation",
        "protocol",
        "standard of care",
        "icmr",
    ]

    MEDICAL_SYNONYMS = {
        "heart attack": ["myocardial infarction", "mi", "acute coronary syndrome"],
        "diabetes": ["diabetes mellitus", "dm", "type 2 diabetes", "t2dm"],
        "high blood pressure": ["hypertension", "htn"],
        "stroke": ["cerebrovascular accident", "cva"],
        "tb": ["tuberculosis", "mycobacterium tuberculosis"],
        "dengue": ["dengue fever", "dengue hemorrhagic fever"],
    }

    def __init__(self) -> None:
        self.settings = get_settings()
        self.nlp = None
        try:
            import spacy

            self.nlp = spacy.load(self.settings.spacy_model)
        except (ImportError, OSError, RuntimeError, ValueError, TypeError):
            self.nlp = None

    def analyze(self, query: str) -> QueryAnalysis:
        query_clean = query.strip()
        query_lower = query_clean.lower()

        intent = self._classify_intent(query_lower)
        entities = self._extract_entities(query_clean)
        metadata_filters = self._infer_metadata_filters(query_lower, intent)
        expanded_terms = self._expand_query(query_lower)
        use_hyde = self.settings.hyde_enabled and intent in {
            QueryIntent.DIAGNOSTIC,
            QueryIntent.PROGNOSTIC,
        }

        return QueryAnalysis(
            original_query=query_clean,
            intent=intent,
            entities=entities,
            rewritten_query=None,
            metadata_filters=metadata_filters,
            use_hyde=use_hyde,
            expanded_terms=expanded_terms,
        )

    def _classify_intent(self, query_lower: str) -> QueryIntent:
        if any(p in query_lower for p in self.DIAGNOSTIC_PATTERNS):
            return QueryIntent.DIAGNOSTIC
        if any(p in query_lower for p in self.THERAPEUTIC_PATTERNS):
            return QueryIntent.THERAPEUTIC
        if any(p in query_lower for p in self.PROGNOSTIC_PATTERNS):
            return QueryIntent.PROGNOSTIC
        if any(p in query_lower for p in self.DRUG_INFO_PATTERNS):
            return QueryIntent.DRUG_INFO
        if any(p in query_lower for p in self.GUIDELINE_PATTERNS):
            return QueryIntent.GUIDELINE
        return QueryIntent.GENERAL

    def _extract_entities(self, query: str) -> dict[str, list[str]]:
        entities = {
            "disease": [],
            "drug": [],
            "symptom": [],
            "procedure": [],
            "lab_value": [],
        }

        if self.nlp is not None:
            try:
                doc = self.nlp(query)
                for ent in doc.ents:
                    label = ent.label_.lower()
                    text = ent.text.lower().strip()
                    if not text:
                        continue
                    if label in {"disease", "disorder"}:
                        entities["disease"].append(text)
                    elif label in {"chemical", "drug", "simple_chemical"}:
                        entities["drug"].append(text)
                    elif label == "sign_symptom":
                        entities["symptom"].append(text)
                    elif label in {"medical_procedure", "diagnostic_procedure"}:
                        entities["procedure"].append(text)
                    elif label == "lab_value":
                        entities["lab_value"].append(text)
            except (RuntimeError, ValueError, TypeError):
                pass

        # Lightweight fallback for obvious tokens
        if not any(entities.values()):
            q = query.lower()
            if "diabetes" in q:
                entities["disease"].append("diabetes")
            if "hypertension" in q:
                entities["disease"].append("hypertension")
            if "metformin" in q:
                entities["drug"].append("metformin")

        return entities

    def _infer_metadata_filters(
        self, query_lower: str, intent: QueryIntent
    ) -> FilterExpression | None:
        conditions: list[FilterCondition] = []

        if any(
            word in query_lower
            for word in ["recent", "latest", "current", "2024", "2025"]
        ):
            conditions.append(
                FilterCondition(
                    field="year",
                    operator=FilterOperator.GTE,
                    value=2020,
                )
            )

        if any(
            word in query_lower for word in ["guideline", "recommendation", "protocol"]
        ):
            conditions.append(
                FilterCondition(
                    field="doc_type",
                    operator=FilterOperator.IN,
                    value=["guideline", "systematic_review", "meta_analysis"],
                )
            )

        if any(word in query_lower for word in ["india", "indian", "indians"]):
            conditions.append(
                FilterCondition(
                    field="india_relevant",
                    operator=FilterOperator.EQ,
                    value=True,
                )
            )

        if intent == QueryIntent.THERAPEUTIC and any(
            word in query_lower for word in ["dose", "dosage"]
        ):
            conditions.append(
                FilterCondition(
                    field="has_drug_dosing",
                    operator=FilterOperator.EQ,
                    value=True,
                )
            )

        if not conditions:
            return None
        return FilterExpression.from_conditions(conditions)

    def _expand_query(self, query_lower: str) -> list[str]:
        expanded_terms: list[str] = []
        for term, synonyms in self.MEDICAL_SYNONYMS.items():
            if term in query_lower:
                expanded_terms.extend(synonyms)

        # Remove duplicates but preserve order
        deduped: list[str] = []
        seen = set()
        for term in expanded_terms:
            if term not in seen:
                deduped.append(term)
                seen.add(term)
        return deduped
