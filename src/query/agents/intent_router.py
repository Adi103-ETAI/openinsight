from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

from src.config.settings import get_settings


class QueryComplexity(str, Enum):
    """Classification of query complexity."""
    SIMPLE = "simple"  # Standard RAG sufficient
    COMPLEX = "complex"  # DeepInsights needed
    MEDIUM = "medium"  # Could go either way


@dataclass
class RoutingDecision:
    """Result of intent routing."""
    complexity: QueryComplexity
    reason: str
    confidence: float
    detected_intent: str
    entities: dict[str, list[str]]
    sub_query_types: list[str]  # What aspects to search for


class IntentRouter:
    """
    Routes queries to either Standard RAG or DeepInsights.
    
    Detection method:
    1. Hardcoded pattern matching (primary)
    2. Entity count heuristics (secondary)
    3. Query structure analysis (tertiary)
    
    This is deterministic and fast - no LLM needed.
    """

    COMPLEX_PATTERNS = {
        # Comparison queries
        r"\bvs\b|\bversus\b|\bcompared to\b|\bversus\b",
        
        # Drug interactions
        r"\binteract(?:ion|ing|s)?\b",
        r"\bwith\b.*\b(medication|drug|pill)\b",
        
        # Multi-condition (requires AND)
        r"\b(and|with)\b.*\b(diabetes|hypertension|ckd|copd|chf)\b.*\b(and|with)\b",
        
        # Contraindications
        r"\bcontraindicat(?:ed|ion|ions)\b",
        
        # Differential diagnosis
        r"\bdifferential\b",
        r"\bwhat could cause\b",
        
        # Protocol/guideline conflicts
        r"\bprotocol\b.*\bversus\b",
        r"\bguideline\b.*\bconflic",
        
        # Monitoring/follow-up with complexity
        r"\bmonitor\b.*\band\b",
        r"\bmanage\b.*\bcomorbidi",
        
        # Side effects + interactions
        r"\bside effect\b.*\binteract",
        r"\badverse\b.*\bcombination\b",
    }

    # Sub-query types to generate based on detected needs
    SUB_QUERY_TEMPLATES = {
        "diagnostic": ["diagnosis", "symptoms", "differential", "etiology"],
        "therapeutic": ["treatment", "drug_of_choice", "dosage", "regimen"],
        "drug_info": ["interactions", "contraindications", "side_effects", "mechanism"],
        "prognostic": ["prognosis", "outcome", "mortality", "survival"],
        "guideline": ["guidelines", "recommendations", "protocols", "standards"],
        "comparative": ["comparison", "versus", "which_is_better", "efficacy"],
    }

    def __init__(self):
        self.settings = get_settings()
        self._compile_patterns()
        self._nlp = None
        self._init_nlp()

    def _init_nlp(self):
        """Initialize spacy model for NLP-based entity extraction."""
        if self.settings.spacy_model:
            try:
                import spacy
                self._nlp = spacy.load(self.settings.spacy_model)
            except (ImportError, OSError, RuntimeError, ValueError, TypeError) as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"spacy model not available: {e}. Using fallback.")

    def _compile_patterns(self):
        """Pre-compile regex patterns for speed."""
        self._complex_regex = [
            re.compile(p, re.IGNORECASE) for p in self.COMPLEX_PATTERNS
        ]

    def route(self, query: str) -> RoutingDecision:
        """Main routing method - returns complexity classification."""
        query_clean = query.strip()
        query_lower = query_clean.lower()

        # Step 1: Pattern matching
        pattern_matches = self._check_complex_patterns(query_lower)
        
        # Step 2: Entity count (use NLP when available)
        nlp_entities = self._extract_nlp_entities(query_clean)
        entity_count = self._count_medical_entities(query_lower) + len(nlp_entities.get("disease", [])) + len(nlp_entities.get("drug", []))
        
        # Step 3: Query structure
        structure_score = self._analyze_structure(query_lower)

        # Combine signals
        complexity, reason, confidence = self._compute_complexity(
            pattern_matches, entity_count, structure_score, query_lower
        )

        # Detect primary intent
        intent = self._detect_intent(query_lower)
        
        # Determine sub-query types needed
        sub_query_types = self._determine_sub_queries(intent, pattern_matches, complexity)

        # Combine simple and NLP entities
        simple_entities = self._extract_entities_simple(query_lower)
        combined_entities = {
            "diseases": list(set(simple_entities.get("diseases", []) + nlp_entities.get("disease", []))),
            "drugs": list(set(simple_entities.get("drugs", []) + nlp_entities.get("drug", []))),
        }

        return RoutingDecision(
            complexity=complexity,
            reason=reason,
            confidence=confidence,
            detected_intent=intent,
            entities=combined_entities,
            sub_query_types=sub_query_types,
        )

    def _extract_nlp_entities(self, query: str) -> dict[str, list[str]]:
        """Extract entities using spacy NLP when available."""
        if self._nlp is None:
            return {}
        
        try:
            doc = self._nlp(query)
            entities = {"disease": [], "drug": [], "procedure": []}
            for ent in doc.ents:
                label = ent.label_.lower()
                text = ent.text.lower().strip()
                if text and len(text) > 2:
                    if label in {"disease", "disorder", "condition"}:
                        entities["disease"].append(text)
                    elif label in {"chemical", "drug", "simple_chemical"}:
                        entities["drug"].append(text)
            return entities
        except Exception:
            return {}

    def _check_complex_patterns(self, query_lower: str) -> list[str]:
        """Check which complex patterns match."""
        matched = []
        for pattern in self._complex_regex:
            if pattern.search(query_lower):
                matched.append(pattern.pattern)
        return matched

    def _count_medical_entities(self, query_lower: str) -> int:
        """Count medical entities to estimate complexity."""
        # Simple keyword-based entity detection
        medical_terms = [
            "diabetes", "hypertension", "ckd", "copd", "chf", "cad",
            "mi", "stroke", "asthma", "arthritis", "cancer", "tb",
            "dengue", "malaria", "fever", "pain", "infection",
            "metformin", "insulin", "atorvastatin", "amlodipine",
            "paracetamol", "ibuprofen", "aspirin", "warfarin",
        ]
        
        count = 0
        for term in medical_terms:
            if term in query_lower:
                count += 1
        
        # Also count "and" occurrences as proxy for multiple conditions
        and_count = query_lower.count(" and ")
        
        return count + and_count

    def _analyze_structure(self, query_lower: str) -> float:
        """Analyze query structure for complexity signals."""
        score = 0.0
        
        # Long queries tend to be more complex
        words = len(query_lower.split())
        if words > 20:
            score += 0.3
        if words > 40:
            score += 0.2
            
        # Multiple questions in one
        question_marks = query_lower.count("?")
        if question_marks > 1:
            score += 0.4
            
        # Contains "or" (alternatives)
        if " or " in query_lower:
            score += 0.2
            
        # Contains "should I" / "can I" (needs nuanced answer)
        if any(p in query_lower for p in ["should i", "can i", "is it safe"]):
            score += 0.3
            
        return min(score, 1.0)

    def _compute_complexity(
        self,
        pattern_matches: list[str],
        entity_count: int,
        structure_score: float,
        query_lower: str,
    ) -> tuple[QueryComplexity, str, float]:
        """Compute final complexity classification."""
        
        # Strong complex signals
        if len(pattern_matches) >= 2:
            return (
                QueryComplexity.COMPLEX,
                f"Multiple complex patterns: {len(pattern_matches)}",
                0.95,
            )
        
        if len(pattern_matches) == 1:
            if entity_count >= 3 or structure_score > 0.5:
                return (
                    QueryComplexity.COMPLEX,
                    f"Complex pattern + multiple entities ({entity_count})",
                    0.90,
                )
            if entity_count >= 2:
                return (
                    QueryComplexity.MEDIUM,
                    f"Complex pattern + entities ({entity_count})",
                    0.70,
                )
            return (
                QueryComplexity.MEDIUM,
                f"Complex pattern detected: {pattern_matches[0][:30]}",
                0.65,
            )
        
        # No patterns - use entity count and structure
        if entity_count >= 4:
            return (
                QueryComplexity.COMPLEX,
                f"Multiple conditions ({entity_count} entities)",
                0.85,
            )
        
        if entity_count >= 3:
            if structure_score > 0.4:
                return (
                    QueryComplexity.COMPLEX,
                    f"Multiple conditions + complex structure",
                    0.80,
                )
            return (
                QueryComplexity.MEDIUM,
                f"Multiple conditions ({entity_count} entities)",
                0.65,
            )
        
        if entity_count >= 2:
            if structure_score > 0.5:
                return (
                    QueryComplexity.MEDIUM,
                    f"Multiple entities + complex structure",
                    0.60,
                )
            return (
                QueryComplexity.SIMPLE,
                f"Standard query with {entity_count} entities",
                0.80,
            )
        
        # Default - simple
        return (
            QueryComplexity.SIMPLE,
            "No complex indicators detected",
            0.75,
        )

    def _detect_intent(self, query_lower: str) -> str:
        """Detect primary clinical intent."""
        intents = {
            "diagnostic": ["diagnos", "symptom", "cause", "differential", "etiology"],
            "therapeutic": ["treat", "therapy", "management", "drug", "medication", "dose"],
            "drug_info": ["interact", "contraind", "side effect", "adverse", "mechanism"],
            "prognostic": ["prognos", "outcome", "survival", "mortality", "risk"],
            "guideline": ["guideline", "recommend", "protocol", "standard"],
            "comparative": ["versus", "vs", "compare", "better", "difference"],
        }
        
        for intent, keywords in intents.items():
            if any(kw in query_lower for kw in keywords):
                return intent
                
        return "general"

    def _determine_sub_queries(
        self,
        intent: str,
        pattern_matches: list[str],
        complexity: QueryComplexity,
    ) -> list[str]:
        """Determine what aspects need to be searched."""
        sub_queries = []
        
        # Base sub-queries based on intent
        if intent in self.SUB_QUERY_TEMPLATES:
            sub_queries.extend(self.SUB_QUERY_TEMPLATES[intent])
        
        # Add extra based on complexity
        if complexity == QueryComplexity.COMPLEX:
            # Always include safety and interactions for complex queries
            if "safety" not in sub_queries:
                sub_queries.append("safety")
            if "interactions" not in sub_queries:
                sub_queries.append("interactions")
            if "contraindications" not in sub_queries:
                sub_queries.append("contraindications")
        
        # Add comparative if patterns detected
        if pattern_matches and any("versus" in p or "compare" in p for p in pattern_matches):
            if "comparison" not in sub_queries:
                sub_queries.append("comparison")
                
        return sub_queries[:6]  # Limit to 6 sub-queries

    def _extract_entities_simple(self, query_lower: str) -> dict[str, list[str]]:
        """Simple entity extraction using keywords."""
        entities = {
            "diseases": [],
            "drugs": [],
            "procedures": [],
        }
        
        # Common diseases
        diseases = [
            "diabetes", "hypertension", "asthma", "copd", "ckd",
            "chf", "cad", "mi", "stroke", "tb", "dengue",
            "cancer", "arthritis", "thyroid", "anemia",
        ]
        for d in diseases:
            if d in query_lower:
                entities["diseases"].append(d)
                
        # Common drugs
        drugs = [
            "metformin", "insulin", "glipizide", "atorvastatin",
            "amlodipine", "losartan", "paracetamol", "ibuprofen",
            "aspirin", "warfarin", "metoprolol", "omeprazole",
        ]
        for d in drugs:
            if d in query_lower:
                entities["drugs"].append(d)
                
        return entities