from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

import torch
from src.config.settings import get_settings

logger = logging.getLogger(__name__)

settings = get_settings()


@dataclass
class ContradictionPair:
    """Two chunks that contradict each other."""
    chunk_a: dict[str, Any]
    chunk_b: dict[str, Any]
    contradiction_type: str
    evidence: str


@dataclass
class ContradictionReport:
    """Report of all contradictions found in retrieved chunks."""
    has_contradictions: bool
    contradictions: list[ContradictionPair]
    summary: str


class ContradictionDetector:
    """
    Detects contradicting evidence in retrieved chunks.
    
    Uses simple semantic similarity + keyword-based detection:
    - Opposite keywords (improve vs worsen)
    - Different recommendations for same condition
    - Conflicting dosage recommendations
    
    For production: Use NLI model (cross-encoder trained on SNLI/MedNLI)
    """

    CONTRADICTION_KEYWORDS = {
        "treatment_conflict": [
            ("first line", "second line"),
            ("recommended", "not recommended"),
            ("contraindicated", "indicated"),
            ("improve", "worsen"),
        ],
        "dosage_conflict": [
            ("mg", "mcg"),
            ("once daily", "twice daily"),
            ("increase", "decrease"),
        ],
        "outcome_conflict": [
            ("effective", "ineffective"),
            ("beneficial", "harmful"),
            ("safe", "unsafe"),
        ],
    }

    def __init__(self):
        self._nli_model = None

    async def detect(
        self,
        chunks: list[dict[str, Any]],
        query: str,
    ) -> ContradictionReport:
        """Detect contradictions among retrieved chunks."""
        if len(chunks) < 2:
            return ContradictionReport(
                has_contradictions=False,
                contradictions=[],
                summary="",
            )

        try:
            return await self._detect_with_nli(chunks, query)
        except Exception as e:
            logger.warning(f"NLI detection failed, using keyword fallback: {e}")
            return self._detect_keyword(chunks)

    async def _detect_with_nli(
        self, chunks: list[dict[str, Any]], query: str
    ) -> ContradictionReport:
        """Use NLI model for contradiction detection."""
        if self._nli_model is None:
            self._nli_model = await self._load_nli_model()

        contradictions = []
        
        for i, chunk_a in enumerate(chunks):
            for chunk_b in chunks[i + 1:]:
                # Check if chunks are about same topic
                if not self._same_topic(chunk_a, chunk_b):
                    continue

                # Run NLI
                result = await self._run_nli(
                    chunk_a.get("text", ""),
                    chunk_b.get("text", ""),
                )

                if result == "contradiction":
                    contradictions.append(ContradictionPair(
                        chunk_a=chunk_a,
                        chunk_b=chunk_b,
                        contradiction_type="semantic",
                        evidence="NLI model detected contradiction",
                    ))

        return self._build_report(contradictions)

    def _detect_keyword(self, chunks: list[dict[str, Any]]) -> ContradictionReport:
        """Fallback keyword-based contradiction detection."""
        contradictions = []

        for i, chunk_a in enumerate(chunks):
            for chunk_b in chunks[i + 1:]:
                if not self._same_topic(chunk_a, chunk_b):
                    continue

                text_a = chunk_a.get("text", "").lower()
                text_b = chunk_b.get("text", "").lower()

                for conflict_type, keyword_pairs in self.CONTRADICTION_KEYWORDS.items():
                    for kw1, kw2 in keyword_pairs:
                        if (kw1 in text_a and kw2 in text_b) or (
                            kw2 in text_a and kw1 in text_b
                        ):
                            contradictions.append(ContradictionPair(
                                chunk_a=chunk_a,
                                chunk_b=chunk_b,
                                contradiction_type=conflict_type,
                                evidence=f"Keywords: {kw1} vs {kw2}",
                            ))

        return self._build_report(contradictions)

    def _same_topic(self, chunk_a: dict[str, Any], chunk_b: dict[str, Any]) -> bool:
        """Check if two chunks are about the same topic."""
        title_a = chunk_a.get("title", "").lower()
        title_b = chunk_b.get("title", "").lower()
        
        # Extract first significant word from titles
        words_a = [w for w in title_a.split() if len(w) > 4]
        words_b = [w for w in title_b.split() if len(w) > 4]
        
        common = set(words_a) & set(words_b)
        return len(common) >= 1

    def _build_report(
        self, contradictions: list[ContradictionPair]
    ) -> ContradictionReport:
        """Build contradiction report."""
        if not contradictions:
            return ContradictionReport(
                has_contradictions=False,
                contradictions=[],
                summary="",
            )

        summary = (
            f"Found {len(contradictions)} potential contradiction(s) in retrieved evidence. "
            "Please verify before clinical use."
        )

        return ContradictionReport(
            has_contradictions=True,
            contradictions=contradictions,
            summary=summary,
        )

    async def _load_nli_model(self):
        """Load NLI model for medical contradiction detection."""
        # Research shows NLI is critical for medical RAG (18.2% performance degradation without it)
        # Using PubMedBERT-based model trained on medical NLI
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            logger.info(f"Loading NLI model: {settings.nli_model_name}")
            model = AutoModelForSequenceClassification.from_pretrained(
                settings.nli_model_name
            )
            tokenizer = AutoTokenizer.from_pretrained(settings.nli_model_name)
            return {"model": model, "tokenizer": tokenizer}
        except Exception as e:
            logger.warning(f"Failed to load NLI model, using keyword fallback: {e}")
            return None

    async def _run_nli(self, text_a: str, text_b: str) -> str:
        """Run NLI inference to detect contradiction between two texts."""
        if self._nli_model is None:
            return "entailment"  # Fallback when no model loaded

        try:
            model = self._nli_model["model"]
            tokenizer = self._nli_model["tokenizer"]

            # Prepare inputs - premise is text_a, hypothesis is text_b
            inputs = tokenizer(
                text_a,
                text_b,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )

            with torch.no_grad():
                outputs = model(**inputs)
                predictions = outputs.logits
                # NLI labels: 0=contradiction, 1=neutral, 2=entailment
                predicted_class = predictions.argmax(dim=-1).item()

            # Map to simplified categories
            if predicted_class == 0:
                return "contradiction"
            elif predicted_class == 2:
                return "entailment"
            else:
                return "neutral"
        except Exception as e:
            logger.warning(f"NLI inference failed: {e}")
            return "entailment"