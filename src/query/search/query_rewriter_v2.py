from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Optional

import httpx

from src.core.config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class QueryRewriteResult:
    """Result of LLM-based query rewriting."""
    original_query: str
    rewritten_query: str
    intent: str
    search_terms: list[str]
    metadata_hints: dict[str, Any]


class LLMQueryRewriter:
    """
    Uses LLM to rewrite clinical queries for better retrieval.
    
    Replaces simple pattern matching with intelligent rewriting:
    - Expand abbreviations (HTN → hypertension)
    - Clarify ambiguous terms
    - Add clinical context
    - Generate multiple search variants
    """

    REWRITE_PROMPT = """You are a clinical query rewriting assistant for a medical RAG system.
Your job is to rewrite physician queries to improve retrieval from medical literature.

Rules:
1. Expand all medical abbreviations (HTN→hypertension, DM→diabetes mellitus, MI→myocardial infarction)
2. Use formal medical terminology
3. Add relevant clinical context (e.g., "in Indian population", "for adult patients")
4. Preserve the original intent
5. Include synonyms that might appear in literature

Input: "{query}"

Output a JSON object with:
{{
    "rewritten_query": "improved query with expanded terms",
    "intent": "diagnostic|therapeutic|prognostic|drug_info|guideline|general",
    "search_terms": ["term1", "term2", "term3"],
    "metadata_hints": {{"year_filter": 2020, "india_relevant": true}}
}}

Only output valid JSON, no additional text:"""

    def __init__(self):
        self.settings = get_settings()
        self._client: Optional[httpx.AsyncClient] = None

    async def rewrite(
        self,
        query: str,
        use_expansion: bool = True,
    ) -> QueryRewriteResult:
        """Rewrite query using LLM."""
        if not use_expansion or not self.settings.nvidia_nim_api_key:
            return self._fallback_rewrite(query)

        try:
            return await self._llm_rewrite(query)
        except Exception as e:
            logger.warning(f"LLM rewrite failed, using fallback: {e}")
            return self._fallback_rewrite(query)

    async def _llm_rewrite(self, query: str) -> QueryRewriteResult:
        """Call LLM to rewrite query."""
        if not self._client:
            self._client = httpx.AsyncClient(timeout=15.0)

        url = f"{self.settings.nvidia_nim_base_url.rstrip('/')}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.settings.nvidia_nim_api_key:
            headers["Authorization"] = f"Bearer {self.settings.nvidia_nim_api_key}"

        body = {
            "model": self.settings.nim_model,
            "messages": [
                {"role": "system", "content": "You are a medical query rewriting assistant. Output valid JSON only."},
                {"role": "user", "content": self.REWRITE_PROMPT.format(query=query)}
            ],
            "temperature": 0.1,
            "max_tokens": 300,
        }

        response = await self._client.post(url, headers=headers, json=body)
        response.raise_for_status()
        
        data = response.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "{}")
        
        import json
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            return self._fallback_rewrite(query)

        return QueryRewriteResult(
            original_query=query,
            rewritten_query=parsed.get("rewritten_query", query),
            intent=parsed.get("intent", "general"),
            search_terms=parsed.get("search_terms", []),
            metadata_hints=parsed.get("metadata_hints", {}),
        )

    def _fallback_rewrite(self, query: str) -> QueryRewriteResult:
        """Fallback to simple pattern-based rewriting."""
        query_lower = query.lower()
        
        # Simple intent detection
        if any(p in query_lower for p in ["treatment", "therapy", "drug", "dose", "medication"]):
            intent = "therapeutic"
        elif any(p in query_lower for p in ["diagnose", "symptom", "cause", "differential"]):
            intent = "diagnostic"
        elif any(p in query_lower for p in ["prognosis", "outcome", "survival", "mortality"]):
            intent = "prognostic"
        elif any(p in query_lower for p in ["side effect", "interaction", "contraindication"]):
            intent = "drug_info"
        elif any(p in query_lower for p in ["guideline", "recommendation", "protocol", "icmr"]):
            intent = "guideline"
        else:
            intent = "general"

        # Simple expansion
        expansions = {
            "htn": "hypertension",
            "dm": "diabetes mellitus",
            "mi": "myocardial infarction",
            "tb": "tuberculosis",
            "copd": "chronic obstructive pulmonary disease",
            "cad": "coronary artery disease",
            "cva": "cerebrovascular accident",
            "chf": "congestive heart failure",
            "afib": "atrial fibrillation",
            "dka": "diabetic ketoacidosis",
        }
        
        rewritten = query
        search_terms = []
        for abbr, full in expansions.items():
            if abbr in query_lower:
                search_terms.append(full)
                rewritten = rewritten.replace(abbr, full)
        
        # Add India context
        india_hints = {}
        if "india" in query_lower or "indian" in query_lower:
            india_hints["india_relevant"] = True

        return QueryRewriteResult(
            original_query=query,
            rewritten_query=rewritten,
            intent=intent,
            search_terms=search_terms,
            metadata_hints=india_hints,
        )

    async def close(self):
        if self._client:
            await self._client.aclose()