from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Optional

import httpx

from src.config.settings import get_settings

logger = logging.getLogger(__name__)


@dataclass
class SubQuery:
    """A decomposed sub-query for parallel retrieval."""
    id: str
    query: str
    focus: str  # What aspect: "drug_interactions", "dosage", etc.
    priority: int  # 1 = highest
    metadata: dict[str, Any]


@dataclass
class DecompositionResult:
    """Result of query decomposition."""
    original_query: str
    sub_queries: list[SubQuery]
    synthesis_prompt: str
    estimated_complexity: str


class QueryDecomposer:
    """
    Decomposes complex queries into parallel sub-queries.
    
    Uses LLM to intelligently break down queries based on:
    - Clinical aspects needed (drug, dose, interactions, etc.)
    - Dependencies between sub-queries
    - Priority ordering
    """

    DECOMPOSE_PROMPT = """You are a clinical query decomposition expert.
Your job is to break complex medical queries into parallel sub-queries that can be 
answered independently then synthesized.

Complex query: "{query}"

Detected intent: {intent}
Entities: {entities}

Generate 3-6 sub-queries, each focused on a specific aspect:
- Each sub-query should be answerable independently
- Cover all clinical aspects: diagnosis, treatment, drug info, safety, guidelines
- Use medical terminology, expand abbreviations

Output JSON:
{{
    "sub_queries": [
        {{
            "id": "q1",
            "query": "specific sub-question",
            "focus": "treatment|dosage|interactions|contraindications|guidelines|safety|diagnosis",
            "priority": 1-3
        }}
    ],
    "synthesis_prompt": "How to combine these answers into a coherent response"
}}

Only output valid JSON:"""

    def __init__(self):
        self.settings = get_settings()
        self._client: Optional[httpx.AsyncClient] = None

    async def decompose(
        self,
        query: str,
        intent: str = "general",
        entities: dict[str, list[str]] | None = None,
    ) -> DecompositionResult:
        """Decompose query into sub-queries."""
        if entities is None:
            entities = {}

        # First try LLM-based decomposition
        try:
            result = await self._llm_decompose(query, intent, entities)
            if result:
                return result
        except Exception as e:
            logger.warning(f"LLM decomposition failed: {e}")

        # Fallback to rule-based decomposition
        return self._rule_based_decompose(query, intent, entities)

    async def _llm_decompose(
        self,
        query: str,
        intent: str,
        entities: dict[str, list[str]],
    ) -> Optional[DecompositionResult]:
        """Use LLM for intelligent decomposition."""
        if not self._client:
            self._client = httpx.AsyncClient(timeout=20.0)

        url = f"{self.settings.nvidia_nim_base_url.rstrip('/')}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.settings.nvidia_nim_api_key:
            headers["Authorization"] = f"Bearer {self.settings.nvidia_nim_api_key}"

        body = {
            "model": self.settings.nim_model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a medical query decomposition expert. Output valid JSON only.",
                },
                {
                    "role": "user",
                    "content": self.DECOMPOSE_PROMPT.format(
                        query=query,
                        intent=intent,
                        entities=entities,
                    ),
                },
            ],
            "temperature": 0.2,
            "max_tokens": 500,
        }

        try:
            response = await self._client.post(url, headers=headers, json=body)
            response.raise_for_status()
        except Exception as e:
            logger.warning(f"LLM decompose HTTP request failed: {e}")
            return None

        try:
            data = response.json()
        except Exception as e:
            logger.warning(f"Failed to parse LLM response as JSON: {e}")
            return None

        content = data.get("choices", [{}])[0].get("message", {}).get("content", "{}")

        import json

        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse decompose response content as JSON: {content[:200]}")
            return None

        sub_queries = []
        for sq in parsed.get("sub_queries", []):
            sub_queries.append(
                SubQuery(
                    id=sq.get("id", "q1"),
                    query=sq.get("query", ""),
                    focus=sq.get("focus", "general"),
                    priority=sq.get("priority", 2),
                    metadata={},
                )
            )

        return DecompositionResult(
            original_query=query,
            sub_queries=sub_queries,
            synthesis_prompt=parsed.get("synthesis_prompt", ""),
            estimated_complexity="high" if len(sub_queries) > 4 else "medium",
        )

    def _rule_based_decompose(
        self,
        query: str,
        intent: str,
        entities: dict[str, list[str]],
    ) -> DecompositionResult:
        """Rule-based fallback decomposition."""
        sub_queries = []
        
        query_lower = query.lower()
        
        # Base queries based on intent
        if intent == "therapeutic" or "treatment" in query_lower:
            sub_queries.append(SubQuery(
                id="q1",
                query=f"treatment options for {query}",
                focus="treatment",
                priority=1,
                metadata={},
            ))
            sub_queries.append(SubQuery(
                id="q2",
                query=f"recommended drug dosages for {query}",
                focus="dosage",
                priority=2,
                metadata={},
            ))
            
        if "interact" in query_lower or "with" in query_lower:
            sub_queries.append(SubQuery(
                id="q3",
                query=f"drug interactions for {query}",
                focus="interactions",
                priority=1,
                metadata={},
            ))
            
        if "contraind" in query_lower or "safe" in query_lower:
            sub_queries.append(SubQuery(
                id="q4",
                query=f"contraindications for {query}",
                focus="contraindications",
                priority=1,
                metadata={},
            ))
            
        # Add guideline search
        sub_queries.append(SubQuery(
            id="q5",
            query=f"ICMR guidelines for {query}",
            focus="guidelines",
            priority=2,
            metadata={"source_filter": "icmr"},
        ))
        
        # Add research search
        sub_queries.append(SubQuery(
            id="q6",
            query=f"latest research on {query}",
            focus="research",
            priority=3,
            metadata={"source_filter": "pubmed"},
        ))

        # If no sub-queries generated, create a general one
        if not sub_queries:
            sub_queries.append(SubQuery(
                id="q1",
                query=query,
                focus=intent,
                priority=1,
                metadata={},
            ))

        return DecompositionResult(
            original_query=query,
            sub_queries=sub_queries[:6],  # Limit to 6
            synthesis_prompt="Synthesize the sub-answers into a comprehensive response covering all clinical aspects.",
            estimated_complexity="medium",
        )

    async def close(self):
        if self._client:
            await self._client.aclose()