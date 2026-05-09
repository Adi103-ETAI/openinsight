from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from .intent_router import IntentRouter, QueryComplexity, RoutingDecision
from .query_decomposer import (
    DecompositionResult,
    QueryDecomposer,
    SubQuery,
)
from src.core.config import get_settings
from src.query.contradiction_detector import ContradictionDetector
from src.query.search.cache import SearchCache
from src.query.search.retriever import HybridRetriever, RetrievedChunk

logger = logging.getLogger(__name__)


@dataclass
class SubQueryResult:
    """Result from a single sub-query retrieval."""
    sub_query: SubQuery
    chunks: list[RetrievedChunk]
    error: str | None = None


def _calculate_confidence(
    sub_results: list[SubQueryResult],
    contradictions: list[dict],
    all_chunks: list[RetrievedChunk],
) -> float:
    """Calculate confidence based on sub-query success, chunk coverage, and contradictions."""
    if not sub_results:
        return 0.0
    
    successful_subqueries = sum(1 for r in sub_results if not r.error and r.chunks)
    subquery_rate = successful_subqueries / len(sub_results)
    
    avg_chunks = sum(len(r.chunks) for r in sub_results) / len(sub_results)
    chunk_score = min(avg_chunks / 5, 1.0)  # 5 chunks = full score
    
    contradiction_penalty = min(len(contradictions) * 0.1, 0.3)  # Max 30% penalty
    
    confidence = (subquery_rate * 0.4 + chunk_score * 0.6) - contradiction_penalty
    return max(0.0, min(1.0, confidence))


@dataclass
class DeepInsightsResult:
    """Final result from DeepInsights processing."""
    answer: str
    sections: dict[str, str]  # Structured answer by topic
    citations: list[dict]
    sub_query_results: list[SubQueryResult]
    contradictions: list[dict] = field(default_factory=list)
    confidence: float = 0.0
    complexity_detected: str = "complex"
    processing_time_ms: float = 0.0


class DeepInsightsOrchestrator:
    """
    Main orchestrator for DeepInsights mode.
    
    Uses LangGraph-style state machine:
    
    1. Route → Determine if complex or simple
    2. Decompose → Break into sub-queries  
    3. Retrieve → Parallel retrieval for each sub-query
    4. Synthesize → Merge results, detect contradictions
    5. Format → Structure final answer
    """

    def __init__(self):
        self.settings = get_settings()
        self.router = IntentRouter()
        self.decomposer = QueryDecomposer()
        self.retriever = HybridRetriever()
        self.cache = SearchCache()
        self.contradiction_detector = ContradictionDetector()

    async def process(
        self,
        query: str,
        top_k: int = 8,
        force_deep: bool = False,
    ) -> DeepInsightsResult:
        """Process query through DeepInsights pipeline."""
        start_time = time.perf_counter()

        try:
            # Step 1: Route
            routing = self.router.route(query)
            logger.info(f"[DeepInsights] Routed as: {routing.complexity.value}")

            # If simple and not forced, could fallback to standard
            if routing.complexity == QueryComplexity.SIMPLE and not force_deep:
                # Still process through deep for consistency
                logger.info("[DeepInsights] Simple query but using deep mode")

            # Step 2: Decompose
            decomposition = await self.decomposer.decompose(
                query=query,
                intent=routing.detected_intent,
                entities=routing.entities,
            )
            logger.info(
                f"[DeepInsights] Decomposed into {len(decomposition.sub_queries)} sub-queries"
            )

            # Step 3: Parallel Retrieval
            sub_results = await self._parallel_retrieve(
                decomposition.sub_queries,
                top_k,
                routing.metadata_filters if hasattr(routing, "metadata_filters") else None,
            )

            # Step 4: Synthesize
            all_chunks = []
            for result in sub_results:
                all_chunks.extend(result.chunks)

            # Detect contradictions
            contradictions = []
            if (
                self.settings.contradiction_detection
                and len(all_chunks) >= self.settings.contradiction_min_chunks
            ):
                contradiction_report = await self.contradiction_detector.detect(
                    chunks=[
                        {
                            "text": c.text,
                            "title": c.metadata.get("title", ""),
                        }
                        for c in all_chunks
                    ],
                    query=query,
                )
                if contradiction_report.has_contradictions:
                    contradictions = [
                        {
                            "type": c.contradiction_type,
                            "evidence": c.evidence,
                            "chunk_a_title": c.chunk_a.get("title", ""),
                            "chunk_b_title": c.chunk_b.get("title", ""),
                        }
                        for c in contradiction_report.contradictions
                    ]

            # Step 5: Generate Answer via LLM
            answer, sections, citations = await self._synthesize_answer(
                query=query,
                chunks=all_chunks,
                sub_results=sub_results,
                synthesis_prompt=decomposition.synthesis_prompt,
            )

            processing_time = (time.perf_counter() - start_time) * 1000

            confidence = _calculate_confidence(sub_results, contradictions, all_chunks)

            return DeepInsightsResult(
                answer=answer,
                sections=sections,
                citations=citations,
                sub_query_results=sub_results,
                contradictions=contradictions,
                confidence=confidence,
                complexity_detected=routing.complexity.value,
                processing_time_ms=processing_time,
            )

        except Exception as e:
            logger.error(f"[DeepInsights] Error: {e}")
            processing_time = (time.perf_counter() - start_time) * 1000
            return DeepInsightsResult(
                answer=f"Error processing query: {str(e)}",
                sections={"error": str(e)},
                citations=[],
                sub_query_results=[],
                contradictions=[],
                confidence=0.0,
                complexity_detected="error",
                processing_time_ms=processing_time,
            )

    async def _parallel_retrieve(
        self,
        sub_queries: list[SubQuery],
        top_k: int,
        filters: Any,
    ) -> list[SubQueryResult]:
        """Execute retrieval for all sub-queries in parallel."""
        tasks = []
        for sq in sub_queries:
            task = self._retrieve_sub_query(sq, top_k, filters)
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        sub_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                sub_results.append(
                    SubQueryResult(
                        sub_query=sub_queries[i],
                        chunks=[],
                        error=str(result),
                    )
                )
            else:
                sub_results.append(result)

        return sub_results

    async def _retrieve_sub_query(
        self,
        sub_query: SubQuery,
        top_k: int,
        filters: Any,
    ) -> SubQueryResult:
        """Retrieve chunks for a single sub-query."""
        try:
            # Create a simple query analysis object
            class SimpleAnalysis:
                def __init__(self):
                    self.use_hyde = False
                    self.expanded_terms = []
                    self.metadata_filters = filters

            analysis = SimpleAnalysis()
            dense_chunks, sparse_chunks = await self.retriever.retrieve(
                query=sub_query.query,
                query_analysis=analysis,
                top_k=top_k,
            )

            # Combine and rank
            all_chunks = dense_chunks + sparse_chunks
            # Dedupe by chunk_id
            seen = set()
            unique_chunks = []
            for c in all_chunks:
                if c.chunk_id not in seen:
                    seen.add(c.chunk_id)
                    unique_chunks.append(c)

            return SubQueryResult(
                sub_query=sub_query,
                chunks=unique_chunks[:top_k],
                error=None,
            )

        except Exception as e:
            return SubQueryResult(
                sub_query=sub_query,
                chunks=[],
                error=str(e),
            )

    async def _synthesize_answer(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        sub_results: list[SubQueryResult],
        synthesis_prompt: str,
    ) -> tuple[str, dict[str, str], list[dict]]:
        """Use LLM to synthesize final answer from retrieved chunks."""
        import httpx

        # Build context
        context_parts = []
        for i, chunk in enumerate(chunks[:20]):  # Limit to top 20
            source = chunk.metadata.get("source_type", "unknown")
            title = chunk.metadata.get("title", "Untitled")
            context_parts.append(
                f"[{i + 1}] Source: {source}, Title: {title}\n{chunk.text[:300]}"
            )

        context = "\n\n".join(context_parts)

        synthesis_prompt_full = f"""You are a clinical decision support AI for Indian physicians.
Synthesize the following retrieved evidence into a comprehensive answer.

Original Query: {query}

{synthesis_prompt}

Guidelines:
- Use evidence from all sub-queries
- Cite sources using [1], [2], etc.
- Flag any conflicting evidence
- Keep answer structured and practical

Evidence:
{context}

Generate a structured answer covering:
1. Key findings from each aspect searched
2. Recommendations with supporting evidence
3. Any warnings or contradictions
4. Confidence assessment

Answer:"""

        # Call LLM
        url = f"{self.settings.nvidia_nim_base_url.rstrip('/')}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.settings.nvidia_nim_api_key:
            headers["Authorization"] = f"Bearer {self.settings.nvidia_nim_api_key}"

        body = {
            "model": self.settings.nim_model,
            "messages": [
                {"role": "system", "content": "You are a medical expert. Provide evidence-based answers."},
                {"role": "user", "content": synthesis_prompt_full},
            ],
            "temperature": self.settings.nim_temperature,
            "max_tokens": self.settings.nim_max_tokens,
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, headers=headers, json=body)
            response.raise_for_status()
            data = response.json()

        answer_text = data.get("choices", [{}])[0].get("message", {}).get("content", "")

        # Extract citations from chunks
        citations = []
        for chunk in chunks[:10]:
            citations.append({
                "id": chunk.chunk_id,
                "title": chunk.metadata.get("title", "Untitled"),
                "source": chunk.metadata.get("source_type", "unknown"),
                "score": chunk.score,
            })

        # Create sections (simplified - could be more sophisticated)
        sections = {
            "summary": answer_text[:500],
            "details": answer_text,
            "recommendations": answer_text[-300:] if len(answer_text) > 300 else answer_text,
        }

        return answer_text, sections, citations