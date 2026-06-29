# BUILT: DeepInsightOrchestrator
"""
DeepInsight Orchestrator — Pure orchestrator that calls agents and validators.
No retrieval logic, no synthesis, no context building inline.
"""
from __future__ import annotations

import asyncio
import hashlib
import re
import time
from dataclasses import dataclass, field, fields
from datetime import datetime, timezone
from typing import Any, Dict

from loguru import logger

from src.config.settings import get_settings
from src.query.deepinsight.agents.intent_router import IntentRouter, QueryComplexity
from src.query.deepinsight.agents.query_decomposer import QueryDecomposer
from src.query.deepinsight.agents.rag_agent import RAGAgent, RAGResult
from src.query.deepinsight.agents.web_search_agent import WebSearchAgent, WebSearchResult
from src.query.deepinsight.agents.synthesis_agent import SynthesisAgent, SynthesisResult
from src.query.deepinsight.agents.citation_validator import CitationValidator, CitationResult
from src.query.deepinsight.agents.docgen_agent import DocGenAgent, DocGenResult
from src.tools import TOOL_REGISTRY, TOOL_FUNCTIONS, get_tool, is_async_tool
from src.query.search.cache import SearchCache
from src.query.validation.validator import validate_answer
from src.query.contradiction_detector import ContradictionDetector
from src.services.llm.router import LLMRouter

# Query sanitization (copied from search.py for consistency)
_DANGEROUS_PATTERNS = re.compile(
    r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]|'
    r'<script|script>|javascript:|on\w+=|'
    r'\$\{|__|SELECT|UNION|INSERT|UPDATE|DELETE|DROP',
    re.IGNORECASE,
)
_WHITESPACE_PATTERN = re.compile(r'\s+')


def _sanitize_query(query: str) -> str:
    """Sanitize query by removing dangerous patterns and normalizing whitespace."""
    sanitized = ''.join(c for c in query if ord(c) >= 32 or c in '\n\t')
    sanitized = _WHITESPACE_PATTERN.sub(' ', sanitized).strip()
    if _DANGEROUS_PATTERNS.search(sanitized):
        raise ValueError("Query contains potentially dangerous characters or patterns")
    return sanitized


@dataclass
class SubQueryResult:
    """Result of a single sub-query execution."""
    sub_query: Any  # SubQuery dataclass
    chunks: list[dict] = field(default_factory=list)
    answer: str = ""
    error: str | None = None


@dataclass
class DeepInsightResponse:
    """Full response from the DeepInsight pipeline."""
    answer: str = ""
    citations: list[dict] = field(default_factory=list)
    sections: dict[str, str] = field(default_factory=dict)
    validation: dict[str, Any] = field(default_factory=dict)
    sub_queries: list[str] = field(default_factory=list)
    sub_query_results: list[SubQueryResult] = field(default_factory=list)
    contradictions: list[dict] = field(default_factory=list)
    sources_used: dict[str, list] = field(default_factory=dict)
    cached: bool = False
    timed_out: bool = False
    synthesis_result: dict = field(default_factory=dict)
    citation_validation: dict = field(default_factory=dict)
    # API-facing fields consumed by routes/deep_insights.py
    confidence: float = 0.0
    complexity_detected: str = "unknown"
    processing_time_ms: float = 0.0


class DeepInsightOrchestrator:
    """
    Pure orchestrator — calls agents and validators, owns the pipeline.

    Pipeline: sanitize → cache check → route → decompose → parallel agents
              → contradiction detect → synthesize → validate → cache write → respond
    """

    def __init__(self):
        self.settings = get_settings()
        self.intent_router = IntentRouter()
        self.query_decomposer = QueryDecomposer()
        self.llm_router = LLMRouter()
        
        # Initialize agents
        self.rag_agent = RAGAgent(settings=self.settings, llm_router=self.llm_router)
        self.web_search_agent = WebSearchAgent(settings=self.settings, llm_router=self.llm_router)
        self.synthesis_agent = SynthesisAgent(settings=self.settings, llm_router=self.llm_router)
        self.citation_validator = CitationValidator(settings=self.settings, llm_router=self.llm_router)
        self.docgen_agent = DocGenAgent(settings=self.settings, llm_router=self.llm_router)
        
        # Initialize tools (function-based registry; full metadata also available)
        self.tools = TOOL_FUNCTIONS
        self.tool_registry = TOOL_REGISTRY
        
        self.contradiction_detector = ContradictionDetector()
        self.cache = SearchCache()

    async def process(
        self,
        query: str,
        top_k: int = 8,
        force_deep: bool = False,
    ) -> DeepInsightResponse:
        """
        Execute the full DeepInsight pipeline.

        Args:
            query: Raw user query
            top_k: Number of chunks to retrieve per sub-query (default 8)
            force_deep: Force deep multi-agent pipeline even for simple queries

        Returns:
            DeepInsightResponse with all fields populated
        """
        start_ts = time.monotonic()

        # 1. Sanitize query
        try:
            sanitized = _sanitize_query(query)
        except ValueError as e:
            return DeepInsightResponse(answer=f"Invalid query: {e}")

        # 2. Check top-level cache
        cache_key = hashlib.sha256(sanitized.lower().encode()).hexdigest()[:16]
        try:
            cached = await self.cache.get_search_result(sanitized, None)
            if cached and "deepinsight" in cached:
                logger.info(f"[DeepInsight] Cache hit for: {sanitized[:60]}...")
                # Filter to known dataclass fields to avoid TypeError on schema drift
                cached_payload = cached["deepinsight"]
                valid_fields = {f.name for f in fields(DeepInsightResponse)}
                filtered = {k: v for k, v in cached_payload.items() if k in valid_fields}
                resp = DeepInsightResponse(**filtered)
                resp.cached = True
                return resp
        except Exception as e:
            logger.warning(f"[DeepInsight] Cache read failed: {e}")

        # 3. Route intent
        routing = self.intent_router.route(sanitized)
        needs_web = routing.complexity in (QueryComplexity.COMPLEX, QueryComplexity.MEDIUM)
        if force_deep:
            needs_web = True
            logger.info("[DeepInsight] force_deep=True — forcing deep multi-agent pipeline")
        logger.info(
            f"[DeepInsight] Routed: complexity={routing.complexity.value}, "
            f"intent={routing.detected_intent}, needs_web={needs_web}"
        )

        # 4. Decompose query
        decomposition = await self.query_decomposer.decompose(
            query=sanitized,
            intent=routing.detected_intent,
            entities=routing.entities,
        )
        sub_query_texts = [sq.query for sq in decomposition.sub_queries]
        metadata_filters = None  # RoutingDecision doesn't have metadata_filters yet

        # Pre-populate sub_query_results from the decomposition so the API always
        # has something to return even if RAG fails.
        sub_query_results: list[SubQueryResult] = [
            SubQueryResult(sub_query=sq, chunks=[], answer="", error=None)
            for sq in decomposition.sub_queries
        ]

        # 5. Parallel execution with timeout
        rag_result: RAGResult | None = None
        web_result: WebSearchResult | None = None

        try:
            # Always run RAG
            rag_coro = self.rag_agent.run(sanitized, sub_query_texts, metadata_filters)
            rag_result = await asyncio.wait_for(
                rag_coro, timeout=self.settings.deep_insights_timeout
            )

            # Run web search if needed or if RAG escalated
            if needs_web or (rag_result and rag_result.escalate):
                web_coro = self.web_search_agent.run(sanitized, sanitized)
                web_result = await asyncio.wait_for(
                    web_coro, timeout=self.settings.deep_insights_timeout
                )

        except asyncio.TimeoutError:
            logger.warning("[DeepInsight] Pipeline timed out")
            return DeepInsightResponse(
                answer="Query timed out. Please try a more specific question.",
                timed_out=True,
                sub_queries=sub_query_texts,
                sub_query_results=sub_query_results,
                complexity_detected=routing.complexity.value,
                processing_time_ms=(time.monotonic() - start_ts) * 1000.0,
            )
        except Exception as e:
            logger.error(f"[DeepInsight] Pipeline error: {e}")
            return DeepInsightResponse(
                answer=f"Error processing query: {str(e)[:200]}",
                sub_queries=sub_query_texts,
                sub_query_results=sub_query_results,
                complexity_detected=routing.complexity.value,
                processing_time_ms=(time.monotonic() - start_ts) * 1000.0,
            )

        # 6. Contradiction detection
        contradictions = []
        if rag_result and web_result and web_result.found:
            try:
                chunks_for_contradiction = [
                    {"text": rag_result.context_used[:2000], "title": "Corpus evidence"},
                    {"text": web_result.summary[:2000], "title": "Web evidence"},
                ]
                report = await self.contradiction_detector.detect(
                    chunks=chunks_for_contradiction, query=sanitized
                )
                if report.has_contradictions:
                    contradictions = [
                        {
                            "type": c.contradiction_type,
                            "evidence": c.evidence,
                            "chunk_a_title": c.chunk_a.get("title", ""),
                            "chunk_b_title": c.chunk_b.get("title", ""),
                        }
                        for c in report.contradictions
                    ]
            except Exception as e:
                logger.warning(f"[DeepInsight] Contradiction detection failed: {e}")

        # 7. Synthesis with new agents
        final_answer = ""
        synthesis_result = None
        
        if rag_result and web_result and web_result.found:
            # Both sources — use synthesis agent
            try:
                synthesis_result = await self.synthesis_agent.run(
                    original_query=sanitized,
                    rag_answer=rag_result.answer,
                    web_context=web_result.summary,
                    conflict_flag=bool(contradictions),
                    conflict_detail=str([c.get('evidence', '') for c in contradictions])
                )
                final_answer = synthesis_result.answer
                logger.info("[DeepInsight] Synthesis agent completed successfully")
            except Exception as e:
                logger.warning(f"[DeepInsight] Synthesis agent failed: {e}")
                final_answer = rag_result.answer
        elif rag_result:
            # Only RAG available
            final_answer = rag_result.answer
        else:
            # No sources
            final_answer = "Unable to generate an answer from available sources."

        # 8. Citation validation
        citation_result = None
        try:
            # Prepare corpus chunks for citation validation
            corpus_chunks = []
            for citation in rag_result.citations if rag_result else []:
                corpus_chunks.append({
                    "id": citation["chunk_id"],
                    "title": citation["title"],
                    "text": citation.get("excerpt", "")[:1000]  # Use excerpt or truncate
                })
            
            # Prepare web sources for citation validation  
            web_sources = []
            if web_result and web_result.found:
                for i, source in enumerate(web_result.sources):
                    web_sources.append({
                        "id": f"WEB_{i+1:03d}",
                        "title": source.get("title", ""),
                        "url": source.get("url", ""),
                        "excerpt": source.get("excerpt", "")[:1000]
                    })
            
            citation_result = await self.citation_validator.run(
                answer_text=final_answer,
                corpus_chunks=corpus_chunks,
                web_sources=web_sources
            )
            logger.info("[DeepInsight] Citation validation completed successfully")
        except Exception as e:
            logger.warning(f"[DeepInsight] Citation validation failed: {e}")
            # Fallback: basic citation extraction
            citation_result = CitationResult(
                validation_complete=False,
                hallucination_detected=False,
                citations=[],
                flagged_claims=[],
                summary={"total_claims": 0, "verified": 0, "assigned": 0, "misattributed": 0, "unsupported": 0}
            )

        # 9. Build citations from validation result
        citations = []
        if citation_result and citation_result.citations:
            citations.extend(citation_result.citations)
        else:
            # Fallback to original citations
            if rag_result:
                citations.extend(rag_result.citations)
            if web_result and web_result.found:
                for i, source in enumerate(web_result.sources, len(citations) + 1):
                    citations.append({
                        "chunk_id": source.get("id", f"web_{i}"),
                        "mongo_id": "",
                        "source_type": "web",
                        "index": i,
                        "title": source.get("title", ""),
                        "score": 0.0,
                        "url": source.get("url", ""),
                        "tier": source.get("tier", 5),
                    })

        # 9. Section splitting
        sections = self._split_sections(final_answer)

        # 10. Validation — DO NOT SKIP
        validation_result = {}
        try:
            source_chunks = [
                {"chunk_text": c.get("excerpt", c.get("title", "")), "title": c.get("title", "")}
                for c in citations
            ]
            # If no chunks from citations, use context from RAG
            if not source_chunks and rag_result and rag_result.context_used:
                source_chunks = [{"chunk_text": rag_result.context_used[:1000], "title": "Corpus"}]

            val = await validate_answer(
                answer=final_answer,
                citations=citations,
                source_chunks=source_chunks,
                verify_citations_in_db=False,
            )
            validation_result = {
                "hallucination_score": val.hallucination_result.hallucination_score if val.hallucination_result else 0.0,
                "safety_status": "safe" if val.is_safe else "unsafe",
                "confidence_score": val.confidence_score,
                "needs_review": val.needs_disclaimer or val.recommendation == "NEEDS_REVIEW",
                "recommendation": val.recommendation,
                "safety_warnings": [
                    {"type": w.warning_type, "severity": w.severity, "message": w.message}
                    for w in val.safety_warnings
                ],
                "citation_validation": {
                    "validation_complete": citation_result.validation_complete if citation_result else False,
                    "hallucination_detected": citation_result.hallucination_detected if citation_result else False,
                    "total_claims": citation_result.summary["total_claims"] if citation_result else 0,
                    "verified_claims": citation_result.summary["verified"] if citation_result else 0,
                    "flagged_claims": len(citation_result.flagged_claims) if citation_result else 0
                } if citation_result else {}
            }
        except Exception as e:
            logger.warning(f"[DeepInsight] Validation failed: {e}")
            validation_result = {"error": str(e)[:200]}

        # 11. Build sources_used from synthesis and validation results
        sources_used: dict[str, list] = {"corpus": [], "web": []}
        if synthesis_result and synthesis_result.sources_used:
            sources_used = synthesis_result.sources_used
        else:
            # Fallback to original sources
            if rag_result:
                sources_used["corpus"] = [
                    {"chunk_id": c["chunk_id"], "title": c["title"]}
                    for c in rag_result.citations
                ]
            if web_result and web_result.found:
                sources_used["web"] = [
                    {"url": s.get("url", ""), "title": s.get("title", ""), "tier": s.get("tier", 5)}
                    for s in web_result.sources
                ]

        # Compute API-facing scalar fields
        processing_time_ms = (time.monotonic() - start_ts) * 1000.0
        confidence_score = 0.0
        if isinstance(validation_result, dict):
            confidence_score = float(validation_result.get("confidence_score", 0.0) or 0.0)

        response = DeepInsightResponse(
            answer=final_answer,
            citations=citations,
            sections=sections,
            validation=validation_result,
            sub_queries=sub_query_texts,
            sub_query_results=sub_query_results,
            contradictions=contradictions,
            sources_used=sources_used,
            cached=False,
            timed_out=False,
            synthesis_result={
                "conflict_resolved": synthesis_result.conflict_resolved if synthesis_result else False,
                "conflict_note": synthesis_result.conflict_note if synthesis_result else "N/A",
                "synthesis_confidence": synthesis_result.synthesis_confidence if synthesis_result else "unknown",
                "synthesis_confidence_reason": synthesis_result.synthesis_confidence_reason if synthesis_result else ""
            } if synthesis_result else {},
            citation_validation={
                "validation_complete": citation_result.validation_complete if citation_result else False,
                "hallucination_detected": citation_result.hallucination_detected if citation_result else False,
                "summary": citation_result.summary if citation_result else {}
            } if citation_result else {},
            confidence=confidence_score,
            complexity_detected=routing.complexity.value,
            processing_time_ms=processing_time_ms,
        )

        # 13. Write to cache
        try:
            await self.cache.set_search_result(
                sanitized, None, {"deepinsight": {
                    "answer": response.answer,
                    "citations": response.citations,
                    "sections": response.sections,
                    "validation": response.validation,
                    "sub_queries": response.sub_queries,
                    "contradictions": response.contradictions,
                    "sources_used": response.sources_used,
                    "cached": False,
                    "timed_out": False,
                    "synthesis_result": response.synthesis_result,
                    "citation_validation": response.citation_validation,
                    "confidence": response.confidence,
                    "complexity_detected": response.complexity_detected,
                    "processing_time_ms": response.processing_time_ms,
                }}
            )
        except Exception as e:
            logger.warning(f"[DeepInsight] Cache write failed: {e}")

        return response

    def _split_sections(self, answer: str) -> dict[str, str]:
        """Parse answer into clinical sections."""
        sections: dict[str, str] = {}

        # Try to find named sections
        section_markers = [
            "Diagnosis", "Treatment", "Dosage", "Monitoring",
            "Recommendation", "Summary", "Limitations", "Key Findings",
            "Protocol", "Contraindications", "Side Effects",
        ]

        current_section = "summary"
        current_lines: list[str] = []

        for line in answer.split("\n"):
            stripped = line.strip()
            matched = False
            for marker in section_markers:
                if stripped.lower().startswith(marker.lower() + ":"):
                    if current_lines:
                        sections[current_section] = "\n".join(current_lines).strip()
                    current_section = marker.lower().replace(" ", "_")
                    current_lines = []
                    matched = True
                    break
            if not matched:
                current_lines.append(line)

        if current_lines:
            sections[current_section] = "\n".join(current_lines).strip()

        # Fallback: split on double newlines if no sections found
        if not sections or (len(sections) == 1 and "summary" in sections):
            paragraphs = [p.strip() for p in answer.split("\n\n") if p.strip()]
            if len(paragraphs) > 1:
                sections = {f"paragraph_{i+1}": p for i, p in enumerate(paragraphs[:5])}
                sections["full_answer"] = answer
            else:
                sections["full_answer"] = answer

        return sections

    async def generate_document(
        self, 
        query: str, 
        doc_format: str = "pdf", 
        title: str = None
    ) -> Dict[str, Any]:
        """
        Generate a document from the last query result.
        
        Args:
            query: Original query (for context)
            doc_format: "pdf" or "docx"
            title: Optional title for document
            
        Returns:
            Dict with document metadata
        """
        # Get cached result if available
        cache_key = hashlib.sha256(query.lower().encode()).hexdigest()[:16]
        try:
            cached = await self.cache.get_search_result(query, None)
            if cached and "deepinsight" in cached:
                result = cached["deepinsight"]
                
                # Prepare document request
                doc_request = {
                    "format": doc_format,
                    "title": title or f"Summary for: {query[:100]}",
                    "content": result.get("answer", ""),
                    "citations": result.get("citations", []),
                    "patient_context": "",  # Would be filled by UI
                    "generated_at": datetime.now().isoformat()
                }
                
                # Generate document
                docgen_result = await self.docgen_agent.run(doc_request)
                
                return {
                    "file_path": docgen_result.file_path,
                    "format": docgen_result.format,
                    "page_count": docgen_result.page_count,
                    "size_bytes": docgen_result.size_bytes,
                    "title": docgen_result.title,
                    "generated_at": docgen_result.generated_at
                }
        except Exception as e:
            logger.error(f"[DeepInsight] Document generation failed: {e}")
            return {"error": str(e)}
        
        return {"error": "No cached result found for document generation"}
