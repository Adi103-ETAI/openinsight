# BUILT: RAGAgent
"""
RAG Agent — Wraps the existing search pipeline into one clean class.
Retrieves, fuses, reranks, diversifies, builds context, generates answer via LLM.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from src.config.settings import get_settings
from src.query.search.retriever import HybridRetriever, RetrievedChunk
from src.query.search.fusion import reciprocal_rank_fusion
from src.query.search.reranker import get_reranker
from src.query.search.mmr import maximal_marginal_relevance
from src.query.search.context_builder import assemble_context, build_citation_list
from src.query.search.cache import SearchCache
from src.services.llm.router import LLMRouter
from src.query.deepinsight.agents.skills import get_system_prompt


@dataclass
class RAGResult:
    """Result from the RAG agent."""
    answer: str = ""
    source_ids: list[str] = field(default_factory=list)
    mongo_ids: list[str] = field(default_factory=list)
    citations: list[dict] = field(default_factory=list)
    context_used: str = ""
    confidence: float = 0.0
    escalate: bool = False
    escalate_reason: str | None = None


class _QueryAnalysis:
    """Minimal query analysis object matching retriever expectations."""

    def __init__(
        self,
        rewritten_query: str | None = None,
        use_hyde: bool = False,
        expanded_terms: list[str] | None = None,
        metadata_filters: Any = None,
    ):
        self.rewritten_query = rewritten_query
        self.use_hyde = use_hyde
        self.expanded_terms = expanded_terms or []
        self.metadata_filters = metadata_filters


class RAGAgent:
    """
    Clinical RAG agent — retrieves from corpus, generates grounded answer.

    Pipeline: cache check → parallel retrieve → fuse → rerank → MMR → context → LLM → cache write
    """

    def __init__(self, settings: Any = None, llm_router: LLMRouter | None = None):
        self.settings = settings or get_settings()
        self.retriever = HybridRetriever()
        self.reranker = get_reranker()
        self.cache = SearchCache()
        self.llm_router = llm_router or LLMRouter()

    async def run(
        self,
        query: str,
        sub_queries: list[str],
        metadata_filters: Any = None,
    ) -> RAGResult:
        """
        Execute the full RAG pipeline.

        Args:
            query: The original clinical query
            sub_queries: Decomposed sub-queries to retrieve for
            metadata_filters: Optional filter expression for retrieval

        Returns:
            RAGResult with answer, citations, confidence, and escalation info
        """
        # 1. Check cache
        try:
            cached = await self.cache.get_search_result(query, metadata_filters)
            if cached and "rag_result" in cached:
                logger.debug(f"[RAG] Cache hit for query: {query[:60]}...")
                return RAGResult(**cached["rag_result"])
        except Exception as e:
            logger.warning(f"[RAG] Cache read failed: {e}")

        # 2. Parallel retrieval for all sub-queries
        analysis = _QueryAnalysis(metadata_filters=metadata_filters)
        all_dense: list[RetrievedChunk] = []
        all_sparse: list[RetrievedChunk] = []

        tasks = [
            self.retriever.retrieve(sq, analysis, top_k=self.settings.top_k_retrieval)
            for sq in sub_queries
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning(f"[RAG] Sub-query {i} retrieval failed: {result}")
                continue
            dense, sparse = result
            all_dense.extend(dense)
            all_sparse.extend(sparse)

        if not all_dense and not all_sparse:
            return RAGResult(
                answer="No relevant clinical information found in the knowledge base.",
                escalate=True,
                escalate_reason="Corpus returned zero chunks for all sub-queries.",
            )

        # 3. Fuse dense + sparse with RRF
        fused = reciprocal_rank_fusion(
            all_dense, all_sparse, top_n=self.settings.top_k_after_fusion
        )

        # 4. Rerank
        try:
            reranked = self.reranker.rerank(
                query, fused, top_k=min(self.settings.top_k_after_rerank, len(fused))
            )
        except Exception as e:
            logger.warning(f"[RAG] Reranking failed, using fused results: {e}")
            reranked = fused

        # 5. MMR for diversity
        diverse = maximal_marginal_relevance(
            self.retriever.embedder,
            reranked,
            lambda_param=self.settings.mmr_lambda,
            n_select=self.settings.top_k_final,
        )

        # 6. Build context + citations
        context = assemble_context(diverse, max_tokens=3000)
        raw_citations = build_citation_list(diverse)

        # 7. Load system prompt
        system_prompt = self._load_system_prompt()

        # 8. Call LLM
        user_prompt = f"Query: {query}\n\nContext:\n{context}"
        client = self.llm_router.get_client_for_agent("rag")

        try:
            answer_text = await client.chat_completions(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.settings.nim_temperature,
                max_tokens=self.settings.nim_max_tokens,
            )
        except Exception as e:
            logger.error(f"[RAG] LLM call failed: {e}")
            return RAGResult(
                answer="Failed to generate answer due to LLM error.",
                escalate=True,
                escalate_reason=f"LLM error: {str(e)[:200]}",
            )

        if not answer_text:
            return RAGResult(
                answer="LLM returned empty response.",
                escalate=True,
                escalate_reason="LLM returned empty response.",
            )

        # 9. Check for escalation signal
        escalate = False
        escalate_reason = None
        if "ESCALATE: true" in answer_text.upper() or "ESCALATE:true" in answer_text.upper():
            escalate = True
            for line in answer_text.split("\n"):
                if line.strip().upper().startswith("REASON:"):
                    escalate_reason = line.strip()[7:].strip()
                    break
            if not escalate_reason:
                escalate_reason = "Model indicated insufficient corpus coverage."

        # 10. Extract source IDs
        source_ids = []
        for line in answer_text.split("\n"):
            if line.strip().upper().startswith("SOURCE_IDS:"):
                ids_str = line.strip()[11:].strip()
                source_ids = [s.strip() for s in ids_str.split(",") if s.strip()]
                break

        # 11. Build citation list with proper format
        citations = []
        for i, chunk in enumerate(diverse, 1):
            meta = chunk.metadata or {}
            citations.append({
                "chunk_id": chunk.chunk_id,
                "mongo_id": str(meta.get("mongo_id", "")),
                "source_type": "corpus",
                "index": i,
                "title": meta.get("title", "Untitled"),
                "score": float(chunk.score),
            })

        # 12. Compute confidence
        confidence = self._compute_confidence(diverse, answer_text, source_ids)

        # 13. Build mongo_ids list
        mongo_ids = [c["mongo_id"] for c in citations if c["mongo_id"]]

        result = RAGResult(
            answer=answer_text,
            source_ids=source_ids,
            mongo_ids=mongo_ids,
            citations=citations,
            context_used=context,
            confidence=confidence,
            escalate=escalate,
            escalate_reason=escalate_reason,
        )

        # 14. Write to cache
        try:
            await self.cache.set_search_result(
                query, metadata_filters, {"rag_result": {
                    "answer": result.answer,
                    "source_ids": result.source_ids,
                    "mongo_ids": result.mongo_ids,
                    "citations": result.citations,
                    "context_used": result.context_used,
                    "confidence": result.confidence,
                    "escalate": result.escalate,
                    "escalate_reason": result.escalate_reason,
                }}
            )
        except Exception as e:
            logger.warning(f"[RAG] Cache write failed: {e}")

        return result

    def _load_system_prompt(self) -> str:
        """Load system prompt from SKILL.md at runtime."""
        fallback = (
            "You are a clinical decision support AI for Indian physicians. "
            "Answer based ONLY on the provided evidence chunks. "
            "Cite sources using [CHUNK_ID] inline. "
            "If the corpus lacks sufficient information, output ESCALATE: true "
            "with a REASON line explaining what is missing. "
            "Never hallucinate drug names, dosages, or study references."
        )
        return get_system_prompt("rag_agent", fallback=fallback)

    def _compute_confidence(
        self, chunks: list[RetrievedChunk], answer: str, source_ids: list[str]
    ) -> float:
        """Compute confidence from chunk quality, count, and citation coverage."""
        if not chunks:
            return 0.0

        # Chunk score component (avg relevance)
        avg_score = sum(c.score for c in chunks) / len(chunks)
        score_component = min(avg_score, 1.0)

        # Coverage component (how many chunks were cited)
        cited_count = len(source_ids)
        coverage_component = min(cited_count / max(len(chunks), 1), 1.0)

        # Evidence level component
        evidence_scores = []
        for c in chunks:
            level = str(c.metadata.get("evidence_level", 5))
            try:
                evidence_scores.append(1.0 - (int(level) - 1) * 0.2)
            except (ValueError, TypeError):
                evidence_scores.append(0.4)
        evidence_component = sum(evidence_scores) / len(evidence_scores) if evidence_scores else 0.4

        # Weighted combination
        confidence = (
            0.35 * score_component
            + 0.30 * coverage_component
            + 0.35 * evidence_component
        )
        return max(0.0, min(1.0, confidence))
