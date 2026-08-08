# Multi-Agent Design — DeepInsights Pipeline

> Last updated: August 2026
> Reflects codebase on `restruct` branch at commit `78f4d6b`

The DeepInsights pipeline uses a **pure orchestrator pattern** with 5 specialized agents. No LangGraph/CrewAI — the orchestrator is a hand-written async coordinator in `src/query/deepinsight/orchestrator.py`.

```
┌─────────────────────────────────────────────────────────────────┐
│                    DeepInsightOrchestrator                       │
│              (Pure coordinator — no inline business logic)       │
│                                                                 │
│  Pipeline:                                                       │
│  1. Sanitize query                                               │
│  2. Cache check (Redis)                                          │
│  3. Intent route → SIMPLE redirects to /search                   │
│  4. Decompose query (3–6 sub-queries)                            │
│  5. Parallel agents: asyncio.gather(RAG + Web Search)            │
│  6. Contradiction detection                                      │
│  7. Synthesis (merge RAG + web)                                  │
│  8. Citation validation                                          │
│  9. Full validation pipeline                                     │
│  10. Cache write + respond                                       │
└─────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────┬───────────┼───────────┬─────────┐
        ▼         ▼           ▼           ▼         ▼
┌─────────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
│  RAG Agent  │ │Web Search│ │Synthesis │ │ Citation │ │ DocGen   │
│             │ │  Agent   │ │  Agent   │ │Validator │ │  Agent   │
│ Full RAG    │ │3-tier    │ │Merge RAG │ │Claim→src │ │PDF/DOCX  │
│ per sub-q   │ │fetch     │ │+ web     │ │mapping   │ │render    │
│ Escalation  │ │conflict  │ │conflict  │ │misattrib │ │no LLM    │
│ detection   │ │detect    │ │resolution│ │detection │ │          │
└─────────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘
```

## Agent Details

### RAG Agent (`rag_agent.py`)
- Full RAG pipeline per sub-query: cache → parallel retrieve → RRF → rerank → MMR → context → LLM → cache write
- **Escalation detection**: If LLM outputs "ESCALATE: true", sets `escalate=True` to signal that corpus coverage is insufficient and web search should be triggered
- **Source ID extraction**: Parses "SOURCE_IDS:" line from LLM output for citation tracking
- **Confidence**: Weighted 0.35 score + 0.30 coverage + 0.35 evidence

### Web Search Agent (`web_search_agent.py`)
- **Tier 1**: HTTPFetcher (httpx) — static medical sites
- **Tier 2**: CDPBrowser — raw WebSocket CDP, zero external dependencies, auto-discovers Lightpanda/Chrome
- **Tier 3**: Gemini Flash — LLM fallback for complex queries
- Target URL construction based on detected entities (ICMR, AHA, NICE, WHO, FDA)
- Medical trust filtering (5-tier source hierarchy)
- Conflict detection via keyword matching

### Synthesis Agent (`synthesis_agent.py`)
- **Conditional activation**: Only runs when BOTH RAG and web return results
- Merges corpus + web results into coherent answer
- Explicit conflict resolution — surfaces contradictions instead of silently picking one
- RAG-only or web-only paths skip synthesis and use raw output directly

### Citation Validator (`citation_validator.py`)
- Post-generation claim→source validation
- Maps every claim to original source chunks/web results
- Detects misattribution (claim cites [C3] but evidence is in [C7] or unsupported)
- Outputs machine-readable citation schema for UI tooltips and "show sources" panel
- LLM-based validation with keyword-overlap fallback

### DocGen Agent (`docgen_agent.py`)
- **No LLM** — pure rendering, cannot introduce new errors
- PDF via reportlab, DOCX via python-docx
- Sections: header, clinical_summary, patient_context, sources_and_citations, disclaimer

## Supporting Components

### IntentRouter (`intent_router.py`)
- **Deterministic** — no LLM, pure pattern matching + entity counting
- Complexity levels: SIMPLE → `/search`, MEDIUM/COMPLEX → DeepInsights
- Confidence scoring: 0.0–1.0 based on pattern count and entity count

### QueryDecomposer (`query_decomposer.py`)
- LLM-based sub-query generation (3–6 sub-queries)
- Each sub-query has: query text, focus, priority (1–3), metadata
- Rule-based fallback when LLM fails

### ContradictionDetector (`contradiction_detector.py`)
- NLI model (PubMedBERT-based) via `asyncio.to_thread`
- Keyword-based fallback
- Detects: treatment_conflict, dosage_conflict, outcome_conflict
