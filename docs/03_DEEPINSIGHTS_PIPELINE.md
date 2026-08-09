# DeepInsights Pipeline

> Last updated: August 2026
> Reflects codebase on `restruct` branch at commit `78f4d6b`

## When to Use DeepInsights

Use DeepInsights for complex clinical queries that require:
- Drug interaction checks (e.g., "Can I give metformin with ACE inhibitors?")
- Differential diagnosis (e.g., "What could cause this presentation?")
- Protocol conflicts (e.g., "ICMR vs WHO guidelines for dengue")
- Multi-condition management (e.g., "DM with HTN and CKD")
- Comparisons across guidelines (e.g., "StatPearls vs ICMR for dengue management")

---

## DeepInsights Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              REQUEST                                        │
│          { "query": "treatment for diabetes with hypertension..." }         │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         INTENT ROUTER                                       │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  IntentRouter — Deterministic (no LLM)                              │    │
│  │                                                                     │    │
│  │  Complex Patterns (regex):                                          │    │
│  │    - "vs / versus" (comparisons)                                    │    │
│  │    - "interaction" (drug interactions)                              │    │
│  │    - Multi-condition ("X and Y and Z")                              │    │
│  │    - "contraindicated"                                              │    │
│  │    - "differential"                                                 │    │
│  │    - "protocol ... versus"                                          │    │
│  │    - "guideline ... conflict"                                       │    │
│  │                                                                     │    │
│  │  Complexity Calculation:                                            │    │
│  │    2+ patterns → COMPLEX (95%)                                     │    │
│  │    1 pattern + 3+ entities → COMPLEX (90%)                         │    │
│  │    1 pattern + 2 entities → MEDIUM (70%)                           │    │
│  │    4+ entities → COMPLEX (85%)                                     │    │
│  │    Default → SIMPLE (75%)                                          │    │
│  │                                                                     │    │
│  │  Output: RoutingDecision                                            │    │
│  │    - complexity: SIMPLE / MEDIUM / COMPLEX                          │    │
│  │    - confidence: 0.0–1.0                                            │    │
│  │    - detected_intent: therapeutic/diagnostic/etc.                   │    │
│  │    - entities: [list of extracted medical entities]                 │    │
│  │    - sub_query_types: [treatment, dosage, interactions, etc.]       │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
                 [SIMPLE]                       [MEDIUM/COMPLEX]
               Use standard                      Continue
                /search
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       QUERY DECOMPOSER                                      │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  QueryDecomposer                                                    │    │
│  │                                                                     │    │
│  │  LLM-based decomposition (rule-based fallback)                     │    │
│  │                                                                     │    │
│  │  Input: "treatment for diabetes with hypertension and CKD"          │    │
│  │                                                                     │    │
│  │  Output: 3–6 Sub-queries, each with:                               │    │
│  │    - query text                                                     │    │
│  │    - focus (treatment/dosage/interaction/guideline/etc.)            │    │
│  │    - priority (1–3)                                                 │    │
│  │    - metadata (intent, entities)                                    │    │
│  │                                                                     │    │
│  │  Example decomposition:                                             │    │
│  │    q1: "diabetes treatment options" (focus: treatment)              │    │
│  │    q2: "hypertension medication dosage" (focus: dosage)             │    │
│  │    q3: "drug interactions diabetes hypertension" (focus: inter)     │    │
│  │    q4: "CKD contraindications diabetes drugs" (focus: contra)       │    │
│  │    q5: "ICMR guidelines diabetes hypertension" (focus: guide)       │    │
│  │    q6: "diabetes CKD management protocols" (focus: protocol)        │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PARALLEL AGENTS (asyncio.gather)                          │
│                                                                             │
│   ┌─────────────────────────┐    ┌─────────────────────────┐                │
│   │      RAG Agent           │    │    Web Search Agent      │                │
│   │                          │    │                          │                │
│   │  Per sub-query:          │    │  Three-tier fetch:       │                │
│   │  1. Cache check          │    │  Tier 1: HTTPFetcher     │                │
│   │  2. Hybrid retrieve      │    │  Tier 2: CDPBrowser      │                │
│   │  3. RRF fusion           │    │  Tier 3: Gemini Flash    │                │
│   │  4. Rerank               │    │                          │                │
│   │  5. MMR diversity        │    │  Medical trust filtering │                │
│   │  6. Context assembly     │    │  Dedup + conflict detect │                │
│   │  7. LLM synthesis        │    │  LLM summarization       │                │
│   │  8. Cache write          │    │                          │                │
│   │                          │    │                          │                │
│   │  Escalation detection    │    │  Target URLs: ICMR, AHA, │                │
│   │  (corpus insufficient)   │    │  NICE, WHO, FDA, etc.   │                │
│   └────────────┬─────────────┘    └────────────┬─────────────┘                │
│                │                               │                             │
│                └───────────────┬───────────────┘                             │
│                                ▼                                             │
│                ┌───────────────────────────────────────┐                      │
│                │  Combined results (RAG + optional web)│                      │
│                └───────────────────────────────────────┘                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    CONTRADICTION DETECTION                                   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  ContradictionDetector                                              │    │
│  │                                                                     │    │
│  │  Primary: NLI model (PubMedBERT-based) via asyncio.to_thread        │    │
│  │  Fallback: Keyword-based detection                                  │    │
│  │    - "improve" vs "worsen"                                          │    │
│  │    - "recommended" vs "not recommended"                             │    │
│  │    - "effective" vs "ineffective"                                   │    │
│  │    - dosage conflicts                                              │    │
│  │                                                                     │    │
│  │  Output: ContradictionReport                                        │    │
│  │    - type: treatment_conflict/dosage_conflict/outcome_conflict      │    │
│  │    - evidence: conflicting chunks and keywords                      │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     SYNTHESIS AGENT                                         │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  SynthesisAgent                                                    │    │
│  │                                                                     │    │
│  │  Only runs when BOTH RAG and Web Search return results.             │    │
│  │  RAG-only or web-only paths skip synthesis and use raw output.      │    │
│  │                                                                     │    │
│  │  Merges corpus + web results into a single coherent answer.         │    │
│  │  Explicitly surfaces contradictions between sources instead of       │    │
│  │  silently picking one.                                              │    │
│  │                                                                     │    │
│  │  Input: original_query, rag_answer, web_context,                    │    │
│  │         conflict_flag, conflict_detail                              │    │
│  │  Output: SynthesisResult (merged answer + conflict tags)            │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    CITATION VALIDATION                                       │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  CitationValidator                                                  │    │
│  │                                                                     │    │
│  │  Post-generation claim→source validation.                           │    │
│  │  Maps every claim in the answer to original source chunks/web.      │    │
│  │  Detects misattribution (e.g., claim cites [C3] but evidence is     │    │
│  │  actually in [C7] or unsupported entirely).                        │    │
│  │                                                                     │    │
│  │  Output: CitationResult — machine-readable citation schema           │    │
│  │  for UI tooltips and "show sources" panel.                          │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    VALIDATION PIPELINE                                       │
│                                                                             │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐        │
│  │  Hallucina-  │ │  Citation    │ │   Safety     │ │  Confidence  │        │
│  │   tion Det   │ │   Check      │ │   Check      │ │   Scoring    │        │
│  └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘        │
│        │                  │                │               │                │
│        └──────────────────┴────────────────┴───────────────┘                │
│                             │                                               │
│                             ▼                                               │
│                    ┌─────────────────┐                                      │
│                    │ Final Response  │                                      │
│                    └─────────────────┘                                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           RESPONSE                                          │
│  {                                                                          │
│    "answer": "...",                                                         │
│    "sections": {                                                            │
│      "summary": "...",                                                      │
│      "diabetes_control": "...",                                             │
│      "hypertension_management": "...",                                      │
│      "kidney_considerations": "..."                                         │
│    },                                                                       │
│    "citations": [...],                                                      │
│    "sub_queries": [                                                         │
│      {"id": "q1", "focus": "treatment", "chunks_retrieved": 8},             │
│      {"id": "q2", "focus": "dosage", "chunks_retrieved": 6},                │
│      ...                                                                    │
│    ],                                                                       │
│    "contradictions": [                                                      │
│      {"type": "dosage_conflict", "evidence": "500mg vs 1000mg"}             │
│    ],                                                                       │
│    "confidence": 0.78,                                                      │
│    "complexity_detected": "complex",                                        │
│    "processing_time_ms": 4500                                               │
│  }                                                                          │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Agent System Detail

### 7 Agents

| Agent | Class | File | LLM? | Key Behavior |
|-------|-------|------|------|-------------|
| **Intent Router** | `IntentRouter` | `agents/intent_router.py` | No | Deterministic complexity classification (SIMPLE/MEDIUM/COMPLEX) via regex pattern matching + entity count. Routes SIMPLE queries to `/search`. |
| **Query Decomposer** | `QueryDecomposer` | `agents/query_decomposer.py` | Yes (fallback: rule-based) | LLM-based sub-query generation (3–6 sub-queries). Each sub-query has focus, priority, and metadata. |
| **RAG Agent** | `RAGAgent` | `agents/rag_agent.py` | Yes | Full RAG per sub-query. Escalation detection when corpus insufficient. Confidence: 0.35 score + 0.30 coverage + 0.35 evidence. |
| **Web Search Agent** | `WebSearchAgent` | `agents/web_search_agent.py` | Yes (Gemini Flash fallback) | 3-tier fetch (HTTP→CDP→Gemini). Medical trust filtering. Auto-discovers browser. Conflict detection. |
| **Synthesis Agent** | `SynthesisAgent` | `agents/synthesis_agent.py` | Yes | Merges RAG + web. Conflict resolution. Only runs when BOTH return results. |
| **Citation Validator** | `CitationValidator` | `agents/citation_validator.py` | Yes (fallback: keyword overlap) | Post-gen claim→source validation. Machine-readable citation schema for UI. Detects misattribution. |
| **DocGen Agent** | `DocGenAgent` | `agents/docgen_agent.py` | No | Pure rendering: PDF (reportlab) / DOCX (python-docx). Cannot introduce new errors. |

### Orchestrator (`orchestrator.py`)

`DeepInsightOrchestrator` — Pure coordinator (no inline business logic):

1. **Sanitize** query (dangerous pattern filtering)
2. **Cache check** (Redis)
3. **Intent route** via `IntentRouter` → SIMPLE redirects to `/search`
4. **Decompose** query via `QueryDecomposer` (3–6 sub-queries)
5. **Parallel agents** via `asyncio.gather`: RAG + optional Web Search
6. **Contradiction detection** via `ContradictionDetector`
7. **Synthesis** via `SynthesisAgent` (if both RAG + web returned results)
8. **Citation validation** via `CitationValidator`
9. **Validation pipeline** via `validate_answer()`
10. **Cache write** to Redis
11. **Response** with sections, contradictions, sub-queries, confidence

Thread-safe initialization via async lock. Binds `self.tools = TOOL_REGISTRY` for dynamic tool lookup.

---

## Agent Tools

The pipeline uses **55 tools** from `src/tools/`. All agents share the common toolbelt.

### Agent → Tool Mapping

| Agent | Filesystem | Web Search | Citation | Doc |
|-------|:---:|:---:|:---:|:---:|
| RAG Agent | `save_chunk`, `load_chunk`, `hash_string`, `cache_key` | — | `extract_chunk_ids` | — |
| Web Search Agent | `write_text`, `write_json` (cache results) | `extract_domain`, `is_medical_domain`, `filter_medical`, `rank_by_keywords`, `top_n`, `deduplicate_by_url`, `deduplicate_by_title`, `group_by_domain` | `extract_web_ids` | — |
| Synthesis Agent | — | — | `extract_all_citations`, `extract_citation_markers` | — |
| Citation Validator | — | — | `claim_supported_by_source`, `is_supported`, `find_best_source`, `build_citation_schema` | — |
| DocGen Agent | `make_reports_dir`, `generate_filename`, `cleanup_temp_files` | — | `format_citations_inline`, `count_citations` | `split_sections`, `build_doc_sections`, `generate_pdf`, `generate_docx`, `get_pdf_metadata` |

### Tool Access Pattern

Agents import only what they need (preferred — explicit, greppable):

```python
from src.tools.doctools.generate_pdf import generate_pdf
from src.tools.citationtools.build_citation_schema import build_citation_schema
```

The orchestrator and routes use the central registry for dynamic lookup:

```python
from src.tools import TOOL_REGISTRY, get_tool, call_tool

# orchestrator.py
self.tools = TOOL_REGISTRY

# api/routes/search.py
build_sections = get_tool("build_doc_sections")
render = get_tool("generate_pdf")  # or "generate_docx"
```

### Async vs Sync Tools

| Class | Count | Examples | How agents handle them |
|-------|------:|----------|------------------------|
| Async (coroutine) | 22 | `write_text`, `read_text`, `save_chunk`, `delete_file`, `make_dir`, `load_chunk` | `await tool_fn(...)` |
| Sync | 33 | `extract_domain`, `filter_medical`, `claim_supported_by_source`, `build_citation_schema`, `generate_pdf` | Plain call: `result = tool_fn(...)` |

### `call_tool()` — Auto-Await Dispatch

For dynamic dispatch where the tool name comes from config/request:

```python
from src.tools import call_tool, is_async_tool

# Uniform call — no need to know if the tool is async
result = await call_tool("write_text", "report.md", body, output_dir="/tmp/openinsight_reports")

# Conditional branch when you need to know ahead of time
if is_async_tool("read_text"):
    text = await call_tool("read_text", "report.md")
else:
    text = call_tool("read_text", "report.md")
```

### Safety Guards on Mutating Tools

Every filesystem tool refuses to operate on paths outside `ALLOWED_ROOTS`:

- **`write_*` and `make_dir*`** raise `ValueError` for unsafe paths — treat as hard failure
- **`delete_directory` and `cleanup_temp_files`** raise `PermissionError` unless `confirm=True`
- **`read_*`, `list_*`, `get_file_*`** silently return `None` / `[]` / `0` for unsafe paths

---

## Web Search Agent Detail

### Three-Tier Fetch Strategy

| Tier | Class | Technology | Target Sites | Dependencies |
|------|-------|-----------|-------------|-------------|
| 1 | `HTTPFetcher` | httpx (async) | Static medical sites (80% coverage) | httpx |
| 2 | `CDPBrowser` | Raw WebSocket CDP protocol | JS-heavy sites (Lightpanda/Chrome) | **Zero** (stdlib only) |
| 3 | Gemini Flash | LLM prompt → JSON | Fallback for complex queries | LLM provider |

### Target URL Construction

The Web Search Agent builds targeted URLs based on query content:
- **ICMR** (icmr.gov.in) — Indian clinical guidelines
- **AHA** (heart.org) — American Heart Association
- **NICE** (nice.org.uk) — UK National Institute for Health and Care Excellence
- **WHO** (who.int) — World Health Organization
- **FDA** (fda.gov) — US Food and Drug Administration
- Plus domain-specific sources based on detected entities

### Browser Auto-Discovery

`discover_cdp_url()` checks ports 9222, 9229, 8080 for Lightpanda or Chrome CDP endpoints.

### Medical Trust Filtering

Results are filtered and ranked by domain trust level (1=highest, 5=lowest):
- **Tier 1**: WHO, ICMR, Cochrane, NICE
- **Tier 2**: PubMed, CDC, AHA
- **Tier 3**: Medical journals, StatPearls
- **Tier 4**: Medical news sites
- **Tier 5**: General web results

---

## Intent Router Logic

### Complexity Patterns (regex)

```python
COMPLEX_PATTERNS = {
    r"\bvs\b|\bversus\b|\bcompared to\b",          # Comparisons
    r"\binteract(?:ion|ing|s)?\b",                   # Drug interactions
    r"\bwith\b.*\b(medication|drug|pill)\b",        # Drug + medication
    r"\b(and|with)\b.*\b(diabetes|hypertension|ckd|copd|chf)\b.*\b(and|with)\b",  # Multi-condition
    r"\bcontraindicat(?:ed|ion|ions)\b",             # Contraindications
    r"\bdifferential\b",                             # Differential diagnosis
    r"\bwhat could cause\b",                         # Etiology
    r"\bprotocol\b.*\bversus\b",                    # Protocol comparison
    r"\bguideline\b.*\bconflic",                    # Guideline conflict
}
```

### Sub-Query Templates

```python
SUB_QUERY_TEMPLATES = {
    "diagnostic": ["diagnosis", "symptoms", "differential", "etiology"],
    "therapeutic": ["treatment", "drug_of_choice", "dosage", "regimen"],
    "drug_info": ["interactions", "contraindications", "side_effects", "mechanism"],
    "prognostic": ["prognosis", "outcome", "mortality", "survival"],
    "guideline": ["guidelines", "recommendations", "protocols", "standards"],
    "comparative": ["comparison", "versus", "which_is_better", "efficacy"],
}
```

---

## Model Assignment Strategy

| Agent | Primary Model | Provider | Fallback Model | Fallback Provider |
|-------|--------------|----------|---------------|------------------|
| Orchestrator | `meta/llama-3.1-8b-instruct` | NVIDIA NIM | `gemini-2.0-flash` | Google |
| Intent Router | No LLM (regex + entity count) | — | — | — |
| Query Decomposer | `meta/llama-3.1-70b-instruct` | NVIDIA NIM | Rule-based fallback | — |
| RAG Agent | `meta/llama-3.1-70b-instruct` | NVIDIA NIM | `meta/llama-3.1-70b-instruct` | Groq |
| Web Search | `gemini-2.0-flash` | Google | `gpt-4o-mini` | OpenAI |
| Synthesis | `meta/llama-3.1-70b-instruct` | NVIDIA NIM | — | — |
| Citation Validator | `gpt-4o-mini` | OpenAI | — | — |
| DocGen | No LLM | — | — | — |

---

## Cost Estimates (per query)

| Path | Models Called | Est. Input Tokens | Est. Cost |
|------|---------------|-------------------|-----------|
| RAG-only | Orchestrator + RAG + Citation | ~4K | ~$0.005 |
| RAG + Web | All 7 agents | ~8K | ~$0.012 |
| RAG + Web + DocGen | All 7 + render | ~10K | ~$0.016 |

---

## Configuration

```python
# DeepInsights
deep_insights_enabled = true
deep_insights_max_sub_queries = 6
deep_insights_sub_query_top_k = 8
deep_insights_timeout = 60           # seconds
deep_insights_context_chars = 300

# Contradiction detection
contradiction_detection = true
contradiction_min_chunks = 3
nli_model_name = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"

# Hallucination detection
hallucination_enabled = true
hallucination_threshold = 0.75

# Search pipeline (shared with /search)
top_k_retrieval = 50
top_k_after_fusion = 20
top_k_after_rerank = 8
top_k_final = 6
mmr_lambda = 0.7
```

---

## Performance

| Stage | Typical Time |
|-------|--------------|
| Intent routing | 5–10ms |
| Query decomposition | 200–500ms |
| Parallel RAG + Web | 2–4 seconds |
| Contradiction detection | 100–200ms |
| Synthesis | 1,500–2,500ms |
| Citation validation | ~500ms |
| Full validation | ~500ms |
| **Total** | **~4–6 seconds** |

---

## Debug Endpoint

`GET /deep-insights/route-check?query=...` returns how a query would be routed without executing the full pipeline:

```json
{
  "query": "metformin vs glipizide for diabetes",
  "complexity": "COMPLEX",
  "intent": "THERAPEUTIC",
  "entities": ["metformin", "glipizide", "diabetes"],
  "sub_query_types": ["treatment", "interactions", "comparison"],
  "confidence": 0.95
}
```
