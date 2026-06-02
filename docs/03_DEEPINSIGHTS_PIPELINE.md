# DeepInsights Pipeline

## When to Use DeepInsights

Use DeepInsights for complex clinical queries that require:
- Drug interaction checks (e.g., "Can I give metformin with ACE inhibitors?")
- Differential diagnosis (e.g., "What could cause this presentation?")
- Protocol conflicts (e.g., "ICMR vs WHO guidelines for dengue")
- Multi-condition management (e.g., "DM with HTN and CKD")

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
│  │  Complex Query Detection (Rule-based + Entity Count)                │    │
│  │                                                                     │    │
│  │  Complex Patterns:                                                  │    │
│  │    - "vs / versus" (comparisons)                                    │    │
│  │    - "interaction" (drug interactions)                              │    │
│  │    - Multi-condition ("X and Y and Z")                              │    │
│  │    - "contraindicated"                                              │    │
│  │    - "differential"                                                 │    │
│  │                                                                     │    │
│  │  Output:                                                            │    │
│  │    - complexity: SIMPLE / MEDIUM / COMPLEX                          │    │
│  │    - confidence: 0.0-1.0                                            │    │
│  │    - detected_intent: therapeutic/diagnostic/etc.                   │    │
│  │    - sub_query_types: [treatment, dosage, interactions, etc.]       │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
                 [SIMPLE]                       [COMPLEX]
               Use standard                      Continue
                /search
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       QUERY DECOMPOSER                                      │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  LLM-based Decomposition                                            │    │
│  │                                                                     │    │
│  │  Input: "treatment for diabetes with hypertension and CKD"          │    │
│  │                                                                     │    │
│  │  Output: 3-6 Sub-queries:                                           │    │
│  │    - q1: "diabetes treatment options" (focus: treatment)            │    │
│  │    - q2: "hypertension medication dosage" (focus: dosage)           │    │
│  │    - q3: "drug interactions diabetes hypertension" (focus: inter)   │    │
│  │    - q4: "CKD contraindications diabetes drugs" (focus: contra)     │    │
│  │    - q5: "ICMR guidelines diabetes hypertension" (focus: guide)     │    │
│  │    - q6: "diabetes CKD management protocols" (focus: protocol)      │    │
│  │                                                                     │    │
│  │  Fallback: Rule-based if LLM fails                                  │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      PARALLEL RETRIEVAL                                     │
│                                                                             │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐      │
│   │ Sub-q 1  │  │ Sub-q 2  │  │ Sub-q 3  │  │ Sub-q 4  │  │ Sub-q 5  │      │
│   │          │  │          │  │          │  │          │  │          │      │
│   │  Dense   │  │  Dense   │  │  Dense   │  │  Dense   │  │  Dense   │      │
│   │  +       │  │  +       │  │  +       │  │  +       │  │  +       │      │
│   │  Sparse  │  │  Sparse  │  │  Sparse  │  │  Sparse  │  │  Sparse  │      │
│   └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘      │
│        │             │             │             │             │            │
│        └─────────────┴─────────────┴─────────────┴─────────────┘            │
│                              │                                              │
│                              ▼                                              │
│              ┌───────────────────────────────────────┐                      │
│              │  All chunks combined (with dedup)     │                      │
│              │  Total: sub_queries × top_k           │                      │
│              └───────────────────────────────────────┘                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    CONTRADICTION DETECTION                                  │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Check for conflicting evidence                                     │    │
│  │                                                                     │    │
│  │  Method 1: NLI Model (future)                                       │    │
│  │  Method 2: Keyword-based (current)                                  │    │
│  │    - "improve" vs "worsen"                                          │    │
│  │    - "recommended" vs "not recommended"                             │    │
│  │    - "effective" vs "ineffective"                                   │    │
│  │    - dosage conflicts                                               │    │
│  │                                                                     │    │
│  │  Output: List of contradiction pairs                                │    │
│  │    - type: treatment_conflict/dosage_conflict/outcome_conflict      │    │
│  │    - evidence: conflicting keywords                                 │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        ANSWER SYNTHESIS                                     │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  LLM Synthesis Prompt                                               │    │
│  │                                                                     │    │
│  │  "Synthesize the following evidence into a comprehensive answer:    │    │
│  │                                                                     │    │
│  │   Original Query: {query}                                           │    │
│  │   Synthesis Guidance: {synthesis_prompt}                            │    │
│  │   Evidence: {all_chunks}                                            │    │
│  │                                                                     │    │
│  │   Include:                                                          │    │
│  │   - Key findings from each sub-query                                │    │
│  │   - Recommendations with citations                                  │    │
│  │   - Warnings about contradictions                                   │    │
│  │   - Confidence assessment                                           │    │
│  │  "                                                                  │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
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

## Agent Tools

The pipeline now runs **5 agents** (RAG, Web Search, Synthesis, Citation Validator, DocGen). All of them share a common toolbelt exposed by `src/tools/`, plus the orchestrator binds the full registry as `self.tools = TOOL_REGISTRY` for dynamic lookup.

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
from src.tools import TOOL_REGISTRY, get_tool, list_tools

# orchestrator.py
self.tools = TOOL_REGISTRY

# api/routes/search.py
build_sections = get_tool("build_doc_sections")
render = get_tool("generate_pdf")  # or "generate_docx"
```

### `POST /search/document` — Document Export Outside the DeepInsights Flow

The standard `/search` endpoint returns JSON. The new `/search/document` endpoint runs the same search, then:

1. `build_doc_sections()` structures the answer into sections
2. `generate_pdf()` or `generate_docx()` (chosen by the request) renders the file
3. Returns the file as a streaming download

This means a basic RAG search can now produce a downloadable clinical report without going through the full multi-agent DeepInsights pipeline.

### Async vs Sync Tools

The 55 tools in `src/tools/` split cleanly along the I/O boundary. Every agent that calls a tool needs to know which is which so it `await`s the right ones.

| Class | Count | Examples | How agents handle them |
|-------|------:|----------|------------------------|
| Async (coroutine) | 22 | `write_text`, `read_text`, `save_chunk`, `delete_file`, `make_dir`, `load_chunk` | `await tool_fn(...)` |
| Sync | 33 | `extract_domain`, `filter_medical`, `claim_supported_by_source`, `build_citation_schema`, `generate_pdf` | Plain call: `result = tool_fn(...)` |

In practice, the sync bucket dominates the hot path: citation extraction, web result filtering, document section building, and PDF/DOCX rendering are all CPU-light and synchronous. The async bucket is exclusively filesystem I/O (the write/read/edit/list/make/save/delete family plus the `save_chunk` / `load_chunk` pair).

### `call_tool()` — Auto-Await Dispatch

For the orchestrator and the `/search/document` route — where the tool name comes from a config / request rather than a static import — use `call_tool()` to dispatch uniformly. It looks up the tool, checks the async flag, and awaits only when needed:

```python
from src.tools import call_tool, is_async_tool

# Uniform call — no need to know if the tool is async
result = await call_tool("write_text", "report.md", body, output_dir="/tmp/openinsight_reports")

# Conditional branch when you need to know ahead of time (e.g. for logging)
if is_async_tool("read_text"):
    text = await call_tool("read_text", "report.md")
else:
    text = call_tool("read_text", "report.md")
```

Prefer **direct imports** (`from src.tools.doctools.generate_pdf import generate_pdf`) when the tool name is known at the call site — they're greppable and let the type checker see the signature. Reach for `call_tool()` when the name is dynamic.

### Safety Guards on Mutating Tools

Every filesystem tool now refuses to operate on paths outside `ALLOWED_ROOTS` (`/tmp/openinsight_temp`, `/tmp/openinsight_reports`, `/tmp`). Agents calling mutating tools must be aware of the new failure modes:

- **`write_*` and `make_dir*`** raise `ValueError` for unsafe paths — the agent's retry logic should treat this as a hard failure, not a transient error
- **`delete_directory` and `cleanup_temp_files`** raise `PermissionError` unless the caller passes `confirm=True`. The DocGen agent's cleanup path is the most likely caller; it must either keep its targets inside the allowed roots or pass `confirm=True` explicitly
- **`read_*`, `list_*`, `get_file_*`** silently return `None` / `[]` / `0` for unsafe paths — agents that need to distinguish "file missing" from "path rejected" should check the log output

If an agent genuinely needs to read or write outside the sandbox (e.g. an ingestion path that lands PDFs in a corpus directory), pass an `allowed_roots=` argument to the safety helpers in `src/tools/safety.py`. The default allowlist is intentionally narrow.

---

## Intent Router Logic

```python
COMPLEX_PATTERNS = {
    # Comparisons
    r"\bvs\b|\bversus\b|\bcompared to\b",
    
    # Drug interactions
    r"\binteract(?:ion|ing|s)?\b",
    r"\bwith\b.*\b(medication|drug|pill)\b",
    
    # Multi-condition
    r"\b(and|with)\b.*\b(diabetes|hypertension|ckd|copd|chf)\b.*\b(and|with)\b",
    
    # Contraindications
    r"\bcontraindicat(?:ed|ion|ions)\b",
    
    # Differential
    r"\bdifferential\b",
}
```

### Complexity Calculation
- 2+ patterns → COMPLEX (95% confidence)
- 1 pattern + 3+ entities → COMPLEX (90%)
- 1 pattern + 2 entities → MEDIUM (70%)
- 4+ entities → COMPLEX (85%)
- Default → SIMPLE (75%)

---

## Configuration

```python
DEEP_INSIGHTS_ENABLED = true
DEEP_INSIGHTS_MAX_SUB_QUERIES = 6
DEEP_INSIGHTS_SUB_QUERY_TOP_K = 8
DEEP_INSIGHTS_TIMEOUT = 60

CONTRADICTION_DETECTION = true
CONTRADICTION_MIN_CHUNKS = 3
```

---

## Performance

| Stage | Typical Time |
|-------|--------------|
| Intent routing | 5-10ms |
| Query decomposition | 200-500ms |
| Parallel retrieval | 800-1500ms |
| Contradiction detection | 100-200ms |
| Answer synthesis | 1500-2500ms |
| **Total** | **~4-5 seconds** |