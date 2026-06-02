# Architecture Overview

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CLIENTS                                        │
│   React UI / Mobile App / API Consumers                                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              FASTAPI LAYER                                  │
│                                                                             │
│   ┌──────────────────────┐      ┌──────────────────────┐                    │
│   │   POST /search       │      │  POST /deep-insights │                    │
│   │   Simple RAG         │      │  Multi-Agent         │                    │
│   └──────────┬───────────┘      └──────────┬───────────┘                    │
│              │                             │                                │
│              ▼                             ▼                                │
│   ┌───────────────────────────┐  ┌─────────────────────────────┐            │
│   │    query/search/*         │  │    query/agents/*           │            │
│   │    - cache.py             │  │    - intent_router.py       │            │
│   │    - retriever.py         │  │    - query_decomposer.py    │            │
│   │    - fusion.py            │  │    - deep_insights.py       │            │
│   │    - reranker.py          │  │                             │            │
│   │    - mmr.py               │  │                             │            │
│   └──────────┬────────────────┘  └──────────┬──────────────────┘            │
│              │                              │                               │
│              └──────────────┬───────────────┘                               │
│                             ▼                                               │
│              ┌────────────────────────────────────────┐                     │
│              │        utils/llm_client.py             │                     │
│              │   NVIDIA NIM (Llama 3.1 70B)           │                     │
│              └────────────────────────────────────────┘                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                             DATA LAYER                                      │
│                                                                             │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                     │
│   │  MongoDB    │    │   Milvus    │    │   Redis     │                     │
│   │             │    │  (Vectors)  │    │  (Cache)    │                     │
│   │ - documents │    │             │    │             │                     │
│   │ - chunks    │    │ - dense idx │    │ - search    │                     │
│   │             │    │ - sparse idx│    │ - embedding │                     │
│   └─────────────┘    └─────────────┘    └─────────────┘                     │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                     VECTORSTORE LAYER                               │   │
│   │   VectorStore (interface) → MilvusVectorStore (implementation)      │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Component Responsibilities

### API Layer (`src/api/`)
- **main.py** - FastAPI app, CORS, lifespan management
- **routes/search.py** - Simple RAG endpoint
- **routes/deep_insights.py** - Multi-agent endpoint

### Query Layer (`src/query/`)

| Module | Responsibility |
|--------|----------------|
| `search/cache.py` | Redis caching for queries and embeddings |
| `search/retriever.py` | Hybrid dense + sparse retrieval |
| `search/fusion.py` | Reciprocal Rank Fusion |
| `search/reranker.py` | Cross-encoder reranking |
| `search/mmr.py` | Maximal Marginal Relevance |
| `search/query_understanding.py` | Intent classification |
| `search/query_rewriter_v2.py` | LLM-based query rewriting |
| `agents/intent_router.py` | Simple vs complex detection |
| `agents/query_decomposer.py` | Sub-query generation |
| `agents/deep_insights.py` | Multi-agent orchestration |
| `validation/validator.py` | Answer validation |

### Ingestion Layer (`src/ingestion/`)

| Module | Responsibility |
|--------|----------------|
| `pipeline.py` | Main orchestration with dead letter queue, OCR fallback, retry logic |
| `run_ingestion.py` | CLI entry point |
| `tasks.py` | Celery tasks for distributed processing |
| `scheduler.py` | Scheduled ingestion jobs |
| `checkpoint.py` | Checkpoint/resume support for long-running jobs |
| `dedupe.py` | Document deduplication |
| `metadata.py` | Metadata enrichment |
| `quality.py` | Quality scoring |
| `vector_indexer.py` | Vector indexing to Milvus |
| `document_db.py` | Document storage operations |
| `deduplication.py` | Advanced deduplication logic |
| `validation.py` | Document validation |
| `monitoring.py` | Metrics and monitoring |
| `llamaindex_integration.py` | Parent-child chunk retrieval (LlamaIndex patterns) |
| `parsers/*` | PDF/XML/HTML parsing (GROBID 0.9.0, ICMR, PubMed, OCR, etc.) |
| `celery_app.py` | Distributed task queue |

### Utility Layer (`src/utils/`)

| Module | Responsibility |
|--------|----------------|
| `pubmed_client.py` | Shared NCBI Entrez API client with rate limiting and retry logic |
| `date_utils.py` | Date parsing and year extraction from medical literature |
| `text_utils.py` | Text cleaning, keyword extraction, quality assessment |

### ML Layer (`src/ml/`)

| Module | Responsibility |
|--------|----------------|
| `chunking/chunker.py` | Hierarchical text chunking |
| `embedding/embedder.py` | Multi-provider embedding (local/HF/Cohere), returns `(embeddings, failed_indices)` |
| `ner.py` | Named entity recognition |

### Data Layer (`src/data/`)

| Module | Responsibility |
|--------|----------------|
| `mongo/doc_store.py` | Document storage |
| `vector/vector_store.py` | Vector storage (Milvus) |

### Config (`src/config/`)
- **settings.py** - All settings from environment variables, including GROBID timeout/retry config, embedding provider selection, dead letter queue settings

### Services (`src/services/`)
- **llm_client.py** - NVIDIA NIM API client

### Tools Package (`src/tools/`)

A shared toolkit of 55 standalone functions used by agents in the DeepInsights pipeline and by the `/search/document` route. Replaces the old `src/query/deepinsight/agents/tools.py` wrapper module.

```
src/tools/
├── __init__.py                 # TOOL_REGISTRY, get_tool(), list_tools()
├── filesystemtools/            # 27 tools / 9 files  — file I/O, hashing, truncation
├── websearchtools/             # 13 tools / 6 files  — result filtering, dedup, ranking
├── citationtools/              #  8 tools / 4 files  — citation ID extraction & validation
└── doctools/                   #  8 tools / 7 files  — PDF/DOCX generation, section building
```

#### Design Principles

- **One tool = one function in one file.** No wrapper classes, no `__init__` boilerplate.
- **Explicit parameters only.** Tools do not depend on a hidden `settings` object — they take what they need as args.
- **Direct imports over registry lookups.** Agents import only the functions they use:
  ```python
  from src.tools.filesystemtools.write_file import write_text
  from src.tools.doctools.generate_pdf import generate_pdf
  ```
- **`TOOL_REGISTRY` in `__init__.py`** maps tool name → function for dynamic lookup by the orchestrator and routes:
  ```python
  from src.tools import TOOL_REGISTRY, get_tool, list_tools

  self.tools = TOOL_REGISTRY  # orchestrator
  pdf_tool = get_tool("generate_pdf")
  ```

#### `aiofiles` Fallback

`aiofiles` is **optional**. Every tool that does I/O tries `import aiofiles` and falls back to stdlib `open()` / `os` calls when it is not installed. No new pip dependency is required for the tools package to work.

#### Why the Old Wrapper Was Removed

The previous `src/query/deepinsight/agents/tools.py` exposed each tool as a class with a `get_tool(settings)` factory. This made call sites verbose, made tests require constructing a wrapper, and made adding a tool a multi-file edit. The function-based layout is greppable, testable in isolation, and makes the dependency graph obvious from import statements.

#### Safety & Hardening

A shared safety layer in `src/tools/safety.py` centralizes path validation, filename sanitization, and allowed-roots enforcement so individual tool files stay small.

**`ALLOWED_ROOTS`** is the default write/delete sandbox:

```python
ALLOWED_ROOTS = [
    Path("/tmp") / "openinsight_temp",
    Path("/tmp") / "openinsight_reports",
    Path("/tmp"),  # broadest fallback
]
```

Every filesystem tool refuses to operate on paths that don't resolve under one of these roots (or a caller-supplied override). The behavior matrix is risk-aware so call sites get predictable feedback:

| Risk class | Tools | Behavior on unsafe path |
|------------|-------|-------------------------|
| Mutating write | `write_text` / `write_json` / `write_bytes` / `make_dir*` | **Raise `ValueError`** |
| Read | `read_text` / `read_json` / `read_bytes` | Return `None` + log warning |
| Edit-in-place | `append_to_file` / `replace_in_file` / `insert_at_line` | Return `False` (return type is now `bool`) |
| Inspect | `list_files` / `list_by_extension` / `get_file_size` / `get_file_info` | Return `[]` / `0` / `None` |
| Destructive | `delete_file` / `delete_directory` / `cleanup_temp_files` | **Raise `PermissionError`** unless `confirm=True` |

Two small helpers are the workhorses — `sanitize_filename(name, max_length=200)` for user-supplied names (strips separators, NUL, control chars; collapses underscores; falls back to `"unnamed"`), and `is_path_safe(path, allowed_roots=None)` which accepts `str | Path`, resolves it, and returns whether it sits under an allowed root. `ensure_safe_path` is the raising variant and `require_confirm(operation, path, confirm)` is the escape hatch for destructive ops that genuinely need to go outside the sandbox.

#### `TOOL_REGISTRY` Metadata

The registry moved from a flat `name → callable` map to a metadata dict per tool, so callers can decide whether to `await` a result without invoking it.

```python
TOOL_REGISTRY: dict[str, dict] = {
    "write_text": {
        "fn": <coroutine>,           # the actual function
        "async": True,               # iscoroutinefunction() at load time
        "desc": "Write text ...",    # short human-readable summary
        "name": "write_text",
    },
    # ...
}
```

New helpers in `src/tools/__init__.py` keep this fast and uniform:

| Helper | Returns | When to use |
|--------|---------|-------------|
| `get_tool(name)` | The callable | Backward-compat direct invocation |
| `get_tool_meta(name)` | Full metadata dict | Inspect `async` / `desc` before calling |
| `is_async_tool(name)` | `bool` | Branch on `await` without calling |
| `call_tool(name, *args, **kwargs)` | Result of the call (auto-awaited) | Uniform dispatch from the orchestrator |
| `list_async_tools()` / `list_sync_tools()` | Sorted `list[str]` | Diagnostics, test selection |
| `TOOL_FUNCTIONS` | `dict[str, Callable]` | Flat alias kept for any code that still iterates a `name → fn` map |

**Final counts:** 22 async tools (all filesystem I/O), 33 sync tools (web / citation / doc helpers and the in-memory `hash_*` / `truncate_*` tools) — 55 total. The async flag is derived automatically from `inspect.iscoroutinefunction` at module load, so adding a new tool stays one-liner simple.

#### Citation Plugin

`claim_supported_by_source` is a token-overlap heuristic by default — and its limitations (no semantic similarity, lexical negation only, no stemming, no numeric claim handling, domain-word inflation) are now documented in the module docstring of `src/tools/citationtools/validate_claim.py`. To replace it with a better check (embeddings, NLI, etc.), register one and it becomes the primary signal — the token-overlap result is still attached under `result["fallback"]` for transparency:

```python
from src.tools.citationtools.validate_claim import register_semantic_check

def my_check(claim: str, source: str) -> dict | None:
    score = my_embedder.similarity(claim, source)
    if score is None:
        return None  # fall through to token overlap
    return {"supported": score > 0.7, "confidence": score, "method": "embedding"}

register_semantic_check(my_check)
# Later: register_semantic_check(None)  # revert to the built-in
```

See `src/tools/README.md` for the full safety and plugin reference.

---

## Logging

All modules use `loguru` for structured logging. Key prefixes:
- `[pipeline]` — Ingestion pipeline orchestration
- `[HFEmbedder]` / `[CohereEmbedder]` — Embedding providers
- `[PubMedClient]` — NCBI API interactions

---

## Data Flow

### Query Flow
1. Request hits `/search` or `/deep-insights`
2. Query analyzed for intent
3. Check Redis cache
4. Vector search (dense + sparse)
5. Results fused and reranked
6. LLM generates answer
7. Answer validated
8. Response returned with citations

### Ingestion Flow
1. Files loaded from directory
2. Parsed (PDF/XML → text) with OCR fallback and retry logic
3. Failed documents stored to dead letter queue for reprocessing
4. Deduplication check
5. Metadata enriched
6. Chunked (350 tokens, 50 overlap)
7. Quality scored
8. Embedded (PubMedBERT or configurable provider) — returns `(embeddings, failed_indices)`
9. Failed embeddings filtered out before indexing
10. Indexed to Milvus with Zilliz verification (expected vs actual count)
11. Stored in MongoDB
12. Metrics saved
13. Checkpoint updated for resume support

---

## Configuration

All config via environment variables (`.env`):
- Database connections
- API keys (NVIDIA, NCBI, HuggingFace, Cohere)
- Model names and embedding provider selection
- Pipeline parameters (top_k, batch sizes, thresholds)
- Feature flags (hyde, contradiction detection, tracing)
- GROBID settings (timeout, max retries, retry delay, health check timeout)
- Dead letter queue configuration
- CPU-based worker defaults (75% of cores)

See `.env.example` for all options.

### Constants (`src/constants/`)
Magic values consolidated here to avoid duplication across modules.