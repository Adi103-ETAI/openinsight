# Architecture Overview

> Last updated: August 2026
> Reflects codebase on `restruct` branch at commit `78f4d6b`

## System Architecture

```
              Clients (React UI / Mobile / API consumers)
                                │
                                ▼
+-----------------------------------------------------------------------+
|                  FastAPI — src/api/main.py                            |
|     /search        /deep-insights         /vault          /reports    |
|     /health        /health/detailed       /health/ready   /metrics    |
+--------+----------------+-------------------+---------------+---------+
         |                |                   |               |
         v                v                   |               v
+----------------+  +------------------+      |        src/reports/
| Search RAG     |  | DeepInsight      |      |        generators.py
| src/query/     |  | Orchestrator     |      |        pdf_renderer.py
| search/*       |  | src/query/       |      |
| cache          |  | deepinsight/*    |      |
| retriever      |  | agents + tools   |      |
| fusion/rerank  |  +--------+---------+      |
+--------+-------+           |                |
         |                   v                |
         +--------> src/services/llm/* ------>+ 10 LLM providers
         |          router + providers.json   | (NVIDIA NIM / OpenAI /
         v                                     Anthropic / Google /
+--------+---------+----------+----------+    Cohere / Together /
| MongoDB          | Milvus   | Redis    |    OpenRouter / Groq /
| docs, chunks,    | dense +  | search & |    AIML / Ollama)
| vault            | sparse   | embed    |
| src/data/mongo/  | vectors  | cache    |
|                  | src/vectorstore/     |
+--------+---------+----------+----------+
         ^
         |
+--------+------------------------------------------------+
| Ingestion — src/ingestion/pipeline.py                   |
| parsers → dedupe → chunk → embed → Milvus + MongoDB     |
| Celery tasks via src/ingestion/celery_app.py            |
+---------------------------------------------------------+
```

---

## API Layer (`src/api/`)

### Endpoints

| Method | Path | Handler | Description |
|--------|------|---------|-------------|
| POST | `/search` | `search_endpoint` | Single-pass RAG with cache, HyDE, fusion, rerank, MMR, validation |
| POST | `/search/document` | `search_document_endpoint` | RAG + PDF/DOCX export (no DeepInsights) |
| POST | `/deep-insights` | `deep_insights_endpoint` | Multi-agent pipeline (7 agents) |
| GET | `/deep-insights/route-check` | `route_check` | Debug: shows how a query would be routed |
| CRUD | `/vault/items` | vault items | Research vault item management |
| CRUD | `/vault/collections` | vault collections | Research collection management |
| POST/DELETE | `/vault/items/{id}/collections/{id}` | linking | Item-collection association |
| POST | `/reports/generate` | `generate_report` | Clinical summary or evidence review (JSON/PDF) |
| GET | `/reports/types` | — | Lists available report types and formats |
| GET | `/health` | `health` | Basic health with degradation status |
| GET | `/health/detailed` | `health_detailed` | MongoDB + Milvus + Redis connectivity check |
| GET | `/health/ready` | `health_ready` | Kubernetes readiness probe (503 if unhealthy) |
| GET | `/metrics` | `metrics` | Request metrics and latency percentiles |

### Request/Response Models

- **SearchRequest**: `query` (1–500 chars), `top_k` (1–50, default 6), `save_to_vault` (bool), `vault_tags` (list[str])
- **SearchResponse**: `answer, citations, query_intent, chunks_retrieved, cache_hit, confidence_score, recommendation, unverified_claims, safety_warnings, evidence_distribution, is_safe, needs_disclaimer, confidence_breakdown`
- **DeepInsightsRequest**: `query, top_k=8, force_deep=False`
- **DeepInsightsResponse**: `answer, sections, citations, sub_queries, sub_query_results, contradictions, sources_used, cached, timed_out, synthesis_result, citation_validation, confidence, complexity_detected, processing_time_ms`
- **ReportRequest**: `report_type` (clinical_summary/evidence_review), `format` (json/pdf), plus query/answer/citations/safety data

### Middleware Stack (order matters)

1. **RequestIDMiddleware** — Generates/propagates `X-Request-ID` via `ContextVar`, binds to loguru per-request
2. **CORSMiddleware** — Origins from `CORS_ORIGINS` env (default `localhost:3000,5173`), allows `X-Request-ID`, `X-User-ID` headers
3. **RateLimitMiddleware** — Token-bucket per client IP:
   - `/search`: ~10 req/min, burst 5
   - `/deep-insights`: ~5 req/min, burst 3
   - `/reports`: ~10 req/min, burst 5
   - Default: 60 req/min, burst 10
   - Excluded: `/health`, `/metrics`, `/docs`

### Lifespan

- **Startup**: Initializes `QueryUnderstanding`, `HybridRetriever`, `get_reranker()`, `SearchCache` as singletons on `app.state.search_components`. Any init failure logs a warning and continues (graceful degradation).
- **Shutdown**: Closes Redis cache (`aclose()`), NIM client, all dynamic LLM provider clients (`close_all_clients()`), LLM router.

---

## Query Layer (`src/query/`)

### Search Pipeline (`src/query/search/`)

| Module | Responsibility |
|--------|----------------|
| `cache.py` | `SearchCache` — Redis-backed caching for search results, embeddings, and reranked outputs. Key format: `openinsight:{version}:{operation}:{sha256_hash}`. Uses `aioredis` with JSON-safe serialization for Enums and Pydantic models. |
| `retriever.py` | `HybridRetriever` — Parallel dense + sparse vector search with HyDE support. `retrieve()` returns `(dense_results, sparse_results)`. Also has `retrieve_with_parent()` for hierarchical parent-child retrieval. |
| `fusion.py` | `reciprocal_rank_fusion()` — RRF with k=60, evidence-level boosting (`EvidenceBoost`), recency boosting (`RecencyBoost`), deduplication by chunk_id. Marks `retrieval_source` as "both"/"dense"/"sparse". |
| `reranker.py` | `BaseReranker` (ABC) with three implementations: `LocalReranker` (cross-encoder, default `BAAI/bge-reranker-v2-m3`), `HuggingFaceReranker` (HF Inference API), `CohereReranker` (Cohere Rerank API). Factory: `create_reranker(provider)`. Singleton: `get_reranker()`. |
| `mmr.py` | `maximal_marginal_relevance()` — Diversity filtering. λ=0.7 by default (70% relevance, 30% diversity). Respects user's requested `top_k`. |
| `query_understanding.py` | `QueryUnderstanding` — Pattern-based intent classification (`QueryIntent` enum: DIAGNOSTIC/THERAPEUTIC/PROGNOSTIC/DRUG_INFO/GUIDELINE/GENERAL), entity extraction (spaCy or fallback), metadata filter inference, query expansion via `MEDICAL_SYNONYMS`, optional LLM-based query rewriting. |
| `context_builder.py` | `assemble_context()` — Formats chunks with evidence level labels and India-relevant flags. `build_citation_list()` for citation extraction. Also has parent-child variants. |

### DeepInsights Pipeline (`src/query/deepinsight/`)

| Module | Responsibility |
|--------|----------------|
| `orchestrator.py` | `DeepInsightOrchestrator` — Pure coordinator: sanitize → cache → intent route → decompose → parallel agents (RAG + optional web) → contradiction detect → synthesize → validate → cache write → respond. Binds `self.tools = TOOL_REGISTRY` for dynamic tool lookup. |
| `agents/rag_agent.py` | `RAGAgent` — Full RAG per sub-query: cache → parallel retrieve → RRF → rerank → MMR → context → LLM → cache write. Escalation detection when corpus is insufficient. Confidence: 0.35 score + 0.30 coverage + 0.35 evidence. |
| `agents/web_search_agent.py` | `WebSearchAgent` — Three-tier web search: Tier 1 HTTPFetcher (static), Tier 2 CDPBrowser (JS-heavy), Tier 3 Gemini Flash fallback. Medical trust filtering, conflict detection. |
| `agents/synthesis_agent.py` | `SynthesisAgent` — Merges RAG + web results. Conflict resolution between corpus and web. Only runs when BOTH RAG and web return results. |
| `agents/citation_validator.py` | `CitationValidator` — Post-generation claim→source validation. Outputs machine-readable citation schema for UI tooltips and "show sources" panel. |
| `agents/docgen_agent.py` | `DocGenAgent` — PDF (reportlab) / DOCX (python-docx) rendering. No LLM — renders exactly what synthesis produced. |
| `agents/intent_router.py` | `IntentRouter` — Deterministic complexity classification (SIMPLE/MEDIUM/COMPLEX) via pattern matching + entity count. No LLM. |
| `agents/query_decomposer.py` | `QueryDecomposer` — LLM-based sub-query generation (3–6 sub-queries) with rule-based fallback. Each sub-query has focus, priority, and metadata. |

### Validation (`src/query/validation/`)

| Module | Responsibility |
|--------|----------------|
| `validator.py` | `validate_answer()` — Orchestrates pipeline: hallucination → citation → safety → confidence → recommendation. `enhance_response()` merges validation into API response. |
| `medical_safety.py` | `check_safety()` — Regex patterns for treatments, dosages, contraindications, drug interactions, monitoring, off-label use. Hardcoded `DANGEROUS_COMBINATIONS` (warfarin+aspirin, methotrexate+NSAID, etc.) and `MONITORING_DRUGS` (19 drugs). |
| `hallucination_detector.py` | `detect_hallucinations()` — Sentence-level semantic similarity via `sentence_transformers.util.cos_sim`, entity grounding, numerical claim verification. Threshold from config (default 0.75). |
| `confidence_scorer.py` | `score_confidence()` — 6-component weighted score: 0.25 base + 0.15 citation + 0.15 evidence + 0.20 grounding + 0.15 quality + 0.10 consistency − safety_penalty. |
| `citation_checker.py` | `check_citations()` — Verifies citations exist in MongoDB, checks trust level, evidence level, freshness (max 5 years). `TRUSTED_SOURCES`: Cochrane/WHO/NICE(5), PubMed/ICMR/CDC(4), StatPearls/Medline/UpToDate(3). |
| `contradiction_detector.py` | `ContradictionDetector` — NLI model (PubMedBERT-based) with keyword fallback. Detects treatment/dosage/outcome conflicts. |

---

## Services Layer (`src/services/`)

### LLM Provider System (`src/services/llm/`)

**10 providers, fully config-driven** — edit `providers.json` to add providers, no Python code changes needed.

| Provider | Auth Type | API Type | Default Model |
|----------|-----------|----------|---------------|
| NVIDIA NIM | bearer | openai | meta/llama-3.1-70b-instruct |
| OpenAI | bearer | openai | gpt-4o |
| Anthropic | x-api-key | openai (compatible) | claude-sonnet-4-20250514 |
| Google Gemini | query_param | google | gemini-2.0-flash |
| Together AI | bearer | openai | meta-llama/Llama-3.3-70B-Instruct-Turbo |
| OpenRouter | bearer | openai | meta-llama/llama-3.3-70b-instruct |
| Groq | bearer | openai | llama-3.3-70b-versatile |
| AIML API | bearer | openai | meta-llama/llama-3.1-70b-instruct |
| Cohere | bearer | cohere | command-r-plus |
| Ollama | none | ollama | llama3.1:70b |

**Key components:**
- `base.py` — `BaseLLMClient` (ABC) with `chat_completions() → str`, `completions() → str`, `chat_completions_detailed() → LLMResponse`
- `registry.py` — `create_llm_client(provider, model, api_key)`, `get_llm_client()`, `register_provider()`. Singleton cache per provider+model.
- `router.py` — `LLMRouter` with `AgentRole` enum (DECOMPOSER, RETRIEVER, SYNTHESIZER, VALIDATOR, CITATION, GENERAL). Load-balanced routing with health tracking (3-strike cooldown, exponential backoff). `get_client_for_agent(role)`.
- `providers/` — One module per provider: `openai.py`, `anthropic.py`, `google.py`, `cohere.py`, `nvidia.py`, `ollama.py`, `openai_compatible.py`
- `llm_client.py` (legacy) — `get_nim_client()` backward-compat wrapper → `get_llm_client()`

### Browser Automation (`src/services/browser/`)

| Module | Responsibility |
|--------|----------------|
| `http_fetcher.py` | `HTTPFetcher` — Concurrent httpx fetch for static medical sites. Regex-based HTML parsing (no BeautifulSoup dependency). Tier 1 of web search. |
| `cdp_browser.py` | `CDPBrowser` — Raw WebSocket CDP protocol client, **zero external dependencies**. Auto-discovers Lightpanda/Chrome at ports 9222, 9229, 8080. Tier 2. |
| `content_extractor.py` | `ContentExtractor` — Unified content extraction from fetched pages. |

---

## Data Layer (`src/data/`)

### MongoDB (`src/data/mongo/`)

| Module | Responsibility |
|--------|----------------|
| `connection.py` | `get_mongo_client()` — Singleton `AsyncIOMotorClient` with connection pool (maxPoolSize=50, minPoolSize=5). `get_mongo_db()`, `close_mongo_client()`. |
| `doc_store.py` | `MongoDocStoreV2` — Collections `documents_v2` and `chunks_v2`. `store_document()` (upsert by doc_id), `store_chunks()` (bulk write), `get_document()`, `get_chunk()`. |
| `vault_store.py` | `VaultStore` — Collections `vault_items` and `vault_collections`. Full CRUD with indexes on user_id+created_at, user_id+item_type, user_id+tags, source_url. |

### Vector Store (`src/vectorstore/`)

| Module | Responsibility |
|--------|----------------|
| `base.py` | `VectorStore` (ABC) — `ensure_collection`, `drop_collection`, `upsert_points`, `search_dense`, `search_sparse`, `health_check` |
| `types.py` | `SparseVector` (frozen), `VectorPoint`, `ScoredPoint` — Shared type definitions |
| `filters.py` | `FilterOperator` (EQ/IN/GT/GTE/LT/LTE), `FilterCondition`, `FilterExpression` — Typed filter building |
| `registry.py` | `get_vector_store()` — Singleton, dispatches by `vector_backend` setting (only `milvus` supported) |
| `backends/milvus_store.py` | `MilvusVectorStore` — Hybrid dense+sparse, 768-dim, COSINE dense + IP sparse. Collection: `openinsight_chunks`. Filter field whitelist via `ALLOWED_FILTER_FIELDS`. |
| `src/data/vector/vector_store.py` | Legacy compatibility layer — all calls delegate to `get_vector_store()` |

---

## Ingestion Layer (`src/ingestion/`)

| Module | Responsibility |
|--------|----------------|
| `pipeline.py` | `IngestionPipeline` — Main orchestration: parse → chunk → embed → validate → quality score → index → checkpoint → store. ThreadPoolExecutor for concurrent batch ingestion. Dead letter queue for failed documents. `tenacity` retry (3 attempts, exponential backoff). |
| `run_ingestion.py` | CLI entry point: `python -m src.ingestion.run_ingestion` |
| `tasks.py` | Celery tasks for distributed processing |
| `scheduler.py` | Scheduled ingestion jobs |
| `checkpoint.py` | Checkpoint/resume support for long-running jobs |
| `deduplication.py` / `dedupe.py` | Document deduplication (content hash + title similarity) |
| `metadata.py` | Metadata enrichment (year, evidence level, journal, DOI, PMID, India-relevance, drug dosing detection) |
| `quality.py` | Quality scoring with high/low value pattern detection |
| `validation.py` | Document and chunk validation |
| `vector_indexer.py` | Vector indexing to Milvus with Zilliz verification (expected vs actual count) |
| `document_db.py` | MongoDB document storage operations |
| `monitoring.py` | Metrics and monitoring for ingestion runs |
| `llamaindex_integration.py` | Parent-child chunk retrieval (LlamaIndex patterns) — `HierarchicalChunkParser`, `ParentChildIndexer`, `ParentChildRetriever` |
| `parsers/*` | PDF/XML/HTML parsing: `grobid.py` (GROBID 0.9.0 with configurable timeout/retry, `/api/health` endpoint), `pubmed.py`, `icmr.py`, `cochrane.py`, `who.py`, `cdc.py`, `statpearls.py`, `ocr.py` |
| `celery_app.py` | Distributed task queue configuration |

### Available Ingestion Sources

`pubmed`, `icmr`, `cochrane`, `nmc_guideline`, `rssdi`, `who`, `cdc`, `statpearls`

---

## ML Layer (`src/ml/`)

| Module | Responsibility |
|--------|----------------|
| `chunking/chunker.py` | `HierarchicalChunkerV3` — Sentence-window splitting with overlap (target 350 tokens, max 500, overlap 50, min 80). Abbreviation protection. Table extraction. Creates doc_summary + per-section + table chunks. |
| `embedding/embedder.py` | `BaseEmbedder` (ABC) with 3 implementations: `LocalEmbedder` (SentenceTransformers, S-PubMedBert-MS-MARCO), `HuggingFaceEmbedder` (HF Inference API), `CohereEmbedder` (Cohere Embed API). All return `(embeddings: ndarray, failed_indices: list[int])`. Shared sparse vector computation with medical tokenization. Factory: `create_embedder(provider)`. Singleton: `get_embedder()`. |
| `ner.py` | `extract_entities()` (diseases, drugs, symptoms, dosages, contraindications, patient_populations, outcomes), `classify_content_type()`, `infer_study_type()` with evidence level mapping. scispaCy when available, rule-based fallback. |

---

## Reports (`src/reports/`)

| Module | Responsibility |
|--------|----------------|
| `models.py` | `ClinicalSummaryReport`, `EvidenceReviewReport` — Pydantic models with metadata, key findings, evidence summary, safety warnings, confidence assessment, disclaimer |
| `generators.py` | `generate_clinical_summary()`, `generate_evidence_review()` — Report content generation |
| `pdf_renderer.py` | `render_report()` — PDF rendering via reportlab with fallbacks |

---

## Tools Package (`src/tools/`)

55 standalone functions across 4 subpackages, replacing the old `src/query/deepinsight/agents/tools.py` wrapper module.

```
src/tools/
├── __init__.py                 # TOOL_REGISTRY, get_tool(), call_tool(), is_async_tool()
├── safety.py                   # ALLOWED_ROOTS, sanitize_filename(), is_path_safe(), ensure_safe_path()
├── filesystemtools/            # 27 tools / 9 files  — file I/O, hashing, truncation
├── websearchtools/             # 13 tools / 6 files  — result filtering, dedup, ranking
├── citationtools/              #  8 tools / 4 files  — citation ID extraction & validation
└── doctools/                   #  8 tools / 7 files  — PDF/DOCX generation, section building
```

### Design Principles

- **One tool = one function in one file.** No wrapper classes, no `__init__` boilerplate.
- **Explicit parameters only.** No hidden `settings` object dependency.
- **Direct imports over registry lookups** (preferred for greppability):
  ```python
  from src.tools.filesystemtools.write_file import write_text
  from src.tools.doctools.generate_pdf import generate_pdf
  ```
- **`TOOL_REGISTRY`** maps name → `{fn, async, desc, name}` metadata dict for dynamic lookup.
- **`call_tool(name, *args, **kwargs)`** — Auto-await dispatch for the orchestrator and routes.
- **22 async tools** (filesystem I/O) + **33 sync tools** (web/citation/doc + hash/truncate) = **55 total**.

### Safety & Hardening

`ALLOWED_ROOTS` sandbox: `/tmp/openinsight_temp`, `/tmp/openinsight_reports`, `/tmp`

| Risk class | Tools | Behavior on unsafe path |
|------------|-------|-------------------------|
| Mutating write | `write_text` / `write_json` / `write_bytes` / `make_dir*` | **Raise `ValueError`** |
| Read | `read_text` / `read_json` / `read_bytes` | Return `None` + log warning |
| Edit-in-place | `append_to_file` / `replace_in_file` / `insert_at_line` | Return `False` |
| Inspect | `list_files` / `list_by_extension` / `get_file_size` / `get_file_info` | Return `[]` / `0` / `None` |
| Destructive | `delete_file` / `delete_directory` / `cleanup_temp_files` | **Raise `PermissionError`** unless `confirm=True` |

### Citation Plugin

`claim_supported_by_source` defaults to token-overlap heuristic. Limitations (no semantic similarity, lexical negation only, no stemming, no numeric claim handling) are documented. Register a better check via `register_semantic_check()`:

```python
from src.tools.citationtools.validate_claim import register_semantic_check

def my_check(claim: str, source: str) -> dict | None:
    score = my_embedder.similarity(claim, source)
    if score is None:
        return None
    return {"supported": score > 0.7, "confidence": score, "method": "embedding"}

register_semantic_check(my_check)
```

When a registered check returns non-`None`, it becomes the primary signal and the token-overlap result is kept under `result["fallback"]` for transparency.

---

## Utility Layer (`src/utils/`)

| Module | Responsibility |
|--------|----------------|
| `pubmed_client.py` | Shared NCBI Entrez API client with rate limiting and retry logic |
| `date_utils.py` | Date parsing and year extraction from medical literature |
| `text_utils.py` | Text cleaning, keyword extraction, quality assessment |
| `metrics.py` | `MetricsCollector` (in-memory request metrics, latency percentiles, 10k cap), `DependencyHealthChecker` (MongoDB/Milvus/Redis health checks), `TimingMiddleware` |

---

## Config (`src/config/`)

### `settings.py` — `Settings` class (pydantic-settings)

**Loading priority** (highest first): Constructor kwargs → OS env vars → `.env` file → `config.base.json` + `config.{APP_ENV}.json` → field defaults

~80+ settings fields organized by category:
- **LLM**: `llm_default_provider="nvidia"`, 10+ API key fields, agent→provider mappings
- **MongoDB**: URL, pool sizes (50 max/5 min), timeouts
- **Vector DB**: `vector_backend="milvus"`, URI, dimensions=768, collection names
- **Redis**: URL
- **PubMed**: API key, email, rate limits
- **Embeddings**: model=`pritamdeka/S-PubMedBert-MS-MARCO`, dim=768, provider (local/huggingface/cohere)
- **Reranker**: model=`BAAI/bge-reranker-v2-m3`, provider (local/huggingface/cohere)
- **Search pipeline**: `top_k_retrieval=50`, `top_k_after_fusion=20`, `top_k_after_rerank=8`, `top_k_final=6`, `mmr_lambda=0.7`
- **DeepInsights**: timeout=60s, max 6 sub-queries, context_chars=300
- **Hallucination**: threshold=0.75, enabled=True
- **Chunking**: target=350 tokens, max=500, overlap=50, min=80
- **Ingestion**: workers=4, batch_size=10, max_retries=3, max_chunks_per_doc=100
- **Dedup**: title_similarity=0.9, content_hash_length=16
- **Quality**: score_threshold=0.3
- **Dead letter**: enabled=True, collection=`failed_documents`
- **Cache**: TTLs (search=1800s, rerank=3600s, embedding=1800s)

`get_settings()` → `@lru_cache` singleton

### `logging_config.py` — `configure_loguru()`

Centralized loguru configuration with request ID support via `ContextVar`.

---

## Constants (`src/constants/`)

Consolidated magic values to avoid duplication across modules:

- **`EvidenceBoost`** — Level-specific boost scores: 1a→1.35, 1b→1.25, 2a→1.15, 2b→1.10, 3→1.05, 4→1.00, 5→1.10
- **`RecencyBoost`** — Year-based boosts: 2026→1.12, 2025→1.10, 2024→1.08, 2023→1.05, 2022→1.03, pre-2022→1.00
- **`RRF_K`** = 60
- **`CACHE_KEY_PREFIX_LENGTH`** = 16
- **`DEFAULT_MAX_WORKERS`** = 4
- **`LLM_TIMEOUT`** = 60.0
- **`EVIDENCE_LEVELS`** mapping

---

## Logging

All modules use `loguru` for structured logging with request ID propagation. Key prefixes:
- `[pipeline]` — Ingestion pipeline orchestration
- `[HFEmbedder]` / `[CohereEmbedder]` — Embedding providers
- `[PubMedClient]` — NCBI API interactions
- `request_id` — Bound per-request via `logger.bind(request_id=...)` in `RequestIDMiddleware`

---

## Data Flow

### Query Flow (Simple Search)
1. Request hits `/search` — query sanitized (XSS, SQL injection, control chars)
2. Cache check (Redis, hash of query + filters)
3. Query understanding (intent classification + entity extraction + metadata filters)
4. Optional LLM query rewrite
5. Hybrid vector search (dense + sparse) with optional HyDE
6. Reciprocal Rank Fusion (k=60)
7. Cross-encoder reranking (top_k=8)
8. MMR diversity filtering (top_k=6)
9. Context assembly with evidence level labels
10. LLM answer generation (NVIDIA NIM Llama 3.1 70B)
11. Answer validation (hallucination + citation + safety + confidence)
12. Optional vault auto-save
13. Cache write
14. Response with citations, confidence, recommendation

### Query Flow (DeepInsights)
1. Intent routing (SIMPLE → redirect to /search, MEDIUM/COMPLEX → continue)
2. Query decomposition (3–6 sub-queries)
3. Parallel execution: RAG agent + Web Search agent (`asyncio.gather`)
4. Contradiction detection across retrieved chunks
5. Synthesis (merge RAG + web, resolve conflicts)
6. Citation validation (claim→source mapping)
7. Validation pipeline (hallucination + safety + confidence)
8. Response with sections, contradictions, sub-queries, confidence

### Ingestion Flow
1. Files loaded from directory
2. Parsed (PDF/XML → text) — GROBID with configurable timeout/retry, OCR fallback
3. Failed documents stored to dead letter queue
4. Deduplication (content hash + title similarity)
5. Metadata enrichment (year, evidence level, journal, DOI, PMID, India-relevance, drug dosing)
6. Chunked (target 350 tokens, 50 overlap, min 80)
7. Quality scored (high/low value pattern detection)
8. Embedded (S-PubMedBert or configured provider) — returns `(embeddings, failed_indices)`
9. Failed embeddings filtered out before indexing
10. Indexed to Milvus with Zilliz verification (expected vs actual count)
11. Stored in MongoDB (documents_v2 + chunks_v2 collections)
12. Metrics saved
13. Checkpoint updated for resume support

---

## Configuration

### Config Files

| File | Purpose |
|------|---------|
| `config.base.json` | Non-secret defaults for all environments (committed to Git) |
| `config.production.json` | Production overrides: `milvus_cloud=true`, `embed_provider=huggingface`, `rerank_provider=cohere`, `hallucination_threshold=0.80` |
| `config.kaggle.json` | Kaggle/Colab environment overrides |
| `.env` | Secrets only: API keys, passwords, connection strings (NEVER committed) |

### Environment Variables

Key env vars (see `.env.example` for full list):
- `MONGODB_URL` — MongoDB connection string
- `VECTOR_URI` / `VECTOR_TOKEN` — Milvus/Zilliz Cloud connection
- `NVIDIA_NIM_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, `COHERE_API_KEY` — LLM provider keys
- `HF_API_KEY` — HuggingFace Inference API
- `CORS_ORIGINS` — Allowed CORS origins
- `APP_ENV` — Environment name (determines which config override to load)

---

## Known Issues

| Severity | Location | Issue |
|----------|----------|-------|
| LOW | `main.py` | `TimingMiddleware` class defined but never added to middleware stack (dead code; also exists in `utils/metrics.py`) |
| LOW | `retriever.py` | `retrieve_with_parent()` and parent-child retrieval infrastructure exists but is not called from the search endpoint |
| LOW | `cache.py` | Embedding and rerank cache methods exist but are not called from the search pipeline |
| LOW | `registry.py` | Dedicated `AnthropicClient` class exists but the registry routes Anthropic through `OpenAICompatibleClient` instead |
| LOW | `vault_store.py` | Creates its own `AsyncIOMotorClient` instead of using the shared connection pool from `connection.py` |
