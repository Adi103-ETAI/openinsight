# OpenInsight Codebase Audit Report — restruct Branch

**Date:** May 30, 2026  
**Branch:** `restruct` (509bfcd)  
**Working Directory:** openinsight/  
**Scope:** Full read-only codebase audit

---

## Table of Contents
1. [Architecture Overview](#1-architecture-overview)
2. [RAG Pipeline Audit](#2-rag-pipeline-audit)
3. [Bug & Problem List](#3-bug--problem-list)
4. [Architectural Improvements (restruct)](#4-architectural-improvements-restruct)
5. [Remaining Architectural Problems](#5-remaining-architectural-problems)
6. [Research Vault Gap Analysis](#6-research-vault-gap-analysis)
7. [Report Generation Plan](#7-report-generation-plan)

---

## 1. Architecture Overview

### 1.1 What Changed in restruct

The `restruct` branch is a significant rewrite of the `main` branch. Key changes:

| Area | Main Branch | restruct Branch |
|------|-------------|-----------------|
| Config | Monolithic `src/core/config.py` (60+ fields, env-only) | 3-file JSON layer (`config.base.json` + `config.{env}.json`) + `.env` secrets, loaded via Pydantic Settings |
| Embedding | Two legacy modules (`embeddings.py`, `embedder_v2.py`) | Pluggable `src/ml/embedding/embedder.py` with Local/HuggingFace/Cohere providers |
| Reranking | Single `CrossEncoderReranker` | Multi-provider: Local/HuggingFace/Cohere via `create_reranker()` factory |
| Ingestion | `pipeline_v4.py` + `run_ingestion_v2.py` | Clean `pipeline.py` + `run_ingestion.py` with checkpoint/resume + dead letter queue |
| Chunking | `chunker_v3.py` (in ingestion) | `src/ml/chunking/chunker.py` (in ml module) |
| NER | `ner.py` (in ingestion) | `src/ml/ner.py` (in ml module) |
| Data Access | Scattered across ingestion module | Centralized `src/data/mongo/` and `src/data/vector/` |
| LLM Client | `src/utils/llm_client.py` | `src/services/llm_client.py` |
| Tests | 8 test files, no shared fixtures | 16 test files with `conftest.py` (shared fixtures, mocks) |
| Utilities | Duplicated across parsers | Centralized `src/utils/` (date_utils, text_utils, metrics, pubmed_client) |
| New | — | LlamaIndex parent-child chunking, Kaggle notebook, checkpoint/resume, dead letter queue |

### 1.2 Directory Structure

```
openinsight/
├── config.base.json              # Shared non-secret defaults (136 fields)
├── config.kaggle.json            # Kaggle/Colab GPU overrides
├── config.production.json        # Production overrides (HuggingFace + Cohere)
├── pyproject.toml                # Build/test/lint config
├── docker-compose.yml            # Milvus + etcd + minio + MongoDB + Redis
├── .env.example                  # Secrets template
├── src/
│   ├── config/
│   │   ├── settings.py           # Pydantic Settings with JSON layer loading
│   │   └── logging_config.py     # Loguru config (console + rotating file)
│   ├── constants/
│   │   └── __init__.py           # EvidenceBoost, RecencyBoost, RRF_K, thresholds
│   ├── data/
│   │   ├── mongo/
│   │   │   ├── connection.py     # MongoDB connection pool manager
│   │   │   └── doc_store.py      # MongoDocStoreV2 (documents_v2 + chunks_v2)
│   │   └── vector/
│   │       └── vector_store.py   # Legacy compat layer (search, hybrid_search)
│   ├── ml/
│   │   ├── embedding/
│   │   │   └── embedder.py       # BaseEmbedder + Local/HF/Cohere implementations
│   │   ├── chunking/
│   │   │   └── chunker.py        # HierarchicalChunkerV3
│   │   └── ner.py                # Medical NER (regex + scispaCy)
│   ├── services/
│   │   └── llm_client.py         # NVIDIA NIM client (httpx)
│   ├── ingestion/
│   │   ├── pipeline.py           # Main ingestion orchestrator
│   │   ├── run_ingestion.py      # CLI entry point
│   │   ├── checkpoint.py         # Resume-on-failure (MongoDB-backed)
│   │   ├── llamaindex_integration.py  # Parent-child chunking
│   │   ├── parsers/              # 8 source-specific parsers
│   │   ├── dedupe.py, deduplication.py, validation.py, quality.py
│   │   ├── metadata.py, monitoring.py, scheduler.py, tasks.py
│   │   └── vector_indexer.py, celery_app.py
│   ├── query/
│   │   ├── search/
│   │   │   ├── retriever.py      # HybridRetriever + parent-child retrieval
│   │   │   ├── fusion.py         # RRF with evidence/recency boosts
│   │   │   ├── reranker.py       # Multi-provider reranking
│   │   │   ├── mmr.py            # Maximal Marginal Relevance
│   │   │   ├── cache.py          # Redis search cache
│   │   │   ├── context_builder.py # Context assembly + parent support
│   │   │   └── query_understanding.py # Intent + entity + filter
│   │   ├── agents/
│   │   │   ├── intent_router.py  # Complexity routing
│   │   │   ├── query_decomposer.py # Sub-query generation
│   │   │   └── deep_insights.py  # Orchestrator
│   │   ├── validation/
│   │   │   ├── validator.py, hallucination_detector.py
│   │   │   ├── citation_checker.py, confidence_scorer.py
│   │   │   └── medical_safety.py
│   │   ├── contradiction_detector.py
│   │   └── prompts.py
│   ├── vectorstore/              # Backend-agnostic vector store abstraction
│   │   ├── base.py, types.py, filters.py, registry.py
│   │   └── backends/milvus_store.py
│   ├── utils/
│   │   ├── date_utils.py, text_utils.py, metrics.py, pubmed_client.py
│   └── api/
│       ├── main.py               # FastAPI + middleware + health checks
│       └── routes/
│           ├── search.py         # POST /search (full RAG pipeline)
│           └── deep_insights.py  # POST /deep-insights
├── tests/                        # 16 test files + conftest.py
├── scripts/                      # run.py, seed_*.py, maintainence.py
├── notebooks/                    # Kaggle ingestion notebook
└── prompts/                      # system.md, query_rewrite.md
```

### 1.3 Data Flow

```
INGESTION:
  Files → Parsers (PubMed, Cochrane, WHO, CDC, StatPearls, ICMR, GROBID, OCR)
    → Deduplication (MongoDB doc_id + content hash + title similarity)
    → Metadata Enrichment (evidence level, doc type, India relevance)
    → Hierarchical Chunking (doc_summary → table → paragraph, 350-token target)
    → Quality Scoring (clinical content weight, evidence bonus, noise penalty)
    → Dual Embedding (PubMedBERT 768d dense + TF-IDF sparse)
    → Vector Store Upsert (Milvus, UUID5 deterministic IDs)
    → MongoDB Storage (documents_v2 + chunks_v2)
    → Checkpoint Resume (MongoDB-backed)

QUERY:
  Doctor Query → Query Understanding (intent, entities, filters, HyDE)
    → Redis Cache Check
    → Hybrid Retrieval (dense + sparse parallel, top_k=50)
    → RRF Fusion (k=60, evidence/recency boosts, top_n=20)
    → Cross-Encoder Reranking (top_k=8)
    → MMR Diversity (λ=0.7, top_k=6)
    → Context Assembly + LLM Generation (NVIDIA NIM)
    → Validation (hallucination, citation, safety, confidence)
    → Response
```

---

## 2. RAG Pipeline Audit

### 2.1 Ingestion Pipeline (Improved)

**Strengths over main:**
- **Checkpoint/resume** — can restart failed ingestion without re-processing
- **Dead letter queue** — failed documents stored for retry, not silently dropped
- **Pluggable embedder** — Local (GPU), HuggingFace (free API), Cohere (paid API)
- **Shared utilities** — date parsing, text processing, PubMed client centralized
- **Kaggle notebook** — can run ingestion on free GPU
- **Cleaner pipeline** — single `pipeline.py` replaces `pipeline_v4.py`

**Remaining issues:**
- **Token estimation** `len(text) // 4` still rough for medical text
- **Section header regex** only matches English headers
- **No PDF page limit** — huge PDFs could OOM
- **Scheduler disabled** — all scheduled ingestion is no-ops

### 2.2 Query Pipeline (Improved)

**Strengths over main:**
- **Multi-provider reranking** — Local/HuggingFace/Cohere
- **Parent-child retrieval** — two-stage: child precision + parent context
- **Thread-safe component creation** — async locks with double-check pattern
- **Query input sanitization** — XSS and SQL injection prevention
- **Enhanced context builder** — supports parent sections with child references

**Remaining issues:**
- **NLI contradiction detection** — improved (PubMedBERT-based) but still has import issues
- **Entity extraction fallback** — only 3 conditions (diabetes, hypertension, metformin)
- **No retry logic** for NIM API calls
- **HyDE uses raw httpx** instead of NVIDIAClient

### 2.3 Embedding System (New)

**Architecture:**
```
BaseEmbedder (ABC)
├── LocalEmbedder       — SentenceTransformers (S-PubMedBert-MS-MARCO, 768d)
├── HuggingFaceEmbedder — HF Inference API (free tier, rate-limited)
└── CohereEmbedder      — Cohere Embed API (1024d, different dimension!)
```

**Issue:** Cohere produces 1024-dim vectors while Local produces 768d. Switching providers requires re-embedding the entire corpus. The `dimension()` method returns provider-specific values, but the Milvus collection schema is fixed at 768d in `config.base.json`.

### 2.4 Reranking System (New)

**Architecture:**
```
BaseReranker (ABC)
├── LocalReranker       — BAAI/bge-reranker-v2-m3 (cross-encoder)
├── HuggingFaceReranker — HF Inference API (text-classification)
└── CohereReranker      — Cohere Rerank API (rerank-english-v3.0)
```

**Improvement over main:** Configurable `reranker_max_chars` is now respected (was hardcoded to 512 on main).

---

## 3. Bug & Problem List

### CRITICAL (Will crash or produce wrong results)

| # | File:Line | Description |
|---|-----------|-------------|
| C1 | `src/ml/embedding/embedder.py:embed_texts()` | **`embed_batch` returns `(ndarray, list[int])` tuple** but `embed_texts()` calls `.tolist()` on the result directly. This will crash with `AttributeError: 'tuple' object has no attribute 'tolist'`. Same issue in `hallucination_detector.py` lines ~188, ~190 where `model.embed_batch()` result is used directly as embeddings without unpacking the tuple. |
| C2 | `src/query/validation/hallucination_detector.py` | **`embed_batch` tuple unpacking missing** — calls `model.embed_batch(chunk_texts)` and `model.embed_batch(sentences)` but doesn't unpack the `(ndarray, failed_indices)` tuple. The `util.cos_sim()` call will receive a tuple instead of a tensor, causing a crash. |
| C3 | `src/ingestion/scheduler.py` | **All scheduled ingestion is disabled** — `_PARSER_BASED_INGESTION_AVAILABLE = False`, all sync functions are no-ops. `start_scheduler()` creates a scheduler but registers zero jobs. The scheduler feature is effectively dead code. |

### HIGH (Functional issues, resource leaks)

| # | File:Line | Description |
|---|-----------|-------------|
| H1 | `src/query/agents/query_decomposer.py` | **Missing `Optional` import** — uses `Optional[httpx.AsyncClient]` in `__init__` and `Optional[DecompositionResult]` in return type, but only imports `Any` from typing. Will crash with `NameError: name 'Optional' is not defined`. |
| H2 | `src/query/validation/medical_safety.py` | **Missing `Optional` import** — uses `Optional[str]` in `SafetyWarning` dataclass but `Optional` is not imported. Will crash with `NameError`. |
| H3 | `src/data/vector/vector_store.py:17` | **Module-level `settings = get_settings()`** — executed at import time. If settings aren't ready (e.g., missing `.env`), the entire module fails to import. Should use lazy initialization. |
| H4 | `src/ingestion/dedupe.py:get_existing_doc_ids()` | **Creates new motor client per call** — `AsyncIOMotorClient(mongo_url)` called inside the method instead of reusing `self.mongo`. Resource leak. (Same bug as main branch.) |
| H5 | `src/query/search/retriever.py:_generate_hyde()` | **Raw httpx call** instead of using `NVIDIAClient` — duplicates HTTP logic, doesn't reuse connection pool. (Same as main branch.) |
| H6 | `src/query/agents/deep_insights.py:_synthesize_answer()` | **Raw httpx call to NIM API** — same duplication issue. (Same as main branch.) |
| H7 | `src/query/contradiction_detector.py` | **`torch` imported at module level** — if torch isn't installed, the entire module fails to import. Should be lazy-imported inside `_run_nli()`. |
| H8 | `src/ingestion/monitoring.py:get_storage_stats()` | **Queries wrong collections** — uses `documents`/`chunks` but v2 pipeline uses `documents_v2`/`chunks_v2`. Stats will be wrong. (Same as main branch.) |

### MEDIUM (Code quality, potential issues at scale)

| # | File:Line | Description |
|---|-----------|-------------|
| M1 | `src/api/main.py` | **CORS `allow_origins=["*"]`** — wide open, insecure for production. |
| M2 | `src/ml/embedding/embedder.py` | **Hash collision risk** — 50K vocab with `abs(hash(term)) % 50000` silently merges weights for colliding terms. (Same as main.) |
| M3 | `src/data/vector/vector_store.py:build_sparse_vector` | **Non-deterministic `hash()`** — Python 3.3+ randomizes hash seeds. Sparse vectors from ingestion won't match query-time vectors. (Same as main.) |
| M4 | `src/query/search/fusion.py:52` | **Mutates `chunk.score`** on original objects — side effect. (Same as main.) |
| M5 | `src/ml/embedding/embedder.py` | **Cohere produces 1024-dim vectors** but Milvus collection schema is fixed at 768d. Switching to Cohere will cause dimension mismatch errors. |
| M6 | `src/query/search/query_understanding.py` | **`rewritten_query` field always None** — query rewriting was folded into understanding but LLM-based rewriting was dropped. The `rewritten_query` field in `QueryAnalysis` is never populated. |
| M7 | `src/vectorstore/backends/milvus_store.py` | **Filter DSL only supports AND (`must`)** — no OR or NOT support. (Same as main.) |
| M8 | `src/query/search/context_builder.py` | **No individual chunk truncation** — a single very long chunk could exceed the context budget. (Same as main.) |
| M9 | `src/services/llm_client.py` | **Return type annotation wrong** — `chat_completions()` annotated as `-> dict[str, Any]` but returns `str`. (Same as main.) |
| M10 | `src/ingestion/validation.py` | **`VALID_SOURCE_TYPES` hardcoded** — may miss new source types. (Same as main.) |
| M11 | `src/constants/__init__.py` | **LEVEL_5 boost = 1.10** higher than LEVEL_3 (1.05) and LEVEL_4 (1.00). Medically unusual. (Same as main.) |
| M12 | `src/query/search/mmr.py` | **`embedder.embed_batch` called synchronously** — could block event loop. (Same as main.) |
| M13 | `src/query/agents/intent_router.py` | **Entity extraction fallback** only handles 3 conditions. (Same as main.) |
| M14 | `src/ingestion/tasks.py` | **`asyncio.run()` inside Celery task** — event loop conflict risk. (Same as main.) |
| M15 | `src/config/settings.py` | **`class Config`** used instead of `model_config = SettingsConfigDict(...)` — deprecated in pydantic v2. |

---

## 4. Architectural Improvements (restruct)

### What Was Fixed

1. **Configuration system** — 3-file JSON layer with environment overrides replaces monolithic env-only config. Much cleaner, supports Kaggle/production/development without code changes.

2. **Module duplication eliminated** — `embeddings.py` + `embedder_v2.py` → single `embedder.py`. `chunker_v3.py` → `ml/chunking/chunker.py`. `ner.py` → `ml/ner.py`. Legacy aliases preserved for backward compatibility.

3. **Data access layer** — `src/data/mongo/` and `src/data/vector/` centralize database access instead of scattering across ingestion modules.

4. **Shared utilities** — `date_utils.py`, `text_utils.py`, `metrics.py`, `pubmed_client.py` eliminate duplication across parsers.

5. **Test suite** — 16 test files (up from 8) with shared fixtures, mock embedders/vector stores, and parametrized test data.

6. **Pluggable providers** — Embedding and reranking now support Local/HuggingFace/Cohere via factory pattern.

7. **Checkpoint/resume** — Ingestion can restart from where it failed.

8. **Dead letter queue** — Failed documents stored for retry instead of being silently dropped.

9. **Health checks** — `/health`, `/health/detailed`, `/health/ready` endpoints for monitoring.

10. **Request middleware** — Request ID tracking, timing headers, structured logging.

---

## 5. Remaining Architectural Problems

### Still Present from Main

1. **NLI contradiction detection** — improved (PubMedBERT model) but has import issues and the model loading is fragile.

2. **Two competing dedup modules** — `deduplication.py` (full MongoDB) vs `dedupe.py` (simpler). Pipeline uses `dedupe.py`.

3. **Validation module not called in pipeline** — `validation.py` defines document/chunk validation but `pipeline.py` doesn't call it.

4. **No user/auth model** — system has no concept of users, sessions, or access control.

5. **No retry logic** for NIM API calls — transient 429/503 errors crash requests.

6. **Broad `except Exception`** in multiple places — silently swallows errors.

7. **Module-level settings loading** — `vector_store.py` and `document_db.py` load settings at import time.

### New Issues from restruct

1. **Dimension mismatch risk** — Cohere embedder produces 1024d but Milvus schema is 768d. Switching providers requires schema migration.

2. **Scheduler is dead code** — all scheduled ingestion disabled, but the code and config remain.

3. **Query rewriter dropped** — LLM-based query rewriting was in main but is missing from restruct. The `rewritten_query` field is never populated.

4. **Kaggle notebook coupling** — notebook hardcodes Zilliz Cloud and MongoDB Atlas endpoints, not portable.

---

## 6. Research Vault Gap Analysis

**Status unchanged from main** — the Research Vault is fully implemented in the frontend (Supabase-backed) but the FastAPI backend has zero vault awareness. The restruct branch doesn't change this.

See main audit report Section 6 for full analysis.

---

## 7. Report Generation Plan

**Status unchanged from main** — report generation is unimplemented.

See main audit report Section 7 for full plan.

---

## Summary of Key Findings

1. **restruct is a significant improvement** — config system, pluggable providers, shared utilities, test suite, checkpoint/resume are all meaningful progress.

2. **3 critical bugs remain** — `embed_batch` tuple unpacking will crash hallucination detection and the convenience `embed_texts()` function. Scheduler is dead code.

3. **2 high-severity import errors** — `Optional` not imported in `query_decomposer.py` and `medical_safety.py`. These will crash at import time.

4. **Cohere dimension mismatch** — switching to Cohere embedder (1024d) will break the 768d Milvus schema.

5. **Query rewriter was dropped** — LLM-based rewriting existed on main but is gone from restruct.

6. **Test coverage improved** — 16 test files with shared fixtures, but integration tests are still minimal.
