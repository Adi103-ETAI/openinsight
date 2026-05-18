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