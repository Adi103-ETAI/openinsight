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
| `pipeline_v4.py` | Main orchestration |
| `chunker_v3.py` | Hierarchical text chunking |
| `embedder_v2.py` | Dual embedding (dense + sparse) |
| `dedupe.py` | Document deduplication |
| `metadata_v2.py` | Metadata enrichment |
| `quality.py` | Quality scoring |
| `parsers/*` | PDF/XML parsing (GROBID, ICMR, PubMed, etc.) |
| `celery_app.py` | Distributed task queue |

### Core (`src/core/`)
- **config.py** - All settings from environment variables

### Utils (`src/utils/`)
- **llm_client.py** - NVIDIA NIM API client

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
2. Parsed (PDF/XML → text)
3. Deduplication check
4. Metadata enriched
5. Chunked (350 tokens, 50 overlap)
6. Quality scored
7. Embedded (PubMedBERT)
8. Indexed to Milvus
9. Stored in MongoDB
10. Metrics saved

---

## Configuration

All config via environment variables (`.env`):
- Database connections
- API keys (NVIDIA, NCBI)
- Model names
- Pipeline parameters (top_k, batch sizes, thresholds)
- Feature flags (hyde, contradiction detection)

See `.env.example` for all options.