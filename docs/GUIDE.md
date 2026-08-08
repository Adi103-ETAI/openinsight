# OpenInsight — Quick Start Guide

> A clinical decision support system for Indian physicians
> Last updated: August 2026

---

## What is OpenInsight?

A medical RAG system that answers clinical questions with cited, evidence-based responses from:
- ICMR guidelines
- PubMed research
- WHO/CDC documents
- Indian clinical literature (Cochrane, StatPearls)

**Four API endpoints:**
- **Search** (`/search`) — Fast single-pass RAG with cache, HyDE, fusion, rerank, MMR
- **Search Document** (`/search/document`) — RAG + PDF/DOCX export
- **DeepInsights** (`/deep-insights`) — Multi-agent pipeline (5 agents) for complex cases
- **Vault** (`/vault`) — Research vault for saving and organizing citations
- **Reports** (`/reports`) — Clinical summary and evidence review generation

---

## Quick Start

### 1. Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Copy env file
cp .env.example .env

# Edit .env with your API keys (NVIDIA NIM, MongoDB, Zilliz, etc.)
```

### 2. Start Infrastructure
```bash
# Start MongoDB, Redis, GROBID, and Milvus
docker compose up -d
```

### 3. Run API
```bash
uvicorn src.api.main:app --reload --port 8000
```

### 4. Test
```bash
# Simple search
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "treatment for dengue fever"}'

# DeepInsights (multi-agent)
curl -X POST http://localhost:8000/deep-insights \
  -H "Content-Type: application/json" \
  -d '{"query": "metformin vs glipizide for diabetes with CKD"}'

# Health check
curl http://localhost:8000/health

# API docs
open http://localhost:8000/docs
```

---

## Running Ingestion

### Using the wrapper script (recommended):
```bash
# Basic
python scripts/run.py pubmed ./data/pdfs

# With options
python scripts/run.py icmr ./data/pdfs -w 8 --recreate --stats

# Dry run (test without indexing)
python scripts/run.py who ./pdfs --dry-run

# Interactive mode
python scripts/run.py
```

### Using the module directly:
```bash
python -m src.ingestion.run_ingestion \
  --dir ./data/pdfs \
  --source pubmed \
  --workers 6 \
  --batch-size 10
```

### Available Sources
```
pubmed, icmr, cochrane, nmc_guideline, rssdi, who, cdc, statpearls
```

---

## Options

| Flag | Description | Default |
|------|-------------|---------|
| `-w, --workers` | Parallel workers | 6 |
| `-b, --batch-size` | Files per batch | 10 |
| `--recreate` | Recreate vector index | false |
| `--dry-run` | Parse only, no indexing | false |
| `--skip-embed` | Skip embedding | false |
| `--skip-index` | Skip vector indexing | false |
| `--stats` | Show detailed stats | false |
| `--resume` | Resume from checkpoint | true |
| `--reset` | Reset checkpoint | false |

> **Warning**: `--recreate` drops the Milvus collection. Never use against shared/production data.

---

## Project Structure

```
src/
├── api/                # FastAPI endpoints + middleware + models
│   ├── main.py         # App, lifespan, middleware, /health, /metrics
│   ├── routes/         # search, deep_insights, vault, reports
│   ├── middleware/     # rate_limit.py
│   └── models/         # vault.py (Pydantic models)
├── config/             # Settings (JSON + .env hybrid), logging config
├── constants/          # Shared magic values (EvidenceBoost, RecencyBoost, RRF_K)
├── data/               # MongoDB stores + vector store compat layer
│   ├── mongo/          # connection, doc_store, vault_store
│   └── vector/         # vector_store.py (legacy compat)
├── ingestion/          # Data pipeline
│   ├── pipeline.py     # Main orchestration
│   ├── parsers/        # PDF/XML parsers (GROBID 0.8.0, ICMR, PubMed, OCR, etc.)
│   ├── llamaindex_integration.py  # Parent-child chunk retrieval
│   └── ...             # tasks, scheduler, checkpoint, dedupe, quality, etc.
├── ml/                 # ML components
│   ├── chunking/       # HierarchicalChunkerV3
│   ├── embedding/      # Multi-provider embedder (local/HF/Cohere)
│   └── ner.py          # Named entity recognition + content classification
├── query/              # Query pipeline
│   ├── search/         # RAG: cache, retriever, fusion, reranker, mmr, query_understanding, context_builder
│   ├── deepinsight/    # Multi-agent: orchestrator + 5 agents (RAG, Web, Synthesis, Citation, DocGen)
│   ├── validation/     # Answer validation: hallucination, citation, safety, confidence
│   └── contradiction_detector.py
├── reports/            # Clinical report generation (PDF/DOCX)
├── services/           # LLM provider system + browser automation
│   ├── llm/            # 10 providers, router, registry, providers.json
│   └── browser/        # HTTPFetcher, CDPBrowser, ContentExtractor
├── tools/              # 55 standalone agent tools
│   ├── safety.py       # ALLOWED_ROOTS sandbox, path validation
│   ├── filesystemtools/  # 27 tools (read/write/edit/delete/list/hash)
│   ├── websearchtools/   # 13 tools (domain, snippet, filter, rank, dedup)
│   ├── citationtools/    # 8 tools (extract, validate, schema, find)
│   └── doctools/         # 8 tools (PDF, DOCX, sections, metadata)
└── utils/              # Shared utilities
    ├── pubmed_client.py   # NCBI Entrez API client
    ├── date_utils.py      # Date parsing helpers
    ├── text_utils.py      # Text processing helpers
    └── metrics.py         # MetricsCollector, DependencyHealthChecker, TimingMiddleware
```

---

## Docker Compose Services

| Service | Image | Port | Purpose |
|---------|-------|------|---------|
| MongoDB | mongo:7 | 27017 | Document + chunk + vault storage |
| Redis | redis:7-alpine | 6379 | Search cache + Celery broker |
| GROBID | lfoppiano/grobid:0.8.0 | 8070 | PDF/XML parsing (4GB heap) |
| Milvus | milvusdb/milvus:v2.5.4 | 19530 | Hybrid dense+sparse vector search |
| etcd | quay.io/coreos/etcd:v3.5.5 | 2379 | Milvus metadata |
| MinIO | minio/minio:RELEASE.2023-03-13 | 9000/9001 | Milvus object storage |

---

## Testing

```bash
# All tests (excluding GPU tests)
pytest tests/ -v -m "not requires_gpu"

# Unit tests only
pytest tests/ -v -m unit

# Integration tests (requires docker compose up -d)
pytest tests/ -v -m integration

# Tool registry tests
pytest tests/test_tools.py -v

# With coverage
pytest tests/ --cov=src --cov-report=term-missing
```

### Test Markers

| Marker | Description |
|--------|-------------|
| `unit` | Unit tests |
| `integration` | Integration tests |
| `slow` | Slow running tests |
| `requires_gpu` | Tests that need GPU |
| `requires_network` | Tests that need API keys |
| `requires_mongodb` | Tests that need MongoDB (docker) |
| `requires_grobid` | Tests that need GROBID (docker) |

---

## LLM Providers

10 providers configured in `src/services/llm/providers.json`:

| Provider | Default Model | Auth |
|----------|--------------|------|
| NVIDIA NIM | meta/llama-3.1-70b-instruct | Bearer token |
| OpenAI | gpt-4o | Bearer token |
| Anthropic | claude-sonnet-4-20250514 | x-api-key |
| Google Gemini | gemini-2.0-flash | Query param |
| Together AI | meta-llama/Llama-3.3-70B-Instruct-Turbo | Bearer token |
| OpenRouter | meta-llama/llama-3.3-70b-instruct | Bearer token |
| Groq | llama-3.3-70b-versatile | Bearer token |
| AIML API | meta-llama/llama-3.1-70b-instruct | Bearer token |
| Cohere | command-r-plus | Bearer token |
| Ollama | llama3.1:70b | None (local) |

---

## Useful Commands

```bash
# Start API
uvicorn src.api.main:app --reload --port 8000

# Run ingestion
python scripts/run.py <source> <directory>

# Seed scripts
python scripts/seed_pubmed.py
python scripts/seed_icmr.py

# Smoke test vector DB
python scripts/zilliz_smoke.py

# Run tests
pytest tests/ -v

# Lint
ruff check src tests
black --check src tests
isort --check-only src tests
```

---

*Built by SentArc Labs*
