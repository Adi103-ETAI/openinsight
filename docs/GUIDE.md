# OpenInsight — Developer Guide
**SentArc Labs | Pune, India**

> This is your single reference for building OpenInsight. Architecture decisions, how to set up your environment, how each component works, and what to build next.

---

## Table of Contents
1. [What We Are Building](#1-what-we-are-building)
2. [Architecture Overview](#2-architecture-overview)
3. [Tech Stack](#3-tech-stack)
4. [Project Structure](#4-project-structure)
5. [Configuration](#5-configuration)
6. [Constants & Magic Values](#6-constants--magic-values)
7. [Query Pipeline](#7-query-pipeline)
8. [DeepInsights Mode](#8-deepinsights-mode)
9. [Data Ingestion](#9-data-ingestion)
10. [API Endpoints](#10-api-endpoints)
11. [Development](#11-development)
12. [Useful Commands](#12-useful-commands)

---

## 1. What We Are Building

OpenInsight is a clinical decision support platform for Indian physicians. A doctor types a clinical question and gets a cited answer — grounded in ICMR guidelines, live PubMed research, and Indian clinical literature — in under 10 seconds.

**Two query modes:**
- **Simple Search** (`POST /search`) — Fast single-pass RAG for straightforward queries
- **DeepInsights** (`POST /deep-insights`) — Multi-agent orchestration for complex cases (drug interactions, differential diagnosis, protocol conflicts)

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         API LAYER                               │
│  POST /search          POST /deep-insights                      │
└─────────────────────────────────────────────────────────────────┘
                              │
         ┌────────────────────┼────────────────────┐
         ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│  query/search/  │  │  query/agents/  │  │  ingestion/     │
│   Simple RAG    │  │  DeepInsights   │  │  Data Pipeline  │
│                 │  │                 │  │                 │
│ - retriever    │  │ - intent_router │  │ - pipeline_v4   │
│ - fusion       │  │ - query_decomp  │  │ - chunker_v3   │
│ - reranker     │  │ - orchestrator │  │ - embedder_v2  │
│ - mmr          │  │                 │  │ - parsers      │
└────────┬────────┘  └────────┬────────┘  └────────┬────────┘
         │                    │                    │
         └────────────────────┼────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     DATA LAYER                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │   MongoDB    │  │   Milvus     │  │    Redis     │       │
│  │ (Documents)  │  │  (Vectors)   │  │   (Cache)    │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.11 |
| Backend | FastAPI |
| Embeddings | PubMedBERT (`pritamdeka/S-PubMedBert-MS-MARCO`) |
| Reranker | BAAI/bge-reranker-base |
| Vector DB | Milvus / Zilliz Cloud |
| Document DB | MongoDB |
| Cache | Redis |
| LLM | NVIDIA NIM (Llama 3.1 70B) |
| NLP | scispacy (`en_core_sci_md`) |

---

## 4. Project Structure

```
src/
├── api/
│   └── routes/
│       ├── search.py         # Simple search endpoint
│       └── deep_insights.py  # DeepInsights endpoint
│
├── query/
│   ├── search/               # RAG pipeline
│   │   ├── cache.py          # Redis caching
│   │   ├── retriever.py      # Hybrid retrieval
│   │   ├── fusion.py         # RRF fusion
│   │   ├── reranker.py       # Cross-encoder reranking
│   │   ├── mmr.py            # Diversity selection
│   │   ├── query_understanding.py  # Intent classification
│   │   └── query_rewriter_v2.py    # LLM query rewriting
│   │
│   ├── agents/               # DeepInsights
│   │   ├── intent_router.py  # Simple vs complex detection
│   │   ├── query_decomposer.py  # Sub-query generation
│   │   └── deep_insights.py  # Orchestrator
│   │
│   ├── validation/           # Answer validation
│   │   ├── validator.py      # Main validator
│   │   ├── citation_checker.py
│   │   ├── confidence_scorer.py
│   │   ├── hallucination_detector.py
│   │   └── medical_safety.py
│   │
│   └── prompts.py            # LLM prompts
│
├── ingestion/                # Data pipeline
│   ├── pipeline_v4.py        # Main pipeline
│   ├── chunker_v3.py         # Hierarchical chunking
│   ├── embedder_v2.py        # Dual embedding
│   ├── dedupe.py             # Document deduplication
│   ├── parsers/              # PDF/XML parsers
│   │   ├── grobid.py
│   │   ├── icmr.py
│   │   ├── pubmed.py
│   │   └── ...
│   └── celery_app.py         # Distributed ingestion
│
├── core/
│   └── config.py             # Settings (from .env)
│
└── utils/
    └── llm_client.py         # NVIDIA NIM client
```

---

## 5. Configuration

All configuration is driven by environment variables. See `.env.example` for all options.

**Key variables:**
```bash
# LLM
NVIDIA_NIM_API_KEY=your_key
NIM_MODEL=meta/llama-3.1-70b-instruct

# Database
MONGODB_URL=mongodb://localhost:27017
REDIS_URL=redis://localhost:6379
VECTOR_URI=https://your-cluster.zillizcloud.com
VECTOR_TOKEN=your_token

# Models
EMBEDDING_MODEL=pritamdeka/S-PubMedBert-MS-MARCO
RERANKER_MODEL_NAME=BAAI/bge-reranker-base
SPACY_MODEL=en_core_sci_md
```

**Tunable parameters:**
- Retrieval: `TOP_K_RETRIEVAL`, `TOP_K_AFTER_FUSION`, `MMR_LAMBDA`
- Chunking: `CHUNK_TARGET_TOKENS`, `CHUNK_OVERLAP_TOKENS`
- Ingestion: `INGESTION_BATCH_SIZE`, `QUALITY_SCORE_THRESHOLD`

---

## 6. Constants & Magic Values

All magic values are consolidated in `src/core/constants.py`. Import from there to avoid duplication.

**Key constants:**
```python
from src.core.constants import (
    EvidenceBoost,      # Boosts for evidence levels (1a→1.35, 1b→1.25, etc.)
    RecencyBoost,      # Boosts for publication recency
    RRF_K,             # Reciprocal Rank Fusion constant (default: 60)
    DEFAULT_TOP_K,     # Default retrieval k
    MMR_LAMBDA,        # Balance relevance vs diversity
)
```

**Usage:**
```python
boost = EvidenceBoost.get_boost("1a")  # returns 1.35
recency = RecencyBoost.get_boost(365)  # returns 1.2 for papers <1yr old
```

**Design principles:**
- All boost values are tunable via config
- RRF_K defaults to 60 (higher than typical 60 means more weight to rank position)
- Constants are classes for flexible lookup tables

---

## 7. Query Pipeline

### Simple Search (`POST /search`)

```
Query → Intent Classification → Cache Check → Hybrid Retrieval
        → RRF Fusion → Reranking → MMR → LLM Generation → Validation
```

**Flow:**
1. **Query Understanding** - Classifies intent (diagnostic/therapeutic/prognostic/drug_info/guideline)
2. **Cache Check** - Returns cached result if available
3. **Hybrid Retrieval** - Parallel dense + sparse vector search
4. **RRF Fusion** - Reciprocal Rank Fusion combining results
5. **Reranking** - Cross-encoder reranking for better relevance
6. **MMR** - Maximal Marginal Relevance for diversity
7. **LLM Generation** - Generate answer using NVIDIA NIM
8. **Validation** - Check citations, hallucinations, safety

---

## 7. DeepInsights Mode

### When to Use
- Drug interaction checks
- Differential diagnosis
- Protocol conflicts
- Multi-condition management

### Flow
```
Query → Intent Router → Complex? 
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
    Simple RAG           DeepInsights
                            │
                    Query Decomposer
                            │
               ┌────────────┼────────────┐
               ▼            ▼            ▼
         Sub-query 1  Sub-query 2  ... Sub-query N
               │            │            │
               └────────────┼────────────┘
                            ▼
                  Evidence Synthesizer
                            │
                            ▼
                  Structured Answer
```

**Agents:**
- **Intent Router** - Detects simple vs complex queries
- **Query Decomposer** - Breaks complex queries into sub-queries
- **Orchestrator** - Coordinates parallel retrieval and synthesis

---

## 8. Data Ingestion

### Pipeline Flow
```
Source Files → Parser → Deduplication → Metadata Enrichment
    → Chunking → Quality Scoring → Embedding → Vector Index → MongoDB
```

**Features:**
- Document deduplication (content hash)
- Quality scoring and filtering
- Dual embeddings (dense + sparse)
- Retry logic for failures
- Monitoring and metrics

### Running Ingestion
```bash
# ICMR PDFs
python scripts/seed_icmr.py

# PubMed
python scripts/seed_pubmed.py

# Re-ingest all
python scripts/reingest_v2.py
```

---

## 9. API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/search` | POST | Simple RAG search |
| `/deep-insights` | POST | Multi-agent complex query |
| `/deep-insights/route-check` | GET | Check query complexity |
| `/health` | GET | Health check |

### Request/Response

**POST /search**
```json
{"query": "What is the treatment for dengue?", "top_k": 6}
```

**Response**
```json
{
  "answer": "...",
  "citations": [...],
  "query_intent": "therapeutic",
  "chunks_retrieved": 6,
  "cache_hit": false,
  "confidence_score": 0.85
}
```

---

## 10. Development

### Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Copy env file
cp .env.example .env

# Start services
docker compose up -d

# Run API
uvicorn src.api.main:app --reload --port 8000
```

### Testing
```bash
# Run tests
pytest tests/ -v

# Smoke test vector DB
python scripts/zilliz_smoke.py
```

---

## 11. Useful Commands

```bash
# Start API
uvicorn src.api.main:app --reload --port 8000

# Ingest ICMR PDFs
python scripts/seed_icmr.py

# Re-ingest all
python scripts/reingest_v2.py

# Test vector backend
python scripts/zilliz_smoke.py

# Format code
black src/
isort src/
```

---

## Notes

- Never commit `.env` — it's in `.gitignore`
- Medical knowledge patterns are hardcoded in respective modules (see `docs/MEDICAL_KNOWLEDGE_HARDCODED.md`)
- For architecture diagrams, see `docs/` folder

---

*OpenInsight — SentArc Labs | Built by Aditya Singh*