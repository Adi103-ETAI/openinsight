# OpenMed — Developer Guide
**SentArc Labs | Pune, India**
> This is your single reference for everything about building OpenMed. Architecture decisions, how to set up your environment, how each component works, what to build next, and why things are structured the way they are.

---

## Table of Contents
1. [What We Are Building](#1-what-we-are-building)
2. [Architecture Overview](#2-architecture-overview)
3. [Tech Stack Decisions](#3-tech-stack-decisions)
4. [Setting Up GitHub Codespaces](#4-setting-up-github-codespaces)
5. [Project File Structure](#5-project-file-structure)
6. [The Two Databases](#6-the-two-databases)
7. [Data Ingestion Pipeline](#7-data-ingestion-pipeline)
8. [Query Flow](#8-query-flow)
9. [Development Phases](#9-development-phases)
10. [Deployment](#10-deployment)
11. [Environment Variables Reference](#11-environment-variables-reference)
12. [Useful Commands](#12-useful-commands)

---

## 1. What We Are Building

OpenMed is a clinical decision support platform for Indian physicians. A doctor types a clinical question and gets a cited answer — grounded in ICMR guidelines, live PubMed research, and Indian clinical literature — in under 10 seconds.

**The core problem it solves:**
- UpToDate is expensive and built for US clinical contexts
- ChatGPT has no citations and hallucinates drug dosages
- Manual PubMed search is slow and returns papers, not answers
- No existing tool knows what ICMR says about managing dengue in India

**The two modes (mirroring OpenEvidence's architecture):**
- **Standard Search** — fast single-pass RAG for straightforward queries
- **DeepConsult** — multi-agent orchestration for complex cases (drug interactions, differential diagnosis, protocol conflicts)

We are building the **prototype** first — no auth, no billing, just validating that the system works and produces good answers for Indian clinical queries.

---

## 2. Architecture Overview

The system has two major pipelines:

```
┌─────────────────────────────────────────────────────────────────┐
│ INGESTION PIPELINE                                              │
│                                                                 │
│  ICMR PDFs ──┐                                                  │
│  PubMed API ─┼──► Parsing/ETL ──► MongoDB (Document DB)        │
│  State docs ─┘         │                                        │
│                         └──► Chunker ──► Embedder ──► Qdrant   │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ QUERY PIPELINE                                                  │
│                                                                 │
│  Doctor query                                                   │
│       │                                                         │
│       ├── Standard Search ──► Embed query                       │
│       │                           │                             │
│       │                    Qdrant semantic search               │
│       │                           │                             │
│       │                    Top-K chunks retrieved               │
│       │                           │                             │
│       │                    Prompt construction                  │
│       │                           │                             │
│       │                    Llama 3.1 70B (NIM)                  │
│       │                           │                             │
│       │                    Answer + Citations                   │
│       │                                                         │
│       └── DeepConsult ──► Agent Orchestrator                    │
│                               │                                 │
│                        Multiple sub-queries                     │
│                               │                                 │
│                        Parallel Qdrant searches                 │
│                               │                                 │
│                        Result Synthesis                         │
│                               │                                 │
│                        Deep Answer + Citations                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Tech Stack Decisions

Every decision here was made deliberately. Here is what we use and why.

### Language: Python 3.11
Every library we need — LangChain, HuggingFace, Qdrant, FastAPI, NCBI Entrez — is Python-native. No other language comes close for AI/ML work.

### Backend: FastAPI
Async, fast, auto-generates API docs at `/docs`, and Pydantic integration is seamless. This is what most production AI APIs use.

### Embeddings: `pritamdeka/S-PubMedBert-MS-MARCO`
This model is trained specifically on PubMed text with retrieval fine-tuning on top. The quality jump over general-purpose models (like `all-MiniLM-L6-v2` from our Colab prototype) is significant for medical queries. It understands that "TB" and "tuberculosis" are the same thing, that "rifampicin" and "rifampin" are the same drug, and that "DOT therapy" means Directly Observed Treatment.

Runs locally on Codespaces. Free. 768-dimensional vectors.

### Document DB: MongoDB
Stores the raw and parsed documents before they go into the vector index. MongoDB is the right tool here because:
- Medical documents are semi-structured (a PubMed paper has different fields than an ICMR PDF)
- No fixed schema needed — different source types have different metadata
- Motor (async MongoDB driver) plays well with FastAPI
- Free tier on Railway for deployment

### Vector DB: Qdrant
Stores embeddings of document chunks for semantic search. Chosen over FAISS (our Colab prototype) because:
- **Payload filtering** — we can query "find top chunks where source_type = icmr AND condition_tag = dengue"
- **Hybrid search** — vector similarity + keyword search combined. Critical for exact drug names
- **Persistent storage** — index survives restarts unlike in-memory FAISS
- **Docker-native** — runs as a container locally and deploys the same way on Railway

### LLM: Llama 3.1 70B via NVIDIA NIM
We call this as an API — no GPU needed on our side. NIM gives us OpenAI-compatible endpoints so the LangChain integration is plug-and-play.

### Cache: Redis
Medical queries repeat. "First-line treatment for TB" will get asked hundreds of times. Cache the answer by query hash. A cache hit drops response time from ~5 seconds to ~200ms.

### Dev Environment: GitHub Codespaces
Full VS Code in the browser, persistent storage, Git built in. Since we call NVIDIA NIM as an API, we don't need a GPU on our dev machine. Student Pack gives 180 core-hours/month free — plenty.

### Deployment: Railway
Deploys Python/FastAPI directly from GitHub. Managed Postgres and Redis add-ons. Persistent volumes for Qdrant. Simple.

---

## 4. Setting Up GitHub Codespaces

### Step 1 — Create the GitHub repo

```bash
# On github.com, create a new repo called: openmed
# Clone it locally or open directly in Codespaces
```

### Step 2 — Push this project into the repo

```bash
git init
git remote add origin https://github.com/YOUR_USERNAME/openmed.git
git add .
git commit -m "init: project scaffold"
git push -u origin main
```

### Step 3 — Open in Codespaces

1. Go to your repo on GitHub
2. Click the green **Code** button
3. Click **Codespaces** tab
4. Click **Create codespace on main**

GitHub will:
- Spin up a container using `.devcontainer/devcontainer.json`
- Run `.devcontainer/setup.sh` automatically
- Install all Python dependencies from `requirements.txt`
- Start Qdrant, MongoDB, and Redis via Docker Compose
- Install all VS Code extensions

This takes about 3-4 minutes the first time.

### Step 4 — Set up your environment variables

```bash
cp .env.example .env
# Edit .env and fill in your actual API keys
```

Keys you need:
- `NVIDIA_NIM_API_KEY` — from build.nvidia.com (free credits available)
- `NCBI_API_KEY` — from ncbi.nlm.nih.gov/account (free, just register)

### Step 5 — Verify everything is running

```bash
# Check Docker services
docker compose ps

# Test Qdrant
curl http://localhost:6333/healthz

# Test MongoDB
python -c "from pymongo import MongoClient; print(MongoClient('mongodb://localhost:27017').server_info()['version'])"

# Start the API
uvicorn src.api.main:app --reload --port 8000

# Visit http://localhost:8000/docs in your browser
# You should see the FastAPI auto-generated docs
```

---

## 5. Project File Structure

```
openmed/
│
├── .devcontainer/
│   ├── devcontainer.json       # Codespaces config — Python 3.11, Docker, Node
│   └── setup.sh                # Post-create: pip install, docker compose up
│
├── src/
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py             # FastAPI app entry point
│   │   └── routes/             # (coming) query.py, ingest.py
│   │
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── document_db.py      # MongoDB client + document/chunk models
│   │   ├── vector_db.py        # Qdrant client + search/upsert helpers
│   │   ├── embeddings.py       # PubMedBERT loader + embed_texts(), embed_query()
│   │   ├── parsers/            # (coming) icmr.py, pubmed.py, pdf.py
│   │   └── pipeline.py         # (coming) orchestrates parse → chunk → embed → store
│   │
│   ├── query/
│   │   ├── __init__.py
│   │   ├── standard.py         # (coming) standard search RAG chain
│   │   ├── deepconsult.py      # (coming) multi-agent orchestration
│   │   └── prompts.py          # (coming) prompt templates
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   └── config.py           # Settings via pydantic-settings, reads .env
│   │
│   └── utils/
│       ├── __init__.py
│       └── chunker.py          # Medical-aware text chunker
│
├── data/
│   ├── raw/                    # Drop ICMR PDFs here (gitignored)
│   └── processed/              # Intermediate parsed outputs (gitignored)
│
├── tests/
│   └── __init__.py             # (coming) unit + integration tests
│
├── scripts/
│   │                           # (coming) one-off scripts
│   │                           # e.g. run_ingestion.py, seed_icmr.py
│
├── docs/
│   └── GUIDE.md                # This file
│
├── docker-compose.yml          # Qdrant + MongoDB + Redis
├── requirements.txt            # All Python dependencies
├── .env.example                # Template — copy to .env and fill in keys
├── .env                        # Your actual keys (gitignored)
└── .gitignore
```

---

## 6. The Two Databases

Understanding why we have two separate databases is important.

### MongoDB — Document Database

**What it stores:** The actual content of every document we ingest — full text of ICMR PDFs, PubMed paper abstracts and metadata, guideline sections.

**Why MongoDB and not just Qdrant?**
Qdrant stores vectors and small payloads. You cannot store a full 40-page ICMR guideline PDF in Qdrant — it's not built for that. MongoDB is where the source of truth lives. When a doctor asks a question and we need to return a citation, we look up the full document in MongoDB to render the citation properly.

**Collections:**

| Collection | Purpose |
|---|---|
| `documents` | One record per source document (ICMR PDF, PubMed paper) |
| `chunks` | Passage-level splits of each document, ready for embedding |
| `sources` | Metadata about each knowledge source (last updated, doc count) |

### Qdrant — Vector Database

**What it stores:** Float vectors (768 dimensions) representing the semantic meaning of each chunk, plus a small payload (chunk_id, source_type, title, condition_tags).

**Why Qdrant and not FAISS?**

| Feature | FAISS (old) | Qdrant (new) |
|---|---|---|
| Persistence | Manual save/load | Built-in |
| Metadata filtering | None | Full payload filter |
| Hybrid search | No | Yes (vector + keyword) |
| Scalability | Single process | Production-ready |
| API | Python only | REST + Python |

**How they connect:** Every chunk in MongoDB has a corresponding point in Qdrant. The Qdrant point payload contains the MongoDB `_id` so we can look up the full text after retrieval.

```
MongoDB chunk._id  ←──────────────────► Qdrant point.payload.mongo_id
MongoDB chunk.text                       Qdrant point.vector (768 floats)
MongoDB chunk.source_type                Qdrant point.payload.source_type
MongoDB chunk.title                      Qdrant point.payload.title
```

---

## 7. Data Ingestion Pipeline

This is Phase 1 of the build. The quality of every answer depends on what is in the index.

### Flow

```
Source document
      │
      ▼
  Parser          ← converts PDF/XML/HTML to clean text + metadata
      │
      ▼
  MongoDB         ← stores full document record
      │
      ▼
  Chunker         ← splits text into 512-token overlapping passages
      │
      ▼
  MongoDB         ← stores each chunk record (embedded=False)
      │
      ▼
  Embedder        ← PubMedBERT generates 768-dim vector per chunk
      │
      ▼
  Qdrant          ← upserts vector + payload, marks chunk embedded=True
```

### Source types we need to ingest

| Source | Format | Priority | Notes |
|---|---|---|---|
| ICMR Clinical Guidelines | PDF | High | Already have some from Colab work |
| PubMed | XML via NCBI API | High | Live search + periodic bulk index |
| National List of Essential Medicines | PDF | High | NMC publishes this |
| State health ministry guidelines | PDF/HTML | Medium | Varies by state |
| WHO India-specific docs | PDF | Medium | SEARO region publications |

### Chunking strategy

Medical text has special requirements. A naive chunker that just splits every 512 tokens will break:
- Dosage tables in half (you lose context about which drug the dose refers to)
- Clinical decision trees mid-branch
- Drug interaction tables

Our chunker in `src/utils/chunker.py` splits at sentence boundaries and avoids common medical abbreviations (`mg`, `BD`, `TDS`, `IV`, etc.). Target chunk size is ~512 words with 80-word overlap.

---

## 8. Query Flow

### Standard Search (Phase 2)

```python
# 1. Embed the doctor's query
query_vector = embed_query("first line treatment for dengue with warning signs")

# 2. Search Qdrant — top 8 most relevant chunks
results = qdrant.search(query_vector, top_k=8)

# 3. Pull full chunk texts from MongoDB
chunks = [mongo.get_chunk(r.payload["mongo_id"]) for r in results]

# 4. Build prompt
context = "\n\n".join([c.chunk_text for c in chunks])
prompt = f"""You are a clinical decision support assistant for Indian physicians.
Answer the question using only the provided context. Cite sources by number.

Context:
{context}

Question: {query}

Answer:"""

# 5. Call Llama 3.1 70B via NIM
answer = nim_client.chat(prompt)

# 6. Return answer + citations
```

### DeepConsult (Phase 3)

For complex queries the agent orchestrator:
1. Breaks the query into sub-questions (e.g. "what is the drug?", "what is the dose?", "what are contraindications?")
2. Runs parallel Qdrant searches for each sub-question
3. Passes all results through a synthesis prompt
4. Returns a comprehensive answer with multi-source citations

---

## 9. Development Phases

### Phase 1 — Ingestion Pipeline (current)
- [ ] ICMR PDF parser (`src/ingestion/parsers/icmr.py`)
- [ ] PubMed XML parser (`src/ingestion/parsers/pubmed.py`)
- [ ] Ingestion pipeline orchestrator (`src/ingestion/pipeline.py`)
- [ ] Script to seed MongoDB + Qdrant from local ICMR PDFs
- [ ] Script to pull and index PubMed articles by condition

### Phase 2 — Standard Search Query Flow
- [ ] Standard search chain (`src/query/standard.py`)
- [ ] Prompt templates for Indian clinical context (`src/query/prompts.py`)
- [ ] FastAPI `/query` endpoint (`src/api/routes/query.py`)
- [ ] Redis caching layer
- [ ] Basic test queries to validate answer quality

### Phase 3 — DeepConsult Mode
- [ ] Agent orchestrator (`src/query/deepconsult.py`)
- [ ] Sub-question decomposition
- [ ] Result synthesis prompt
- [ ] FastAPI `/query/deep` endpoint

### Phase 4 — UI
- [ ] React/Next.js frontend on Vercel
- [ ] Simple chat interface
- [ ] Citation display with links back to source

### Phase 5 — Deployment
- [ ] Dockerize the FastAPI app
- [ ] Railway project setup (FastAPI + MongoDB + Redis + Qdrant)
- [ ] GitHub Actions CI/CD pipeline
- [ ] Environment variable management on Railway

---

## 10. Deployment

### Railway setup (when ready)

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login
railway login

# Create new project
railway new openmed

# Add services
railway add --service mongodb
railway add --service redis

# Deploy FastAPI app
railway up
```

Qdrant runs as a Docker container on Railway using the `qdrant/qdrant` image with a persistent volume.

### GitHub Actions (auto-deploy on push)

Create `.github/workflows/deploy.yml`:
```yaml
on:
  push:
    branches: [main]
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: bervProject/railway-deploy@main
        with:
          railway_token: ${{ secrets.RAILWAY_TOKEN }}
          service: openmed-api
```

---

## 11. Environment Variables Reference

| Variable | Required | Description |
|---|---|---|
| `NVIDIA_NIM_API_KEY` | Yes | From build.nvidia.com |
| `NVIDIA_NIM_BASE_URL` | No | Default: NVIDIA's endpoint |
| `MONGODB_URL` | No | Default: localhost:27017 |
| `MONGODB_DB` | No | Default: openmed |
| `QDRANT_URL` | No | Default: localhost:6333 |
| `QDRANT_COLLECTION` | No | Default: openmed_chunks |
| `REDIS_URL` | No | Default: localhost:6379 |
| `NCBI_API_KEY` | Recommended | Rate limit: 3 req/s without, 10 with |
| `NCBI_EMAIL` | Yes | Required by NCBI Entrez policy |
| `EMBEDDING_MODEL` | No | Default: PubMedBERT |

---

## 12. Useful Commands

```bash
# Start all services
docker compose up -d

# Stop all services
docker compose down

# View service logs
docker compose logs qdrant -f
docker compose logs mongodb -f

# Start FastAPI dev server
uvicorn src.api.main:app --reload --port 8000

# Run tests
pytest tests/ -v

# Check Qdrant collections
curl http://localhost:6333/collections

# Check MongoDB collections
mongosh openmed --eval "db.getCollectionNames()"

# Format code
black src/
isort src/

# Run ingestion script (Phase 1)
python scripts/seed_icmr.py

# Rebuild Docker services (after docker-compose.yml changes)
docker compose down && docker compose up -d --build
```

---

## Notes

- Never commit `.env` — it is in `.gitignore`
- The `data/` folder is gitignored — ICMR PDFs are large and should not be in the repo. Store them in the Codespace and later in Railway volumes or an S3 bucket
- When you first open the Codespace, `setup.sh` runs automatically. If something fails, run `bash .devcontainer/setup.sh` manually in the terminal
- Qdrant data persists in a Docker volume (`qdrant_data`). If you want to reset the vector index: `docker compose down -v && docker compose up -d`

---

*OpenMed — SentArc Labs, Pune | Built by Aditya Singh | adi.singh1426@gmail.com*
