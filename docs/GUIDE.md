# OpenInsight — Quick Start Guide

> A clinical decision support system for Indian physicians

---

## What is OpenInsight?

A medical RAG system that answers clinical questions with cited, evidence-based responses from:
- ICMR guidelines
- PubMed research
- Indian clinical literature

**Two modes:**
- **Search** (`/search`) - Fast single-pass RAG
- **DeepInsights** (`/deep-insights`) - Multi-agent for complex cases

---

## Quick Start

### 1. Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Copy env file
cp .env.example .env

# Edit .env with your API keys (NVIDIA, MongoDB, Zilliz)
```

### 2. Run API
```bash
uvicorn src.api.main:app --reload --port 8000
```

### 3. Test
```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "treatment for dengue fever"}'
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

# Limit files
python scripts/run.py pubmed ./pdfs --limit 100

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
| `--limit N` | Limit files to process | none |
| `--stats` | Show detailed stats | false |
| `--resume` | Resume from checkpoint | true |

---

## Project Structure

```
src/
├── api/              # FastAPI endpoints
├── config/           # Settings (.env)
├── constants/        # Magic values
├── data/             # MongoDB & vector storage
├── ingestion/        # Data pipeline
│   └── parsers/      # PDF/XML parsers
├── ml/               # ML components
│   ├── chunking/     # Document chunking
│   ├── embedding/    # Text embeddings
│   └── ner.py        # Named entity recognition
├── query/            # Query pipeline
│   ├── search/       # RAG retrieval
│   ├── agents/       # DeepInsights agents
│   └── validation/  # Answer validation
├── services/         # LLM client
└── utils/            # Utilities
```

---

## Useful Commands

```bash
# Start API
uvicorn src.api.main:app --reload --port 8000

# Run ingestion
python scripts/run.py <source> <directory>

# Run tests
pytest tests/ -v

# Smoke test vector DB
python scripts/zilliz_smoke.py
```

---

*Built by SentArc Labs*