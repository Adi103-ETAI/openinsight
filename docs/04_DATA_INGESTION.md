# Data Ingestion Pipeline

> Last updated: August 2026
> Reflects codebase on `restruct` branch at commit `78f4d6b`

## Overview

The ingestion pipeline transforms raw source documents into searchable vector embeddings stored in Milvus, with full documents and metadata stored in MongoDB.

### Key Features
- **Multi-provider embeddings**: Local (SentenceTransformers), HuggingFace Inference API, or Cohere
- **Dead letter queue**: Failed documents tracked in `failed_documents` MongoDB collection for reprocessing
- **OCR fallback**: Scanned PDFs automatically detected and processed via pytesseract
- **Checkpoint/resume**: Long-running jobs can be paused and resumed via checkpoint files
- **Zilliz verification**: Post-upsert count validation (expected vs actual) for data integrity
- **GROBID 0.8.0**: Configurable timeouts, retries (exponential backoff), and health check with fallback to ICMRParser
- **Quality scoring**: High/low value pattern detection for document quality assessment
- **Deduplication**: Content hash + title similarity (threshold 0.9) to prevent duplicate indexing

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         INGESTION PIPELINE                                   │
│                                                                             │
│  ┌──────────────┐                                                           │
│  │  Load Files  │  Directory scan or Kaggle dataset                         │
│  └──────┬───────┘                                                           │
│         ▼                                                                   │
│  ┌──────────────┐                                                           │
│  │    Parse     │  Source-aware: GROBID / ICMR / PubMed / Cochrane /        │
│  │              │  WHO / CDC / StatPearls / OCR fallback                    │
│  └──────┬───────┘                                                           │
│         ▼                                                                   │
│  ┌──────────────┐                                                           │
│  │   Dedupe     │  Content hash + title similarity (0.9 threshold)         │
│  └──────┬───────┘                                                           │
│         ▼                                                                   │
│  ┌──────────────┐                                                           │
│  │  Metadata    │  Year, evidence level, journal, DOI, PMID,               │
│  │  Enrichment  │  India-relevance flag, drug dosing detection             │
│  └──────┬───────┘                                                           │
│         ▼                                                                   │
│  ┌──────────────┐                                                           │
│  │    Chunk     │  HierarchicalChunkerV3: target 350 tokens,               │
│  │              │  max 500, overlap 50, min 80. Doc summary +              │
│  │              │  per-section + table chunks.                              │
│  └──────┬───────┘                                                           │
│         ▼                                                                   │
│  ┌──────────────┐                                                           │
│  │    Embed     │  S-PubMedBert-MS-MARCO (768d dense) +                    │
│  │              │  TF-IDF sparse (medical tokenization, IDF weighting)     │
│  │              │  Returns (embeddings, failed_indices)                     │
│  └──────┬───────┘                                                           │
│         ▼                                                                   │
│  ┌──────────────┐                                                           │
│  │   Validate   │  Document and chunk validation, quality scoring          │
│  └──────┬───────┘                                                           │
│         ▼                                                                   │
│  ┌──────────────┐     ┌──────────────┐                                      │
│  │  Milvus      │     │   MongoDB    │                                      │
│  │  Index       │     │   Store      │                                      │
│  │  (dense +    │     │  documents_v2│                                      │
│  │   sparse)    │     │  chunks_v2   │                                      │
│  └──────────────┘     └──────────────┘                                      │
│         │                                                                   │
│         ▼                                                                   │
│  ┌──────────────┐                                                           │
│  │  Checkpoint  │  Update progress for resume support                       │
│  └──────────────┘                                                           │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Dead Letter Queue

Failed documents are stored in the `failed_documents` MongoDB collection with:
- Original document path/content
- Error message and traceback
- Timestamp
- Source type
- Retry count

These can be reprocessed later by querying the dead letter queue and re-submitting.

---

## Parsers (`src/ingestion/parsers/`)

| Parser | File | Input Format | Source | Key Details |
|--------|------|-------------|--------|-------------|
| **GROBID** | `grobid.py` | PDF → TEI XML | Any academic PDF | Primary PDF parser. GROBID 0.8.0 with configurable timeout (120s default), max retries (3), exponential backoff. Falls back to ICMRParser if GROBID unavailable. |
| **ICMR** | `icmr.py` | PDF → text | ICMR guidelines | pdfplumber-based extraction for Indian clinical guidelines. |
| **PubMed** | `pubmed.py` | XML | PubMed Central | Biopython Entrez + XML parsing for research articles. |
| **Cochrane** | `cochrane.py` | PDF/XML | Cochrane Library | Systematic review extraction. |
| **WHO** | `who.py` | PDF | WHO guidelines | WHO guideline PDF parsing. |
| **CDC** | `cdc.py` | PDF | CDC documents | CDC guideline extraction. |
| **StatPearls** | `statpearls.py` | PDF | StatPearls (NCBI) | StatPearls reference article parsing. |
| **OCR** | `ocr.py` | Scanned PDF → text | Any | pytesseract fallback for scanned/image PDFs. |

### GROBID Configuration

```python
grobid_url = "http://localhost:8070"      # GROBID server (Docker container)
grobid_timeout = 120                       # seconds
grobid_max_retries = 3
grobid_retry_delay = 2.0                   # seconds (exponential backoff base)
grobid_health_check_timeout = 10           # seconds
```

### Base Parser (`base.py`)

All parsers extend a common base class with shared error handling and logging.

---

## Chunking (`src/ml/chunking/chunker.py`)

### HierarchicalChunkerV3

- **Sentence-window splitting** with configurable overlap
- **Abbreviation protection** — common medical abbreviations (et al., e.g., i.e., mg/dl, p.o., i.v., b.d., t.d.s.) are protected from sentence splitting
- **Table extraction** — tables get their own chunks with preserved structure
- **Doc summary chunk** — first chunk is always a document-level summary

### Chunk Types

| Type | Description | Typical Size |
|------|-------------|-------------|
| `doc_summary` | Document-level overview | ~350 tokens |
| `section` | Per-section content | 80–500 tokens |
| `table` | Extracted table data | Variable |

### Configuration

```python
chunk_target_tokens = 350    # Target chunk size
chunk_max_tokens = 500       # Maximum chunk size
chunk_overlap_tokens = 50    # Overlap between chunks
chunk_min_tokens = 80        # Minimum viable chunk size
```

### Chunk Data Model

```python
@dataclass
class ChunkV3:
    chunk_id: str
    doc_id: str
    chunk_type: str           # "doc_summary", "section", "table"
    section_title: str
    text: str
    contextual_text: str      # Context window for retrieval
    char_count: int
    token_estimate: int
    chunk_index: int
    total_chunks: int
    metadata: dict
```

---

## Embedding (`src/ml/embedding/embedder.py`)

### Providers

| Provider | Class | Model | Dimension | GPU? | Batch Size |
|----------|-------|-------|-----------|------|-----------|
| `local` | `LocalEmbedder` | S-PubMedBert-MS-MARCO | 768 | Yes (CUDA) | 32 (retry: 16) |
| `huggingface` | `HuggingFaceEmbedder` | S-PubMedBert-MS-MARCO | 768 | No (API) | 32 |
| `cohere` | `CohereEmbedder` | embed-english-v3.0 | 1024 | No (API) | 32 |

### API Contract

Every provider's `embed_batch()` returns `(embeddings: np.ndarray, failed_indices: list[int])`:
- `embeddings`: numpy array with zero vectors at failed indices
- `failed_indices`: list of indices that failed (empty = all succeeded)

Callers in ingestion and retrieval must filter out failed indices before indexing to Milvus to avoid zero-vector pollution.

### Sparse Vectors

Computed CPU-side via shared `compute_sparse_vector()` regardless of dense provider:
- Medical tokenization (compound term splitting)
- IDF weighting
- Stopword removal
- Configurable vocab size (default 50,000)

---

## Deduplication

Two mechanisms prevent duplicate content in the index:

1. **Content hash** — SHA-256 hash of normalized text content (first 16 chars stored as `dedup_content_hash_length`)
2. **Title similarity** — Fuzzy matching on document titles with threshold `dedup_title_similarity = 0.9`

Both `deduplication.py` and `dedupe.py` exist in the ingestion directory (the latter is the primary one used in the pipeline).

---

## Quality Scoring (`src/ingestion/quality.py`)

### High-Value Patterns
Documents matching these patterns get boosted scores:
- `randomized`, `controlled`, `meta-analysis`, `systematic review`, `prospective`, `cohort`, `double-blind`, `multicenter`, `guideline`, `meta:analysis`

### Low-Value Patterns
Documents matching these patterns get penalized:
- `opinion`, `case report`, `editorial`, `letter`, `abstract only`, `retracted`

### Configuration
```python
quality_score_threshold = 0.3           # Minimum quality score to index
quality_high_value_patterns_count = 10  # Number of high-value patterns
quality_low_value_penalty = 0.5         # Penalty multiplier for low-value content
```

---

## Metadata Enrichment (`src/ingestion/metadata.py`)

### Extracted Fields

| Field | Method | Description |
|-------|--------|-------------|
| `year` | Date parsing | Publication year from various date formats |
| `evidence_level` | Pattern matching | 1a, 1b, 2a, 2b, 3, 4, 5 based on study type |
| `journal` | Metadata extraction | Journal name |
| `doi` | Metadata extraction | Digital Object Identifier |
| `pmid` | Metadata extraction | PubMed ID |
| `india_relevant` | Source/author check | Flag for Indian clinical relevance |
| `has_drug_dosing` | Regex detection | Flags documents with dosage information |
| `authors` | Metadata extraction | Author list for credibility assessment |
| `source_type` | Config | Source classification (icmr, pubmed, who, etc.) |

### Evidence Level Detection

| Pattern | Level |
|---------|-------|
| Systematic review, meta-analysis | 1a |
| RCT | 1b |
| High-quality cohort | 2a |
| Cohort / case-control | 2b |
| Case series | 3 |
| Expert opinion / case report | 4 |
| Unknown | 5 |

---

## LlamaIndex Integration (`src/ingestion/llamaindex_integration.py`)

Experimental parent-child chunk retrieval using LlamaIndex patterns:

- **HierarchicalChunkParser**: ~1000 token parent chunks + ~350 token child chunks
- **ParentChildIndexer**: Indexes to separate `{collection}_child` and `{collection}_parent` Milvus collections
- **ParentChildRetriever**: Two-stage retrieval (child search → parent fetch for context)
- **Backward compatible**: `HybridRetrieverWithParent` wraps existing `HybridRetriever`

> Note: This integration exists but is not currently wired into the main search endpoint. The `HybridRetriever.retrieve_with_parent()` method is available but not called from `/search`.

---

## Vector Indexing (`src/ingestion/vector_indexer.py`)

### Milvus Schema

| Field | Type | Description |
|-------|------|-------------|
| `id` (configurable) | VARCHAR | Chunk ID |
| `dense` (configurable) | FLOAT_VECTOR(768) | Dense embedding (COSINE metric) |
| `sparse` (configurable) | SPARSE_FLOAT_VECTOR | Sparse TF-IDF vector (IP metric) |
| payload fields | VARCHAR/INT | Metadata fields (source, year, evidence_level, etc.) |

### Zilliz Verification

After upsert, the indexer compares:
- **Expected count** (number of chunks submitted)
- **Actual count** (number of entities in collection)

A mismatch logs a warning and increments a metric.

### Configuration
```python
vector_backend = "milvus"                # Only supported backend
vector_collection = "openinsight_chunks"  # Default collection
vector_collection_v2 = "openinsight_v2"   # V2 collection name
vector_dim = 768                          # Dense vector dimension
vector_dense_metric = "COSINE"
vector_sparse_metric = "IP"
milvus_cloud = false                      # true for Zilliz Cloud
```

---

## MongoDB Storage (`src/data/mongo/doc_store.py`)

### Collections

| Collection | Documents | Key Fields |
|------------|-----------|------------|
| `documents_v2` | Full document metadata | doc_id, title, source, year, evidence_level, authors, doi, pmid |
| `chunks_v2` | Individual chunks | chunk_id, doc_id, text, chunk_type, chunk_index, metadata |
| `failed_documents` | Dead letter queue | doc_id, error, traceback, timestamp, retry_count |

### Connection

Singleton `AsyncIOMotorClient` with connection pool:
- `maxPoolSize = 50`
- `minPoolSize = 5`
- `maxIdleTimeMS = 30000`
- `connectTimeoutMS = 5000`
- `serverSelectionTimeoutMS = 5000`

---

## Checkpoint/Resume (`src/ingestion/checkpoint.py`)

- Tracks which files have been successfully ingested
- On resume, skips already-processed files
- Checkpoint file location: configurable, defaults to `.ingestion_checkpoint.json`
- Reset with `--reset` flag

---

## Celery Tasks (`src/ingestion/tasks.py` + `celery_app.py`)

- Distributed task queue for production ingestion
- Broker: Redis (`redis://localhost:6379/0`)
- Concurrency: 4 workers (default), configurable via `celery_concurrency`
- Tasks: document parsing, embedding, indexing
- Monitoring via Flower (optional)

### Configuration
```python
celery_broker_url = "redis://localhost:6379/0"
celery_concurrency = 4
ingestion_workers = 4
ingestion_batch_size = 10
ingestion_max_retries = 3
ingestion_retry_delay = 2.0
ingestion_max_chunks_per_doc = 100
max_concurrent_docs = 6
retry_backoff_multiplier = 2.0
retry_max_delay = 60.0
```

---

## Monitoring (`src/ingestion/monitoring.py`)

- Per-source document count and chunk count
- Embedding success/failure rates
- Processing time per document
- Dead letter queue size
- Quality score distribution

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

### CLI Options

| Flag | Description | Default |
|------|-------------|---------|
| `-w, --workers` | Parallel workers | 6 |
| `-b, --batch-size` | Files per batch | 10 |
| `--recreate` | Drop and rebuild Milvus collection | false |
| `--dry-run` | Parse only, no indexing | false |
| `--skip-embed` | Skip embedding step | false |
| `--skip-index` | Skip vector indexing | false |
| `--stats` | Show detailed stats | false |
| `--resume` | Resume from checkpoint | true |
| `--reset` | Reset checkpoint | false |

> **Warning**: `--recreate` calls `drop_collection` on the Milvus collection. Never use this against shared or production collections.

---

## Kaggle Notebook

`notebooks/kaggle_ingestion.ipynb` — Pre-configured for Kaggle/Colab environments:
- Installs dependencies with loose pins (Kaggle compatibility)
- Uses `config.kaggle.json` environment overrides
- Connects to Zilliz Cloud and MongoDB Atlas
- Processes PDFs from Kaggle datasets
- Saves checkpoint to `/kaggle/working` (5 GB persistent)

See `docs/RERANKER_AND_PLATFORM_RESEARCH.md` for platform comparison (Kaggle vs Colab).
