# Data Ingestion Pipeline

## Overview

The ingestion pipeline transforms raw source documents into searchable vector embeddings stored in Milvus, with full documents and metadata stored in MongoDB.

### Key Features
- **Multi-provider embeddings**: Local (SentenceTransformers), HuggingFace Inference API, or Cohere
- **Dead letter queue**: Failed documents tracked and reprocessable
- **OCR fallback**: Scanned PDFs automatically detected and processed via OCR
- **Checkpoint/resume**: Long-running jobs can be paused and resumed
- **Zilliz verification**: Post-upsert count validation for data integrity
- **GROBID 0.9.0**: Updated parser with configurable timeouts, retries, and health check with fallback

---

## Recent Changes

### GROBID 0.9.0 Update
- **New API endpoint**: `/api/processFulltextDocument` (was `/api/processFulltextDocument` with different TEI structure)
- **Health check**: Uses `/api/health` endpoint with fallback to `/api/isalive` for older versions
- **Configurable settings**: `GROBID_TIMEOUT`, `GROBID_MAX_RETRIES`, `GROBID_RETRY_DELAY`, `GROBID_HEALTH_CHECK_TIMEOUT`
- **Retry logic**: Exponential backoff with configurable max retries and delay cap
- **Fallback**: Falls back to ICMRParser (pdfplumber) if GROBID is unavailable

### LlamaIndex Integration (`src/ingestion/llamaindex_integration.py`)
- **Parent-child chunking**: Hierarchical retrieval with ~1000 token parent chunks and ~350 token child chunks
- **Three classes**: `HierarchicalChunkParser`, `ParentChildIndexer`, `ParentChildRetriever`
- **Backward compatible**: `HybridRetrieverWithParent` wraps existing retriever
- **Separate Milvus collections**: `{collection}_child` and `{collection}_parent`

### Parser Utilities (`src/utils/`)
- **`pubmed_client.py`**: Shared NCBI Entrez client with rate limiting, retry logic, and structured `PubMedArticle` dataclass
- **`date_utils.py`**: Year extraction from various date formats (ISO, US, free text, MedlineDate)
- **`text_utils.py`**: Keyword extraction, garble ratio calculation, text cleaning, chunking helpers

### Embedder Changes
- **Returns tuple**: `embed_batch()` now returns `(embeddings, failed_indices)` instead of just embeddings
- **Multi-provider**: `LocalEmbedder`, `HuggingFaceEmbedder`, `CohereEmbedder` via `EMBED_PROVIDER` setting
- **Failed embedding filtering**: Pipeline filters out failed embeddings before indexing

### Pipeline Fixes
- **Zero vector tracking**: Failed embeddings tracked via `failed_indices` — zero vectors are filtered before indexing
- **Zilliz verification**: Post-upsert count validation (`expected_count` vs `indexed`)
- **Dead letter reprocessing**: `reprocess_dead_letter()` method for retrying failed documents

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         SOURCE DOCUMENTS                                    │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │  ICMR PDFs  │  │   PubMed    │  │    WHO      │  │   Other     │         │
│  │   (.pdf)    │  │    (.xml)   │  │   (.pdf)    │  │  Sources    │         │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘         │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            PARSING                                          │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Source-specific Parsers                                            │    │
│  │                                                                     │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                  │    │
  │  │  │  ICMRParser │  │PubMedParser │  │ GROBIDParser│                  │    │
  │  │  │  (pdfplumb) │  │   (XML)     │  │  (TEI 0.9.0)│                  │    │
  │  │  └─────────────┘  └─────────────┘  └─────────────┘                  │    │
  │  │                                                                     │    │
  │  │  GROBID: /api/health → /api/isalive fallback, retries, timeout      │    │
  │  │  OCR: Auto-detected scanned PDFs → OCR fallback                     │    │
  │  │  Workers: 75% of CPU cores (configurable)                           │    │
  │  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  Output: Document {title, abstract, content, authors, year, doi, pmid}      │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DEDUPLICATION                                     │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  DocumentDeduplicator                                               │    │
│  │                                                                     │    │
│  │  1. Check if doc_id exists in MongoDB                               │    │
│  │  2. If exists, compute content hash (SHA256)                        │    │
│  │  3. Compare with stored hash                                        │    │
│  │  4. Skip if unchanged (or re-index if force_reindex=True)           │    │
│  │                                                                     │    │
│  │  dedup_title_similarity = 0.9 (for fuzzy matching)                  │    │
│  │  dedup_content_hash_length = 16                                     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       METADATA ENRICHMENT                                   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  MetadataEnricherV2                                                 │    │
│  │                                                                     │    │
│  │  Evidence Level Detection:                                          │    │
│  │    - RCT_TITLE_PATTERNS ("randomized controlled trial")             │    │
│  │    - SYSTEMATIC_REVIEW_PATTERNS ("meta-analysis")                   │    │
│  │    - GUIDELINE_PATTERNS ("clinical practice guideline")             │    │
│  │    - COHORT_PATTERNS ("prospective cohort")                         │    │
│  │                                                                     │    │
│  │  India Relevance:                                                   │    │
│  │    - Check for India-specific terms                                 │    │
│  │    - Source type (ICMR, Indian journals)                            │    │
│  │                                                                     │    │
│  │  Drug Dosing Detection:                                             │    │
│  │    - mg, ml, units patterns                                         │    │
│  │    - dosage ranges                                                  │    │
│  │                                                                     │    │
│  │  Specialty Tagging:                                                 │    │
│  │    - cardiology, neurology, infectious_disease, etc.                │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  Output: Enriched document with metadata                                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            CHUNKING                                         │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  HierarchicalChunkerV3                                              │    │
│  │                                                                     │    │
│  │  Config:                                                            │    │
│  │    - TARGET_TOKENS = 350                                            │    │
│  │    - MAX_TOKENS = 500                                               │    │
│  │    - MIN_TOKENS = 80                                                │    │
│  │    - OVERLAP_TOKENS = 50                                            │    │
│  │                                                                     │    │
│  │  Process:                                                           │    │
│  │    1. Extract summary (title + abstract + MeSH + keywords)          │    │
│  │    2. Identify tables (up to 20 per section)                        │    │
│  │    3. Split sections by sentence boundaries                         │    │
│  │    4. Enforce token constraints                                     │    │
│  │    5. Add context: title, section, source type                      │    │
│  │                                                                     │    │
│  │  Chunk Types:                                                       │    │
│  │    - doc_summary (full document summary)                            │    │
│  │    - paragraph (regular text)                                       │    │
│  │    - table (extracted tables)                                       │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  Output: List of ChunkV3 {text, contextual_text, chunk_type, etc.}          │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         QUALITY SCORING                                     │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  score_chunks()                                                     │    │
│  │                                                                     │    │
│  │  High-value patterns (+1 each):                                     │    │
│  │    - randomized, controlled, meta-analysis, systematic review       │    │
│  │    - prospective, cohort, case-control                              │    │
│  │    - guidelines, recommendations                                    │    │
│  │                                                                     │    │
│  │  Low-value patterns (-0.5 each):                                    │    │
│  │    - opinion, case report, editorial, letter                        │    │
│  │    - animal study, in vitro                                         │    │
│  │                                                                     │    │
│  │  Base score: 0.5, Clamped: [0.0, 1.0]                               │    │
│  │                                                                     │    │
│  │  Filter: quality_score_threshold = 0.3                              │    │
│  │  (chunks below threshold are dropped)                               │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           EMBEDDING                                         │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Embedder (configurable via EMBED_PROVIDER)                         │    │
│  │                                                                     │    │
│  │  Providers:                                                         │    │
│  │    - local:       SentenceTransformers on GPU (default)             │    │
│  │    - huggingface: HF Inference API (free tier)                      │    │
│  │    - cohere:      Cohere Embed API (1k calls/mo free)               │    │
│  │                                                                     │    │
│  │  Dense Embedding (PubMedBERT for local/HF):                         │    │
│  │    - Model: pritamdeka/S-PubMedBert-MS-MARCO                        │    │
│  │    - Dimension: 768                                                 │    │
│  │    - Batch size: 32                                                 │    │
│  │    - Input: contextual_text (title + section + content)             │    │
│  │                                                                     │    │
│  │  Sparse Embedding (TF-IDF, shared across providers):                │    │
│  │    - Vocabulary: 50,000                                             │    │
│  │    - Medical compound terms: 3.5x weight                            │    │
│  │    - Long terms (>10 chars): 3.0x weight                            │    │
│  │    - Stopwords: 96 removed                                          │    │
│  │                                                                     │    │
│  │  Return: (embeddings_array, failed_indices)                         │    │
│  │  Failed embeddings are filtered before indexing                     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  Output: (dense_vector [768d], sparse_vector) per valid chunk               │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          VECTOR INDEXING                                    │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  VectorIndexerV2                                                    │    │
│  │                                                                     │    │
│  │  Milvus Collection: configurable (default: openinsight_v2)        │    │
│  │                                                                     │    │
│  │  Schema:                                                            │    │
│  │    - id: VARCHAR (primary key)                                      │    │
│  │    - dense: FLOAT_VECTOR (768d)                                     │    │
│  │    - sparse: SPARSE_FLOAT_VECTOR                                    │    │
│  │    - year: INT64                                                    │    │
│  │    - source_type: VARCHAR                                           │    │
│  │    - evidence_level: VARCHAR                                        │    │
│  │    - india_relevant: BOOL                                           │    │
│  │    - has_drug_dosing: BOOL                                          │    │
│  │    - [dynamic fields: title, abstract, chunk_text, etc.]            │    │
│  │                                                                     │    │
│  │  Indexes:                                                           │    │
│  │    - Dense: COSINE metric, V5_CM                                    │    │
│  │    - Sparse: IP metric, SPARSE_INVERTED_INDEX                       │    │
│  │                                                                     │    │
│  │  Batch upsert: 100 docs                                             │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
                                   │
              ┌────────────────────┴────────────────────┐
              ▼                                         ▼
┌─────────────────────────────┐         ┌─────────────────────────────┐
│         MONGODB             │         │         MILVUS              │
│                             │         │                             │
│  documents_v2:              │         │  Collection:                │
│  - doc_id (unique)          │         │  openinsight_v2             │
│  - title                    │         │                             │
│  - abstract                 │         │  Points:                    │
│  - content                  │         │  - dense vectors            │
│  - authors                  │         │  - sparse vectors           │
│  - year                     │         │  - metadata                 │
│  - metadata (enriched)      │         │                             │
│  - content_hash             │         │                             │
│  - ingested_at              │         │                             │
│                             │         │                             │
│  chunks_v2:                 │         │                             │
│  - chunk_id                 │         │                             │
│  - doc_id                   │         │                             │
│  - text                     │         │                             │
│  - chunk_index              │         │                             │
│  - metadata                 │         │                             │
│                             │         │                             │
└─────────────────────────────┘         └─────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MONITORING                                        │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  RunMetrics saved to MongoDB                                        │    │
│  │                                                                     │    │
│  │  {                                                                  │    │
│  │    "run_id": "uuid",                                                │    │
│  │    "source": "icmr",                                                │    │
│  │    "start_time": "2024-01-01T00:00:00",                             │    │
│  │    "files_total": 50,                                               │    │
│  │    "files_parsed": 48,                                              │    │
│  │    "documents_stored": 150,                                         │    │
│  │    "chunks_created": 2500,                                          │    │
│  │    "chunks_indexed": 2480,                                          │    │
│  │    "chunks_deduped": 20,                                            │    │
│  │    "chunks_quality_filtered": 15                                    │    │
│  │    "files_failed": 2                                                │    │
│  │  }                                                                  │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Dead Letter Queue (failed_documents collection)                    │    │
│  │                                                                     │    │
│  │  Tracks documents that failed at any stage:                         │    │
│  │    - parse_error: Primary parser + OCR both failed                  │    │
│  │    - embed_error: Embedding API failures after retries              │    │
│  │    - index_error: Vector indexing failures                          │    │
│  │    - ocr_error: OCR-specific failures                               │    │
│  │                                                                     │    │
│  │  Reprocessing:                                                      │    │
│  │    pipeline.reprocess_dead_letter(                                  │    │
│  │        error_type="parse_error",  # or None for all                 │    │
│  │        max_retry_count=2,                                           │    │
│  │        source="pubmed"          # or None for all                   │    │
│  │    )                                                                │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Running Ingestion

### Quick Start (Recommended)
```bash
# Using the wrapper script
python scripts/run.py <source> <directory> [options]

# Examples
python scripts/run.py pubmed ./data/pdfs
python scripts/run.py icmr ./data/pdfs -w 8 --recreate
python scripts/run.py who ./pdfs --dry-run
python scripts/run.py pubmed ./pdfs --limit 100
```

### Using the module directly
```bash
python -m src.ingestion.run_ingestion \
  --source pubmed \
  --dir ./data/pdfs \
  --workers 6 \
  --batch-size 10
```

### Options
| Flag | Description | Default |
|------|-------------|---------|
| `--source` | Source (pubmed, icmr, cochrane, etc.) | required |
| `--dir` | Directory with files | required |
| `-w, --workers` | Parallel workers | 6 |
| `-b, --batch-size` | Files per batch | 10 |
| `--recreate` | Recreate vector index | false |
| `--dry-run` | Parse only, no embedding/indexing | false |
| `--skip-embed` | Skip embedding | false |
| `--skip-index` | Skip vector indexing | false |
| `--stats` | Show statistics | false |
| `--resume/--no-resume` | Checkpoint resume | enabled |
| `--reset` | Reset checkpoint | false |

> **Note**: The `--limit` argument was removed from the ingestion pipeline. To limit files, use shell commands like `ls | head -n N` or filter at the directory level.

### Available Sources
```
pubmed, icmr, cochrane, nmc_guideline, rssdi, who, cdc, statpearls
```

### Celery (Distributed)
```bash
python -m src.ingestion.tasks run
```

---

## Configuration

```python
# Ingestion
INGESTION_BATCH_SIZE = 10         # Files per batch
INGESTION_WORKERS = 4             # Thread pool workers (deprecated, use below)
INGESTION_MAX_RETRIES = 3         # Retry attempts
PARSING_THREAD_WORKERS = auto     # 75% of CPU cores (min 2, max 16)
INGESTION_THREAD_WORKERS = auto   # 75% of CPU cores for embedding
MAX_CONCURRENT_DOCS = 6           # Max concurrent document processing

# Collections (configurable via settings)
VECTOR_COLLECTION_V2 = "openinsight_v2"   # Milvus collection name
DOCUMENTS_COLLECTION = "documents_v2"       # MongoDB documents
CHUNKS_COLLECTION = "chunks_v2"             # MongoDB chunks

# Deduplication
DEDUP_ENABLED = true
DEDUP_TITLE_SIMILARITY = 0.9

# Quality
QUALITY_SCORE_THRESHOLD = 0.3     # Drop chunks below this

# Chunking
CHUNK_TARGET_TOKENS = 350
CHUNK_MAX_TOKENS = 500
CHUNK_OVERLAP_TOKENS = 50
CHUNK_MIN_TOKENS = 80

# Embedding
EMBEDDING_BATCH_SIZE = 32
EMBED_PROVIDER = "local"          # local | huggingface | cohere
EMBEDDING_RETRY_BATCH_SIZE = 16   # Reduced batch size on retry
EMBEDDING_TIMEOUT = 60            # Timeout in seconds

# GROBID
GROBID_URL = "http://localhost:8070"
GROBID_TIMEOUT = 120              # Timeout for GROBID API calls (seconds)
GROBID_MAX_RETRIES = 3            # Max retries for failed GROBID calls
GROBID_RETRY_DELAY = 2.0          # Initial delay between retries (seconds)
GROBID_HEALTH_CHECK_TIMEOUT = 10  # Timeout for health check requests

# Dead Letter Queue
DEAD_LETTER_ENABLED = true
DEAD_LETTER_COLLECTION = "failed_documents"

# PubMed API Rate Limiting (configurable per API tier)
PUBMED_RATE_LIMIT_SECONDS = 0.34  # Without API key (~3 requests/sec)
PUBMED_RATE_LIMIT_WITH_KEY = 0.1  # With API key (~10 requests/sec)
```

### Rate Limiting for External APIs

The CDC and WHO parsers use configurable rate limiting when accessing PubMed:

- **Without NCBI API key**: Uses `pubmed_rate_limit_seconds` (default: 0.34s between requests)
- **With NCBI API key**: Uses `pubmed_rate_limit_with_key` (default: 0.1s between requests)

This allows adjusting request rates based on your NCBI API tier to avoid rate limiting errors.

### GROBID Configuration

GROBID 0.9.0 settings are configurable via environment variables:

| Setting | Default | Description |
|---------|---------|-------------|
| `GROBID_URL` | `http://localhost:8070` | GROBID service URL |
| `GROBID_TIMEOUT` | `120` | Timeout for API calls in seconds |
| `GROBID_MAX_RETRIES` | `3` | Max retry attempts on failure |
| `GROBID_RETRY_DELAY` | `2.0` | Initial delay between retries (exponential backoff) |
| `GROBID_HEALTH_CHECK_TIMEOUT` | `10` | Timeout for health check requests |

The health check tries `/api/health` first (GROBID 0.9.0+), then falls back to `/api/isalive` for older versions.

---

## Performance

| Stage | Throughput |
|-------|------------|
| Parsing | ~10 docs/sec |
| Chunking | ~50 docs/sec |
| Embedding | ~20 docs/sec |
| Indexing | ~30 docs/sec |
| **Total** | ~8-10 docs/sec |

---

## Notebooks

Ingestion notebooks are available for cloud-based processing:

- **Kaggle**: `notebooks/kaggle_ingestion.ipynb` — Updated with GROBID 0.9.0 fixes and health check
- **Colab**: See `RERANKER_AND_PLATFORM_RESEARCH.md` for platform comparison

Both notebooks include:
- GROBID health check with fallback to `/api/isalive`
- Configurable timeout and retry settings
- Connection to Zilliz Cloud and MongoDB Atlas

---

## Logging

All ingestion modules use `loguru` for structured logging. Log messages include:
- `[pipeline]` prefix for pipeline orchestration
- `[HFEmbedder]`, `[CohereEmbedder]` prefixes for embedder providers
- `[PubMedClient]` prefix for NCBI API interactions
- `[pipeline]` dead letter queue operations logged at WARNING level