# Data Ingestion Pipeline

## Overview

The ingestion pipeline transforms raw source documents into searchable vector embeddings stored in Milvus, with full documents and metadata stored in MongoDB.

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
│  │  │  (pdfplumb) │  │   (XML)     │  │  (TEI)      │                  │    │
│  │  └─────────────┘  └─────────────┘  └─────────────┘                  │    │
│  │                                                                     │    │
│  │  Fallback: OCRParser for scanned PDFs                               │    │
│  │  Parallel: 4 worker threads per batch                               │    │
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
│  │  DualEmbedderV2                                                     │    │
│  │                                                                     │    │
│  │  Dense Embedding (PubMedBERT):                                      │    │
│  │    - Model: pritamdeka/S-PubMedBert-MS-MARCO                        │    │
│  │    - Dimension: 768                                                 │    │
│  │    - Batch size: 32                                                 │    │
│  │    - Input: contextual_text (title + section + content)             │    │
│  │                                                                     │    │
│  │  Sparse Embedding (TF-IDF):                                         │    │
│  │    - Vocabulary: 50,000                                             │    │
│  │    - Medical compound terms: 3.5x weight                            │    │
│  │    - Long terms (>10 chars): 3.0x weight                            │    │
│  │    - Stopwords: 96 removed                                          │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  Output: (dense_vector [768d], sparse_vector) per chunk                     │
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
│  │  }                                                                  │    │
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
INGESTION_WORKERS = 4             # Thread pool workers
INGESTION_MAX_RETRIES = 3         # Retry attempts

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

# PubMed API Rate Limiting (configurable per API tier)
PUBMED_RATE_LIMIT_SECONDS = 0.34  # Without API key (~3 requests/sec)
PUBMED_RATE_LIMIT_WITH_KEY = 0.1 # With API key (~10 requests/sec)
```

### Rate Limiting for External APIs

The CDC and WHO parsers use configurable rate limiting when accessing PubMed:

- **Without NCBI API key**: Uses `pubmed_rate_limit_seconds` (default: 0.34s between requests)
- **With NCBI API key**: Uses `pubmed_rate_limit_with_key` (default: 0.1s between requests)

This allows adjusting request rates based on your NCBI API tier to avoid rate limiting errors.

---

## Performance

| Stage | Throughput |
|-------|------------|
| Parsing | ~10 docs/sec |
| Chunking | ~50 docs/sec |
| Embedding | ~20 docs/sec |
| Indexing | ~30 docs/sec |
| **Total** | ~8-10 docs/sec |