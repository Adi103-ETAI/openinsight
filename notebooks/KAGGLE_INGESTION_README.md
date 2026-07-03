# OpenInsight — Kaggle Ingestion Notebook

**Purpose**: Run the OpenInsight ingestion pipeline on Kaggle's free GPU (T4 × 2, 30h/week) to populate Milvus + MongoDB with indexed medical content.

**This is the only way to actually ingest data without paying for GPUs.**

## What This Notebook Does

1. **Clones the OpenInsight repo** (latest `re-insight` branch) into the Kaggle environment
2. **Installs dependencies** (sentence-transformers, pymilvus, motor, etc.) — ~5 min
3. **Starts GROBID** as a background subprocess (for PDF parsing) — ~3 min startup
4. **Connects to cloud services**: MongoDB Atlas + Zilliz Cloud (via Kaggle Secrets)
5. **Discovers + fetches + parses + embeds + stores** articles from any of the 21 sources
6. **Checkpoints every 50 docs** to `/kaggle/working/checkpoint.json` — resume after session death
7. **Reports progress** every batch with timing + counts

## Prerequisites

### 1. MongoDB Atlas (free, 512MB)
- Sign up at mongodb.com/atlas
- Create M0 free cluster in **Mumbai region** (aws-ap-south-1) for DPDP compliance
- Database user: read+write
- Network access: allow `0.0.0.0/0` (Kaggle IPs vary)
- Connection string: `mongodb+srv://<user>:<pass>@<cluster>.mongodb.net/openinsight`

### 2. Zilliz Cloud (free, 2M vectors)
- Sign up at zilliz.com
- Create free cluster in **Mumbai region** (gcp-asia-south1)
- Get URI + token
- URI: `https://in03-xxxxx.server.gcp-asia-south1.cloud.zilliz.com`
- Token: `db-xxxxx:yyyyy`

### 3. NCBI API Key (free, raises rate limit 3→10 req/sec)
- Sign up at ncbi.nlm.nih.gov/account
- Generate API key
- Without this: PubMed/PMC scraping works but is 3x slower

### 4. Kaggle Secrets
Add these as Kaggle Secrets (Add-ons → Secrets):
- `MONGODB_URL`: your Atlas connection string
- `ZILLIZ_URI`: your Zilliz cluster URI
- `ZILLIZ_TOKEN`: your Zilliz token
- `NCBI_API_KEY`: your NCBI key

## Kaggle Settings
- **Accelerator**: GPU T4 × 2 (or T4 × 1)
- **Internet**: ON (required for git clone + NCBI + cloud services)
- **Persistence**: ON (so checkpoints survive across sessions)
- **Environment**: Use default (no custom Docker)

## Usage

### Run all sources (small test — 50 articles each)
```python
SOURCES = ["pubmed", "medknow", "statpearls", "cdsco", "ctri"]
LIMIT_PER_SOURCE = 50
```

### Targeted run — Indian journals for diabetes
```python
SOURCES = ["pubmed"]
JOURNALS = ["Indian J Endocr Metab", "J Assoc Physicians India"]
LIMIT_PER_SOURCE = 200
```

### Full PubMed Indian journal run (will take multiple sessions)
```python
SOURCES = ["pubmed"]
LIMIT_PER_SOURCE = 500  # 30 journals × 500 = 15K articles
# Session 1: journals 0-5 (5K articles, ~6 hours)
# Session 2: journals 6-12
# Session 3: journals 13-20
# Session 4: journals 21-30
```

## Checkpointing

The notebook writes `/kaggle/working/checkpoint.json` after every batch:
```json
{
  "source": "pubmed",
  "journal_index": 5,
  "articles_ingested": 2340,
  "articles_failed": 12,
  "last_url": "https://eutils.ncbi.nlm.nih.gov/...",
  "timestamp": "2026-07-03T14:30:00Z"
}
```

If a session dies, the next session reads this file and resumes from `journal_index`.

## GROBID on Kaggle

GROBID is a Java application. We:
1. Download the latest release ZIP (~400MB)
2. Unzip to `/kaggle/working/grobid/`
3. Start with `./gradlew run --args="server live"` in the background
4. Wait for `http://localhost:8070/api/isalive` to return 200
5. Point `src/ingestion/parsers/grobid.py` at it

GROBID uses ~4GB RAM + 2GB VRAM (it has a CRF model for entity recognition). On Kaggle T4, this coexists with sentence-transformers fine.

## Quota Math

- **Per article**: ~3s (1s scrape + 1s embed + 0.5s index + 0.5s store)
- **Per 9h session**: ~10,800 articles (9h × 3600s / 3s)
- **Per week (30h)**: ~36,000 articles
- **Phase 1 target (Indian journals)**: ~30K articles → 1 session (9h)
- **Phase 2 target (StatPearls + Bookshelf + NMC + manuals)**: ~10K → 1 session
- **Phase 3 target (CDSCO + CTRI + PvPI + NFI)**: ~15K → 1.5 sessions
- **Phase 4 target (specialty guidelines)**: ~100 PDFs → 30 min (GROBID bottleneck)

**Total: ~4-5 Kaggle sessions to index everything.**

## Troubleshooting

**"GROBID not starting"** — Java version mismatch. Use `JAVA_HOME=/opt/conda` if default Java is wrong.

**"Zilliz connection timeout"** — Free tier has cold-start latency (30s). Retry logic in `http_client.py` handles this.

**"Embedding CUDA OOM"** — Reduce batch_size in `embedder.embed_batch()` from 32 to 16.

**"PubMed 429 Too Many Requests"** — Set NCBI_API_KEY Kaggle Secret (raises rate from 3→10 req/sec).
