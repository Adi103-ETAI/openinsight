# OpenInsight — Kaggle Ingestion Quick Start

**Goal**: Get 50 real articles indexed in Milvus + MongoDB in under 1 hour, for free, on Kaggle.

## Step 0: Accounts (15 min, one-time)

Sign up for these 3 free services:

| Service | Sign up URL | What you get | Time |
|---|---|---|---|
| MongoDB Atlas | mongodb.com/atlas | 512MB free database (Mumbai region) | 5 min |
| Zilliz Cloud | zilliz.com | 2M free vectors (Mumbai region) | 5 min |
| NCBI Account | ncbi.nlm.nih.gov/account | API key (10 req/sec vs 3) | 3 min |

## Step 1: MongoDB Atlas Setup (5 min)

1. Create **M0 Free** cluster
2. **Region**: AWS / Mumbai (ap-south-1) — required for DPDP compliance
3. **Database User**: create username + password (write these down)
4. **Network Access**: add `0.0.0.0/0` (allow access from anywhere — Kaggle IPs vary)
5. **Connect → Python → Driver**: copy the connection string
   - Looks like: `mongodb+srv://user:pass@cluster0.xxxxx.mongodb.net/?retryWrites=true&w=majority`

## Step 2: Zilliz Cloud Setup (5 min)

1. Create **Free** cluster
2. **Region**: GCP / asia-south1 (Mumbai)
3. Copy **Cluster URI** (looks like `https://in03-xxxxx.server.gcp-asia-south1.cloud.zilliz.com`)
4. Copy **API Token** (looks like `db-xxxxx:yyyyy`)

## Step 3: NCBI API Key (3 min)

1. Sign in at ncbi.nlm.nih.gov/account
2. Settings → API Key Management → Create
3. Copy the key (looks like `abc123def456...`)

## Step 4: Kaggle Secrets (3 min)

1. Go to kaggle.com → Create New Notebook
2. **Settings** (right sidebar):
   - Accelerator: **GPU T4 × 2**
   - Internet: **ON**
   - Persistence: **ON** (V2)
3. **Add-ons → Secrets** (or click "Secrets" in right sidebar):
   - Add `MONGODB_URL` → paste your Atlas connection string
   - Add `ZILLIZ_URI` → paste your Zilliz cluster URI
   - Add `ZILLIZ_TOKEN` → paste your Zilliz API token
   - Add `NCBI_API_KEY` → paste your NCBI key

## Step 5: Upload the Notebook (2 min)

1. Download `notebooks/kaggle_ingestion_v2.ipynb` from the repo
2. In Kaggle: **File → Import Notebook → Upload** the .ipynb file
3. Open it

## Step 6: Verify Setup (3 min)

1. Add a new cell at the top of the notebook
2. Paste this:
   ```python
   !python notebooks/verify_kaggle_setup.py
   ```
   (Actually, since the repo isn't cloned yet, run the verification as a cell:)
   ```python
   # Quick check that Kaggle Secrets are accessible
   from kaggle_secrets import UserSecretsClient
   sc = UserSecretsClient()
   for name in ["MONGODB_URL", "ZILLIZ_URI", "ZILLIZ_TOKEN", "NCBI_API_KEY"]:
       val = sc.get_secret(name)
       print(f"  {name}: {'set (' + str(len(val)) + ' chars)' if val else 'NOT SET'}")
   ```
3. Run it — all 4 should say "set"
4. If any say "NOT SET", go back to Step 4

## Step 7: Run a Small Test (20 min)

In the notebook, edit the CONFIGURATION cell:

```python
SOURCES = ["pubmed"]           # Just PubMed for the test
LIMIT_PER_SOURCE = 10          # Only 10 articles
JOURNALS = ["Indian J Med Res"]  # Just one journal
BATCH_SIZE = 5
NEEDS_GROBID = False           # PubMed doesn't need GROBID
```

Then **Run All**. You should see:

```
=== Installing dependencies ===  (5 min)
✓ Dependencies installed
=== Cloning re-insight branch ===  (30 sec)
✓ Repo at /kaggle/working/openinsight
=== Setting up environment ===
=== GROBID not needed (no PDF sources) — skipping ===
=== Verifying setup ===
✓ GPU: Tesla T4 (15.0 GB VRAM)
✓ MongoDB reachable
✓ Zilliz/Milvus reachable

============================================================
OpenInsight Kaggle Ingestion
============================================================
Sources: ['pubmed']
Limit per source: 10
Journals: ['Indian J Med Res']
============================================================

--- Ingesting from: pubmed (limit=10) ---
  Discovered 10 URLs
  [5/10] ingested=5 failed=0
  [10/10] ingested=10 failed=0

✓ pubmed: {'source': 'pubmed', 'ingested': 10, 'failed': 0, 'discovered': 10}

============================================================
Ingestion Complete
============================================================
  pubmed: ingested=10 failed=0

Total articles ingested: 10
```

**If this works — congratulations, the system is end-to-end functional.** You now have 10 real IJMR articles indexed and searchable.

## Step 8: Verify the Data (5 min)

In a new cell:

```python
import sys
sys.path.insert(0, "/kaggle/working/openinsight")
import os
os.environ["VECTOR_URI"] = UserSecretsClient().get_secret("ZILLIZ_URI")
os.environ["VECTOR_TOKEN"] = UserSecretsClient().get_secret("ZILLIZ_TOKEN")

from pymilvus import connections, Collection
connections.connect(alias="default", uri=os.environ["VECTOR_URI"], token=os.environ["VECTOR_TOKEN"])
coll = Collection("openinsight_v2")
coll.load()
print(f"Total vectors in Milvus: {coll.num_entities}")

# Search for "diabetes"
from src.ml.embedding.embedder import get_embedder
embedder = get_embedder()
query_emb = embedder.embed_query("diabetes treatment in India")
results = coll.search(
    data=[query_emb.tolist()],
    anns_field="dense",
    param={"metric_type": "COSINE", "params": {"nprobe": 10}},
    limit=5,
    output_fields=["raw_text", "title", "source_type", "journal"]
)
for hit in results[0]:
    print(f"  [{hit.score:.3f}] {hit.entity.get('title', '')[:60]}")
    print(f"    source: {hit.entity.get('source_type')}")
```

You should see 5 results from the 10 articles you just ingested.

## Step 9: Scale Up (when ready)

Once the test works, scale up:

```python
# Small run — 200 articles from 4 sources
SOURCES = ["pubmed", "statpearls", "cdsco", "ctri"]
LIMIT_PER_SOURCE = 50
# ~30 minutes, ~200 articles

# Medium run — 1000 articles
SOURCES = ["pubmed", "statpearls", "ncbi_bookshelf", "cdsco", "ctri"]
LIMIT_PER_SOURCE = 200
# ~2 hours, ~1000 articles

# Full PubMed Indian journals run — 15K articles (takes multiple sessions)
SOURCES = ["pubmed"]
LIMIT_PER_SOURCE = 500  # 30 journals × 500 = 15K
JOURNALS = None  # all 30 Indian journals
# ~6 hours per session, ~3 sessions
```

## Troubleshooting

**"ModuleNotFoundError: No module named 'src'"** — Add `sys.path.insert(0, "/kaggle/working/openinsight")` at the top of every cell that imports from `src.*`.

**"pymilvus connection timeout"** — Zilliz free tier has cold-start latency. The first connection takes 30s. Subsequent connections are fast.

**"CUDA out of memory"** — Reduce `BATCH_SIZE` from 10 to 5, and reduce embedding batch size in `embedder.embed_batch()` from 32 to 16.

**"PubMed 429 Too Many Requests"** — Your NCBI_API_KEY isn't being picked up. Verify it's in Kaggle Secrets and the env var is set in `setup_env()`.

**"GROBID failed to start"** — Java version issue. Run `!java -version` in a cell. If it's not Java 11+, install it: `!apt-get install -y openjdk-11-jre`.

**Session dies at hour 8** — This is normal. The checkpoint file at `/kaggle/working/checkpoint.json` records which sources completed. Just re-run the notebook — it'll skip completed sources and resume.

## What's Next

After the test run works:
1. Scale up to a real run (Step 9)
2. Build the `/search` API server on a DigitalOcean droplet (separate from Kaggle — Kaggle is only for ingestion)
3. Point the frontend at the API
4. You have a working clinical RAG system
