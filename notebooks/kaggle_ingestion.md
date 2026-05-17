# OpenInsight Data Ingestion Pipeline - Kaggle Edition

## Knowledge

This guide explains how to run the full ingestion pipeline on Kaggle's free GPU (T4/T4x2) while storing data in Zilliz Cloud (Milvus) and MongoDB Atlas.

## Architecture

```text
Kaggle GPU (T4, 16GB VRAM)
  └── PDFs (Kaggle Dataset)
       ↓
  GROBID/OCR → Structured Text
       ↓
  Dedup → Enrich → Chunk (350 tokens)
       ↓
  S-PubMedBert (768d dense) ──→ Zilliz Cloud
  TF-IDF (sparse, CPU-only)  ──→ Zilliz Cloud
       ↓
  Metadata ──→ MongoDB Atlas
  Checkpoints ──→ /kaggle/working (5GB persistent)
```

## Setup Instructions

1. Upload your medical PDFs as a private Kaggle Dataset.
2. Enable Internet in Settings so Kaggle can reach Zilliz and MongoDB.
3. Enable GPU acceleration in Settings.
4. Fill in the configuration cell below with your credentials.
5. Run the code cells in order.

## Quotas

- Kaggle Free: 30 GPU hours per week, 12 hour max session, 100GB private datasets.
- Zilliz Cloud Free: 1 collection, 1M vectors.
- MongoDB Atlas Free: 512MB storage.

---

## 1. Configuration

### Knowledge

Replace the placeholder values below with your own credentials and dataset path. Never commit these secrets to a public repository.

### Code

```python
import os

# ============ REQUIRED: Fill in your credentials ============

# Zilliz Cloud (Milvus)
ZILLIZ_URI = "https://in01-xxxx.api.gcp-us-west1.zillizcloud.com"  # Your Zilliz Cloud endpoint
ZILLIZ_TOKEN = "your_zilliz_api_key"  # Your Zilliz Cloud API key

# MongoDB Atlas
MONGODB_URL = "mongodb+srv://user:password@cluster.mongodb.net/openinsight"  # Your Atlas connection string
MONGODB_DB = "openinsight"

# NCBI (optional, for PubMed fetching)
NCBI_API_KEY = ""  # Get free at https://www.ncbi.nlm.nih.gov/account/
NCBI_EMAIL = "your_email@example.com"

# ============ Ingestion Settings ============

# Source type: pubmed, icmr, cochrane, who, cdc, statpearls, nmc_guideline, rssdi
SOURCE_TYPE = "pubmed"

# Path to PDFs (Kaggle auto-mounts datasets under /kaggle/input/)
DATA_DIR = "/kaggle/input/your-dataset-name/"  # Change to your dataset name

# Batch size for processing
BATCH_SIZE = 10

# Set environment variables for the pipeline
os.environ["EMBED_PROVIDER"] = "local"  # Use local GPU for ingestion
os.environ["RERANK_PROVIDER"] = "local"  # Use local GPU for reranking
os.environ["DENSE_MODEL_NAME"] = "pritamdeka/S-PubMedBert-MS-MARCO"
os.environ["RERANKER_MODEL_NAME"] = "BAAI/bge-reranker-v2-m3"
os.environ["VECTOR_URI"] = ZILLIZ_URI
os.environ["VECTOR_TOKEN"] = ZILLIZ_TOKEN
os.environ["MONGODB_URL"] = MONGODB_URL
os.environ["MONGODB_DB"] = MONGODB_DB
os.environ["NCBI_API_KEY"] = NCBI_API_KEY
os.environ["NCBI_EMAIL"] = NCBI_EMAIL
os.environ["MILVUS_CLOUD"] = "true"
os.environ["VECTOR_COLLECTION_V2"] = "openinsight_v2"

print("Configuration set!")
print(f"Source type: {SOURCE_TYPE}")
print(f"Data directory: {DATA_DIR}")
```

---

## 2. Install Dependencies

### Knowledge

This cell installs the Python packages needed by the notebook and starts a local GROBID server for PDF parsing. Kaggle does not provide these services by default.

### Code

```python
# Install OpenInsight dependencies
!pip install -q sentence-transformers pymilvus pymongo httpx loguru pydantic-settings transformers

# Install GROBID dependencies (Java + GROBID server)
!apt-get update -qq && apt-get install -qq -y default-jre wget > /dev/null 2>&1

# Download and start GROBID server
!wget -q https://github.com/grobidOrg/grobid/releases/download/0.9.0/grobid-0.9.0.zip
!unzip -q -o grobid-0.9.0.zip -d grobid

import subprocess
import time

# Start GROBID in background
grobid_proc = subprocess.Popen(
    ["./grobid/grobid-0.9.0/bin/grobid-server", "--port", "8070"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
)

# Wait for GROBID to start
print("Starting GROBID server...")
time.sleep(30)

# Verify GROBID is running
import httpx
try:
    resp = httpx.get("http://localhost:8070/api/isalive", timeout=10)
    print(f"GROBID server status: {'ALIVE' if resp.status_code == 200 else 'NOT RESPONDING'}")
except Exception as e:
    print(f"GROBID server may still be starting: {e}")

os.environ["GROBID_URL"] = "http://localhost:8070"

# Verify GPU
import torch
print(f"\nGPU available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
```

---

## 3. Clone and Setup OpenInsight

### Knowledge

This step clones the repository into Kaggle's working directory and adds it to the Python path so the notebook can import the project code.

### Code

```python
import sys

# Clone the repo (or upload as a Kaggle dataset for faster access)
!git clone -b restruct https://github.com/Adi103-ETAI/openinsight.git /kaggle/working/openinsight

# Add to Python path
sys.path.insert(0, "/kaggle/working/openinsight")

print("OpenInsight repo cloned and added to path!")
```

---

## 4. Verify Connections

### Knowledge

Before running ingestion, confirm that Kaggle can reach both Zilliz Cloud and MongoDB Atlas.

### Code

```python
from pymilvus import MilvusClient
from pymongo import MongoClient

# Test Zilliz Cloud connection
try:
    zilliz_client = MilvusClient(uri=ZILLIZ_URI, token=ZILLIZ_TOKEN)
    collections = zilliz_client.list_collections()
    print(f"Zilliz Cloud: Connected! Existing collections: {collections}")
except Exception as e:
    print(f"Zilliz Cloud: FAILED - {e}")
    print("Check your VECTOR_URI and VECTOR_TOKEN")

# Test MongoDB Atlas connection
try:
    mongo_client = MongoClient(MONGODB_URL, serverSelectionTimeoutMS=5000)
    mongo_client.server_info()  # Forces connection test
    print(f"MongoDB Atlas: Connected! Database: {MONGODB_DB}")
except Exception as e:
    print(f"MongoDB Atlas: FAILED - {e}")
    print("Check your MONGODB_URL")
```

---

## 5. Run Ingestion Pipeline

### Knowledge

This is the main ingestion step. It parses source files, deduplicates them, enriches metadata, chunks text, scores quality, generates embeddings, and stores everything in Zilliz Cloud and MongoDB Atlas.

### Code

```python
import asyncio
from src.ingestion.pipeline import IngestionPipeline

async def run_ingestion():
    pipeline = IngestionPipeline()
    
    print(f"Starting ingestion for source: {SOURCE_TYPE}")
    print(f"Data directory: {DATA_DIR}")
    print(f"Batch size: {BATCH_SIZE}")
    print("=" * 50)
    
    summary = await pipeline.ingest_directory(
        directory=DATA_DIR,
        source=SOURCE_TYPE,
        recreate_index=False,  # Set True to recreate collection
        batch_size=BATCH_SIZE,
        resume=True,  # Resume from checkpoint if interrupted
        reset=False,  # Set True to start fresh
    )
    
    print("\n" + "=" * 50)
    print("INGESTION COMPLETE")
    print("=" * 50)
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    return summary

# Run the pipeline
summary = await run_ingestion()
```

---

## 6. Verify Ingested Data

### Knowledge

After ingestion, this checks whether the Milvus collection and MongoDB collections contain data.

### Code

```python
# Verify data in Zilliz Cloud
from pymilvus import MilvusClient

zilliz_client = MilvusClient(uri=ZILLIZ_URI, token=ZILLIZ_TOKEN)

collection_name = os.environ.get("VECTOR_COLLECTION_V2", "openinsight_v2")

if zilliz_client.has_collection(collection_name):
    stats = zilliz_client.get_collection_stats(collection_name)
    print(f"Zilliz Cloud - Collection: {collection_name}")
    print(f"  Row count: {stats.get('row_count', 'unknown')}")
else:
    print(f"Collection {collection_name} not found!")

# Verify data in MongoDB Atlas
from pymongo import MongoClient

mongo_client = MongoClient(MONGODB_URL)
db = mongo_client[MONGODB_DB]

doc_count = db.documents.count_documents({})
chunk_count = db.chunks.count_documents({})
failed_count = db.failed_documents.count_documents({})

print(f"\nMongoDB Atlas - Database: {MONGODB_DB}")
print(f"  Documents: {doc_count}")
print(f"  Chunks: {chunk_count}")
print(f"  Failed (dead letter): {failed_count}")
```

---

## 7. Quick Search Test

### Knowledge

This optional check confirms that the ingested chunks are searchable with a sample query.

### Code

```python
# Quick search test
from src.ml.embedding.embedder import LocalEmbedder
from pymilvus import MilvusClient

embedder = LocalEmbedder("pritamdeka/S-PubMedBert-MS-MARCO")
zilliz_client = MilvusClient(uri=ZILLIZ_URI, token=ZILLIZ_TOKEN)

test_query = "What are the current guidelines for type 2 diabetes management in India?"

# Embed query
query_vector = embedder.embed_query(test_query).tolist()

# Search Zilliz Cloud
results = zilliz_client.search(
    collection_name=collection_name,
    data=[query_vector],
    anns_field="dense",
    limit=5,
    output_fields=["chunk_id", "source", "source_type", "chunk_type"],
)

print(f"Query: {test_query}")
print(f"\nTop 5 results:")
if results and results[0]:
    for i, hit in enumerate(results[0]):
        entity = hit.get("entity", {})
        print(f"  {i+1}. Score: {hit.get('distance', 0):.4f} | "
              f"Source: {entity.get('source_type', '?')} | "
              f"Type: {entity.get('chunk_type', '?')}")
else:
    print("  No results found. Check your ingestion.")
```

---

## 8. Ingest Multiple Sources

### Knowledge

Use this section if you want to run several source folders one after another.

### Code

```python
# Ingest multiple sources
# Update paths to match your Kaggle dataset structure

SOURCES = {
    # "pubmed": "/kaggle/input/your-dataset/pubmed/",
    # "icmr": "/kaggle/input/your-dataset/icmr/",
    # "who": "/kaggle/input/your-dataset/who/",
    # "cochrane": "/kaggle/input/your-dataset/cochrane/",
    # "cdc": "/kaggle/input/your-dataset/cdc/",
    # "statpearls": "/kaggle/input/your-dataset/statpearls/",
}

async def ingest_all_sources():
    pipeline = IngestionPipeline()
    all_summaries = {}
    
    for source, data_dir in SOURCES.items():
        print(f"\n{'='*50}")
        print(f"Ingesting source: {source}")
        print(f"Directory: {data_dir}")
        print('='*50)
        
        try:
            summary = await pipeline.ingest_directory(
                directory=data_dir,
                source=source,
                batch_size=BATCH_SIZE,
                resume=True,
            )
            all_summaries[source] = summary
            print(f"  Completed: {summary.get('chunks_indexed', 0)} chunks indexed")
        except Exception as e:
            print(f"  FAILED: {e}")
            all_summaries[source] = {"error": str(e)}
    
    return all_summaries

# Uncomment to run:
# summaries = await ingest_all_sources()
# for source, summary in summaries.items():
#     print(f"{source}: {summary}")
```

---

## 9. Save Progress and Cleanup

### Knowledge

Kaggle automatically persists files in `/kaggle/working/`. Saving a summary there makes it easier to inspect the previous run later.

### Code

```python
# Save ingestion summary to persistent storage
import json

summary_path = "/kaggle/working/ingestion_summary.json"

try:
    with open(summary_path, "w") as f:
        json.dump({"last_run": summary}, f, indent=2, default=str)
    print(f"Summary saved to {summary_path}")
except Exception as e:
    print(f"Could not save summary: {e}")

# Stop GROBID server
try:
    grobid_proc.terminate()
    print("GROBID server stopped.")
except:
    pass

print("\nDone! Your data is now in Zilliz Cloud and MongoDB Atlas.")
print("Connect your query pipeline to these databases to start searching.")
```
