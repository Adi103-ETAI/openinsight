# (B) Colab Notebook: GPU-based Ingestion to Zilliz Cloud

This notebook runs data ingestion in Google Colab (with GPU) and stores embeddings & metadata in Zilliz Cloud (Milvus managed) + MongoDB.

---

## Setup

### Cell 1: Install dependencies

```python
# Install required packages
!pip install -q torch sentence-transformers pymilvus pymongo python-dotenv scikit-learn loguru

# Verify GPU is available
import torch
print(f"GPU available: {torch.cuda.is_available()}")
print(f"GPU device: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'N/A'}")
```

### Cell 2: Set environment variables

```python
import os
from google.colab import userdata

# Fetch secrets from Colab Secrets
ZILLIZ_ENDPOINT = userdata.get('ZILLIZ_ENDPOINT')  # e.g., https://in02-abc.api.gcp-us-west1.zilliz.cloud
ZILLIZ_TOKEN = userdata.get('ZILLIZ_TOKEN')
ZILLIZ_DB_NAME = "default"

MONGODB_URL = userdata.get('MONGODB_URL')  # e.g., mongodb+srv://user:pass@cluster.mongodb.net/dbname?retryWrites=true
MONGODB_DB = "openinsight"

# Set environment for ingestion config
os.environ['VECTOR_URI'] = ZILLIZ_ENDPOINT
os.environ['VECTOR_TOKEN'] = ZILLIZ_TOKEN
os.environ['MILVUS_DB_NAME'] = ZILLIZ_DB_NAME
os.environ['MILVUS_CLOUD'] = 'true'

os.environ['MONGODB_URL'] = MONGODB_URL
os.environ['MONGODB_DB'] = MONGODB_DB

print("✓ Environment variables set")
print(f"  Zilliz endpoint: {ZILLIZ_ENDPOINT}")
print(f"  MongoDB: {MONGODB_URL.split('@')[1] if '@' in MONGODB_URL else 'configured'}")
```

---

## Ingestion Setup

### Cell 3: Import ingestion modules

```python
import sys
sys.path.insert(0, '/content')  # If cloning repo into /content

from pymilvus import MilvusClient
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
import numpy as np
from loguru import logger
import json
from datetime import datetime

logger.remove()
logger.add(lambda msg: print(msg, end=""), level="INFO")

print("✓ Imports successful")
```

### Cell 4: Initialize clients

```python
# Initialize Milvus (Zilliz Cloud)
milvus_client = MilvusClient(
    uri=ZILLIZ_ENDPOINT,
    token=ZILLIZ_TOKEN,
    db_name=ZILLIZ_DB_NAME,
)

# Initialize MongoDB
mongo_client = MongoClient(MONGODB_URL)
mongo_db = mongo_client[MONGODB_DB]
mongo_docs_collection = mongo_db["documents"]  # For metadata

# Initialize embedding model (loads on GPU if available)
embed_model = SentenceTransformer("pritamdeka/S-PubMedBert-MS-MARCO")
if torch.cuda.is_available():
    embed_model = embed_model.cuda()

print("✓ Clients initialized")
print(f"  Milvus collections: {milvus_client.list_collections()}")
print(f"  MongoDB collections: {mongo_db.list_collection_names()}")
print(f"  Embedding model device: {next(embed_model.parameters()).device}")
```

---

## Data Ingestion

### Cell 5: Create vector collection (if not exists)

```python
from pymilvus import DataType

COLLECTION_NAME = "openinsight_v2"
VECTOR_DIM = 768

if milvus_client.has_collection(COLLECTION_NAME):
    print(f"✓ Collection '{COLLECTION_NAME}' already exists")
else:
    # Create schema
    schema = MilvusClient.create_schema(
        auto_id=False,
        enable_dynamic_field=True,
    )
    schema.add_field(
        field_name="id",
        datatype=DataType.VARCHAR,
        is_primary=True,
        max_length=128,
    )
    schema.add_field(
        field_name="dense",
        datatype=DataType.FLOAT_VECTOR,
        dim=VECTOR_DIM,
    )
    schema.add_field(
        field_name="sparse",
        datatype=DataType.SPARSE_FLOAT_VECTOR,
    )
    schema.add_field(field_name="source_type", datatype=DataType.VARCHAR, max_length=64)
    schema.add_field(field_name="year", datatype=DataType.INT64)
    schema.add_field(field_name="evidence_level", datatype=DataType.VARCHAR, max_length=16)
    
    # Create index params
    index_params = milvus_client.prepare_index_params()
    index_params.add_index(field_name="dense", metric_type="COSINE")
    index_params.add_index(
        field_name="sparse",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="IP",
    )
    
    # Create collection
    milvus_client.create_collection(
        collection_name=COLLECTION_NAME,
        schema=schema,
        index_params=index_params,
    )
    print(f"✓ Created collection '{COLLECTION_NAME}'")
```

### Cell 6: Sample data (replace with your own)

```python
# Example sample data — replace with your actual ingestion source
SAMPLE_DOCUMENTS = [
    {
        "chunk_id": "doc_001_chunk_1",
        "doc_id": "doc_001",
        "raw_text": "Type 2 diabetes is a chronic disease characterized by insulin resistance. Treatment typically involves lifestyle modifications and medications.",
        "contextual_text": "Type 2 diabetes is a chronic disease characterized by insulin resistance. Treatment typically involves lifestyle modifications and medications.",
        "title": "Management of Type 2 Diabetes",
        "source_type": "pubmed",
        "year": 2023,
        "evidence_level": "1",
        "pmid": "12345678",
    },
    {
        "chunk_id": "doc_002_chunk_1",
        "doc_id": "doc_002",
        "raw_text": "Atrial fibrillation increases stroke risk significantly. Anticoagulation therapy is recommended for most patients.",
        "contextual_text": "Atrial fibrillation increases stroke risk significantly. Anticoagulation therapy is recommended for most patients.",
        "title": "Atrial Fibrillation Management",
        "source_type": "pubmed",
        "year": 2022,
        "evidence_level": "2a",
        "pmid": "87654321",
    },
]

print(f"✓ Loaded {len(SAMPLE_DOCUMENTS)} sample documents")
```

### Cell 7: Embed and upsert to Milvus

```python
def embed_documents(documents: list[dict], batch_size: int = 16) -> list[dict]:
    """
    Embed documents using SentenceTransformer.
    
    Args:
        documents: List of document dicts with 'contextual_text' key.
        batch_size: Batch size for embedding.
    
    Returns:
        List of docs with 'dense_embedding' added.
    """
    texts = [doc.get("contextual_text", "") for doc in documents]
    
    embeddings = embed_model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    
    for doc, emb in zip(documents, embeddings):
        doc["dense_embedding"] = emb.tolist()
    
    return documents


def prepare_milvus_rows(documents: list[dict]) -> list[dict]:
    """
    Prepare documents for Milvus upsert.
    
    Args:
        documents: List of embedded documents.
    
    Returns:
        List of rows ready for Milvus.
    """
    rows = []
    for doc in documents:
        row = {
            "id": doc.get("chunk_id", ""),
            "dense": doc.get("dense_embedding", []),
            "sparse": {},  # Empty for remote embedding (can add TF-IDF if needed)
            "raw_text": doc.get("raw_text", ""),
            "chunk_id": doc.get("chunk_id", ""),
            "doc_id": doc.get("doc_id", ""),
            "title": doc.get("title", ""),
            "source_type": doc.get("source_type", ""),
            "year": int(doc.get("year", 0)),
            "evidence_level": doc.get("evidence_level", "unknown"),
            "pmid": doc.get("pmid", ""),
        }
        rows.append(row)
    return rows


# Embed documents
print("Embedding documents...")
embedded_docs = embed_documents(SAMPLE_DOCUMENTS, batch_size=16)

# Prepare rows for Milvus
milvus_rows = prepare_milvus_rows(embedded_docs)

# Upsert to Milvus
print(f"\nUpserting {len(milvus_rows)} vectors to Milvus...")
result = milvus_client.upsert(
    collection_name=COLLECTION_NAME,
    data=milvus_rows,
)
print(f"✓ Upserted {result} vectors")
```

### Cell 8: Store metadata in MongoDB

```python
def store_metadata_mongodb(documents: list[dict], collection_name: str = "documents"):
    """Store document metadata in MongoDB."""
    coll = mongo_db[collection_name]
    
    metadata_docs = []
    for doc in documents:
        metadata = {
            "chunk_id": doc.get("chunk_id", ""),
            "doc_id": doc.get("doc_id", ""),
            "title": doc.get("title", ""),
            "source_type": doc.get("source_type", ""),
            "year": doc.get("year", 0),
            "evidence_level": doc.get("evidence_level", ""),
            "pmid": doc.get("pmid", ""),
            "ingested_at": datetime.utcnow(),
            "embedding_model": "pritamdeka/S-PubMedBert-MS-MARCO",
        }
        metadata_docs.append(metadata)
    
    # Upsert to MongoDB (replace if exists)
    for meta in metadata_docs:
        coll.replace_one(
            {"chunk_id": meta["chunk_id"]},
            meta,
            upsert=True,
        )
    
    return len(metadata_docs)


# Store metadata
num_stored = store_metadata_mongodb(SAMPLE_DOCUMENTS)
print(f"✓ Stored {num_stored} metadata docs in MongoDB")
```

---

## Validation

### Cell 9: Verify ingestion

```python
# Check Milvus stats
collection_stats = milvus_client.get_collection_stats(COLLECTION_NAME)
print(f"Milvus collection stats:")
print(f"  Total vectors: {collection_stats}")

# Check MongoDB stats
mongo_count = mongo_db["documents"].count_documents({})
print(f"\nMongoDB metadata docs: {mongo_count}")

# Sample query to validate
query_text = "Type 2 diabetes treatment"
query_embedding = embed_model.encode(
    query_text,
    normalize_embeddings=True,
    convert_to_numpy=True,
).tolist()

results = milvus_client.search(
    collection_name=COLLECTION_NAME,
    data=[query_embedding],
    anns_field="dense",
    limit=3,
    output_fields=["*"],
    search_params={"metric_type": "COSINE", "params": {"level": 2}},
)

print(f"\nSample search results for '{query_text}':")
if results and len(results) > 0:
    for hit in results[0]:
        print(f"  - {hit.get('entity', {}).get('title', 'N/A')} (score: {hit.get('distance', 0):.3f})")
else:
    print("  No results found")
```

### Cell 10: Export ingestion stats (optional)

```python
import json

# Create summary report
stats = {
    "timestamp": datetime.utcnow().isoformat(),
    "collection_name": COLLECTION_NAME,
    "total_vectors": collection_stats,
    "total_metadata": mongo_count,
    "embedding_model": "pritamdeka/S-PubMedBert-MS-MARCO",
    "vector_dimension": VECTOR_DIM,
    "milvus_endpoint": ZILLIZ_ENDPOINT,
    "mongodb_url": MONGODB_URL.split("@")[0] + "://***@" + (MONGODB_URL.split("@")[1] if "@" in MONGODB_URL else "***"),
}

# Save to file
report_json = json.dumps(stats, indent=2)
print("Ingestion Report:")
print(report_json)

# Save to MongoDB as a summary
mongo_db["ingestion_runs"].insert_one(stats)
print("\n✓ Summary saved to MongoDB (ingestion_runs collection)")
```

---

## Production Setup (simplified)

To run this on a schedule in production:

### Option 1: GitHub Actions (run periodically)

Create `.github/workflows/colab-ingestion.yml`:

```yaml
name: Colab Ingestion
on:
  schedule:
    - cron: "0 2 * * 0"  # Every Sunday at 2 AM UTC

jobs:
  ingest:
    runs-on: ubuntu-latest
    steps:
      - uses: colab-actions/run-notebook@main
        with:
          notebook: scripts/colab_ingestion.ipynb
          env:
            ZILLIZ_ENDPOINT: ${{ secrets.ZILLIZ_ENDPOINT }}
            ZILLIZ_TOKEN: ${{ secrets.ZILLIZ_TOKEN }}
            MONGODB_URL: ${{ secrets.MONGODB_URL }}
```

### Option 2: Manual local run

Save notebook as `.ipynb` and run:

```bash
jupyter nbconvert --to notebook --execute colab_ingestion.ipynb --output colab_ingestion_run.ipynb
```

### Option 3: Docker (if GPU available)

```dockerfile
FROM nvidia/cuda:12.0-runtime-ubuntu22.04

WORKDIR /app
COPY requirements.txt .
RUN pip install -q -r requirements.txt

COPY scripts/colab_ingestion_script.py .

CMD ["python", "colab_ingestion_script.py"]
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Embedding model not on GPU | Check `torch.cuda.is_available()` in cell 1 |
| Milvus connection refused | Verify `ZILLIZ_ENDPOINT` and `ZILLIZ_TOKEN` are correct |
| MongoDB connection timeout | Check VPC/network rules; use IP allowlist in MongoDB Atlas |
| Embedding dimension mismatch | Ensure model dim matches `VECTOR_DIM` (768 for S-PubMedBert) |
| Out of memory | Reduce `batch_size` in `embed_documents()` (try 8 or 4) |

---

## Next Steps

1. Replace `SAMPLE_DOCUMENTS` with your actual data source (PubMed, WHO, etc.)
2. Add error handling and retry logic for Milvus/MongoDB calls
3. Schedule the notebook to run periodically (see Production Setup)
4. Validate ingestion results in [openinsight-ui](https://github.com/Adi103-ETAI/openinsight-ui) by querying the API
5. (Optional) Add monitoring and alerting for failed ingestions

---

## Full notebook Python script (alternative: no Colab)

If running locally, save as `scripts/ingest_to_zilliz.py`:

```python
#!/usr/bin/env python3
"""
Standalone ingestion script for Zilliz Cloud + MongoDB.
Usage: python ingest_to_zilliz.py --data-file samples.json
"""

import argparse
import json
import sys
import os
from datetime import datetime

import torch
from pymilvus import MilvusClient, DataType
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from loguru import logger

logger.remove()
logger.add(sys.stderr, level="INFO")


def main():
    parser = argparse.ArgumentParser(description="Ingest to Zilliz Cloud")
    parser.add_argument("--data-file", required=True, help="JSON file with documents")
    parser.add_argument("--collection", default="openinsight_v2")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    # Load config from env
    zilliz_uri = os.getenv("VECTOR_URI")
    zilliz_token = os.getenv("VECTOR_TOKEN")
    mongo_url = os.getenv("MONGODB_URL")

    if not all([zilliz_uri, zilliz_token, mongo_url]):
        logger.error("Missing env vars: VECTOR_URI, VECTOR_TOKEN, MONGODB_URL")
        sys.exit(1)

    # Initialize clients
    milvus = MilvusClient(uri=zilliz_uri, token=zilliz_token)
    mongo = MongoClient(mongo_url)
    db = mongo["openinsight"]

    embed_model = SentenceTransformer("pritamdeka/S-PubMedBert-MS-MARCO")
    if torch.cuda.is_available():
        embed_model = embed_model.cuda()

    # Load data
    with open(args.data_file) as f:
        documents = json.load(f)

    logger.info(f"Loaded {len(documents)} documents")

    # Embed & upsert
    texts = [d.get("contextual_text", "") for d in documents]
    embeddings = embed_model.encode(
        texts, batch_size=args.batch_size, show_progress_bar=True,
        normalize_embeddings=True, convert_to_numpy=True
    )

    rows = [
        {
            "id": d.get("chunk_id", ""),
            "dense": emb.tolist(),
            "sparse": {},
            "raw_text": d.get("raw_text", ""),
            "title": d.get("title", ""),
            "source_type": d.get("source_type", ""),
            "year": int(d.get("year", 0)),
        }
        for d, emb in zip(documents, embeddings)
    ]

    result = milvus.upsert(collection_name=args.collection, data=rows)
    logger.info(f"✓ Upserted {result} vectors to {args.collection}")

    # Store metadata
    coll = db["documents"]
    for d in documents:
        coll.replace_one(
            {"chunk_id": d.get("chunk_id")},
            {**d, "ingested_at": datetime.utcnow()},
            upsert=True,
        )
    logger.info(f"✓ Stored {len(documents)} metadata docs")


if __name__ == "__main__":
    main()
```

Run it:

```bash
export VECTOR_URI="https://in02-abc.api.gcp-us-west1.zilliz.cloud"
export VECTOR_TOKEN="<token>"
export MONGODB_URL="mongodb+srv://..."
python scripts/ingest_to_zilliz.py --data-file samples.json --batch-size 32
```
