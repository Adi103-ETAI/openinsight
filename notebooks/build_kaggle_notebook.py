#!/usr/bin/env python3
"""Build kaggle_ingestion_v2.ipynb directly (not from a .py file).

Key fixes vs the broken version:
1. Inline code per cell (not wrapped in main() function)
2. Shell commands use !pip install / !git clone (shows output in notebook)
3. Async code uses top-level await (Jupyter supports this since IPython 7)
4. Each cell produces visible output
"""
import json
from pathlib import Path

cells: list[dict] = []

def md(text: str) -> None:
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text})

def code(text: str) -> None:
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": text,
    })

# === Intro ===
md("""# OpenInsight — Kaggle Ingestion

**Purpose**: Run the OpenInsight ingestion pipeline on Kaggle's free GPU to populate Milvus + MongoDB with indexed medical content.

## Setup (one-time, see KAGGLE_QUICKSTART.md)

1. **MongoDB Atlas** (free, 512MB) — M0 cluster in **Mumbai region**, connection string
2. **Zilliz Cloud** (free, 2M vectors) — cluster in **Mumbai region**, URI + token
3. **NCBI API Key** (free) — ncbi.nlm.nih.gov/account, raises rate 3→10 req/sec
4. **Kaggle Secrets** — add: `MONGODB_URL`, `ZILLIZ_URI`, `ZILLIZ_TOKEN`, `NCBI_API_KEY`
5. **Kaggle Settings** — GPU T4 × 2, Internet ON, Persistence ON

## How to use this notebook

Run cells top-to-bottom. Edit the **Configuration** cell (cell 5) to pick sources + limits.
""")

# === Cell 1: Install dependencies ===
md("## 1. Install dependencies")

code("""# Install Python deps not pre-installed on Kaggle
# Using !pip so output is visible in the notebook
!pip install -q pymilvus>=2.5,<2.6 motor==3.6.0 sentence-transformers==3.1.1 \\
    transformers==4.45.1 biopython==1.84 pdfplumber==0.11.4 \\
    beautifulsoup4==4.12.3 lxml==5.3.0 loguru==0.7.2 tenacity==8.5.0 \\
    trafilatura>=1.12.0 httpx==0.27.2

print("✓ Dependencies installed")
import torch
print(f"✓ PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB VRAM)")
""")

# === Cell 2: Clone repo ===
md("## 2. Clone the OpenInsight repo")

code("""import os
from pathlib import Path

REPO_DIR = "/kaggle/working/openinsight" if os.path.exists("/kaggle") else "/tmp/openinsight"
REPO_BRANCH = "re-insight"

if Path(REPO_DIR).exists():
    print(f"Repo exists at {REPO_DIR} — pulling latest")
    !cd {REPO_DIR} && git fetch origin {REPO_BRANCH} && git checkout {REPO_BRANCH} && git pull origin {REPO_BRANCH}
else:
    print(f"Cloning {REPO_BRANCH} branch to {REPO_DIR}")
    !git clone --branch {REPO_BRANCH} --depth 1 https://github.com/Adi103-ETAI/openinsight.git {REPO_DIR}

print(f"\\n✓ Repo ready at {REPO_DIR}")
""")

# === Cell 3: Load secrets + set env ===
md("""## 3. Load Kaggle Secrets + set environment variables

This reads secrets from Kaggle's Secrets store and sets them as environment variables that the OpenInsight pipeline reads.
""")

code("""import os

# Read secrets from Kaggle
try:
    from kaggle_secrets import UserSecretsClient
    secrets = UserSecretsClient()
    MONGODB_URL = secrets.get_secret("MONGODB_URL")
    ZILLIZ_URI = secrets.get_secret("ZILLIZ_URI")
    ZILLIZ_TOKEN = secrets.get_secret("ZILLIZ_TOKEN")
    NCBI_API_KEY = secrets.get_secret("NCBI_API_KEY") or ""
    print("✓ Read secrets from Kaggle Secrets store")
except ImportError:
    # Not on Kaggle — read from env (for local testing)
    MONGODB_URL = os.environ.get("MONGODB_URL", "")
    ZILLIZ_URI = os.environ.get("ZILLIZ_URI", os.environ.get("VECTOR_URI", ""))
    ZILLIZ_TOKEN = os.environ.get("ZILLIZ_TOKEN", os.environ.get("VECTOR_TOKEN", ""))
    NCBI_API_KEY = os.environ.get("NCBI_API_KEY", "")
    print("⚠ Not on Kaggle — reading from env vars")

# Set env vars for OpenInsight pipeline
os.environ["MONGODB_URL"] = MONGODB_URL
os.environ["VECTOR_URI"] = ZILLIZ_URI
os.environ["VECTOR_TOKEN"] = ZILLIZ_TOKEN
os.environ["NCBI_API_KEY"] = NCBI_API_KEY
os.environ["NVIDIA_NIM_API_KEY"] = "dummy-for-ingestion-only"
os.environ["APP_ENV"] = "kaggle"
os.environ["PYTHONPATH"] = REPO_DIR

# Add repo to Python path
import sys
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

# Show status (don't print actual values — they're secrets)
print(f"\\nMONGODB_URL: {'✓ set (' + str(len(MONGODB_URL)) + ' chars)' if MONGODB_URL else '✗ NOT SET'}")
print(f"ZILLIZ_URI:  {'✓ set (' + str(len(ZILLIZ_URI)) + ' chars)' if ZILLIZ_URI else '✗ NOT SET'}")
print(f"ZILLIZ_TOKEN:{' ✓ set (' + str(len(ZILLIZ_TOKEN)) + ' chars)' if ZILLIZ_TOKEN else ' ✗ NOT SET'}")
print(f"NCBI_API_KEY:{' ✓ set (10 req/sec)' if NCBI_API_KEY else ' ✗ not set (3 req/sec — slower)'}")
""")

# === Cell 4: Verify connections ===
md("""## 4. Verify connections

Confirms MongoDB Atlas + Zilliz Cloud are reachable before starting ingestion. **If this cell fails, fix the issue before continuing** — no point running a 6-hour GPU job if the database is unreachable.
""")

code("""import sys
sys.path.insert(0, REPO_DIR)

# --- MongoDB check (sync — uses pymongo directly for the ping) ---
try:
    from pymongo import MongoClient
    client = MongoClient(MONGODB_URL, serverSelectionTimeoutMS=10000)
    client.admin.command("ping")
    db = client["openinsight"]
    db["test_collection"].insert_one({"test": True})
    db["test_collection"].delete_many({})
    client.close()
    print("✓ MongoDB Atlas reachable + writable")
except Exception as e:
    print(f"✗ MongoDB Atlas: {e}")

# --- Zilliz check ---
try:
    from pymilvus import connections, utility
    connections.connect(alias="default", uri=ZILLIZ_URI, token=ZILLIZ_TOKEN)
    collections = utility.list_collections()
    print(f"✓ Zilliz/Milvus reachable ({len(collections)} existing collections: {collections})")
    connections.disconnect("default")
except Exception as e:
    print(f"✗ Zilliz/Milvus: {e}")

# --- GPU check ---
import torch
if torch.cuda.is_available():
    print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
else:
    print("✗ No GPU — enable GPU T4 in Kaggle settings")
""")

# === Cell 5: Configuration ===
md("""## 5. Configuration

Edit this cell to pick which sources to ingest and how many articles per source.

**For first test**: keep LIMIT_PER_SOURCE = 10 and SOURCES = ["pubmed"] to verify the pipeline works end-to-end in ~5 minutes.

**For real runs**: see KAGGLE_QUICKSTART.md for recommended configs.
""")

code("""# ============ EDIT THIS CELL ============

# Sources to ingest (any subset of the 21 registered sources)
# HTML sources (no GROBID needed): pubmed, medknow, pmc_india, statpearls, ncbi_bookshelf, cdsco, ctri
# PDF sources (need GROBID):       nmc_curriculum, pvpi, rssdi, csi, isccm, iap, fogsi, aios, isn, ntep, nvbdcp, nhm, npcds
SOURCES = [
    "pubmed",
    "statpearls",
    # "cdsco",
    # "ctri",
]

# Cap per source (10 for test, 200+ for real runs)
LIMIT_PER_SOURCE = 10

# For PubMed: specific journals (None = all 30 Indian journals)
JOURNALS = ["Indian J Med Res"]  # None for all

# Batch size for ingestion pipeline
BATCH_SIZE = 5

# ============ DON'T EDIT BELOW ============

# Does any selected source need GROBID (PDF parsing)?
PDF_SOURCES = {"nmc_curriculum", "pvpi", "rssdi", "csi", "isccm", "iap", "fogsi", "aios", "isn", "ntep", "nvbdcp", "nhm", "npcds"}
NEEDS_GROBID = any(s in PDF_SOURCES for s in SOURCES)

# Checkpoint file (survives across Kaggle sessions)
CHECKPOINT_FILE = "/kaggle/working/checkpoint.json" if os.path.exists("/kaggle") else "/tmp/checkpoint.json"

print(f"Sources: {SOURCES}")
print(f"Limit per source: {LIMIT_PER_SOURCE}")
if JOURNALS:
    print(f"PubMed journals: {JOURNALS}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Needs GROBID: {NEEDS_GROBID}")
print(f"Checkpoint file: {CHECKPOINT_FILE}")
""")

# === Cell 6: Start GROBID (only if needed) ===
md("""## 6. Start GROBID (only if PDF sources selected)

GROBID is a Java app for parsing PDFs. Only needed for sources that publish content as PDFs (NMC, PvPI, specialty society guidelines). Skipped automatically if you only selected HTML sources.
""")

code("""import subprocess
import time
import httpx

grobid_proc = None

if not NEEDS_GROBID:
    print("⏭  GROBID not needed (no PDF sources selected) — skipping")
else:
    grobid_dir = Path("/kaggle/working/grobid")
    grobid_version = "0.9.0"

    if not grobid_dir.exists():
        print(f"Downloading GROBID {grobid_version} (~400MB)...")
        !wget -q -O /tmp/grobid.zip https://github.com/kermitt2/grobid/releases/download/{grobid_version}/grobid-{grobid_version}.zip
        !unzip -q /tmp/grobid.zip -d /kaggle/working/
        !mv /kaggle/working/grobid-{grobid_version} {grobid_dir}
        !rm /tmp/grobid.zip

    print("Starting GROBID server (takes ~2 min)...")
    grobid_proc = subprocess.Popen(
        ["./gradlew", "run", "--args=server live"],
        cwd=str(grobid_dir),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env={**os.environ, "JAVA_OPTS": "-Xmx4g"},
    )

    # Wait for GROBID to be healthy (max 3 min)
    for i in range(36):
        try:
            r = httpx.get("http://localhost:8070/api/isalive", timeout=2)
            if r.status_code == 200:
                print(f"✓ GROBID healthy (took {i*5}s)")
                break
        except Exception:
            pass
        time.sleep(5)
    else:
        print("✗ GROBID failed to start within 3 minutes")
        if grobid_proc:
            grobid_proc.kill()
            grobid_proc = None
""")

# === Cell 7: Initialize pipeline ===
md("""## 7. Initialize the ingestion pipeline

This loads the embedding model (`pritamdeka/S-PubMedBert-MS-MARCO`, 768-dim) onto the GPU and connects to Milvus + MongoDB.
""")

code("""import sys
sys.path.insert(0, REPO_DIR)

from src.ingestion.pipeline import IngestionPipeline
from src.ingestion.scrapers import get_scraper

print("Initializing pipeline (loads embedding model — takes ~30s)...")
pipeline = IngestionPipeline()
print("✓ Pipeline ready")
print(f"  Embedder: {pipeline.embedder.__class__.__name__}")
print(f"  Vector collection: {pipeline.settings.vector_collection_v2}")
""")

# === Cell 8: Checkpoint helpers ===
md("""## 8. Checkpoint helpers

Writes progress to `/kaggle/working/checkpoint.json` after each source completes. If the Kaggle session dies (9-hour limit), the next run skips completed sources.
""")

code("""import json
from pathlib import Path

def load_checkpoint():
    if not Path(CHECKPOINT_FILE).exists():
        return {"completed_sources": [], "results": []}
    with open(CHECKPOINT_FILE) as f:
        return json.load(f)

def save_checkpoint(checkpoint):
    Path(CHECKPOINT_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(checkpoint, f, indent=2, default=str)

checkpoint = load_checkpoint()
print(f"Checkpoint loaded: {len(checkpoint['completed_sources'])} sources already completed")
if checkpoint["completed_sources"]:
    print(f"  Completed: {checkpoint['completed_sources']}")
""")

# === Cell 9: Ingestion ===
md("""## 9. Run ingestion

This cell iterates over the selected sources, discovers articles, fetches + parses + embeds + stores them. Progress is printed every batch.

**This is the main cell.** It uses `await` directly (Jupyter supports top-level await) — no `asyncio.run()` needed.
""")

code("""import hashlib
import importlib
from src.ingestion.document_db import ChunkRecord, DocumentRecord

def get_parser(source_name):
    \"\"\"Get the parser for a source, or None if no dedicated parser.\"\"\"
    parser_map = {
        "indmed": "src.ingestion.parsers.indmed:IndMEDParser",
        "medknow": "src.ingestion.parsers.medknow:MedknowParser",
        "pmc_india": "src.ingestion.parsers.pmc_india:PMCIndiaParser",
        "statpearls": "src.ingestion.parsers.statpearls_v2:StatPearlsParser",
        "ncbi_bookshelf": "src.ingestion.parsers.ncbi_bookshelf:NCBIBookshelfParser",
        "cdsco": "src.ingestion.parsers.cdsco:CDSCOParser",
        "ctri": "src.ingestion.parsers.ctri:CTRIParser",
        "nfi": "src.ingestion.parsers.nfi:NFIParser",
    }
    path = parser_map.get(source_name)
    if not path:
        return None
    module_path, class_name = path.split(":")
    module = importlib.import_module(module_path)
    return getattr(module, class_name)()


async def ingest_source(source_name, limit, journals=None):
    \"\"\"Ingest articles from one source.\"\"\"
    print(f"\\n{'='*60}")
    print(f"Ingesting from: {source_name} (limit={limit})")
    print(f"{'='*60}")

    scraper = get_scraper(source_name)
    total_ingested = 0
    total_failed = 0

    try:
        # Discover URLs
        if source_name == "pubmed" and journals:
            jobs = []
            per_journal = max(1, limit // len(journals))
            for journal in journals:
                j = await scraper.discover_by_journal(journal, max_results=per_journal)
                jobs.extend(j)
        else:
            jobs = await scraper.discover(max_results=limit)

        print(f"Discovered {len(jobs)} URLs")

        # Fetch + parse + batch-ingest
        parsed_docs = []
        for i, job in enumerate(jobs[:limit]):
            try:
                scraped = await scraper.fetch_one(job)
                if not scraped:
                    total_failed += 1
                    continue

                parser = get_parser(source_name)
                if parser:
                    record, chunks = parser.parse(scraped)
                    if chunks:
                        parsed_docs.append((record, chunks))
                    else:
                        total_failed += 1
                else:
                    # No dedicated parser — create a minimal DocumentRecord + ChunkRecord
                    content = scraped.content.decode("utf-8", errors="replace") if scraped.content else ""
                    record = DocumentRecord(
                        source_type=source_name,
                        title=scraped.title or "Untitled",
                        content=content[:50000],
                        url=scraped.url,
                        doi=scraped.doi,
                        published_date=scraped.pubdate,
                        journal=scraped.journal,
                        is_india_specific=True,
                        parser_version=f"{source_name}-scraped-v1",
                        condition_tags=[],
                        specialty_tags=[],
                    )
                    chunk_text = record.content[:8000]
                    if len(chunk_text.strip()) < 80:
                        total_failed += 1
                        continue
                    chunk = ChunkRecord(
                        document_id="",
                        source_type=source_name,
                        title=record.title,
                        chunk_text=chunk_text,
                        chunk_index=0,
                    )
                    parsed_docs.append((record, [chunk]))

                # Batch ingest
                if len(parsed_docs) >= BATCH_SIZE:
                    result = await pipeline.ingest_scraped_documents(
                        documents=parsed_docs, source=source_name,
                    )
                    total_ingested += result.get("documents_stored", 0)
                    total_failed += result.get("files_failed", 0)
                    parsed_docs = []
                    print(f"  [{i+1}/{min(len(jobs), limit)}] ingested={total_ingested} failed={total_failed}")

            except Exception as e:
                print(f"  [{i+1}] failed: {e}")
                total_failed += 1

        # Flush remaining
        if parsed_docs:
            result = await pipeline.ingest_scraped_documents(
                documents=parsed_docs, source=source_name,
            )
            total_ingested += result.get("documents_stored", 0)
            total_failed += result.get("files_failed", 0)

    finally:
        await scraper.close()

    print(f"\\n✓ {source_name}: ingested={total_ingested} failed={total_failed}")
    return {"source": source_name, "ingested": total_ingested, "failed": total_failed}


# Run ingestion for each source (skipping completed ones)
for source in SOURCES:
    if source in checkpoint["completed_sources"]:
        print(f"\\n⏭  Skipping {source} (already completed)")
        continue

    result = await ingest_source(
        source_name=source,
        limit=LIMIT_PER_SOURCE,
        journals=JOURNALS if source == "pubmed" else None,
    )
    checkpoint["completed_sources"].append(source)
    checkpoint["results"].append(result)
    save_checkpoint(checkpoint)
    print(f"  Checkpoint saved")

print(f"\\n{'='*60}")
print("INGESTION COMPLETE")
print(f"{'='*60}")
for r in checkpoint["results"]:
    print(f"  {r['source']}: ingested={r['ingested']} failed={r['failed']}")
print(f"\\nTotal ingested: {sum(r['ingested'] for r in checkpoint['results'])}")
""")

# === Cell 10: Verify data ===
md("""## 10. Verify the data

Searches Milvus for a test query to confirm articles were actually indexed and are retrievable.
""")

code("""from pymilvus import connections, Collection
from src.ml.embedding.embedder import get_embedder

print("Connecting to Zilliz...")
connections.connect(alias="default", uri=ZILLIZ_URI, token=ZILLIZ_TOKEN)
coll = Collection(pipeline.settings.vector_collection_v2)
coll.load()
print(f"✓ Connected. Total vectors in '{pipeline.settings.vector_collection_v2}': {coll.num_entities}")

if coll.num_entities > 0:
    print("\\nSearching for 'diabetes treatment in India'...")
    embedder = get_embedder()
    query_emb = embedder.embed_query("diabetes treatment in India")

    results = coll.search(
        data=[query_emb.tolist() if hasattr(query_emb, 'tolist') else list(query_emb)],
        anns_field="dense",
        param={"metric_type": "COSINE", "params": {"nprobe": 10}},
        limit=5,
        output_fields=["raw_text", "title", "source_type"],
    )

    print(f"\\nTop {len(results[0])} results:")
    for i, hit in enumerate(results[0]):
        title = hit.entity.get('title', '')[:70]
        source = hit.entity.get('source_type', '?')
        text = hit.entity.get('raw_text', '')[:120].replace('\\n', ' ')
        print(f"  [{i+1}] score={hit.score:.3f} source={source}")
        print(f"      title: {title}")
        print(f"      text:  {text}...")
        print()
else:
    print("\\n⚠ No vectors in collection — ingestion may have failed. Check the output above.")
""")

# === Cell 11: Cleanup ===
md("""## 11. Cleanup

Stops GROBID if it was started. The checkpoint file is preserved in `/kaggle/working/` for the next session.
""")

code("""# Stop GROBID if running
if grobid_proc:
    grobid_proc.kill()
    print("✓ GROBID stopped")

# Show checkpoint state
print(f"\\nCheckpoint file: {CHECKPOINT_FILE}")
print(f"Completed sources: {checkpoint['completed_sources']}")
print(f"Total articles ingested: {sum(r['ingested'] for r in checkpoint['results'])}")

print("\\n✓ Done. If the session was interrupted, re-run the notebook — it will skip completed sources.")
""")

# === Build notebook ===
notebook = {
    "nbformat": 4,
    "nbformat_minor": 4,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0",
        },
        "kaggle": {
            "accelerator": "gpu-t4x2",
            "dataSources": [],
            "isGpuEnabled": True,
            "isInternetEnabled": True,
            "language": "python",
        },
    },
    "cells": cells,
}

out_path = Path(__file__).parent / "kaggle_ingestion_v2.ipynb"
out_path.write_text(json.dumps(notebook, indent=1) + "\n")
print(f"✓ Generated {out_path} ({len(cells)} cells)")
