"""
OpenInsight — Kaggle Ingestion Script
======================================

This is the Kaggle-edition ingestion script. It runs the OpenInsight
pipeline on Kaggle's free GPU (T4 × 2) and stores to Zilliz Cloud +
MongoDB Atlas.

This script is also a runnable .py file (for testing locally) but is
designed to be pasted into a Kaggle notebook cell-by-cell.

Cells are marked with # === CELL N === comments.

Usage on Kaggle:
    1. Create new notebook
    2. Settings: GPU T4 × 2, Internet ON, Persistence ON
    3. Add Kaggle Secrets: MONGODB_URL, ZILLIZ_URI, ZILLIZ_TOKEN, NCBI_API_KEY
    4. Paste cells below in order
    5. Run all

Usage locally (for testing):
    python notebooks/kaggle_ingestion_v2.py
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
# === CELL 1: Configuration ================================================

# Sources to ingest (pick from list_sources())
SOURCES = [
    "pubmed",           # Indian journals via PubMed
    "statpearls",       # Clinical overviews
    "ncbi_bookshelf",   # GeneReviews + Medical Genetics
    "cdsco",            # Approved drugs (HTML, no GROBID needed)
    "ctri",             # Clinical trials (HTML, no GROBID needed)
    # "nmc_curriculum",  # PDF — needs GROBID
    # "pvpi",            # PDF — needs GROBID
    # "rssdi",           # PDF — needs GROBID
    # "csi",             # PDF — needs GROBID
]

# Cap per source (set low for testing, raise for production)
LIMIT_PER_SOURCE = 50

# For PubMed: specific journals (None = all 30 Indian journals)
JOURNALS = None  # or ["Indian J Med Res", "Natl Med J India"]

# Batch size for ingestion pipeline
BATCH_SIZE = 10

# GROBID: only start if any source needs PDF parsing
NEEDS_GROBID = any(s in SOURCES for s in ["nmc_curriculum", "pvpi", "rssdi", "csi", "isccm", "iap", "fogsi", "aios", "isn", "ntep", "nvbdcp", "nhm", "npcds"])

# Repo config
REPO_URL = "https://github.com/Adi103-ETAI/openinsight.git"
REPO_BRANCH = "re-insight"
REPO_DIR = "/kaggle/working/openinsight" if os.path.exists("/kaggle") else "/tmp/openinsight"

# Checkpoint file
CHECKPOINT_FILE = "/kaggle/working/checkpoint.json" if os.path.exists("/kaggle") else "/tmp/checkpoint.json"

# === CELL 2: Install dependencies ========================================

def install_dependencies() -> None:
    """Install Python deps not pre-installed on Kaggle."""
    print("=== Installing dependencies ===")
    deps = [
        "pymilvus>=2.5,<2.6",
        "motor==3.6.0",
        "sentence-transformers==3.1.1",
        "transformers==4.45.1",
        "biopython==1.84",
        "pdfplumber==0.11.4",
        "beautifulsoup4==4.12.3",
        "lxml==5.3.0",
        "loguru==0.7.2",
        "tenacity==8.5.0",
        "trafilatura>=1.12.0",
    ]
    for dep in deps:
        print(f"  pip install {dep}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", dep])
    print("✓ Dependencies installed")

# === CELL 3: Clone repo ===================================================

def clone_repo() -> None:
    """Clone the OpenInsight repo (re-insight branch)."""
    print(f"=== Cloning {REPO_BRANCH} branch ===")
    if Path(REPO_DIR).exists():
        print(f"  {REPO_DIR} already exists — pulling latest")
        subprocess.check_call(["git", "-C", REPO_DIR, "fetch", "origin", REPO_BRANCH])
        subprocess.check_call(["git", "-C", REPO_DIR, "checkout", REPO_BRANCH])
        subprocess.check_call(["git", "-C", REPO_DIR, "pull", "origin", REPO_BRANCH])
    else:
        subprocess.check_call([
            "git", "clone", "--branch", REPO_BRANCH, "--depth", "1",
            REPO_URL, REPO_DIR,
        ])
    print(f"✓ Repo at {REPO_DIR}")

# === CELL 4: Set up environment variables =================================

def setup_env() -> None:
    """Set environment variables from Kaggle Secrets."""
    print("=== Setting up environment ===")

    def get_secret(name: str, default: str = "") -> str:
        """Read from Kaggle Secrets, fallback to env, fallback to default."""
        # Try Kaggle Secrets
        try:
            from kaggle_secrets import UserSecretsClient
            secrets = UserSecretsClient()
            try:
                return secrets.get_secret(name) or default
            except Exception:
                pass
        except ImportError:
            pass
        # Try env var
        return os.environ.get(name, default)

    os.environ["MONGODB_URL"] = get_secret("MONGODB_URL", "mongodb://localhost:27017")
    os.environ["VECTOR_URI"] = get_secret("ZILLIZ_URI", "http://localhost:19530")
    os.environ["VECTOR_TOKEN"] = get_secret("ZILLIZ_TOKEN", "")
    os.environ["NCBI_API_KEY"] = get_secret("NCBI_API_KEY", "")
    os.environ["NVIDIA_NIM_API_KEY"] = get_secret("NVIDIA_NIM_API_KEY", "dummy-for-ingestion")
    os.environ["APP_ENV"] = "kaggle"
    os.environ["PYTHONPATH"] = REPO_DIR

    print(f"  MONGODB_URL: {'set (' + os.environ['MONGODB_URL'][:30] + '...)' if os.environ['MONGODB_URL'] else 'NOT SET'}")
    print(f"  VECTOR_URI: {os.environ['VECTOR_URI']}")
    print(f"  VECTOR_TOKEN: {'set' if os.environ['VECTOR_TOKEN'] else 'NOT SET'}")
    print(f"  NCBI_API_KEY: {'set' if os.environ['NCBI_API_KEY'] else 'NOT SET (3 req/sec rate limit)'}")

# === CELL 5: Start GROBID (only if needed) ===============================

def start_grobid() -> subprocess.Popen | None:
    """Download + start GROBID as a background subprocess."""
    if not NEEDS_GROBID:
        print("=== GROBID not needed (no PDF sources) — skipping ===")
        return None

    print("=== Starting GROBID ===")
    grobid_dir = Path("/kaggle/working/grobid") if os.path.exists("/kaggle") else Path("/tmp/grobid")
    grobid_version = "0.9.0"

    if not grobid_dir.exists():
        print(f"  Downloading GROBID {grobid_version}...")
        url = f"https://github.com/kermitt2/grobid/releases/download/{grobid_version}/grobid-{grobid_version}.zip"
        zip_path = str(grobid_dir.parent / "grobid.zip")
        subprocess.check_call(["wget", "-q", "-O", zip_path, url])
        subprocess.check_call(["unzip", "-q", zip_path, "-d", str(grobid_dir.parent)])
        grobid_dir.parent.joinpath(f"grobid-{grobid_version}").rename(grobid_dir)
        Path(zip_path).unlink()

    print("  Starting GROBID server (this takes ~2 min)...")
    proc = subprocess.Popen(
        ["./gradlew", "run", "--args=server live"],
        cwd=str(grobid_dir),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        # Limit Java heap to 4GB (leave room for sentence-transformers)
        env={**os.environ, "JAVA_OPTS": "-Xmx4g"},
    )

    # Wait for GROBID to be healthy (max 3 min)
    import httpx
    for i in range(36):  # 36 × 5s = 3 min
        try:
            r = httpx.get("http://localhost:8070/api/isalive", timeout=2)
            if r.status_code == 200:
                print(f"✓ GROBID healthy (took {i*5}s)")
                return proc
        except Exception:
            pass
        time.sleep(5)

    print("✗ GROBID failed to start within 3 minutes")
    proc.kill()
    return None

# === CELL 6: Verify GPU + connections ====================================

def verify_setup() -> bool:
    """Verify GPU, MongoDB, and Zilliz are all reachable."""
    print("=== Verifying setup ===")
    all_ok = True

    # GPU check
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"✓ GPU: {gpu_name} ({vram:.1f} GB VRAM)")
        else:
            print("✗ No GPU available — check Kaggle accelerator setting")
            all_ok = False
    except ImportError:
        print("✗ torch not installed")
        all_ok = False

    # MongoDB check
    try:
        import asyncio
        from motor.motor_asyncio import AsyncIOMotorClient
        async def check_mongo():
            client = AsyncIOMotorClient(os.environ["MONGODB_URL"], serverSelectionTimeoutMS=5000)
            await client.admin.command("ping")
            client.close()
        asyncio.run(check_mongo())
        print("✓ MongoDB reachable")
    except Exception as e:
        print(f"✗ MongoDB unreachable: {e}")
        all_ok = False

    # Zilliz check
    try:
        from pymilvus import connections
        connections.connect(
            alias="default",
            uri=os.environ["VECTOR_URI"],
            token=os.environ.get("VECTOR_TOKEN", ""),
        )
        print("✓ Zilliz/Milvus reachable")
    except Exception as e:
        print(f"✗ Zilliz/Milvus unreachable: {e}")
        all_ok = False

    return all_ok

# === CELL 7: Checkpoint helpers ==========================================

def load_checkpoint() -> dict:
    """Load checkpoint from previous session."""
    if not Path(CHECKPOINT_FILE).exists():
        return {"completed_sources": [], "current_source": None, "current_offset": 0}
    with open(CHECKPOINT_FILE) as f:
        return json.load(f)

def save_checkpoint(checkpoint: dict) -> None:
    """Save checkpoint (called after each source completes in this script)."""
    Path(CHECKPOINT_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(checkpoint, f, indent=2, default=str)

# === CELL 8: Main ingestion loop =========================================

async def ingest_from_source(
    source_name: str,
    limit: int,
    pipeline,
    journals: list[str] | None = None,
) -> dict:
    """Ingest articles from one source. Returns summary dict."""
    import sys
    sys.path.insert(0, REPO_DIR)

    from src.ingestion.scrapers import get_scraper

    print(f"\n--- Ingesting from: {source_name} (limit={limit}) ---")

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

        print(f"  Discovered {len(jobs)} URLs")

        # Fetch + parse + batch-ingest
        parsed_docs = []
        for i, job in enumerate(jobs[:limit]):
            try:
                scraped = await scraper.fetch_one(job)
                if not scraped:
                    total_failed += 1
                    continue

                # Get the right parser
                parser = get_parser(source_name)
                if parser:
                    record, chunks = parser.parse(scraped)
                    if chunks:
                        parsed_docs.append((record, chunks))
                    else:
                        total_failed += 1
                else:
                    # No parser (e.g., PubMed — handled differently)
                    # Create a minimal DocumentRecord
                    from src.ingestion.document_db import DocumentRecord
                    record = DocumentRecord(
                        source_type=source_name,
                        title=scraped.title or "Untitled",
                        content=scraped.content.decode("utf-8", errors="replace")[:50000] if scraped.content else "",
                        url=scraped.url,
                        doi=scraped.doi,
                        published_date=scraped.pubdate,
                        journal=scraped.journal,
                        is_india_specific=True,
                        parser_version=f"{source_name}-scraped-v1",
                        content_hash=hashlib.sha256(scraped.content or b"").hexdigest()[:16] if scraped.content else "",
                        condition_tags=[],
                        specialty_tags=[],
                    )
                    parsed_docs.append((record, []))

                # Batch ingest every BATCH_SIZE docs
                if len(parsed_docs) >= BATCH_SIZE:
                    result = await pipeline.ingest_scraped_documents(
                        documents=parsed_docs,
                        source=source_name,
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
                documents=parsed_docs,
                source=source_name,
            )
            total_ingested += result.get("documents_stored", 0)
            total_failed += result.get("files_failed", 0)

    finally:
        await scraper.close()

    return {
        "source": source_name,
        "ingested": total_ingested,
        "failed": total_failed,
        "discovered": len(jobs) if "jobs" in dir() else 0,
    }


def get_parser(source_name: str):
    """Get the right parser for a source (or None for sources handled inline)."""
    import importlib
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

# === CELL 9: Run it ======================================================

async def main():
    """Main entrypoint — run from Kaggle notebook."""
    import hashlib

    # Setup
    install_dependencies()
    clone_repo()
    setup_env()

    # Start GROBID if needed
    grobid_proc = start_grobid()

    # Verify everything works
    if not verify_setup():
        print("✗ Setup verification failed — fix errors above before continuing")
        return

    # Import pipeline (after repo is cloned)
    import sys
    sys.path.insert(0, REPO_DIR)
    from src.ingestion.pipeline import IngestionPipeline

    pipeline = IngestionPipeline()
    checkpoint = load_checkpoint()

    print(f"\n{'='*60}")
    print(f"OpenInsight Kaggle Ingestion")
    print(f"{'='*60}")
    print(f"Sources: {SOURCES}")
    print(f"Limit per source: {LIMIT_PER_SOURCE}")
    if JOURNALS:
        print(f"Journals: {JOURNALS}")
    print(f"{'='*60}\n")

    results = []
    for source in SOURCES:
        if source in checkpoint.get("completed_sources", []):
            print(f"Skipping {source} (already completed in previous session)")
            continue

        result = await ingest_from_source(
            source_name=source,
            limit=LIMIT_PER_SOURCE,
            pipeline=pipeline,
            journals=JOURNALS if source == "pubmed" else None,
        )
        results.append(result)

        # Update checkpoint
        checkpoint["completed_sources"].append(source)
        checkpoint["current_source"] = source
        save_checkpoint(checkpoint)

        print(f"\n✓ {source}: {result}")

    # Cleanup
    if grobid_proc:
        grobid_proc.kill()

    print(f"\n{'='*60}")
    print(f"Ingestion Complete")
    print(f"{'='*60}")
    for r in results:
        print(f"  {r['source']}: ingested={r['ingested']} failed={r['failed']}")
    total = sum(r["ingested"] for r in results)
    print(f"\nTotal articles ingested: {total}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
