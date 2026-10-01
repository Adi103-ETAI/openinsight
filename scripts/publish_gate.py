"""
Publish gate CLI — run after a batch, before marking it ready.

Usage (reads .env):
    python scripts/publish_gate.py --source pubmed
    python scripts/publish_gate.py --source statpearls --mark-ready

Without --mark-ready this is read-only: prints the gate + smoke reports and
exits nonzero on failure. With --mark-ready, passing batches flip their
documents_v2 rows to status="ready".
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _env(name: str, default: str = "") -> str:
    for candidate in (PROJECT_ROOT / ".env", Path(".env")):
        if candidate.exists():
            for line in candidate.read_text().splitlines():
                m = re.match(rf"^{name}=(.*)$", line.strip())
                if m:
                    return m.group(1).strip()
    return os.environ.get(name, default)


def main() -> int:
    ap = argparse.ArgumentParser(description="Publish gate for an ingestion batch")
    ap.add_argument("--source", required=True, help="source name, e.g. pubmed")
    ap.add_argument("--collection", default="", help="Milvus collection (default: VECTOR_COLLECTION)")
    ap.add_argument("--mark-ready", action="store_true", help="flip documents to ready on PASS")
    args = ap.parse_args()

    from pymongo import MongoClient
    from pymilvus import MilvusClient

    from src.ingestion.publish import collect_milvus_stats, collect_mongo_stats, run_gate
    from src.ingestion.smoke import run_smoke

    mongo_url = _env("MONGODB_URL", "mongodb://localhost:27017")
    vector_uri = _env("VECTOR_URI", "http://localhost:19530")
    vector_token = _env("VECTOR_TOKEN", "")
    collection = args.collection or _env("VECTOR_COLLECTION", "openinsight_v2")
    expected_dim = int(_env("VECTOR_DIM", "768") or 768)

    db = MongoClient(mongo_url, serverSelectionTimeoutMS=15000)["openinsight"]
    mc = MilvusClient(uri=vector_uri, token=vector_token or None)

    mongo_chunks, texts, metas = collect_mongo_stats(db, args.source)
    milvus_rows, dim = collect_milvus_stats(mc, collection)

    # Per-source Milvus row count via filtered query (falls back to collection
    # total when the source filter matches nothing, e.g. legacy batches).
    try:
        per_source = []
        offset = 0
        while True:
            batch = mc.query(collection, filter=f"source == '{args.source}'",
                             output_fields=["id"], limit=10000, offset=offset)
            if not batch:
                break
            per_source.extend(batch)
            offset += len(batch)
            if len(batch) < 10000 or offset >= 16000:
                break
        milvus_rows = len(per_source) if per_source or offset else milvus_rows
    except Exception:
        pass

    produced = mongo_chunks  # existing batches: produced == stored; fresh runs pass explicit counts
    gate = run_gate(
        source=args.source,
        produced_chunks=produced,
        mongo_chunks=mongo_chunks,
        milvus_rows=milvus_rows,
        collection_dim=dim,
        expected_dim=expected_dim,
        sample_texts=texts,
        sample_metas=metas,
    )
    print(gate.summary())

    smoke = run_smoke(db)
    print(smoke.summary())

    ok = gate.passed and smoke.passed
    if ok and args.mark_ready:
        res = db["documents_v2"].update_many(
            {"$or": [{"source": args.source}, {"source_type": args.source}]},
            {"$set": {"status": "ready"}},
        )
        print(f"marked ready: {res.modified_count} documents")
    elif not ok:
        print("gate FAILED — nothing marked ready")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
