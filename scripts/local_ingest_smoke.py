"""
Local 1-doc ingestion smoke — validates fetch → parse → chunk → embed →
upsert → mongo end-to-end against CLOUD backends, without touching prod data.

- Milvus: writes to `smoke_test` collection (dropped at the end)
- Mongo: uses `openinsight_smoke` database (dropped at the end)
- Embeds on CPU (slow, fine for 1 doc)

Usage: python scripts/local_ingest_smoke.py
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SMOKE_COLLECTION = "smoke_test"
SMOKE_DB = "openinsight_smoke"
# A short, stable PubMed Central open-access article for the smoke
SMOKE_PMID = "41666468"  # already in corpus family; tiny fetch


async def main() -> int:
    from src.config.settings import get_settings

    settings = get_settings()
    print(f"embed_provider={settings.embed_provider} model={settings.dense_model_name}")

    from src.ingestion.scrapers import get_scraper

    scraper = get_scraper("pubmed")
    try:
        jobs = await scraper.discover_by_journal("Indian J Med Res", max_results=1)
        assert jobs, "discover returned zero jobs"
        scraped = await scraper.fetch_one(jobs[0])
        assert scraped and scraped.content, "fetch returned empty"
        print(f"fetch OK: {(scraped.title or 'untitled')[:80]} ({len(scraped.content)} bytes)")
    finally:
        await scraper.close()

    from src.ml.chunking.chunker import HierarchicalChunkerV3

    chunker = HierarchicalChunkerV3()
    text = scraped.content.decode("utf-8", errors="replace") if isinstance(scraped.content, bytes) else scraped.content
    doc = {"title": scraped.title or "smoke", "content": text, "abstract": ""}
    chunks = chunker.chunk_document(doc, {"source": "pubmed", "title": doc["title"]})
    assert chunks, "chunker produced zero chunks"
    print(f"chunk OK: {len(chunks)} chunks")

    from src.ml.embedding.embedder import create_embedder

    embedder = create_embedder()
    vecs, failed = embedder.embed_batch([c.text for c in chunks])
    assert not failed, f"embed failures: {failed}"
    dim = len(vecs[0])
    print(f"embed OK: {len(vecs)} vectors dim={dim}")
    assert dim == settings.embedding_dim, f"dim {dim} != settings {settings.embedding_dim}"

    import time

    from pymilvus import MilvusClient

    def _with_wake_retry(label, fn, tries=3):
        for attempt in range(1, tries + 1):
            try:
                return fn()
            except Exception as e:
                print(f"{label} attempt {attempt}/{tries} failed: {str(e)[:100]}")
                if attempt == tries:
                    raise
                time.sleep(10)  # serverless wake-up window

    mc = MilvusClient(uri=settings.vector_uri, token=settings.vector_token or None)
    _with_wake_retry("milvus-connect", lambda: mc.list_collections())
    if mc.has_collection(SMOKE_COLLECTION):
        mc.drop_collection(SMOKE_COLLECTION)
    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=True)
    from pymilvus import DataType

    schema.add_field("id", DataType.VARCHAR, max_length=128, is_primary=True)
    schema.add_field("dense", DataType.FLOAT_VECTOR, dim=dim)
    index_params = mc.prepare_index_params()
    index_params.add_index("dense", index_type="AUTOINDEX", metric_type="COSINE")
    mc.create_collection(SMOKE_COLLECTION, schema=schema, index_params=index_params)
    rows = [{"id": f"smoke-{i}", "dense": list(map(float, v))} for i, v in enumerate(vecs)]
    mc.insert(SMOKE_COLLECTION, rows)
    mc.flush(SMOKE_COLLECTION)
    n = mc.get_collection_stats(SMOKE_COLLECTION).get("row_count")
    print(f"milvus OK: {n} rows in {SMOKE_COLLECTION}")
    assert int(n) == len(vecs)

    from pymongo import MongoClient

    mdb = MongoClient(settings.mongodb_url, serverSelectionTimeoutMS=15000)[SMOKE_DB]
    mdb["smoke_chunks"].insert_many([{"id": f"smoke-{i}", "text": c.text[:200]} for i, c in enumerate(chunks)])
    print(f"mongo OK: {mdb['smoke_chunks'].estimated_document_count()} docs in {SMOKE_DB}")

    # Cleanup: leave no trace (Atlas user lacks dropDatabase — drop contents).
    # Best-effort: cleanup failures warn but never fail the smoke verdict.
    try:
        _with_wake_retry("milvus-drop", lambda: mc.drop_collection(SMOKE_COLLECTION))
        from pymongo import MongoClient as _MC

        _sdb = _MC(settings.mongodb_url, serverSelectionTimeoutMS=15000)[SMOKE_DB]
        for coll in _sdb.list_collection_names():
            _sdb[coll].delete_many({})
        print("cleanup OK: smoke collection dropped, smoke db emptied")
    except Exception as e:
        print(f"cleanup WARNING (manual tidy needed): {str(e)[:150]}")

    print("\nSMOKE PASS — pipeline works end-to-end")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
