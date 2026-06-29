# Scraper Framework → IngestionPipeline Integration Design

**Status**: Design doc — implementation deferred to a focused follow-up PR.
**Why deferred**: The integration requires careful surgery on
`IngestionPipeline` (which is currently file-path based) to expose a clean
`ingest_scraped_documents()` entry point. Rushing it would risk breaking
the existing `/search` pipeline. This doc captures the design so the work
can be picked up with full context.

## Problem Statement

The scraper framework (Phase 0.5) produces `ScrapedDocument` objects via
`BaseScraper.fetch_one()`. The Phase 1 parsers (`IndMEDParser`,
`MedknowParser`, `PMCIndiaParser`) consume `ScrapedDocument` and produce
`(DocumentRecord, list[ChunkRecord])`. The pipeline then needs to:

1. Enrich metadata (NER, evidence level, quality score)
2. Embed chunks (dense + sparse vectors)
3. Store document + chunks to MongoDB
4. Upsert vectors to Milvus
5. Update cross-source dedup index
6. Track in ingestion monitor

The existing `IngestionPipeline.ingest_directory()` does all of this but is
coupled to file-path input — it scans a directory, picks a parser based on
file extension + source, parses each file, then enriches + embeds + stores.

## Current State

```python
# What works today (Phase 1 final state):
scraper = get_scraper("indmed")
jobs = await scraper.discover(journals=["ijp"], max_articles_per_journal=10)
for job in jobs:
    scraped_doc = await scraper.fetch_one(job)  # ScrapedDocument
    if scraped_doc:
        parser = IndMEDParser()
        record, chunks = parser.parse(scraped_doc)  # (DocumentRecord, list[ChunkRecord])
        # ❌ STUCK HERE — no pipeline method to ingest (record, chunks) directly
```

## Design Options Considered

### Option A: Write ScrapedDocument to temp file, call `ingest_directory()`

```python
# Write scraped HTML to /tmp/indmed/<doc_id>.html
# Then: pipeline.ingest_directory(directory="/tmp/indmed", source="indmed")
```

**Pros**: Zero pipeline changes. Reuses all existing enrich+embed+store logic.
**Cons**: Loses the structured metadata extracted by the scraper (title, authors,
DOI, PMID — these would have to be re-extracted from the HTML by the parser,
which the parser already does but the pipeline wouldn't know to use the
structured path). Round-trips through disk unnecessarily.

**Verdict**: Quick to ship but loses the structured-data advantage. Not
recommended as the primary path.

### Option B: Add `ingest_scraped_documents()` method to `IngestionPipeline` ✅ RECOMMENDED

```python
class IngestionPipeline:
    async def ingest_scraped_documents(
        self,
        documents: list[tuple[ScrapedDocument, type]],  # (doc, parser_class)
        source: str,
        batch_size: int = 10,
    ) -> dict[str, int]:
        """Ingest pre-fetched ScrapedDocuments using their source-specific parser.

        Bypasses the file-path scanning + parser routing (those are handled by
        the scraper framework). Reuses the existing enrich + embed + store
        internal helpers.
        """
```

**Pros**:
- Clean separation: scraper framework handles fetch + parse; pipeline handles enrich + embed + store
- Reuses all existing pipeline internals (metadata enrichment, NER, embedder, vector indexer, Mongo storage, dead letter queue, monitor)
- Doesn't touch the file-path-based `ingest_directory()` flow (no regression risk)
- Future scrapers (CDSCO, CTRI, NFHS, etc.) all use the same entry point

**Cons**:
- Requires refactoring `ingest_directory()` to extract its enrich+embed+store logic into reusable internal methods (currently inline)
- ~3-4 hours of careful work with regression testing

**Verdict**: Right long-term design. Worth doing properly.

### Option C: Refactor `ingest_directory()` into composable stages

Split `ingest_directory()` into:
- `_parse_stage(file_paths, source)` → list[dict] (already exists as `_parse_batch`)
- `_enrich_stage(docs)` → list[dict] (currently inline)
- `_embed_stage(chunks)` → embeddings (currently inline)
- `_store_stage(docs, chunks, embeddings)` → None (currently inline)

Then both `ingest_directory()` and a new `ingest_scraped_documents()` become
thin orchestrators that call these stages.

**Pros**: Cleanest design. Each stage is independently testable.
**Cons**: Largest refactor. Highest regression risk.

**Verdict**: Aspirational. Defer until we have a test coverage on the existing
pipeline that gives confidence for the refactor.

## Recommended Implementation Plan (Option B)

### Step 1: Extract internal helpers from `ingest_directory()`

Pull these code blocks out of `ingest_directory()` into named methods:

```python
async def _enrich_documents(self, docs: list[dict]) -> list[dict]:
    """Apply metadata enrichment (NER, evidence level, India relevance)."""
    # Currently inline at lines ~580-620 of pipeline.py

async def _embed_and_store_chunks(
    self,
    docs_for_batch: list[dict],
    chunks_for_batch: list[ChunkRecord],
    source: str,
) -> tuple[int, list[dict]]:
    """Embed chunks (dense + sparse) and store to Mongo + Milvus.
    Returns (indexed_count, dead_letter_entries).
    """
    # Currently inline at lines ~620-720 of pipeline.py
```

### Step 2: Add `ingest_scraped_documents()` method

```python
async def ingest_scraped_documents(
    self,
    documents: list[tuple[ScrapedDocument, type]],
    source: str,
    batch_size: int = 10,
    recreate_index: bool = False,
) -> dict[str, int]:
    """Ingest pre-fetched ScrapedDocuments.

    Args:
        documents: list of (ScrapedDocument, parser_class) tuples
        source: source name (e.g., "indmed", "medknow", "pmc_india")
        batch_size: docs per batch (default 10)
        recreate_index: whether to recreate the Milvus collection

    Returns:
        Summary dict with counts
    """
    # 1. Ensure Milvus collection exists
    self.indexer.create_collection(
        recreate=recreate_index,
        collection_name=self.settings.vector_collection_v2,
    )

    # 2. Parse each ScrapedDocument with its parser
    parsed: list[tuple[DocumentRecord, list[ChunkRecord]]] = []
    for scraped_doc, parser_cls in documents:
        parser = parser_cls()
        record, chunks = parser.parse(scraped_doc)
        if chunks:
            parsed.append((record, chunks))

    # 3. Process in batches (same batching logic as ingest_directory)
    summary = {"documents_total": len(parsed), "documents_stored": 0,
               "chunks_created": 0, "chunks_indexed": 0, "files_failed": 0}

    for i in range(0, len(parsed), batch_size):
        batch = parsed[i:i + batch_size]
        # 4. For each batch: enrich + embed + store (reuse the new internal helpers)
        docs_for_batch = [self._normalize_document(r, source) for r, _ in batch]
        all_chunks = [c for _, chunks in batch for c in chunks]

        # Apply metadata enrichment
        enriched = await self._enrich_documents(docs_for_batch)

        # Quality score + filter
        valid_chunks, rejected = filter_valid_chunks(all_chunks)

        # Embed + store
        indexed, _ = await self._embed_and_store_chunks(enriched, valid_chunks, source)

        summary["documents_stored"] += len(enriched)
        summary["chunks_created"] += len(valid_chunks)
        summary["chunks_indexed"] += indexed

    return summary
```

### Step 3: Wire `seed_indian_journals.py` to call the new method

Replace the TODO blocks in `seed_indian_journals.py` with:

```python
async def ingest_pubmed(journals, max_per_journal, discover_only):
    scraper = get_scraper("pubmed")
    pipeline = IngestionPipeline()
    # ... discovery ...
    if not discover_only and jobs:
        # Fetch + parse in batches, then ingest
        for batch_start in range(0, len(jobs), batch_size=10):
            batch_jobs = jobs[batch_start:batch_size]
            scraped_docs = []
            for job in batch_jobs:
                doc = await scraper.fetch_one(job)
                if doc:
                    scraped_docs.append((doc, PubMedParser))
            if scraped_docs:
                result = await pipeline.ingest_scraped_documents(
                    documents=scraped_docs,
                    source="pubmed",
                )
                logger.info(f"Ingested batch: {result}")
```

### Step 4: Tests

- Unit test: `ingest_scraped_documents()` with mocked embedder + Mongo + Milvus
- Integration test: end-to-end with a small set of fixture HTML files, verify chunks land in Mongo + Milvus
- Regression test: ensure `ingest_directory()` still works after the refactor

## Estimation

| Step | Hours |
|---|---|
| Step 1: Extract `_enrich_documents` + `_embed_and_store_chunks` from `ingest_directory()` | 1.5 |
| Step 2: Add `ingest_scraped_documents()` method | 1.0 |
| Step 3: Wire `seed_indian_journals.py` (all 4 source paths) | 0.5 |
| Step 4: Tests (unit + integration + regression) | 2.0 |
| Buffer for debugging | 1.0 |
| **Total** | **6 hours** |

## What This Enables

Once `ingest_scraped_documents()` ships:

1. `python scripts/seed_indian_journals.py --source indmed --max-per-journal 10` actually ingests 10 IndMED articles per journal
2. Cross-source dedup runs automatically (Phase 0 framework) — same article in PubMed + IndMED appears once
3. Per-journal `trust_tier` flows from scraper → parser → chunk → Milvus payload → retrieval boost
4. Frontend can render `trust_tier` badge on each citation (per `docs/FRONTEND_CHANGES_NEEDED.md`)
5. Phase 2 (NCBI Bookshelf, NMC curriculum, govt manuals) can use the same `ingest_scraped_documents()` entry point — no per-source pipeline work needed
6. Phase 3 (NFI, CDSCO, CTRI, PvPI) same
7. Phase 5 (NFHS, NCDIR) same

## Decision

Defer to a focused PR titled `feat(pipeline): ingest_scraped_documents() — scraper framework integration`. This is the single highest-leverage piece of remaining backend work — it unlocks actual data ingestion for every subsequent phase. Don't start Phase 2 until this lands.
