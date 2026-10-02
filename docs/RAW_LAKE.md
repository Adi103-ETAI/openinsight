# Raw Lake Spec — scraper/ingestion separation

Status: SPEC (awaiting approval) · Task: TASKS.md queued/high
Law: **scrapers collect, ingestion transforms.** Scraper PRs that parse get
rejected; pipeline PRs that fetch get rejected.

## 1. Layout

```
data/raw/{source}/               # one dir per source (pubmed, statpearls, …)
  {doc_id}.bin                   # raw bytes exactly as fetched (XML, HTML, PDF…)
  manifest.jsonl                 # ONE line per doc (append-only)
data/raw/manual/                 # Feynman / human drops (PDFs, DOCs + manifest)
```

`data/raw/` is gitignored (runtime); manifests are the audit trail.

## 2. Manifest schema (one JSON object per line)

| Field | Required | Notes |
|---|---|---|
| `doc_id` | yes | stable id (`pmid_123`, `NBK…`, slug) — joins raw to index |
| `source` | yes | `pubmed`, `statpearls`, `manual`, … |
| `url` | yes for scraped | origin URL; `file://…` + donor note for manual |
| `fetched_at` | yes | UTC ISO timestamp |
| `sha256` | yes | of the `.bin` bytes (re-download detection) |
| `content_type` | yes | `application/xml`, `text/html`, `application/pdf`, … |
| `license` | no | `open`, `government`, `unknown` — commercial full-text stays link-only |
| `title` | no | best-effort at fetch time (may be empty; parser owns truth) |
| `error` | no | fetch failures recorded here too (no silent drops) |

## 3. Cut lines (code changes)

1. **New `src/ingestion/raw_lake.py`**: `write_raw(source, doc_id, content, meta)` →
   bytes + manifest line (atomic append); `iter_raw(source)` → yields
   `(meta, bytes)`; `find_by_hash(sha256)` for re-download detection.
2. **`BaseScraper` (`scrapers/framework/base.py`)**: after successful fetch,
   call `write_raw` and return the manifest entry (not parsed content).
   Scrapers keep: discovery, fetch, retry, rate limit, robots, cache.
   Scrapers lose: all parsing/chunking calls.
3. **`pipeline.py`**: `ingest_scraped_documents` deprecated in favor of
   `ingest_raw_batch(source, limit)` reading from `iter_raw`; parse via
   existing per-source parsers keyed off manifest `source` + `content_type`.
   Network imports (`HttpClient`) removed from pipeline module.
4. **Notebook**: cell 18 becomes two cells — FETCH (scrapers → raw lake,
   prints bytes + manifest counts) then INGEST (`ingest_raw_batch` →
   vectors). Either cell re-runnable alone.
5. **Checkpoints**: existing checkpoint keys on fetch completion per source;
   add ingest cursor per source (raw manifest offset).

## 4. Feynman/manual drops

Drop files into `data/raw/manual/` + append manifest lines (same schema,
`source: manual`, `url: file://…`, license mandatory). Publish gate treats
them identically (metadata check requires title+source_type — parser must
supply from manifest title or filename).

## 5. Migration (no data loss)

1. Land `raw_lake.py` + manifest writer (pure additive).
2. Add fetch-only entry point; prove byte-identical re-fetch of 2 PubMed docs.
3. Switch notebook to fetch→ingest cells on a 5-doc trial; gate must PASS.
4. Delete parse calls from scraper path; remove `HttpClient` from pipeline.
5. Cleaning stage (next task) plugs between parse and chunk in the ingest half.

## 6. Non-goals

No new infra (no S3/MinIO — local disk + gitignored dir suffices to 100k docs),
no backfill of old runs (May corpus stays as-is; new runs use the lake).
