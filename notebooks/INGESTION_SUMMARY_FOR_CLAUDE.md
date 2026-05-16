# Ingestion Pipeline — Summary for Claude

Purpose
-------
This document summarizes the current state, intention, and runnable steps for the OpenInsight data ingestion pipeline so you (Claude) can reason about next steps, reproduce the work, and provide guidance. It focuses on the runtime flow, entrypoints we use in notebooks, configuration/secrets needed, verification checks, common failure modes, and recommended next actions.

High-level goal
---------------
- Run the repo's authoritative ingestion code (not a simplified demo) from cloud notebook environments (Colab and Kaggle) so we can temporarily process data with GPU acceleration and external services (Milvus/Mongo/GROBID).
- Preserve pipeline semantics: parsing (GROBID/ICMR/OCR) → deduplication → metadata enrichment → chunking → quality filtering → embed → index (Milvus/Zilliz) → store (MongoDB) → monitoring/checkpointing.

What we changed / created
-------------------------
- Two runnable notebooks (full repo entrypoints):
  - `notebooks/colab_ingestion.ipynb` — Colab-ready notebook that installs deps, sets environment, starts GROBID (if desired), and calls `IngestionPipeline.ingest_directory(...)` from `src.ingestion.pipeline`.
  - `notebooks/kaggle_ingestion.ipynb` — Kaggle-ready version using Kaggle secrets and paths.
- A human-readable guide: `notebooks/kaggle_ingestion.md` — markdown separation of explanation vs code.
- This summary file: `notebooks/INGESTION_SUMMARY_FOR_CLAUDE.md` (you are reading it).

Key code entrypoints referenced
-------------------------------
- `src/ingestion/pipeline.py` — `IngestionPipeline` and `ingest_directory(...)` (primary entrypoint used by notebooks).
- `src/ingestion/run_ingestion.py` — CLI wrapper that mirrors notebook usage.
- `src/ingestion/checkpoint.py` — checkpointing logic persisted to MongoDB.
- `src/ingestion/vector_indexer.py` and `src/vectorstore/registry.py` — vector indexing and vectorstore selection (Milvus / Zilliz cloud).
- `src/ml/embedding/embedder.py` — embedding providers (local SentenceTransformers or HF providers).
- `src/data/mongo/doc_store.py` — how parsed documents & chunks are persisted to MongoDB.

Runtime flow (what a notebook runs)
----------------------------------
1. Install Python dependencies (notebooks use `%pip install -r requirements.txt` / selected extras).
2. Configure environment variables (either via Colab UI or Kaggle secrets). Key vars listed below.
3. Optionally start a local GROBID server (`http://localhost:8070`) via subprocess inside the notebook (not required if an external GROBID is available).
4. Instantiate pipeline: `pipeline = IngestionPipeline()` (loads chunker, embedder, vector indexer, doc store, monitor, checkpoint manager based on settings).
5. Run ingestion: `await pipeline.ingest_directory(directory=DATA_DIR, source=SOURCE, recreate_index=False, batch_size=..., resume=True)`
6. Pipeline writes checkpoints and RunMetrics into MongoDB; vectors are upserted into Milvus/Zilliz.
7. Notebook uses `pymilvus` and `pymongo` clients to verify connectivity and basic upsert counts.

Required environment variables / secrets
--------------------------------------
These must be set before running the notebooks (Colab: use UI variables or `os.environ`; Kaggle: use Kaggle Secrets or mount them):

- `MONGODB_URL` — MongoDB connection string (Atlas preferred).
- `MONGODB_DB` — database name used for ingestion checkpoints and documents.
- `ZILLIZ_VECTOR_URI` or `VECTOR_URI` — Milvus / Zilliz Cloud endpoint (depending on vectorstore registry settings).
- `ZILLIZ_VECTOR_TOKEN` or `VECTOR_TOKEN` — token for vector service.
- `NVIDIA_NIM_API_KEY` — required by the repo's settings (marked required in `settings.py`).
- `GROBID_URL` — URL for GROBID (if using external); otherwise notebook starts a local GROBID at `http://localhost:8070`.
- Optional provider keys: `HUGGINGFACE_API_KEY`, `COHERE_API_KEY`, `NCBI_API_KEY`, etc., depending on configured embedder / parsing helpers.

Exact call used in the notebooks
--------------------------------
Example invocation used in the notebooks (Colab/Kaggle):

```python
from src.ingestion.pipeline import IngestionPipeline
pipeline = IngestionPipeline()
summary = await pipeline.ingest_directory(
    directory=DATA_DIR,
    source=SOURCE_TYPE,
    recreate_index=False,
    batch_size=10,
    resume=True,
    reset=False,
)
print(summary)
```

What the pipeline returns / where data lands
-------------------------------------------
- `ingest_directory` returns a dict-summary including keys: `files_total`, `files_parsed`, `documents_stored`, `chunks_created`, `chunks_indexed`, `files_failed`, `chunks_deduped`, `chunks_quality_filtered`.
- Checkpoints stored in MongoDB collection `ingestion_checkpoints` (via `CheckpointManager`).
- Run metrics stored in MongoDB collection `ingestion_metrics` (via `IngestionMonitor`).
- Documents & chunk metadata persisted by `MongoDocStoreV2` in the configured `MONGODB_DB`.
- Vector points upserted to Milvus/Zilliz via `VectorIndexer` / vectorstore registry.

Verification steps (quick checklist executed by notebook)
-------------------------------------------------------
1. Confirm env vars present: attempt `pymongo.MongoClient(MONGODB_URL)` and `MilvusClient(uri, token)` connection.
2. After pipeline run, query Mongo `ingestion_checkpoints` for recent checkpoint and `ingestion_metrics` for run summary.
3. Query Milvus collection for approximate vector count for the target namespace/collection.
4. Print the `summary` returned by `ingest_directory` and cross-check with metric documents in Mongo.

Common failure modes & debugging tips
------------------------------------
- Missing `NVIDIA_NIM_API_KEY` or other provider keys: pipeline may error during embedder initialization. Ensure keys are set.
- GROBID not reachable: parsing files that need XML extraction may fail; either start local GROBID (notebooks include subprocess start) or set `GROBID_URL` to a reachable instance.
- Milvus/Zilliz connectivity: ensure the correct URI and token; check network egress from Colab/Kaggle to the vector service.
- Mongo auth/URI issues: use atlas connection string with SRV form or proper credentials; verify from a small `pymongo` probe before full ingestion.
- Notebook magic for Colab: use `%pip install` (not `!pip`) to avoid JSON lint errors when editing notebooks programmatically.

Notes on environment differences
--------------------------------
- Colab: GPU available (if you select a GPU runtime). Notebook uses `%pip` and can start a local GROBID subprocess. Use the Colab UI to set environment variables or write them into a hidden cell with `os.environ` (avoid committing secrets).
- Kaggle: datasets mounted under `/kaggle/input`. Working dir is `/kaggle/working`; some mounts are read-only — copy data into `working` if needed. Use Kaggle Secrets for credentials.

Files inspected and relevant locations
------------------------------------
- `src/ingestion/pipeline.py` — pipeline implementation and `ingest_directory` signature.
- `src/config/settings.py` — pydantic settings and list of required env vars (notably `nvidia_nim_api_key`).
- `src/ml/embedding/embedder.py` — embedder selection and GPU/HF options.
- `src/ingestion/checkpoint.py`, `src/data/mongo/doc_store.py`, `src/ingestion/vector_indexer.py`, `src/vectorstore/registry.py` — storage, checkpoint, and indexing details.
- Notebooks: `notebooks/colab_ingestion.ipynb`, `notebooks/kaggle_ingestion.ipynb`, `notebooks/kaggle_ingestion.md`.

Suggested next steps for Claude
------------------------------
1. Review the notebooks to ensure parameter choices (batch_size, recreate_index) match operational goals.
2. Confirm list of data sources and dataset paths to process (what `SOURCE_TYPE` and `DATA_DIR` should be for each run).
3. Validate credentials and run a small smoke ingestion (one small directory or single file) to exercise parsing, embedding, upsert, and checkpointing.
4. After smoke run, inspect `ingestion_metrics` and a few persisted chunk documents to validate chunk quality and metadata enrichment.
5. If scale runs are needed, recommend increasing batch_size and monitoring Milvus load and Mongo write throughput; consider rate-limiting or staged ingestion.

Minimal run checklist for a smoke test (copyable)
------------------------------------------------
1. Open `notebooks/colab_ingestion.ipynb` in Colab.
2. Set env vars in Colab UI: `MONGODB_URL`, `MONGODB_DB`, `VECTOR_URI`, `VECTOR_TOKEN`, `NVIDIA_NIM_API_KEY`, `GROBID_URL` (if external).
3. Run the install cell (first cell uses `%pip install -r requirements.txt`).
4. Run the config cell and verify connectivity probes (pymongo and Milvus client tests included).
5. Run the `IngestionPipeline.ingest_directory(...)` cell against a small local data folder.

Contact points for more detail
-----------------------------
If you need specific code snippets (exact parser fallback logic, chunker params, or the full `ingest_directory` implementation), open these files and ask for targeted extracts:

- `src/ingestion/pipeline.py`
- `src/config/settings.py`
- `src/ingestion/checkpoint.py`

— End of summary —
