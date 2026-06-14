## AGENTS.md - OpenInsight Multi-Agent Coordination Instructions

## Scope

This file applies to the entire OpenInsight repository. It gives shared rules, architecture context, and coordination guidance for all AI coding agents working here. Per-folder `AGENTS.md` files take precedence over this one when they exist.

## Required Git Rules

- Commit every turn of work
- Do not amend commits
- Do not change branches without explicit user permission
- Do not push, pull, or rebase unless explicitly requested
- Stay on `restruct` during active experimentation on this branch
- Do not merge to `main` without explicit user permission

## Commit Expectations

- Focused commits scoped to the requested task
- Conventional commit messages (`feat:`, `fix:`, `refactor:`, `chore:`, `docs:`, `test:`)
- No generated-with lines, attribution blocks, or command transcripts in commit messages

## Validation

```bash
pip install -r requirements.txt
pip install ruff
docker compose up -d
pytest tests/ -v -m "not requires_gpu"
ruff check src tests
black --check src tests
isort --check-only src tests
```

## Safety

- Do not revert user-authored or unrelated local changes unless explicitly requested
- Avoid destructive git commands unless explicitly requested
- Never commit `.env` — secrets belong in `.env` or OS environment variables only; tunable defaults live in `config.base.json` and `config.{APP_ENV}.json`
- Never run ingestion with `--recreate` against shared or production Milvus collections — `MilvusVectorStore.ensure_collection(recreate=True)` in `src/vectorstore/backends/milvus_store.py` calls `drop_collection`
- Never drop MongoDB collections in the `openinsight` database (documents, chunks, vault items, `failed_documents` dead-letter queue) without explicit user instruction
- Never weaken path checks in `src/tools/safety.py` or bypass `ALLOWED_ROOTS` in filesystem tools under `src/tools/filesystemtools/`
- Do not rename keys in `TOOL_REGISTRY` (`src/tools/__init__.py`) — the DeepInsights orchestrator and `tests/test_tools.py` resolve tools by string name at runtime
- Never manually edit `requirements.txt` pins — update dependencies with `pip install <package>` then `pip freeze` or deliberate pin edits followed by `pip install -r requirements.txt` verification

## Project Overview

OpenInsight is a medical knowledge retrieval and question-answering system built by SentArc Labs for clinical decision support, targeting Indian physicians. It ingests evidence from ICMR guidelines, PubMed, WHO, and other medical literature, indexes it for hybrid vector search, and returns cited answers. The FastAPI backend exposes fast single-pass RAG at `/search`, a multi-agent DeepInsights pipeline at `/deep-insights`, plus research vault and clinical report endpoints.

## Architecture

```
              Clients (React UI / Mobile / API consumers)
                                |
                                v
+-----------------------------------------------------------------------+
|                  FastAPI — src/api/main.py                            |
|     /search        /deep-insights         /vault          /reports    |
+--------+----------------+-------------------+---------------+---------+
         |                |                   |               |
         v                v                   |               v
+----------------+  +------------------+      |        src/reports/
| Search RAG     |  | DeepInsight      |      |        generators +
| src/query/     |  | Orchestrator     |      |        pdf_renderer
| search/*       |  | src/query/       |      |
| cache          |  | deepinsight/*    |      |
| retriever      |  | agents + tools   |      |
| fusion/rerank  |  +--------+---------+      |
+--------+-------+           |                |
         |                   v                |
         +--------> src/services/llm/* ------> NVIDIA NIM / OpenAI /
         |          router + providers.json    Anthropic / Cohere / Ollama
         v
+--------+---------+----------+----------+
| MongoDB          | Milvus   | Redis    | GROBID (docker :8070)
| docs, chunks,    | dense +  | search & | PDF/XML parsing
| vault            | sparse   | embed    |
| src/data/mongo/  | vectors  | cache    |
|                  | src/vectorstore/     |
+--------+---------+----------+----------+
         ^
         |
+--------+------------------------------------------------+
| Ingestion — src/ingestion/pipeline.py                   |
| parsers → dedupe → chunk → embed → Milvus + MongoDB     |
| Celery tasks via src/ingestion/celery_app.py            |
+---------------------------------------------------------+
```

## Multi-Agent Role Table

| Agent | Role | Owns (files/folders) | Never touches |
|-------|------|----------------------|---------------|
| OpenCode | Orchestrator / Architecture | `docs/`, `changelog.md`, `src/query/deepinsight/agents/`, `src/query/deepinsight/agents/skills/`, `src/query/deepinsight/orchestrator.py` | `AGENTS.md`, `TASKS.md`, `config.base.json`, `docker-compose.yml` |
| Cursor Agent | Feature implementation | `src/`, `tests/`, `prompts/` | `AGENTS.md`, `TASKS.md`, `src/query/deepinsight/orchestrator.py` without coordination |
| Copilot | Inline completions | No file ownership | `docs/01_ARCHITECTURE.md`, `AGENTS.md`, `TASKS.md`, `src/query/deepinsight/orchestrator.py` |
| BlackboxAI | Config / Boilerplate | `config.base.json`, `config.production.json`, `config.kaggle.json`, `.env.example`, `docker-compose.yml`, `pyproject.toml`, `requirements.txt`, `scripts/`, `.devcontainer/` | `AGENTS.md`, `TASKS.md`, `src/query/`, `tests/` |
| Antigravity | Experimental / Notebooks | `notebooks/`, exploratory one-off scripts outside `src/` | `AGENTS.md`, `TASKS.md`, `src/`, `config.base.json`, `docker-compose.yml` |

Hard File Locks — do not modify without explicit user instruction:

- `AGENTS.md`
- `TASKS.md`
- `src/tools/__init__.py` — `TOOL_REGISTRY` with 55 tool names consumed by orchestrator and routes
- `src/config/settings.py` — settings loading priority and all field definitions
- `src/services/llm/providers.json` — LLM provider registry loaded at runtime
- `src/query/deepinsight/orchestrator.py` — DeepInsights agent coordination loop
- `src/vectorstore/backends/milvus_store.py` — Milvus collection schema and field contract

## Project Structure

| Path | Purpose |
|------|---------|
| `src/api/` | FastAPI app (`main.py`), routes (`search`, `deep_insights`, `vault`, `reports`), middleware, Pydantic models |
| `src/config/` | `settings.py` (JSON + `.env` hybrid config), `logging_config.py` |
| `src/constants/` | Shared magic values used across modules |
| `src/data/` | MongoDB stores: `mongo/doc_store.py`, `mongo/vault_store.py`, `mongo/connection.py` |
| `src/ingestion/` | Ingestion pipeline, Celery tasks, checkpointing, parsers (PubMed, ICMR, GROBID, OCR, WHO, CDC, etc.) |
| `src/ml/` | Chunking (`chunking/chunker.py`), embeddings (`embedding/embedder.py`), NER (`ner.py`) |
| `src/query/` | Search RAG (`search/`), DeepInsights agents (`deepinsight/`), validation, contradiction detection |
| `src/reports/` | Clinical summary and evidence review generation, PDF rendering |
| `src/services/` | LLM client, multi-provider router (`llm/`), browser/CDP fetchers (`browser/`) |
| `src/tools/` | 55 standalone agent tools: `filesystemtools/`, `websearchtools/`, `citationtools/`, `doctools/`, `safety.py` |
| `src/utils/` | PubMed client, date/text utilities, metrics and health checking |
| `src/vectorstore/` | Vector store interface, Milvus backend, types, registry |
| `tests/` | Pytest suite with shared fixtures in `conftest.py` |
| `docs/` | Architecture, query pipeline, DeepInsights, ingestion, and evaluation docs |
| `scripts/` | `run.py` (ingestion wrapper), `seed_pubmed.py`, `seed_icmr.py`, `zilliz_smoke.py`, `maintainence.py` |
| `prompts/` | LLM prompt templates: `system.md`, `query_rewrite.md` |
| `data/` | Local data directory; `data/raw/` and `data/processed/` are gitignored |
| `notebooks/` | Kaggle/Colab ingestion notebooks (`kaggle_ingestion.ipynb`, `kaggle_ingestion.py`) |
| `.devcontainer/` | VS Code dev container config and `setup.sh` |
| `config.base.json` | Versioned non-secret defaults for all environments |
| `config.production.json` | Production overrides (`milvus_cloud`, provider switches) |
| `config.kaggle.json` | Kaggle/Colab environment overrides |
| `docker-compose.yml` | Local MongoDB, Redis, GROBID, Milvus standalone stack |
| `pyproject.toml` | Project metadata, pytest/coverage/ruff configuration |
| `requirements.txt` | Python dependency pins |
| `.env.example` | Secret and connection-string template |
| `changelog.md` | Release and refactor history |

## Key Files

| Path | Purpose |
|------|---------|
| `src/api/main.py` | FastAPI entry point, middleware, `/health`, `/metrics`, router registration |
| `src/api/routes/search.py` | `POST /search` and `POST /search/document` RAG endpoints |
| `src/api/routes/deep_insights.py` | `POST /deep-insights` multi-agent endpoint |
| `src/api/routes/vault.py` | Research vault CRUD for citations, notes, collections |
| `src/api/routes/reports.py` | `POST /reports/generate` clinical report endpoint |
| `src/config/settings.py` | `Settings` class and `get_settings()` with JSON + `.env` loading |
| `src/query/search/retriever.py` | Hybrid dense + sparse retrieval with HYDE |
| `src/query/search/fusion.py` | Reciprocal Rank Fusion and evidence boosting |
| `src/query/search/reranker.py` | Cross-encoder reranking via `local` or `cohere` provider |
| `src/query/deepinsight/orchestrator.py` | DeepInsights orchestrator coordinating agents and `TOOL_REGISTRY` |
| `src/ingestion/pipeline.py` | Main ingestion orchestration with dead-letter queue and checkpointing |
| `src/ingestion/run_ingestion.py` | CLI entry: `python -m src.ingestion.run_ingestion` |
| `src/ml/embedding/embedder.py` | Multi-provider embeddings (`local`, `huggingface`, `cohere`) |
| `src/vectorstore/backends/milvus_store.py` | Milvus/Zilliz implementation with hybrid dense+sparse schema |
| `src/data/mongo/doc_store.py` | MongoDB document and chunk persistence |
| `src/services/llm/router.py` | LLM provider routing by agent role |
| `src/services/llm/providers.json` | Provider definitions: models, API key env vars, timeouts |
| `src/tools/__init__.py` | `TOOL_REGISTRY`, `get_tool()`, `call_tool()`, async/sync metadata |
| `src/tools/safety.py` | Filesystem sandbox (`ALLOWED_ROOTS`) for all agent file tools |
| `tests/conftest.py` | Shared fixtures, env isolation via `isolate_settings` autouse fixture |
| `scripts/run.py` | Unified ingestion runner wrapper |

## Development

```bash
pip install -r requirements.txt
cp .env.example .env
docker compose up -d
uvicorn src.api.main:app --reload --port 8000
python scripts/run.py pubmed ./data/pdfs
python scripts/run.py icmr ./data/pdfs -w 8 --recreate --stats
python -m src.ingestion.run_ingestion --dir ./data/pdfs --source pubmed --workers 6 --batch-size 10
python scripts/seed_pubmed.py
python scripts/seed_icmr.py
python scripts/zilliz_smoke.py
black src tests
isort src tests
```

Ingestion sources: `pubmed`, `icmr`, `cochrane`, `nmc_guideline`, `rssdi`, `who`, `cdc`, `statpearls`.

API docs: `http://localhost:8000/docs`

## Testing

```bash
docker compose up -d
pytest tests/ -v -m "not requires_gpu"
pytest tests/ -v -m unit
pytest tests/ -v -m integration
pytest tests/test_tools.py -v
pytest tests/ --cov=src --cov-report=term-missing
```

Tests marked `requires_mongodb` or `requires_grobid` need `docker compose up -d` (MongoDB on `27017`, GROBID on `8070`). Tests marked `requires_gpu` are excluded from the pre-commit gate. Tests marked `requires_network` may need API keys in `.env`.

- **Assertion library:** plain `pytest` assertions and `pytest.raises`; `unittest.mock` (`MagicMock`, `AsyncMock`, `patch`)
- **Location:** test files in top-level `tests/`, named `test_*.py` (not co-located with source)
- **Structure:** class-based groupings (`class TestRetrievedChunk`) for related cases; standalone functions for smoke tests
- **Parametrization:** `@pytest.mark.parametrize` and shared tuples in `conftest.py` (`MEDICAL_QUERY_INTENT_PAIRS`, `EVIDENCE_LEVEL_BOOST_PAIRS`)
- **Markers:** `unit`, `integration`, `slow`, `requires_gpu`, `requires_network`, `requires_mongodb`, `requires_grobid` (defined in `pyproject.toml`)
- **Async:** `@pytest.mark.asyncio` for async tool and API tests
- **Optional deps:** `pytest.importorskip("torch")` guards embedding/retriever tests
- **Fixtures:** use `isolate_settings` (autouse), `mock_vector_store`, `mock_embedder`, `sample_document`, `sample_chunk`, `sample_retrieved_chunks` from `conftest.py`

No separate e2e test runner is configured.

## Build Requirements

- Python `>=3.11` (`requires-python` in `pyproject.toml`)
- Package manager: `pip` with `requirements.txt` (no `uv.lock` or `poetry.lock`)
- `pip install -r requirements.txt` into a virtual environment (`.venv/` recommended)
- Docker Compose for local MongoDB (`27017`), Redis (`6379`), GROBID (`8070`), Milvus (`19530`)
- `MONGODB_URL` in `.env` for MongoDB; `VECTOR_URI` and `VECTOR_TOKEN` for Zilliz Cloud (local Milvus uses `http://localhost:19530` from `config.base.json`)
- `NVIDIA_NIM_API_KEY` or another provider key from `src/services/llm/providers.json` for LLM calls
- `torch`, `sentence-transformers`, `transformers` for local embedding and reranking
- spaCy/sciSpaCy model `en_core_sci_md` for production NER (tests override to `en_core_web_sm` via `conftest.py`)
- Milvus container requires `seccomp: unconfined` in `docker-compose.yml` on Linux/WSL2
- No type checker configured (no mypy or pyright in `pyproject.toml`)

## Conventions

- Start Python files with `from __future__ import annotations`
- Use `loguru` (`from loguru import logger`) with module-specific prefixes (`[pipeline]`, `[HFEmbedder]`, `[PubMedClient]`)
- Load config via `get_settings()` from `src.config.settings`; never hardcode secrets
- Pydantic models for API schemas and settings fields
- Formatter: Black (`black==24.10.0` in `requirements.txt`); devcontainer sets format-on-save
- Import sorter: isort (`isort==5.13.2` in `requirements.txt`)
- Linter: Ruff (`target-version = "py311"`, `line-length = 120`, rules `E`, `F`, `W`, `I` in `pyproject.toml`)
- `snake_case` for modules, functions, variables; `PascalCase` for classes; test classes prefixed `Test`
- One tool = one function in one file under `src/tools/`; prefer direct imports in agents over `TOOL_REGISTRY` lookups
- Absolute imports from `src.*` (e.g. `from src.query.search.retriever import HybridRetriever`)
- Consolidate magic numbers in `src/constants/` rather than duplicating across modules
- Agent skill docs as `*.SKILL.md` under `src/query/deepinsight/agents/skills/`
- No emojis in code or terminal output except where existing CLI already uses them (`scripts/run.py` interactive mode)

## Configuration and Secrets Split

Settings load in this priority (highest first): OS environment variables, `.env` file, `config.{APP_ENV}.json`, `config.base.json`, Pydantic field defaults in `src/config/settings.py`. API keys, passwords, and connection strings with credentials go in `.env` only. Tunable parameters (retrieval `top_k`, chunk sizes, model names, cache TTLs) go in `config.base.json` or environment-specific JSON overrides. Do not move secrets into JSON config files — `config.base.json` is committed to Git and must stay secret-free.

## Tool Registry and Async Dispatch

`TOOL_REGISTRY` in `src/tools/__init__.py` maps 55 string names to `{fn, async, desc, name}` metadata. The DeepInsights orchestrator resolves tools by name at runtime. Do not rename registry keys without updating `src/query/deepinsight/orchestrator.py`, route handlers, and `tests/test_tools.py`. Of the 55 tools, 22 filesystem tools are async coroutines — they must be called via `await call_tool(name, ...)` or `await fn(...)` after checking `is_async_tool(name)`. Calling an async tool without awaiting returns an unawaited coroutine object.

## Milvus Collection and Vector Schema

The only supported `vector_backend` is `milvus` (`src/vectorstore/registry.py` raises on anything else). Collection field names come from settings: `id` (`vector_id_field`), `dense` (`vector_dense_field`), `sparse` (`vector_sparse_field`), dimension `768` (`vector_dim`). Default collection is `openinsight_chunks`. Ingestion `--recreate` and `MilvusVectorStore.ensure_collection(recreate=True)` drop and rebuild the collection. Filter fields are whitelisted in `ALLOWED_FILTER_FIELDS` inside `milvus_store.py` — do not add filter fields without updating that frozenset.

## Embedding and Reranker Provider Contract

`embed_provider` accepts `local`, `huggingface`, or `cohere` (`src/ml/embedding/embedder.py`). Every provider's `embed_batch()` returns `(embeddings: np.ndarray, failed_indices: list[int])` — callers in ingestion and retrieval must filter out failed indices before indexing to Milvus. Sparse vectors are computed CPU-side via shared `_compute_sparse_vector()` regardless of dense provider. `rerank_provider` accepts `local` or `cohere` (`src/query/search/reranker.py`). Production (`config.production.json`) uses `huggingface` + `cohere`; local dev (`config.base.json`) uses `local` for both — providers must produce compatible 768-dim dense vectors and rerank scores without changing caller logic.

## Filesystem Tool Sandbox

All filesystem tools in `src/tools/filesystemtools/` enforce `ALLOWED_ROOTS` from `src/tools/safety.py`: `/tmp/openinsight_temp`, `/tmp/openinsight_reports`, and `/tmp`. Writes outside these roots raise `ValueError`; reads return `None`; deletes raise `PermissionError` unless `confirm=True`. New filesystem tools must import `ensure_safe_path` or `is_path_safe` from `safety.py` — do not implement ad-hoc path checks in individual tool files.

## Branch and Merge Policy

Active development runs on `restruct`. Do not switch to `main` or merge `restruct` into `main` without explicit user permission. `origin/HEAD` points to `main`, but agent work targets `restruct` until the user directs otherwise.

## Pull Requests / Handoffs

- When handing off between agents, update `TASKS.md` before stopping
- No test plans or checklists in commit messages or handoff notes
- Record which files changed, which agent owns the next step, and any blocked dependencies in `TASKS.md`
