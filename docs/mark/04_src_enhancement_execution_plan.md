# OpenInsight v2 in src/ - Enhancement-First Execution Plan

## Intent
This plan adapts the v2 architecture to the current codebase without unnecessary rewrites.

Primary constraints:
- Keep current `src/` structure.
- Keep parser split in `src/ingestion/parsers/` (do not collapse into a single parser file).
- Prefer enhancement and selective replacement over rebuilding from scratch.
- Keep existing `/query` endpoint functional while building new `/search` pipeline.

---

## Current Architecture Fit (What Is Good Already)

The following are already strong and should be reused:
- Modular parser structure in `src/ingestion/parsers/`.
- Mature ingestion orchestration in `src/ingestion/pipeline_v3.py`.
- Hybrid vector support (dense+sparse) in `src/ingestion/vector_db.py`.
- Reranker module in `src/query/reranker.py`.
- Existing query flow in `src/query/standard.py` and API route in `src/api/routes/query.py`.

The v2 work should therefore be an additive upgrade path, not a reset.

---

## Preservation Rules (Hard Rules)

1. Do not remove or flatten parser files under `src/ingestion/parsers/`.
2. Do not break current ingestion and `/query` behavior until v2 `/search` is verified.
3. Add new v2 modules in parallel where possible, then switch wiring.
4. Keep backward-compatible config defaults.

---

## Proposed Module Map (src Adaptation)

### Keep as-is (enhance only)
- `src/ingestion/parsers/base.py`
- `src/ingestion/parsers/pubmed.py`
- `src/ingestion/parsers/grobid.py`
- `src/ingestion/parsers/ocr.py`
- `src/ingestion/parsers/icmr.py`
- `src/ingestion/parsers/cochrane.py`
- `src/ingestion/parsers/who.py`
- `src/ingestion/parsers/cdc.py`
- `src/ingestion/parsers/statpearls.py`

### Add new v2 modules
- `src/search/query_understanding.py`
- `src/search/retriever.py`
- `src/search/fusion.py`
- `src/search/mmr.py`
- `src/search/context_builder.py`
- `src/search/cache.py`
- `src/api/routes/search.py`

### Replace or supersede gradually
- `src/query/standard.py` -> remains for legacy path initially; v2 pipeline lives in new `src/search/*` and new route.
- `src/ingestion/vector_db.py` -> enhance in-place or split responsibilities with new index helper while preserving current imports.
- `src/utils/chunker_v2.py` -> enhance or wrap from a new ingestion chunk module, but keep compatibility for existing pipeline.

### Update in place
- `src/core/config.py` (new v2 settings, backward-compatible defaults)
- `src/api/main.py` (add `/search` router and lifespan singletons)

---

## Execution Phases

## Phase 0 - Baseline and Safety

Goal:
- Prepare for additive v2 work without destabilizing current behavior.

Tasks:
- Add v2 config keys to `src/core/config.py`:
  - `qdrant_collection_v2` default `openinsight_v2`
  - `dense_model_name`
  - `reranker_model_name`
  - `cache_version`
  - `cache_ttl_search`
  - `cache_ttl_rerank`
  - `top_k_retrieval`
  - `top_k_after_fusion`
  - `top_k_after_rerank`
  - `top_k_final`
  - `mmr_lambda`
  - `hyde_enabled`
- Keep existing keys untouched.

Validation:
- App imports cleanly.
- Existing `/query` endpoint unchanged.

---

## Phase 1 - Ingestion Metadata and Chunk Quality Upgrade

Goal:
- Improve chunk payload richness and retrieval readiness while preserving parser modules.

Tasks:
- Add `src/ingestion/metadata_v2.py`:
  - `DocType`, `EvidenceLevel`
  - evidence mapping and boost mapping
  - `ChunkMetadataV2` serializer for Qdrant payload
  - `MetadataEnricherV2` with specialty and India relevance detection
- Add `src/ingestion/chunker_v3.py` (or enhance current chunker with a compatibility wrapper):
  - enforce doc_summary + section-aware + paragraph windows
  - table-aware chunk formatting
  - contextual text prefix for embedding input
- Keep `src/utils/chunker_v2.py` available; do not remove.

Parser enhancement scope (without changing file layout):
- Ensure parser outputs expose enough fields for metadata enrichment:
  - title, abstract, sections, year, journal, doi/pmid where possible
  - table detection from GROBID TEI where available

Validation:
- Unit checks for metadata inference.
- Sample document yields multiple chunks and complete payload fields.

---

## Phase 2 - Embedding and Indexing Upgrade

Goal:
- Ensure ingestion writes v2-ready dense+sparse vectors with proper payload indexing.

Tasks:
- Add `src/ingestion/embedder_v2.py`:
  - dense embedding with `torch.inference_mode()`
  - normalized vectors
  - medical-aware sparse tokenization
- Add `src/ingestion/qdrant_indexer_v2.py` or enhance `src/ingestion/vector_db.py`:
  - named vectors (`dense`, `sparse`)
  - payload indexes for v2 filter fields
  - batched upsert and deterministic point ids from chunk ids
- Add `src/ingestion/mongo_store_v2.py` (or extend existing doc store model) for full document and chunk persistence linked by chunk ids.

Validation:
- `openinsight_v2` collection exists with both vector types.
- Payload indexes created.
- Upserted points include all required metadata flags.

---

## Phase 3 - Ingestion Orchestration in src

Goal:
- Build a dedicated v2 ingestion entrypoint while preserving current pipelines.

Tasks:
- Add `src/ingestion/pipeline_v4.py`:
  - reuse parser modules from `src/ingestion/parsers/`
  - parse -> enrich metadata -> chunk -> embed -> store mongo -> upsert qdrant
  - source-aware input handling (`pubmed`, `icmr`, `cochrane`, `nmc_guideline`, `rssdi`, `who`)
- Add `src/ingestion/run_ingestion_v2.py` CLI with `--dir`, `--source`, `--recreate`.
- Keep `pipeline.py`, `pipeline_v2.py`, `pipeline_v3.py` intact.

Validation:
- Dry-run imports pass.
- CLI help works.
- Small sample ingestion succeeds end to end.

---

## Phase 4 - v2 Search Stack (New Package)

Goal:
- Implement complete standard search path as isolated new modules.

Tasks:
- Create `src/search/` package with:
  - `query_understanding.py`
  - `retriever.py`
  - `fusion.py`
  - `reranker.py` (can wrap/reuse logic from `src/query/reranker.py`)
  - `mmr.py`
  - `context_builder.py`
  - `cache.py`

Detailed behavior targets:
- Intent detection + metadata filter inference + optional HyDE.
- Parallel dense/sparse retrieval.
- RRF fusion with evidence and recency boosts.
- Cross-encoder rerank on post-fusion candidates.
- MMR dedup for final context diversity.
- Citation list and context assembly.
- Redis caches for reranked and full results.

Validation:
- Module-level tests for each component.
- Search pipeline returns structured chunks and citations.

---

## Phase 5 - API Wiring and Safe Rollout

Goal:
- Add v2 API path without breaking existing users.

Tasks:
- Add `src/api/routes/search.py` with `POST /search` response contract:
  - `answer`
  - `citations`
  - `query_intent`
  - `chunks_retrieved`
  - `cache_hit`
- Update `src/api/main.py`:
  - register new search router
  - add lifespan startup for singletons (query understanding, retriever, reranker, cache)
- Keep `src/api/routes/query.py` as legacy route for fallback.

Validation:
- `/query` continues to work.
- `/search` works and returns expected shape.
- Repeat `/search` query shows cache hit.

---

## Phase 6 - Cutover Criteria

Switch default caller from `/query` to `/search` only when:
- Ingestion to `openinsight_v2` is complete and payload validated.
- Search latency targets are met:
  - first call under target budget
  - repeat call significantly faster via cache
- Citation quality and diversity improve against baseline.

After cutover:
- Keep `/query` for one release behind feature flag.
- Deprecate legacy path after stability window.

---

## File-Level Action Table

Enhance:
- `src/core/config.py`
- `src/api/main.py`
- `src/ingestion/vector_db.py` (if chosen over dedicated indexer file)
- Parser files in `src/ingestion/parsers/` (field-level improvements only)

Add:
- `src/search/__init__.py`
- `src/search/query_understanding.py`
- `src/search/retriever.py`
- `src/search/fusion.py`
- `src/search/reranker.py`
- `src/search/mmr.py`
- `src/search/context_builder.py`
- `src/search/cache.py`
- `src/api/routes/search.py`
- `src/ingestion/metadata_v2.py`
- `src/ingestion/chunker_v3.py`
- `src/ingestion/embedder_v2.py`
- `src/ingestion/qdrant_indexer_v2.py` (optional if enhancing `vector_db.py`)
- `src/ingestion/mongo_store_v2.py`
- `src/ingestion/pipeline_v4.py`
- `src/ingestion/run_ingestion_v2.py`

Keep unchanged initially:
- `src/api/routes/query.py`
- `src/query/standard.py`
- `src/ingestion/pipeline_v3.py`

---

## Suggested Implementation Order (Task Tickets)

1. Config expansion in `src/core/config.py`.
2. `src/search/` scaffolding + query understanding.
3. Retriever + fusion + reranker adapter.
4. MMR + context builder + cache.
5. New `/search` route + lifespan singletons.
6. Ingestion metadata/chunker/embedder/indexer upgrades.
7. Ingestion v2 CLI and sample run.
8. End-to-end validation + cutover decision.

---

## Notes for Implementation Sessions

- All work should be additive-first; only replace code paths when an equivalent v2 path is already validated.
- Do not refactor parser file boundaries.
- Preserve existing docs and scripts; add new scripts for v2 workflows.
- Prefer small, testable PRs by phase.
