# Migration: Offload embeddings & rerank to remote providers (Cohere / HF)

Goal: Allow ingestion to run on GPU in Colab and use cloud vector DB (Zilliz/Milvus + MongoDB). Make query-serving optionally GPU-free by using remote embedding/rerank APIs while keeping NVIDIA NIM for LLM generation.

---

## Summary

- Current: ingestion and query use local PyTorch models:
  - Dense embeddings: `src/ml/embedding/embedder.py` (SentenceTransformers, CUDA if available)
  - Reranker: `src/query/search/reranker.py` (transformers CrossEncoder, CUDA if available)
  - HYDE + answer LLM: NVIDIA NIM (`src/services/llm_client.py`, used by `src/api/routes/search.py`)
  - Vector DB: Milvus/Zilliz (`src/vectorstore/backends/milvus_store.py`, configured in `src/config/settings.py`)
- Plan: produce embeddings in Colab (GPU), store vectors in Zilliz Cloud + metadata in MongoDB; make query-time embedding & rerank pluggable so server can call remote APIs (Cohere / HF) to avoid local GPUs.

---

## Architecture Diagrams

Current (simplified)

```mermaid
flowchart LR
  A[Client UI] --> B[API Server]
  B --> C[HybridRetriever]
  C --> D[Milvus (local / cloud)]
  C --> E[Local Embedder (SentenceTransformers)]
  B --> F[Cross-Encoder Reranker (local)]
  B --> G[NVIDIA NIM LLM]
  D --> H[MongoDB metadata]
```

Proposed with remote providers

```mermaid
flowchart LR
  subgraph Ingestion
    IColab[Colab (GPU)] -->|dense vectors| MilvusCloud[Zilliz/Milvus Cloud]
    IColab -->|metadata| MongoDB
  end

  subgraph Serving
    Client --> APIServer[API Server]
    APIServer --> Retriever[HybridRetriever]
    Retriever -->|dense search| MilvusCloud
    Retriever -->|sparse search| MilvusCloud
    Retriever -->|maybe remote embed| RemoteEmbed[Remote Embed API (Cohere/HF)]
    APIServer -->|optional remote| RemoteRerank[Remote Reranker API]
    APIServer --> NIM[NVIDIA NIM (LLM)]
    MilvusCloud --> MongoDB
  end
```

---

## Migration Plan (phased)

### Phase 0 — Prep & config

- Add config keys in `src/config/settings.py`:
  - `embed_provider: str` (options: `local`, `cohere`, `hf`)
  - `embed_api_key`, `embed_api_url`, `rerank_provider`, `rerank_api_key`, timeouts, fallback modes.
  - `milvus_cloud: bool` and `vector_token` already exist; set them for Zilliz Cloud.
- Document Colab environment requirements (Python deps, `requirements.txt` entries: `sentence-transformers`, `pymilvus`, `pymongo`, `torch`).

### Phase 1 — Ingestion (Colab)

- Run ingestion in Colab using existing ingestion tasks:
  - Ensure `vector_uri` points to Zilliz Cloud and `vector_token` set.
  - Set `milvus_cloud=True` to skip explicit `load_collection` calls.
- Validate upserts by querying Milvus Cloud after ingestion (see tests below).
- Save a snapshot of which embedding model + seed used (store model name in collection metadata / Mongo).

### Phase 2 — Pluggable embedder + remote impl

- Add an `Embedder` interface (module e.g., `src/ml/embedding/base.py`) and keep `DualEmbedderV2` as local impl.
- Implement `RemoteEmbedder` (e.g., `src/ml/embedding/remote.py`) that:
  - Calls Cohere / HF inference to get embeddings for queries.
  - Has batching + retry + timeout + metric normalization (cosine norm if needed).
- Wire `HybridRetriever` to use configured embedder (inject via settings or factory) instead of constructing `DualEmbedderV2` directly.

### Phase 3 — Remote reranker (optional, recommended)

- Implement `RemoteReranker` (`src/query/search/remote_reranker.py`) that accepts query+candidate texts and calls remote service:
  - Cohere has rerank endpoints; HF Inference can host cross-encoders.
  - Return scores and preserve `RetrievedChunk.score`.
- Update `_get_or_create_component` in `src/api/routes/search.py` to instantiate `RemoteReranker` when configured.

### Phase 4 — Fallbacks & caching

- Keep local reranker/embedder as fallback when remote provider fails.
- Use caching for embeddings/rerank outputs (settings: `cache_ttl_embedding`, `cache_ttl_rerank`) to reduce API calls and cost.
- Add circuit-breaker logic and metrics.

### Phase 5 — Testing & canary rollout

- Run unit + integration tests (see test plan).
- Canary using 5-10% traffic or separate staging instance with `embed_provider=cohere`.
- Compare results (quality metrics below) vs local models.

### Phase 6 — Full rollout & monitoring

- Roll out, monitor latency, cost, correctness. Tune caching and batching.

---

## Implementation pointers & code locations

- Local embedder: `src/ml/embedding/embedder.py` — used by ingestion and `HybridRetriever`.
- Local reranker: `src/query/search/reranker.py` — loaded in `src/api/routes/search.py`.
- Where to change:
  - `src/query/search/retriever.py`: replace `self.embedder = DualEmbedderV2(settings.dense_model_name)` with a factory: `get_embedder()` from new factory module.
  - `src/api/routes/search.py`: in `_get_or_create_component`, return `RemoteReranker` when `settings.rerank_provider != 'local'`.
  - Add new modules:
    - `src/ml/embedding/base.py` (interface)
    - `src/ml/embedding/remote.py` (Cohere/HF impl)
    - `src/query/search/remote_reranker.py` (remote scoring)
    - `src/utils/http_clients.py` (shared http client + retries/timeouts)
- Example pseudocode for remote embed call (Cohere):

```python
# src/ml/embedding/remote.py (concept)
import httpx
class RemoteEmbedder:
  def __init__(self, api_key, url="https://api.cohere.ai/embeddings"):
    self.client = httpx.Client(timeout=10)
    self.api_key = api_key
  def embed_batch(self, texts, batch_size=32):
    # batch, call API, parse vectors, normalize if needed
    ...
```

---

## Testing & Validation Plan

1. Unit tests
   - Add unit tests for `RemoteEmbedder.embed_batch` and `RemoteReranker.rerank` using mocked HTTP responses.
   - Existing tests in `tests/` should continue to pass.

2. Integration tests (staging)
   - Ingest a representative dataset in Colab (small sample).
   - Query sample queries and compare:
     - Top-k overlap between local and remote rerank (e.g., top-10 intersection)
     - Embedding cosine similarity distribution for same-document pairs.
   - Commands/examples:

```bash
# run API server in staging env:
export VECTOR_URI="https://<zilliz-cloud-endpoint>"
export VECTOR_TOKEN="<token>"
export MILVUS_CLOUD="true"
export EMBED_PROVIDER="cohere"
export EMBED_API_KEY="<key>"
# start uvicorn / docker-compose as usual
```

3. Performance & latency
   - Measure p50/p95/p99 for:
     - embedding call (remote vs local)
     - rerank call (remote vs local)
     - full search request (end-to-end)
   - Acceptable thresholds:
     - P50 < 200–400ms for embed + rerank (depends on provider)
     - If >1s, consider caching or partial local fallback.

4. Quality metrics
   - Use relevance annotations (small labeled set) to compute:
     - NDCG@k, Precision@k, MRR for local vs remote.
   - If drop in quality > X% (define X, e.g., 5–10%), investigate model mismatch.

5. Consistency checks
   - Ensure the embedding model used in Colab for ingestion == the runtime embedding provider model or is well-matched.
   - If different, validate nearest-neighbor recall loss: measure recall@k for known positives.

6. Canary rollout
   - Start with a small percentage of traffic or a staging endpoint where `embed_provider=cohere`.
   - Monitor:
     - Error rates, latency, cost per request, quality metrics.
   - If regressions appear, roll back to local.

7. Monitoring & alerts
   - Instrument:
     - Count of remote API calls, failures, latencies.
     - Cache hit rate.
     - Cost estimate per day/week.
   - Alert on >5% remote-call error rate or >30% increase in average latency.

---

## Verification checklist (quick)
- [ ] Ingestion in Colab successfully upserts vectors to Milvus Cloud.
- [ ] Queries return results and answer generation still calls NVIDIA NIM.
- [ ] Remote embed + rerank endpoints respond within SLA.
- [ ] Unit + integration tests pass.
- [ ] Quality metrics acceptable vs baseline.

---

## Risks & Tradeoffs
- Latency increase due to network calls to embedding/rerank APIs.
- Cost (per-embedding / per-rerank) can be significant — use batching/caching.
- Model mismatch between ingestion embeddings and query-time embeddings reduces retrieval recall.
- Reranker differences can change ranking behavior; test carefully.

---

## Quick next steps
- (A) Draft precise code patches to add a `RemoteEmbedder` and `RemoteReranker` and wire config.
- (B) Produce a small Colab notebook snippet (requirements + minimal ingestion snippet) to run embeddings and upsert to Zilliz Cloud.
- (C) Provide test scripts to compare local vs remote results automatically.

Pick one and I will implement it.
