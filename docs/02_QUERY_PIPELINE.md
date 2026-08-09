# Query Pipeline

> Last updated: August 2026
> Reflects codebase on `restruct` branch at commit `78f4d6b`

## Simple Search Pipeline (`POST /search`)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              REQUEST                                        │
│          { "query": "...", "top_k": 6, "save_to_vault": false }            │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         QUERY SANITIZATION                                  │
│  Validation: strip control chars + normalize whitespace; reject XSS/SQL patterns │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         QUERY UNDERSTANDING                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Intent Classification (QueryIntent enum)                           │    │
│  │    DIAGNOSTIC / THERAPEUTIC / PROGNOSTIC / DRUG_INFO /             │    │
│  │    GUIDELINE / GENERAL                                             │    │
│  │  Entity Extraction (spaCy or rule-based fallback)                   │    │
│  │  Metadata Filters (year/evidence_level/source_type)                 │    │
│  │  Query Expansion (MEDICAL_SYNONYMS dict)                            │    │
│  │  Optional LLM-based query rewrite                                  │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CACHE CHECK                                    │
│                    ┌─────────────────────────────────┐                      │
│                    │  Redis: hash(query + filters)   │                      │
│                    │  Key: openinsight:v2:search:sha256│                     │
│                    └─────────────────────────────────┘                      │
│                                    │                                        │
│                      ┌─────────────┴─────────────┐                          │
│                      ▼                           ▼                          │
│                    [HIT]                        [MISS]                      │
│                Return cached                    Continue                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼ (on miss)
┌─────────────────────────────────────────────────────────────────────────────┐
│                          HYBRID RETRIEVAL                                   │
│                                                                             │
│  ┌─────────────────────┐              ┌─────────────────────┐               │
│  │   DENSE SEARCH      │              │   SPARSE SEARCH     │               │
│  │   (Semantic)        │              │   (Keyword/TF-IDF)  │               │
│  │                     │              │                     │               │
│  │ Embed query         │              │ Compute sparse      │               │
│  │ S-PubMedBert → 768d │              │ vector (medical     │               │
│  │ or HyDE embedding   │              │ tokenization, IDF   │               │
│  │                     │              │ weighting)          │               │
│  │ Milvus search       │              │ Milvus sparse search│               │
│  │ top_k=50            │              │ top_k=50            │               │
│  └─────────┬───────────┘              └─────────┬───────────┘               │
│            │                                    │                           │
│            └──────────────┬─────────────────────┘                           │
│                           ▼                                                 │
│              ┌────────────────────────────────────┐                         │
│              │  Reciprocal Rank Fusion (k=60)     │                         │
│              │  + EvidenceBoost (1a→1.35, etc.)   │                         │
│              │  + RecencyBoost (2026→1.12, etc.)  │                         │
│              │  Dedup by chunk_id                  │                         │
│              │  Mark retrieval_source              │                         │
│              └─────────────┬──────────────────────┘                         │
│                            ▼                                                │ 
│                   top_k=20 results                                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            RERANKING                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Provider-based (configurable):                                     │    │
│  │                                                                     │    │
│  │  local:    Cross-encoder BAAI/bge-reranker-v2-m3 (568M params)     │    │
│  │            GPU inference, max_length=1024                            │    │
│  │                                                                     │    │
│  │  huggingface: HF Inference API (free tier, 300 req/hr)             │    │
│  │              Sends pairs individually, rate limit every 10 requests  │    │
│  │                                                                     │    │
│  │  cohere:   Cohere Rerank API (rerank-english-v3.0)                  │    │
│  │            Single batch call, proper /rerank endpoint                │    │
│  │                                                                     │    │
│  │  Keep top_k=8                                                       │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DIVERSITY (MMR)                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Maximal Marginal Relevance                                         │    │
│  │                                                                     │    │
│  │  MMR(d) = λ * relevance(d) - (1-λ) * max_similarity(d, selected)    │    │
│  │  λ = 0.7 (70% relevance, 30% diversity)                             │    │
│  │                                                                     │    │
│  │  Result: top_k diverse, relevant chunks (respects user's top_k)     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     CONTEXT ASSEMBLY                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  assemble_context(chunks, max_tokens=3000)                          │    │
│  │  - Evidence level labels ([Level 1a], [Level 2b], etc.)            │    │
│  │  - India-relevant flag ([INDIA-RELEVANT])                          │    │
│  │  - Citation markers ([C1], [C2], etc.)                             │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          LLM GENERATION                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Provider: LLMRouter → default NVIDIA NIM (Llama 3.1 70B)          │    │
│  │  Legacy: get_nim_client() (backward compat)                        │    │
│  │                                                                     │    │
│  │  Prompt:                                                            │    │
│  │  "You are a clinical decision support assistant..."                 │    │
│  │  "Context: [assembled chunks]"                                      │    │
│  │  "Question: [query]"                                                │    │
│  │  "Answer with numbered citations [1][2]..."                         │    │
│  │                                                                     │    │
│  │  Temperature: 0.1, Max tokens: 1024                                 │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          VALIDATION                                         │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐        │
│  │  Citation    │ │  Hallucina-  │ │   Safety     │ │  Confidence  │        │
│  │   Check      │ │   tion Det   │ │   Check      │ │   Scoring    │        │
│  └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘        │
│        │                  │                │               │                │
│        └──────────────────┴────────────────┴───────────────┘                │
│                             │                                               │
│                             ▼                                               │
│                    ┌─────────────────┐                                      │
│                    │ Final Response  │                                      │
│                    └─────────────────┘                                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     POST-PROCESSING                                         │
│  - Optional vault auto-save (if save_to_vault=true)                        │
│  - Cache write to Redis                                                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                             RESPONSE                                        │
│  {                                                                          │
│    "answer": "...",                                                         │
│    "citations": [...],                                                      │
│    "query_intent": "THERAPEUTIC",                                           │
│    "chunks_retrieved": 8,                                                   │
│    "cache_hit": false,                                                      │
│    "confidence_score": 0.85,                                                │
│    "recommendation": "SAFE",                                                │
│    "unverified_claims": [],                                                 │
│    "safety_warnings": [],                                                   │
│    "evidence_distribution": {"1a": 3, "2b": 2, ...},                       │
│    "is_safe": true,                                                         │
│    "needs_disclaimer": false,                                               │
│    "confidence_breakdown": {...}                                            │
│  }                                                                          │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Document Export (`POST /search/document`)

Runs the standard `/search` pipeline, then:
1. `build_doc_sections()` structures the answer into labeled sections
2. `generate_pdf()` or `generate_docx()` (chosen by request format) renders the file
3. Returns the file as a streaming download

This provides downloadable clinical reports without going through the full DeepInsights pipeline.

---

## Empty Response Behavior

When no relevant results are found, the search endpoint returns a complete response with all fields set to safe defaults:

```json
{
  "answer": "No relevant clinical information found in the knowledge base for this query.",
  "citations": [],
  "query_intent": "...",
  "chunks_retrieved": 0,
  "cache_hit": false,
  "confidence_score": 0.0,
  "recommendation": "NEEDS_REVIEW",
  "unverified_claims": [],
  "safety_warnings": [],
  "evidence_distribution": {},
  "is_safe": true,
  "needs_disclaimer": false,
  "confidence_breakdown": null
}
```

---

## Logging Behavior

The query pipeline includes structured logging for debugging and monitoring:

- **HYDE Failures**: When HYDE (Hypothetical Document Embeddings) generation fails (network timeout, API error, etc.), the failure is logged with a warning level and the system falls back to using the original query
- **Reranker Failures**: When the reranker fails (e.g., GPU out of memory), exceptions are logged before falling back to CPU-based score sorting
- **Request ID**: Every request gets a unique `X-Request-ID` propagated through the pipeline via `ContextVar` and bound to loguru

Example log output:
```
HYDE generation failed (Timeout(15.0s)), falling back to original query
Reranker failed (CUDA out of memory), falling back to score sorting
```

---

## Embedder API

The embedder's `embed_batch()` method returns a tuple `(embeddings, failed_indices)`:
- `embeddings`: numpy array with zero vectors at failed indices
- `failed_indices`: list of indices that failed (empty list means all succeeded)

Callers must filter out failed embeddings before indexing to avoid zero-vector pollution in Milvus.

### Providers

| Provider | Model | Dimension | Notes |
|----------|-------|-----------|-------|
| `local` | S-PubMedBert-MS-MARCO | 768 | SentenceTransformers, GPU support, normalized |
| `huggingface` | S-PubMedBert-MS-MARCO | 768 | HF Inference API, rate limiting, handles 3D/2D/1D responses |
| `cohere` | embed-english-v3.0 | 1024 | Cohere Embed API, dimension validated against Milvus schema |

Sparse vectors computed CPU-side via shared `compute_sparse_vector()` with medical tokenization, IDF weighting, and stopword removal — regardless of dense provider.

---

## Reranker Providers

| Provider | Model | Notes |
|----------|-------|-------|
| `local` | BAAI/bge-reranker-v2-m3 (568M params) | Cross-encoder, GPU inference, max_length=1024 |
| `huggingface` | BAAI/bge-reranker-v2-m3 | HF Inference API free tier, rate limit every 10 requests |
| `cohere` | rerank-english-v3.0 | Proper `/rerank` endpoint, single batch call |

---

## MMR Behavior

The Maximal Marginal Relevance (MMR) diversity step respects the user's requested `top_k` from the request. The final result count is determined by `payload.top_k` rather than a fixed internal value, ensuring users get exactly the number of results they request.

---

## Cache Architecture

### Search Cache (`SearchCache`)

Redis-backed with `aioredis`. Key format: `openinsight:{version}:{operation}:{sha256_hash}`

| Method | TTL | Used By |
|--------|-----|---------|
| `get_search_result` / `set_search_result` | 1800s (30 min) | Search endpoint, DeepInsights orchestrator |
| `get_query_embedding` / `set_query_embedding` | 1800s | Available but not currently called from search pipeline |
| `get_reranked` / `set_reranked` | 3600s (1 hour) | Available but not currently called from search pipeline |

Serialization: `_json_safe()` handles Enums, Pydantic models (`.dict()` / `.model_dump()`), and nested objects recursively.

---

## Performance Metrics

| Stage | Typical Latency |
|-------|------------------|
| Query understanding | 10–20ms |
| Cache check | 5–10ms |
| Dense search | 200–400ms |
| Sparse search | 150–300ms |
| RRF fusion | 50ms |
| Reranking (local) | 500–800ms |
| Reranking (cohere) | 100–200ms |
| MMR | 100–150ms |
| LLM generation | 1000–1500ms |
| Validation | ~500ms |
| **Total (local reranker)** | **~2–3 seconds** |
| **Total (cohere reranker)** | **~2–2.5 seconds** |

---

## Configuration Parameters

All values from `config.base.json` unless overridden by `config.{APP_ENV}.json` or env vars:

```python
# Retrieval
top_k_retrieval = 50          # Initial hybrid retrieval per branch
top_k_after_fusion = 20       # After RRF
top_k_after_rerank = 8        # After cross-encoder reranking
top_k_final = 6               # Default final answer chunks

# Diversity
mmr_lambda = 0.7              # 70% relevance, 30% diversity

# RRF
rrf_k = 60                    # Reciprocal Rank Fusion constant

# Cache
cache_version = "v2"
cache_ttl_search = 1800       # 30 minutes
cache_ttl_rerank = 3600       # 1 hour
cache_ttl_embedding = 1800    # 30 minutes

# HyDE (Hypothetical Document Embeddings)
hyde_enabled = true            # Generate synthetic doc for better retrieval
hyde_timeout = 15.0            # Timeout for HyDE generation

# Query rewrite
llm_query_rewrite = true       # Enable LLM-based query rewriting
query_rewrite_fallback = true  # Fall back to original on failure
query_rewrite_max_tokens = 64
query_rewrite_temperature = 0.0

# Embedding
embed_provider = "local"       # local | huggingface | cohere
embedding_model = "pritamdeka/S-PubMedBert-MS-MARCO"
embedding_dim = 768
embedding_batch_size = 32
embedding_retry_batch_size = 16
embedding_timeout = 60

# Reranker
rerank_provider = "local"      # local | huggingface | cohere
reranker_model_name = "BAAI/bge-reranker-v2-m3"
cohere_rerank_model = "rerank-english-v3.0"
reranker_max_length = 1024
reranker_batch_size = 16
reranker_max_chars = 1200
reranker_top_n = 8
```

---

## Internal Components

### LLM Client

The search endpoint uses `get_nim_client()` (legacy wrapper) for answer generation, which delegates to `get_llm_client()` with the default provider from settings. The `chat_completions()` method returns a **string** (the generated answer content), not an OpenAI response object.

```python
content = await client.chat_completions(
    messages=[...],
    temperature=0.1,
    max_tokens=1024,
)  # Returns: str
```

For DeepInsights, the `LLMRouter` routes requests to providers based on agent role, with health tracking and automatic failover.

### Cache Serialization

The search cache properly serializes `FilterExpression` objects when storing results. The filter is converted to a dictionary using `model_dump()` or fallback serialization before being used as a cache key. Custom types, Enums, and Pydantic models are handled recursively by `_json_safe()`.
