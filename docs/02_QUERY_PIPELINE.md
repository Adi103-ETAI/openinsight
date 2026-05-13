# Query Pipeline

## Simple Search Pipeline (`POST /search`)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              REQUEST                                        │
│                    { "query": "...", "top_k": 6 }                           │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         QUERY UNDERSTANDING                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Intent Classification (DIAGNOSTIC/THERAPEUTIC/PROGNOSTIC/DRUG_INFO)│    │
│  │  Entity Extraction (disease/drug/symptom)                           │    │
│  │  Metadata Filters (year/evidence_level/source_type)                 │    │
│  │  Query Expansion (medical synonyms)                                 │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CACHE CHECK                                    │
│                    ┌─────────────────────────────────┐                      │
│                    │  Redis: hash(query + filters)   │                      │
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
│  │   (Semantic)        │              │   (Keyword)         │               │
│  │                     │              │                     │               │
│  │ Embed query         │              │ Compute TF-IDF      │               │
│  │ PubMedBERT → 768d   │              │ sparse vector       │               │
│  │                     │              │                     │               │
│  │ Milvus search       │              │ Milvus sparse search│               │
│  │ top_k=50            │              │ top_k=50            │               │
│  └─────────┬───────────┘              └─────────┬───────────┘               │
│            │                                    │                           │
│            └──────────────┬─────────────────────┘                           │
│                           ▼                                                 │
│              ┌────────────────────────────┐                                 │
│              │  Reciprocal Rank Fusion    │                                 │
│              │  k=60, combine dense+sparse│                                 │
│              └─────────────┬──────────────┘                                 │
│                            ▼                                                │ 
│                   top_k=20 results                                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            RERANKING                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Cross-Encoder (BAAI/bge-reranker-base)                             │    │
│  │                                                                     │    │
│  │  For each (query, chunk) pair:                                      │    │
│  │    - Tokenize (max 512 tokens)                                      │    │
│  │    - GPU inference                                                  │    │
│  │    - Score [0-1]                                                    │    │
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
│  │  Result: top_k=6 diverse, relevant chunks                           │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          LLM GENERATION                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  NVIDIA NIM (Llama 3.1 70B)                                         │    │
│  │                                                                     │    │
│  │  Prompt:                                                            │    │
│  │  "You are a clinical decision support assistant..."                 │    │
│  │  "Context: [chunks]"                                                │    │
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
│                             RESPONSE                                        │
│  {                                                                          │
│    "answer": "...",                                                         │
│    "citations": [...],                                                      │
│    "confidence_score": 0.85,                                                │
│    "recommendation": "SAFE"                                                 │
│  }                                                                          │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Empty Response Behavior

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

### Logging Behavior

The query pipeline now includes structured logging for debugging and monitoring:

- **HYDE Failures**: When HYDE (Hypothetical Document Embeddings) generation fails (network timeout, API error, etc.), the failure is logged with a warning level and the system falls back to using the original query
- **Reranker Failures**: When the reranker fails (e.g., GPU out of memory), exceptions are logged before falling back to CPU-based score sorting

Example log output:
```
HYDE generation failed (Timeout(15.0s)), falling back to original query
Reranker failed (CUDA out of memory), falling back to score sorting
```

### MMR Behavior

The Maximal Marginal Relevance (MMR) diversity step now respects the user's requested `top_k` from the request. The final result count is determined by `payload.top_k` rather than a fixed internal value, ensuring users get exactly the number of results they request.

---

## Performance Metrics

| Stage | Typical Latency |
|-------|------------------|
| Query understanding | 10-20ms |
| Cache check | 5-10ms |
| Dense search | 200-400ms |
| Sparse search | 150-300ms |
| RRF fusion | 50ms |
| Reranking | 500-800ms |
| MMR | 100-150ms |
| LLM generation | 1000-1500ms |
| **Total** | **~2-3 seconds** |

---

## Configuration Parameters

```python
# Retrieval
TOP_K_RETRIEVAL = 50          # Initial retrieval
TOP_K_AFTER_FUSION = 20        # After RRF
TOP_K_AFTER_RERANK = 8         # After reranker
TOP_K_FINAL = 6                # Final answer

# Diversity
MMR_LAMBDA = 0.7              # 70% relevance, 30% diversity

# Cache
CACHE_TTL_SEARCH = 1800       # 30 minutes
CACHE_TTL_RERANK = 3600       # 1 hour

# HyDE (Hypothetical Document Embeddings)
HYDE_ENABLED = true            # Generate synthetic doc for better retrieval

---

## Internal Components

### LLM Client (NVIDIA NIM)

The LLM client uses the NVIDIA NIM API to generate answers. The client returns a string response containing the generated answer text.

**Note**: The `chat_completions` method has a return type of `str`, not `dict[str, Any]`. The client extracts the content from the response message directly.

```python
content = await client.chat_completions(
    messages=[...],
    temperature=0.1,
    max_tokens=1024,
)  # Returns: str (the generated answer content)
```

### Cache Serialization

The search cache now properly serializes `FilterExpression` objects when storing results. The filter is converted to a dictionary using `model_dump()` or fallback serialization before being used as a cache key.