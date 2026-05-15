# Migration: Offload embeddings & rerank to remote providers (Cohere / HF)

Goal: Allow ingestion to run on GPU in Kaggle/Colab and use cloud vector DB (Zilliz/Milvus + MongoDB). Make query-serving GPU-free by using remote embedding/rerank APIs while keeping NVIDIA NIM for LLM generation.

---

## Status: IMPLEMENTED ✅

The pluggable provider architecture has been implemented. The following changes are now in the `restruct` branch:

### Files Modified
- `src/config/settings.py` — Added `embed_provider`, `rerank_provider`, `hf_api_token`, `cohere_api_key`, and model-specific settings
- `src/ml/embedding/embedder.py` — Refactored to `BaseEmbedder` abstract class + `LocalEmbedder`, `HuggingFaceEmbedder`, `CohereEmbedder` + factory
- `src/query/search/reranker.py` — Refactored to `BaseReranker` abstract class + `LocalReranker`, `HuggingFaceReranker`, `CohereReranker` + factory; upgraded default model to `BAAI/bge-reranker-v2-m3`
- `src/query/search/retriever.py` — Now uses `get_embedder()` factory instead of hardcoded `DualEmbedderV2`
- `src/api/routes/search.py` — Uses `get_reranker()` factory instead of hardcoded `CrossEncoderReranker`
- `.env.example` — Added all new provider configuration variables

### Files Added
- `notebooks/kaggle_ingestion.py` — Self-contained Kaggle notebook for data ingestion

---

## Architecture

### Ingestion (Kaggle/Colab with GPU)

```mermaid
flowchart LR
  subgraph Ingestion
    Kaggle[Kaggle T4/T4x2 GPU] -->|S-PubMedBert 768d| Zilliz[Zilliz Cloud]
    Kaggle -->|TF-IDF sparse CPU| Zilliz
    Kaggle -->|metadata| MongoDB[MongoDB Atlas]
    Kaggle -->|checkpoints| KagglePersist[/kaggle/working 5GB]
  end
```

### Query Pipeline (CPU server + Free APIs)

```mermaid
flowchart LR
  subgraph Query
    Client --> API[API Server]
    API --> Retriever[HybridRetriever]
    Retriever -->|embed query| EmbedProvider[Embed Provider]
    Retriever -->|dense+sparse search| Zilliz[Zilliz Cloud]
    API -->|rerank| RerankProvider[Rerank Provider]
    API -->|LLM gen| NIM[NVIDIA NIM]
    Zilliz --> MongoDB[MongoDB Atlas]
  end
```

---

## Provider Configuration

### Embedding Providers

| Provider | `EMBED_PROVIDER` | Model | Dimension | Free Tier | Notes |
|----------|-----------------|-------|-----------|-----------|-------|
| Local | `local` | S-PubMedBert-MS-MARCO | 768 | N/A (GPU needed) | Best quality, requires GPU |
| HuggingFace | `huggingface` | S-PubMedBert-MS-MARCO | 768 | 300 req/hr | Same model as ingestion, **recommended for consistency** |
| Cohere | `cohere` | embed-english-v3.0 | 1024 | 1,000 calls/mo | Different vector space, DO NOT mix with S-PubMedBert |

**⚠️ Critical**: If you used S-PubMedBert during ingestion (Kaggle), you MUST use the same model at query time. Use `EMBED_PROVIDER=huggingface` with `HF_EMBED_MODEL=pritamdeka/S-PubMedBert-MS-MARCO` for query-time to ensure embedding consistency. Using Cohere at query time with S-PubMedBert-indexed vectors will produce garbage results.

### Reranking Providers

| Provider | `RERANK_PROVIDER` | Model | Free Tier | Notes |
|----------|-------------------|-------|-----------|-------|
| Local | `local` | bge-reranker-v2-m3 | N/A (GPU needed) | Upgraded from bge-reranker-base |
| HuggingFace | `huggingface` | bge-reranker-v2-m3 | 300 req/hr | Uses text-classification endpoint |
| Cohere | `cohere` | rerank-english-v3.0 | 1,000 calls/mo | Proper /rerank API, **recommended** |

---

## Reranker Upgrade: bge-reranker-base → bge-reranker-v2-m3

| Factor | bge-reranker-base (old) | bge-reranker-v2-m3 (new) |
|--------|------------------------|--------------------------|
| Parameters | 278M | 568M |
| Model size | ~1.1 GB | ~2.27 GB |
| VRAM (FP16) | ~0.6 GB | ~1.5 GB |
| Benchmark ΔHit@1 | +11.7pp | +14.7pp |
| HF Free API fit | ✅ | ✅ (well within 10GB limit) |
| CPU speed | Baseline | 40-60x faster than Gemma variant |

---

## Quick Start

### For Query Pipeline (CPU server with free APIs)

```bash
# .env configuration for free-tier query pipeline
EMBED_PROVIDER=huggingface
HF_API_TOKEN=hf_your_token_here
HF_EMBED_MODEL=pritamdeka/S-PubMedBert-MS-MARCO

RERANK_PROVIDER=cohere
COHERE_API_KEY=your_cohere_key_here
COHERE_RERANK_MODEL=rerank-english-v3.0

# Zilliz Cloud (from ingestion)
VECTOR_URI=https://your-cluster.api.gcp-us-west1.zillizcloud.com
VECTOR_TOKEN=your_zilliz_api_key
MILVUS_CLOUD=true

# MongoDB Atlas (from ingestion)
MONGODB_URL=mongodb+srv://user:pass@cluster.mongodb.net/openinsight

# LLM (already cloud-based)
NVIDIA_NIM_API_KEY=your_nim_key
```

### For Ingestion (Kaggle notebook)

Use the notebook at `notebooks/kaggle_ingestion.py`. Key settings:
```bash
EMBED_PROVIDER=local  # Use Kaggle GPU
RERANK_PROVIDER=local  # Use Kaggle GPU
DENSE_MODEL_NAME=pritamdeka/S-PubMedBert-MS-MARCO
RERANKER_MODEL_NAME=BAAI/bge-reranker-v2-m3
```

---

## Testing & Validation Plan

1. **Unit tests** — Add tests for `HuggingFaceEmbedder`, `CohereEmbedder`, `HuggingFaceReranker`, `CohereReranker` with mocked HTTP responses.

2. **Integration tests** — Ingest a small sample in Kaggle, then query from CPU server with remote providers.

3. **Quality metrics** — Compare retrieval quality between local and remote providers using labeled queries.

4. **Consistency checks** — Ensure embedding model used during ingestion matches query-time model.

---

## Risks & Tradeoffs

- **Latency**: Remote API calls add network latency (50-200ms per call)
- **Rate limits**: HF free tier = 300 req/hr; Cohere = 1,000 calls/mo
- **Embedding consistency**: Using different models at ingestion vs query time degrades retrieval quality
- **Availability**: Free tier APIs may have downtime or throttling

---

## Backward Compatibility

- `DualEmbedderV2` is now an alias for `LocalEmbedder`
- `CrossEncoderReranker` is now an alias for `LocalReranker`
- Default behavior (`EMBED_PROVIDER=local`, `RERANK_PROVIDER=local`) is unchanged
- All existing code using `DualEmbedderV2` or `CrossEncoderReranker` continues to work
