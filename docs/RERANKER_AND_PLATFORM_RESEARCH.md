# Research Report: Reranker Models & Ingestion Platforms for OpenInsight Medical RAG

> Research conducted: March 2026  
> Last updated: May 2026  
> Context: OpenInsight medical RAG system processing Indian clinical guidelines, PubMed papers, WHO/CDC docs

> **Status**: The recommended `bge-reranker-v2-m3` has been adopted as the default reranker model in production.

---

## TOPIC 1: BAAI/bge-reranker-v2-gemma vs BAAI/bge-reranker-v2-m3

### 1.1 Architecture & Model Specifications

| Specification | bge-reranker-v2-gemma | bge-reranker-v2-m3 |
|---|---|---|
| **Base Model** | Google gemma-2b | BAAI/bge-m3 (XLM-RoBERTa) |
| **Architecture Type** | LLM-based cross-encoder (decoder) | Encoder-only cross-encoder (XLM-RoBERTa) |
| **Parameters** | **2.51B** | **568M** |
| **Model Size (disk)** | **~10 GB** | **~2.27 GB** |
| **Language Support** | Multilingual | Multilingual |
| **Max Sequence Length** | 1024 tokens (default code) | 8192 tokens (architecture); fine-tuned at **1024 tokens** (recommended) |
| **Python Class** | `FlagLLMReranker` | `FlagReranker` |
| **HF Pipeline** | `text-classification` / `AutoModelForCausalLM` | `text-classification` / `AutoModelForSequenceClassification` |
| **License** | Apache-2.0 | Apache-2.0 |
| **HF Likes** | 84 | 987 |

**Key Architecture Difference:** The Gemma variant uses an LLM decoder architecture that frames reranking as a "Yes/No" prompt-based task (`"Given a query A and passage B, determine whether the passage contains an answer..."`), while the M3 variant uses a standard encoder cross-encoder that directly outputs a relevance score via `SequenceClassification`.

### 1.2 Memory / VRAM Requirements

| Resource | bge-reranker-v2-gemma | bge-reranker-v2-m3 |
|---|---|---|
| **GPU VRAM (FP16 inference)** | **~5-6 GB** (model weights ~5GB in FP16) | **~1.5 GB** (confirmed by user reports on GitHub TEI issue #280) |
| **CPU RAM** | ~12-15 GB | ~11 GB (reported in GitHub TEI issue #280) |
| **Fits on T4 (16GB VRAM)?** | ✅ Yes, but tight with batch processing | ✅ Yes, easily |
| **Fits on free Colab/Kaggle?** | ✅ Yes, but less headroom for embeddings model alongside | ✅ Yes, plenty of headroom |

**Critical for your use case:** If you need to run BOTH S-PubMedBert (768d embeddings, ~1.3GB VRAM) AND the reranker simultaneously on a T4 GPU, the M3 model leaves ~13.7 GB free while the Gemma model leaves only ~10 GB free. Both work, but M3 gives significantly more breathing room.

### 1.3 Inference Speed / Latency

| Environment | bge-reranker-v2-gemma | bge-reranker-v2-m3 |
|---|---|---|
| **CPU (MacBook M1)** | 20-30 seconds per query | ~0.5 seconds per query |
| **GPU (A10)** | ~1 second per query | <0.5 seconds per query |
| **Relative Speed** | **~40-60x slower** on CPU | **Baseline (fastest)** |

Source: Official BAAI discussion (#22) — user `shaunxu` reported 20-30s for Gemma vs 0.5s for M3 on MacBook M1 CPU. BAAI developer `Shitao` confirmed M3 is smaller and faster.

**The Gemma model's LLM-decoder architecture makes it dramatically slower**, especially on CPU. This is because it needs to autoregressively generate "Yes"/"No" tokens, while the M3 model directly computes a scalar score via sequence classification.

### 1.4 Benchmark Performance

#### AIMultiple Reranker Benchmark (English, ~145k Amazon Reviews)
Benchmarked 8 rerankers using multilingual-e5-base retrieval + reranking top-100 → top-10:

| Model | ΔHit@1 | Hit@1 | nDCG@10 |
|---|---|---|---|
| nemotron_rerank_1b | +20.3pp | 83.00% | Highest |
| gte_modernbert_base | +20.3pp | ~83% | Highest |
| jina_reranker_v3 | +18.7pp | ~81% | Very high |
| qwen3_reranker_4b | +15.0pp | ~78% | High |
| **bge-reranker-v2-m3** | **+14.7pp** | **~77%** | **Good** |
| bge-reranker-base | +11.7pp | ~74% | Moderate |
| baseline (no reranker) | 0.0pp | 62.67% | — |

Note: bge-reranker-v2-gemma was not included in this specific benchmark. The M3 variant scored competitively but was outperformed by newer/larger models.

#### MAIR Benchmark (EMNLP 2024)
- **bge-reranker-v2-gemma achieved the best results among all rerankers**, outperforming all embedding models when no instructions were provided (source: ACL Anthology 2024.emnlp-main.778)
- bge-reranker-v2-gemma outperformed bge-reranker-v2-m3 in general capabilities

#### BAAI Official Recommendation
From the official BGE documentation (bge-model.com):
- **For better performance:** Use `bge-reranker-v2-gemma` (or `minicpm-layerwise`)
- **For efficiency:** Use `bge-reranker-v2-m3`
- **For English or Chinese:** Both M3 and minicpm-layerwise are recommended

### 1.5 Suitability for English-Only Medical Text

| Factor | bge-reranker-v2-gemma | bge-reranker-v2-m3 |
|---|---|---|
| **English performance** | Slightly better (per BAAI & MAIR) | Strong, but marginally lower |
| **Medical domain specificity** | No specific medical training | No specific medical training |
| **Token limit for medical docs** | 1024 tokens (~768 words) | 1024 recommended (8192 possible but fine-tuned at 1024) |
| **Chunk compatibility** | Works well with 512-token chunks | Works well with 512-token chunks |

**For English-only medical text, the Gemma variant has a slight quality edge, but the M3 variant is very close.** Neither model was specifically trained on medical text, so the difference is marginal for your domain.

### 1.6 HuggingFace Inference API (Free Tier) Compatibility

| Aspect | bge-reranker-v2-gemma | bge-reranker-v2-m3 |
|---|---|---|
| **Available on HF Inference API?** | ⚠️ Partially (text-classification endpoint) | ✅ Yes (text-classification endpoint) |
| **Free tier model size limit** | 10 GB (this model is ~10 GB — **right at the limit!**) | 2.27 GB (well within limit) |
| **Rate limit (registered user)** | 300 requests/hour | 300 requests/hour |
| **Rate limit (unregistered)** | 1 request/hour | 1 request/hour |
| **TEI (Text Embeddings Inference) support** | ❌ NOT supported | ❌ NOT officially listed* |
| **Dedicated /rerank endpoint** | ❌ No | ❌ No |

*Note: The M3 model is tagged with "text-embeddings-inference" on its HF card but is NOT in the official TEI supported models list (which only includes `BAAI/bge-reranker-large` and `BAAI/bge-reranker-base` for re-ranking). This means deployment via TEI may require custom configuration or patches.

**Critical Issue:** The HF Inference API free tier exposes these as `text-classification` models, NOT as dedicated reranking endpoints. You cannot use the standard `/rerank` API endpoint with these models on the free tier. You'd need to:
1. Use the `text-classification` endpoint and parse scores manually, OR
2. Use Cohere Rerank API (which has a proper `/rerank` endpoint), OR
3. Self-host via TEI/Docker (not on free tier)

**The bge-reranker-v2-gemma at ~10 GB is RIGHT AT the free tier size limit** and may be unreliable or unavailable on the free HF Inference API.

### 1.7 Cohere Rerank Free Tier (Alternative)

| Aspect | Details |
|---|---|
| **Trial API key** | Free, **1,000 API calls/month** across ALL models |
| **Production pricing** | $2.00 per 1,000 searches (Rerank 3.5) |
| **Dedicated /rerank endpoint** | ✅ Yes, proper reranking API |
| **Quality** | Competitive with BGE models; often easier to integrate |
| **Best for** | Query-time reranking with proper API support |

**1,000 free calls/month on Cohere is very limited** for production use but sufficient for testing/development.

### 1.8 Known Issues & Limitations

**bge-reranker-v2-gemma:**
- Very slow on CPU (20-30s per query) — unsuitable for CPU-only inference
- Uses LLM decoder architecture, so it's memory-hungry
- Fine-tuning can cause CUDA OOM errors even on 4x A10G (24GB each) — reported in HF discussions
- At 10 GB model size, sits right at the HF free tier limit
- No proper `/rerank` endpoint on HF Inference API

**bge-reranker-v2-m3:**
- Fine-tuned at max_length=1024, so using 8192 tokens degrades quality (confirmed in HF discussion #9: "we recommend to set max_length as 1024")
- Not officially supported by TEI (Text Embeddings Inference) for production deployment
- Marginally lower accuracy than the Gemma variant on some benchmarks
- Some users report worsened RAG results compared to no reranker in certain domains (Reddit reports)
- No proper `/rerank` endpoint on HF Inference API

---

## TOPIC 2: Kaggle vs Google Colab for Data Ingestion Pipeline

### 2.1 Hardware Specifications

| Specification | Kaggle (Free) | Google Colab (Free) |
|---|---|---|
| **GPU Types** | T4 (16GB), P100 (16GB), T4 x2 (32GB combined) | T4 (15-16GB VRAM) |
| **GPU VRAM** | 16 GB (T4/P100) | ~15 GB (T4) |
| **GPU Availability** | ✅ Guaranteed (when quota available) | ⚠️ Not guaranteed; dynamic allocation |
| **CPU Cores** | 4 cores | 2 vCPU cores |
| **System RAM (GPU notebook)** | **13 GB** | **~12.7 GB** (sometimes upgradable to ~25 GB) |
| **System RAM (CPU-only)** | 16-30 GB | ~12.7 GB |
| **TPU** | Available (9hr limit) | Available |

### 2.2 Time Quotas

| Quota | Kaggle (Free) | Google Colab (Free) |
|---|---|---|
| **Session timeout** | 12 hours (CPU/GPU), 9 hours (TPU) | ~12 hours max (varies; often less) |
| **Idle timeout** | No strict idle timeout | ~90 minutes of inactivity = disconnected |
| **Weekly GPU quota** | **30 hours/week** (sometimes up to 40 hours) | **Unpublished**; dynamic; heavy users get cooldowns |
| **Concurrent GPU sessions** | 1 interactive + 2 commit sessions | 1 session typically |
| **Quota reset** | Weekly (Saturday midnight UTC) | No fixed reset; variable cooldown |

### 2.3 Storage

| Storage | Kaggle (Free) | Google Colab (Free) |
|---|---|---|
| **Working disk space (temporary)** | **20 GB** scratch (/kaggle/temp) | ~**100 GB** temporary (ephemeral VM) |
| **Persistent storage** | **5 GB** auto-saved (/kaggle/working) | Google Drive integration (15 GB free) |
| **Private dataset upload** | **100 GB** total private datasets | Direct upload to runtime; Drive for persistence |
| **Maximum single dataset** | Up to 100 GB | No explicit limit (Drive-bound) |

### 2.4 Network Access & External Services

| Capability | Kaggle (Free) | Google Colab (Free) |
|---|---|---|
| **Internet access** | ✅ Available (must enable in Settings → Internet) | ✅ Always available |
| **Phone verification required** | ✅ Yes (for internet access) | ❌ No |
| **Connect to MongoDB Atlas** | ✅ Yes (confirmed by tutorials on Kaggle) | ✅ Yes (confirmed by multiple tutorials) |
| **Connect to Zilliz Cloud** | ✅ Yes (via pymilvus SDK) | ✅ Yes (via pymilvus SDK) |
| **Connect to external APIs** | ✅ Yes | ✅ Yes |
| **SSH/tunneling** | ❌ Limited | ❌ Limited |

**Both platforms can connect to Zilliz Cloud and MongoDB Atlas** since they both support pip installing `pymilvus` and `pymongo`, and both have internet access (Kaggle requires enabling it and phone verification).

### 2.5 Package Installation & Environment

| Capability | Kaggle (Free) | Google Colab (Free) |
|---|---|---|
| **pip install** | ✅ Yes | ✅ Yes |
| **apt-get install** | ✅ Yes | ✅ Yes |
| **Python version** | 3.10-3.11 (managed) | 3.10+ (managed) |
| **Docker support** | ❌ No | ❌ No |
| **GROBID server** | ⚠️ Possible (java install) | ⚠️ Possible (java install) |
| **Pre-installed ML libs** | PyTorch, TensorFlow, scikit-learn, etc. | PyTorch, TensorFlow, etc. |
| **Package persistence** | ❌ Must reinstall each session | ❌ Must reinstall each session |

### 2.6 Persistence Between Sessions

| Aspect | Kaggle (Free) | Google Colab (Free) |
|---|---|---|
| **Auto-saved outputs** | /kaggle/working (5 GB, persists) | ❌ No auto-save |
| **Manual save** | Commit notebook → outputs saved | Save to Google Drive |
| **Dataset persistence** | ✅ Datasets persist between sessions | Via Google Drive |
| **Model weights persistence** | Save to /kaggle/working or as dataset | Save to Google Drive |
| **Notebook versioning** | ✅ Built-in version control | ✅ Google Drive version history |

### 2.7 PDF Upload & Processing

| Aspect | Kaggle (Free) | Google Colab (Free) |
|---|---|---|
| **Upload PDFs as dataset** | ✅ Yes (up to 100 GB private datasets) | ✅ Yes (upload to runtime or Drive) |
| **Bulk PDF upload** | ✅ Via Kaggle CLI or web UI | ✅ Via Google Drive mount |
| **GROBID server feasibility** | ⚠️ Possible but uses significant RAM | ⚠️ Possible but uses significant RAM |
| **OCR support** | ✅ pip install pytesseract + tesseract | ✅ pip install pytesseract + tesseract |

### 2.8 Decision Matrix for Your Ingestion Pipeline

Given your pipeline (PDF parsing → S-PubMedBert embeddings → TF-IDF sparse → Zilliz Cloud + MongoDB Atlas):

| Criterion | Winner | Reasoning |
|---|---|---|
| **GPU compute for embeddings** | **Kaggle** | Guaranteed GPU; T4 x2 option (32GB VRAM); 30hr/week is predictable |
| **Long-running batch jobs** | **Kaggle** | 12hr session + 30hr/week is explicit; Colab's limits are opaque |
| **Disk space for PDFs** | **Colab** | ~100 GB temporary vs Kaggle's 20 GB scratch + 5 GB saved |
| **RAM for PDF parsing** | **Kaggle** | 13-30 GB vs Colab's ~12.7 GB |
| **External service connectivity** | **Tie** | Both support MongoDB Atlas + Zilliz Cloud |
| **Persistence between sessions** | **Kaggle** | Auto-saved outputs + dataset versioning |
| **Predictability** | **Kaggle** | Clear, published quotas; Colab's are dynamic/undocumented |
| **Ease of use** | **Colab** | No phone verification; Google Drive integration |
| **Upload large PDF collections** | **Kaggle** | 100 GB private datasets via CLI |
| **GROBID server** | **Colab** | More RAM headroom (25 GB possible) + more disk |

---

## RECOMMENDATIONS

### Reranker Selection

**For the OpenInsight medical RAG system, I recommend bge-reranker-v2-m3 as the primary choice:**

1. **Ingestion (Kaggle/Colab with GPU):** M3 uses only ~1.5 GB VRAM, leaving ample room for S-PubMedBert alongside it. Gemma's ~5-6 GB VRAM consumption makes co-loading tight on a T4.

2. **Query-time (HF Inference API free tier):** M3 at 2.27 GB is well within the 10 GB free tier limit. Gemma at 10 GB is right at the boundary and may be unreliable.

3. **Speed:** M3 is 40-60x faster on CPU and significantly faster on GPU. For query-time reranking, this translates to sub-second vs multi-second latency.

4. **Quality gap is marginal for English medical text:** The Gemma variant has a slight edge in general benchmarks, but the difference is small for English-only content. The MAIR paper shows Gemma as the best overall reranker, but M3 is still very competitive (ΔHit@1 of +14.7pp is strong).

5. **Fallback strategy:** Use Cohere Rerank (1,000 free calls/month) for production query-time reranking with a proper `/rerank` API endpoint, and use M3 for ingestion-time reranking where you control the infrastructure.

### Platform Selection

**For your ingestion pipeline, I recommend Kaggle as the primary platform:**

1. **Predictable quotas** (30 GPU hrs/week, 12-hr sessions) let you plan batch processing
2. **T4 x2 option** provides 32 GB combined VRAM — enough for S-PubMedBert + M3 reranker simultaneously
3. **100 GB private datasets** for uploading your medical PDF collection
4. **5 GB auto-saved** outputs persist between sessions (processed embeddings, metadata)
5. **Explicit internet access** confirmed for MongoDB Atlas + Zilliz Cloud connectivity

**Use Colab as a supplement** when:
- You need more temporary disk space (~100 GB) for large batch operations
- You hit Kaggle's weekly GPU quota and need a few extra hours
- You need the 25 GB RAM option for memory-intensive PDF parsing (GROBID)

### Suggested Architecture

```
INGESTION (Kaggle T4/T4x2):
  PDFs (Kaggle Dataset) → GROBID/OCR → Chunks
  → S-PubMedBert (768d dense, GPU) → Zilliz Cloud
  → TF-IDF (sparse, CPU) → Zilliz Cloud
  → bge-reranker-v2-m3 (optional, for quality scoring during ingestion)
  → Metadata → MongoDB Atlas
  → Checkpoint/progress → /kaggle/working (5 GB persistent)

QUERY-TIME (Production API):
  User Query → Embed query (S-PubMedBert or API)
  → Hybrid search Zilliz Cloud (dense + sparse)
  → Rerank top-k candidates via:
     Option A: Cohere Rerank API (proper /rerank endpoint, 1k free/mo)
     Option B: HF Inference API + bge-reranker-v2-m3 (text-classification endpoint)
     Option C: Self-hosted TEI with M3 (if you have a server)
  → LLM generation
```
