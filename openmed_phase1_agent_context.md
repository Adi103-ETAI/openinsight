# OpenMed — Agent Build Context
## Phase 1: Retrieval Quality (Steps 1.2, 1.3, 1.4)

You are continuing to build **OpenMed** — an AI clinical decision support platform for Indian physicians by SentArc Labs, Pune. Phase 1.1 (system prompt refactor) is already complete.

This document covers the remaining three steps of Phase 1. Build them **in order**. Each step has its own verification. Do not move to the next step until the current one verifies cleanly.

---

## What Is Already Built (Do Not Touch)

```
openmed/
├── prompts/
│   └── system.md                  ← EXISTS — system prompt in Markdown
├── src/
│   ├── core/
│   │   └── config.py              ← EXISTS — has nim_temperature, nim_max_tokens, retrieval_top_k, reranker_top_n
│   ├── ingestion/
│   │   ├── document_db.py         ← EXISTS — MongoDB client, DocumentRecord, ChunkRecord
│   │   ├── vector_db.py           ← EXISTS — Qdrant client, search(), upsert_chunks()
│   │   └── embeddings.py          ← EXISTS — embed_query(), embed_texts()
│   └── query/
│       ├── prompts.py             ← EXISTS — loads system.md, has build_prompt()
│       └── standard.py            ← EXISTS — standard_search(), calls NIM API
├── src/api/
│   ├── main.py                    ← EXISTS — FastAPI app
│   └── routes/
│       └── query.py               ← EXISTS — POST /query endpoint
```

### Existing `src/core/config.py` Settings (relevant fields)
```python
nvidia_nim_api_key: str
nvidia_nim_base_url: str = "https://integrate.api.nvidia.com/v1"
nim_model: str = "meta/llama-3.1-70b-instruct"
qdrant_url: str = "http://localhost:6333"
qdrant_collection: str = "openmed_chunks"
embedding_model: str = "pritamdeka/S-PubMedBert-MS-MARCO"
embedding_dim: int = 768
nim_temperature: float = 0.1
nim_max_tokens: int = 1024
retrieval_top_k: int = 8
reranker_top_n: int = 8
```

### Existing `src/query/standard.py` (current flow)
```python
async def standard_search(query: str, top_k: int = 8) -> dict:
    # 1. embed_query(query) → vector
    # 2. search(vector, top_k) → Qdrant results (top_k chunks)
    # 3. build_prompt(query, chunks) → prompt string
    # 4. Call NIM API → answer
    # 5. Return { answer, citations, query, model, chunks_retrieved }
```

### Existing `src/ingestion/vector_db.py` — `search()` signature
```python
def search(
    query_vector: list[float],
    top_k: int = 8,
    source_type: Optional[str] = None
) -> list:
    # Returns list of Qdrant ScoredPoint
    # result.score — float cosine similarity
    # result.payload — dict with: mongo_id, source_type, title, condition_tags, chunk_text
```

---

## Installed Packages (requirements.txt — do not add new ones without checking first)

```
fastapi, uvicorn, pydantic, pydantic-settings
pymongo, motor, pdfplumber, pypdf2, beautifulsoup4
requests, httpx, lxml, biopython
sentence-transformers, transformers, torch, numpy
qdrant-client
langchain, langchain-community, openai
redis, tqdm, loguru, tenacity, python-slugify
pytest, pytest-asyncio, black, isort
```

---

## Step 1.2 — Query Rewriting Layer

### What It Does
Before the doctor's raw query hits the vector index, rewrite it into a clean, expanded medical query. This improves retrieval recall significantly.

**Examples:**
- "best treatment for sugar" → "diabetes mellitus type 2 treatment guidelines India"
- "TB drugs" → "tuberculosis first-line drug treatment DOTS regimen India ICMR"
- "fever in child" → "pediatric fever management diagnosis treatment India"
- "hospital infection" → "hospital acquired infection prevention control protocols HAI India ICMR"

### Files To Create
- `prompts/query_rewrite.md` — prompt for the rewriting LLM call
- `src/query/rewriter.py` — async function that rewrites a query

### Files To Modify
- `src/query/standard.py` — add rewrite step before embed_query()

---

### File 1: `prompts/query_rewrite.md`

Create this file with exactly this content:

```markdown
You are a medical query normalisation assistant for an Indian clinical decision support system.

Your job is to rewrite a doctor's raw query into a clean, expanded medical query that will retrieve better results from a knowledge base containing ICMR guidelines, PubMed research, and Indian clinical literature.

Rules:
1. Expand abbreviations to full medical terms (TB → tuberculosis, DM → diabetes mellitus, HTN → hypertension, MI → myocardial infarction)
2. Add the primary disease name if implied but not stated ("sugar problem" → "diabetes mellitus")
3. Add "India" or "Indian" if the query is about treatment/management and doesn't already specify a country context
4. Add "ICMR guidelines" if the query is about treatment protocols or management
5. Keep the core clinical intent — do not change what the doctor is asking
6. Output ONLY the rewritten query. No explanation, no preamble, no punctuation at the end.
7. If the query is already specific and well-formed, return it unchanged.
8. Maximum output: 20 words.

Examples:
Input: best treatment for sugar
Output: diabetes mellitus type 2 treatment guidelines India ICMR

Input: TB drugs dosage
Output: tuberculosis first-line drug regimen dosage duration India ICMR

Input: dengue warning signs
Output: dengue hemorrhagic fever warning signs management India ICMR

Input: what are hospital infection guidelines
Output: hospital acquired infection prevention control protocols India ICMR

Input: first line treatment for drug resistant TB in adults
Output: drug resistant tuberculosis MDR-TB treatment regimen adults India ICMR
```

---

### File 2: `src/query/rewriter.py`

Create this file:

```python
"""
Query Rewriter
Rewrites raw doctor queries into expanded medical queries before vector search.
Uses a fast NIM call with a dedicated rewrite prompt.
"""
from loguru import logger
from openai import AsyncOpenAI
from pathlib import Path

from src.core.config import get_settings

settings = get_settings()


def _load_rewrite_prompt() -> str:
    prompts_dir = Path(__file__).resolve().parents[2] / "prompts"
    return (prompts_dir / "query_rewrite.md").read_text(encoding="utf-8")


REWRITE_PROMPT = _load_rewrite_prompt()


async def rewrite_query(query: str) -> str:
    """
    Rewrite a raw doctor query into an expanded medical query.
    Returns the original query unchanged if rewriting fails.
    """
    if len(query.strip()) < 4:
        return query

    try:
        client = AsyncOpenAI(
            api_key=settings.nvidia_nim_api_key,
            base_url=settings.nvidia_nim_base_url,
        )
        response = await client.chat.completions.create(
            model=settings.nim_model,
            messages=[
                {"role": "system", "content": REWRITE_PROMPT},
                {"role": "user", "content": query.strip()},
            ],
            temperature=0.0,
            max_tokens=64,
        )
        rewritten = response.choices[0].message.content.strip()
        if rewritten and len(rewritten) > 3:
            logger.info(f"Query rewritten: '{query}' → '{rewritten}'")
            return rewritten
        return query
    except Exception as exc:
        logger.warning(f"Query rewrite failed, using original: {exc}")
        return query
```

---

### Modify `src/query/standard.py`

Add the rewrite step. Find the line:
```python
query_vector = embed_query(query)
```

Replace it with:
```python
from src.query.rewriter import rewrite_query

rewritten_query = await rewrite_query(query)
query_vector = embed_query(rewritten_query)
logger.info(f"Using rewritten query for embedding: {rewritten_query}")
```

Also update the return dict to include the rewritten query for debugging:
```python
return {
    "answer": answer,
    "citations": citations,
    "query": query,
    "rewritten_query": rewritten_query,   # ADD THIS LINE
    "model": settings.nim_model,
    "chunks_retrieved": len(chunks),
}
```

And add `rewritten_query: str = ""` to `QueryResponse` in `src/api/routes/query.py`.

---

### Verify Step 1.2

```bash
python -c "
import asyncio
from src.query.rewriter import rewrite_query
async def test():
    tests = [
        'best treatment for sugar',
        'TB drugs',
        'fever in child',
        'hospital infection guidelines',
    ]
    for q in tests:
        result = await rewrite_query(q)
        print(f'  IN:  {q}')
        print(f'  OUT: {result}')
        print()
asyncio.run(test())
"
```

Expected — each query should come back expanded with medical terminology. Paste the output before moving to Step 1.3.

---

## Step 1.3 — Cross-Encoder Reranker

### What It Does
After vector search retrieves top-K candidates, a cross-encoder model scores each (query, chunk) pair individually and reranks them by true relevance. This is the highest-impact quality improvement in the entire Phase 1 plan.

**Why it matters:**
Vector search finds semantically similar chunks — but similarity is not the same as relevance. A chunk about "hospital infection history in India" is semantically close to "hospital infection treatment protocols" but clinically useless. The cross-encoder catches this.

**Flow change:**
```
Before: embed → search(top_8) → prompt → LLM
After:  embed → search(top_50) → rerank(top_50 → top_8) → prompt → LLM
```

### Install Required Package

```bash
pip install sentence-transformers --upgrade
```

`sentence-transformers` is already in requirements.txt — the CrossEncoder class is included in it.

Also add to requirements.txt:
```
# (sentence-transformers already listed — CrossEncoder is included)
```

No new package needed.

---

### File To Create: `src/query/reranker.py`

```python
"""
Cross-Encoder Reranker
Reranks retrieved chunks by true relevance to the query.
Uses BAAI/bge-reranker-base — free, runs on CPU, good medical text performance.
"""
from functools import lru_cache
from loguru import logger
from sentence_transformers import CrossEncoder

from src.core.config import get_settings

settings = get_settings()

RERANKER_MODEL = "BAAI/bge-reranker-base"


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    logger.info(f"Loading reranker model: {RERANKER_MODEL}")
    model = CrossEncoder(RERANKER_MODEL, max_length=512)
    logger.info("Reranker model loaded.")
    return model


def rerank_chunks(query: str, chunks: list[dict], top_n: int = 8) -> list[dict]:
    """
    Rerank chunks by relevance to query using cross-encoder.
    
    Args:
        query: the (rewritten) doctor query
        chunks: list of chunk dicts with 'chunk_text' key
        top_n: how many top chunks to return after reranking
    
    Returns:
        top_n chunks sorted by reranker score descending
    """
    if not chunks:
        return chunks

    if len(chunks) <= top_n:
        return chunks

    try:
        model = get_reranker()
        pairs = [(query, chunk["chunk_text"]) for chunk in chunks]
        scores = model.predict(pairs, show_progress_bar=False)

        scored_chunks = [
            {**chunk, "reranker_score": float(score)}
            for chunk, score in zip(chunks, scores)
        ]

        scored_chunks.sort(key=lambda x: x["reranker_score"], reverse=True)
        top_chunks = scored_chunks[:top_n]

        logger.info(
            f"Reranked {len(chunks)} chunks → kept top {len(top_chunks)}. "
            f"Top score: {top_chunks[0]['reranker_score']:.3f}, "
            f"Bottom score: {top_chunks[-1]['reranker_score']:.3f}"
        )
        return top_chunks

    except Exception as exc:
        logger.error(f"Reranking failed, returning original chunks: {exc}")
        return chunks[:top_n]
```

---

### Modify `src/query/standard.py`

**Change 1** — increase initial retrieval from top_k to top_50 for reranking:

Find:
```python
results = search(query_vector, top_k=top_k)
```

Replace with:
```python
# Retrieve more candidates for reranking — reranker will reduce to top_k
retrieval_k = max(top_k * 6, 50)
results = search(query_vector, top_k=retrieval_k)
```

**Change 2** — add reranking step after building chunks list:

Find the block that builds the `chunks` list (the for loop over results). After that block ends and before `if not chunks:`, add:

```python
from src.query.reranker import rerank_chunks

# Rerank retrieved chunks by true relevance to query
chunks = rerank_chunks(rewritten_query, chunks, top_n=settings.reranker_top_n)
```

---

### Verify Step 1.3

```bash
# 1. Test reranker loads
python -c "
from src.query.reranker import get_reranker
model = get_reranker()
print('Reranker loaded OK')
"

# 2. Test reranking logic
python -c "
from src.query.reranker import rerank_chunks
chunks = [
    {'chunk_text': 'The foreword of this guideline acknowledges the task force members.', 'title': 'Test', 'source_type': 'icmr', 'score': 0.91},
    {'chunk_text': 'Doxycycline 200mg/day in two divided doses for 7 days is recommended for treatment of scrub typhus.', 'title': 'Test', 'source_type': 'icmr', 'score': 0.89},
    {'chunk_text': 'The historical context of rickettsial disease dates back to the Peloponnesian War.', 'title': 'Test', 'source_type': 'icmr', 'score': 0.88},
]
result = rerank_chunks('treatment for scrub typhus dosage', chunks, top_n=2)
print('Top chunk after reranking:')
print(result[0]['chunk_text'][:100])
print()
print('Expected: the doxycycline dosage chunk should be first, not the foreword')
"
```

The doxycycline chunk must come first after reranking. If the foreword chunk is still first, something is wrong. Paste the output.

---

## Step 1.4 — Qdrant Hybrid Search (Dense + Sparse)

### What It Does
Adds sparse vector (keyword/BM25-style) search alongside existing dense vector search. Combined hybrid search catches exact medical terms, drug names, and abbreviations that pure semantic search misses.

**Why this matters for medical text:**
- "rifampicin" and "rifampin" are the same drug — semantic search catches this
- "DOTS" (Directly Observed Treatment) — semantic search might miss this abbreviation
- "INH 300mg" — exact dosage strings need keyword matching
- ICD codes — purely symbolic, need keyword search

### How Qdrant Hybrid Search Works

Qdrant supports sparse vectors natively. You store both a dense vector (768-dim float) and a sparse vector (term frequencies as a sparse dict) per chunk. At query time you search both indexes and fuse scores.

### Install Required Package

```bash
pip install qdrant-client --upgrade
```

Already in requirements.txt — just ensure it's up to date.

---

### Modify `src/ingestion/vector_db.py`

**Change 1** — update `ensure_collection()` to create collection with both dense and sparse vectors:

Replace the entire `ensure_collection()` function with:

```python
def ensure_collection():
    """Create the Qdrant collection with dense + sparse vectors if it doesn't exist."""
    from qdrant_client.models import (
        VectorParams, SparseVectorParams, Distance,
        SparseIndexParams
    )
    client = get_qdrant()
    existing = [c.name for c in client.get_collections().collections]
    if settings.qdrant_collection not in existing:
        client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config={
                "dense": VectorParams(
                    size=settings.embedding_dim,
                    distance=Distance.COSINE,
                )
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(
                    index=SparseIndexParams(on_disk=False)
                )
            }
        )
        logger.info(f"Created Qdrant collection with hybrid search: {settings.qdrant_collection}")
    else:
        logger.info(f"Collection already exists: {settings.qdrant_collection}")
```

**Change 2** — update `upsert_chunks()` to accept the new named vector format:

The existing upsert function passes vectors directly. Update it to use named vectors:

```python
def upsert_chunks(points: list[PointStruct]):
    """Batch upsert chunks into Qdrant."""
    client = get_qdrant()
    client.upsert(
        collection_name=settings.qdrant_collection,
        points=points,
    )
```

This stays the same — PointStruct handles both formats.

**Change 3** — add a new `hybrid_search()` function alongside the existing `search()`:

```python
def build_sparse_vector(text: str) -> dict:
    """
    Build a simple sparse vector from text using term frequencies.
    Maps term hash → frequency. Used for BM25-style keyword search.
    """
    import re
    from collections import Counter
    tokens = re.findall(r'\b[a-zA-Z0-9]{2,}\b', text.lower())
    tf = Counter(tokens)
    # Use positive integer indices (hash mod large prime)
    sparse = {abs(hash(term)) % 2_000_000: float(freq) for term, freq in tf.items()}
    return sparse


def hybrid_search(
    query_text: str,
    query_vector: list[float],
    top_k: int = 50,
    source_type: Optional[str] = None,
) -> list:
    """
    Hybrid search — combine dense semantic search with sparse keyword search.
    Falls back to dense-only search if hybrid fails.
    """
    from qdrant_client.models import (
        SparseVector, NamedVector, NamedSparseVector,
        SearchRequest, Filter, FieldCondition, MatchValue
    )
    client = get_qdrant()

    query_filter = None
    if source_type:
        query_filter = Filter(
            must=[FieldCondition(key="source_type", match=MatchValue(value=source_type))]
        )

    try:
        sparse_vec = build_sparse_vector(query_text)

        results = client.query_points(
            collection_name=settings.qdrant_collection,
            prefetch=[
                {
                    "query": query_vector,
                    "using": "dense",
                    "limit": top_k,
                },
                {
                    "query": SparseVector(
                        indices=list(sparse_vec.keys()),
                        values=list(sparse_vec.values())
                    ),
                    "using": "sparse",
                    "limit": top_k,
                },
            ],
            query={"fusion": "rrf"},
            limit=top_k,
            with_payload=True,
            query_filter=query_filter,
        )
        return results.points if hasattr(results, 'points') else results
    except Exception as exc:
        logger.warning(f"Hybrid search failed, falling back to dense: {exc}")
        return search(query_vector, top_k=top_k, source_type=source_type)
```

---

### Modify `src/query/standard.py`

Replace the existing search call with the hybrid search:

Find:
```python
retrieval_k = max(top_k * 6, 50)
results = search(query_vector, top_k=retrieval_k)
```

Replace with:
```python
from src.ingestion.vector_db import hybrid_search

retrieval_k = max(top_k * 6, 50)
results = hybrid_search(
    query_text=rewritten_query,
    query_vector=query_vector,
    top_k=retrieval_k,
)
```

---

### Important Note On Existing Data

The existing Qdrant collection was created without sparse vectors. Hybrid search on the existing collection will fall back to dense-only gracefully (the `except` block in `hybrid_search()`). New data ingested after this change will have sparse vectors automatically.

To get full hybrid search on existing data, re-run:
```bash
python scripts/seed_icmr.py
python scripts/seed_pubmed.py
```

This is optional for now — dense-only is acceptable until re-ingestion happens as part of Phase 2.

---

### Verify Step 1.4

```bash
# 1. Test hybrid search imports
python -c "
from src.ingestion.vector_db import hybrid_search, build_sparse_vector
sparse = build_sparse_vector('tuberculosis doxycycline treatment India')
print(f'Sparse vector built: {len(sparse)} terms')
print('hybrid_search imported OK')
"

# 2. End-to-end test
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "doxycycline dosage for scrub typhus", "top_k": 5}'
```

Check the answer — it should cite specific dosage information (200mg/day, 7 days) from the ICMR Rickettsial Disease guidelines. If it does, hybrid search + reranker + query rewriting are all working together.

---

## Final Verification — All Phase 1 Steps Together

After all three steps are complete, run this sequence:

```bash
# 1. All imports clean
python -c "
from src.query.rewriter import rewrite_query
from src.query.reranker import rerank_chunks
from src.ingestion.vector_db import hybrid_search
print('All Phase 1 imports OK')
"

# 2. Query that previously returned poor citations
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "what are hospital infection guidelines", "top_k": 5}'

# 3. Specific clinical query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "TB treatment first line drugs dosage duration", "top_k": 5}'

# 4. Colloquial query test
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "sugar treatment", "top_k": 5}'
```

For each query paste the `answer` field. The hospital infection query should now return specific protocols, not forewords. The sugar query should return diabetes treatment, not confectionery.

---

## Rules

- Absolute imports only (`src.*`) — never relative imports
- Never hardcode model names, URLs, or API keys — always use `get_settings()`
- Use `loguru` throughout — never `print()` inside `src/`
- If any step fails verification, fix it before moving to the next step
- Do not modify any existing file not listed in this document

---

*OpenMed Phase 1 — SentArc Labs | Director: Aditya Singh | adi.singh1426@gmail.com*
