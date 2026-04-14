# OpenInsight v2 — Agent Execution Instructions
> Feed this file to your GitHub Copilot agent. It contains only what the agent needs: precise tasks, file targets, no theory.

---

## How to Use This File

This document is a sequential task list for a code-generation agent. Each task is self-contained and references the full implementation guide (`01_full_implementation_guide.md`) for the complete code. The agent should:

1. Read the task description
2. Create or modify the specified file
3. Run the specified validation command
4. Only proceed to the next task when validation passes

Do **not** skip tasks. Do **not** reorder tasks. Dependencies exist between steps.

---

## Environment Setup (Run Once)

Before starting any task, ensure the following are installed in the Codespaces environment.

```bash
# Python packages
pip install \
  sentence-transformers>=2.7.0 \
  transformers>=4.40.0 \
  torch>=2.2.0 \
  qdrant-client>=1.9.0 \
  scispacy>=0.5.4 \
  lxml>=5.2.0 \
  httpx>=0.27.0 \
  pdf2image>=1.17.0 \
  pytesseract>=0.3.10 \
  pymongo>=4.7.0 \
  motor>=3.4.0 \
  redis>=5.0.0

# scispaCy model
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_md-0.5.4.tar.gz

# Verify installs
python -c "import sentence_transformers, qdrant_client, motor, redis; print('All OK')"
```

Add GROBID to `docker-compose.yml` under `services:`:

```yaml
  grobid:
    image: lfoppiano/grobid:0.8.0
    ports:
      - "8070:8070"
    environment:
      - JAVA_OPTS=-Xms512m -Xmx2g
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8070/api/isalive"]
      interval: 30s
      timeout: 10s
      retries: 3
```

Add to `.env`:

```env
QDRANT_COLLECTION=openinsight_v2
DENSE_MODEL_NAME=pritamdeka/S-PubMedBert-MS-MARCO
RERANKER_MODEL_NAME=BAAI/bge-reranker-base
GROBID_URL=http://grobid:8070
CACHE_VERSION=v2
TOP_K_RETRIEVAL=50
TOP_K_AFTER_FUSION=20
TOP_K_AFTER_RERANK=8
TOP_K_FINAL=6
MMR_LAMBDA=0.7
```

---

## Task 1 — Create Source-Aware Parser

**File to create:** `app/ingestion/parsers.py`

**What this file must contain:**
- `SourceType` enum with values: `PUBMED_XML`, `PDF_GUIDELINE`, `PDF_PAPER`, `COCHRANE_HTML`, `NMC_PDF`
- `ParsedSection` dataclass with fields: `title`, `text`, `section_index`, `has_table`, `tables`
- `ParsedDocument` dataclass with fields: `doc_id`, `source_type`, `title`, `abstract`, `authors`, `year`, `journal`, `doi`, `pmid`, `sections`, `mesh_terms`, `keywords`, `raw_text`, `metadata`
- `DocumentParser` class with:
  - `parse(path, source_type) -> ParsedDocument`
  - `_parse_pubmed_xml(path) -> ParsedDocument`
  - `_parse_grobid_pdf(path, source_type) -> ParsedDocument`
  - `_parse_tei_xml(tei_xml, path, source_type) -> ParsedDocument`
  - `_ocr_fallback(path) -> str`

**Implementation notes for agent:**
- PubMed XML parser uses `xml.etree.ElementTree` (stdlib)
- GROBID parser sends POST to `http://grobid:8070/api/processFulltextDocument`
- TEI XML parser uses `lxml.etree` with namespace `http://www.tei-c.org/ns/1.0`
- OCR fallback uses `pdf2image.convert_from_path` + `pytesseract.image_to_string`
- `ParsedDocument.doc_id` format: `"pmid_{pmid}"` for PubMed, `"pdf_{md5hash[:12]}"` for PDFs
- Tables found in GROBID TEI XML at `.//tei:figure[@type="table"]`

**Validation command:**
```bash
python -c "
from app.ingestion.parsers import DocumentParser, SourceType, ParsedDocument
print('Parser import OK')
p = DocumentParser()
print('Parser instantiation OK')
"
```

---

## Task 2 — Create Metadata Schema and Enricher

**File to create:** `app/ingestion/metadata.py`

**What this file must contain:**
- `DocType` enum: `RCT`, `SYSTEMATIC_REVIEW`, `META_ANALYSIS`, `GUIDELINE`, `REVIEW`, `CASE_REPORT`, `EDITORIAL`, `COHORT`, `UNKNOWN`
- `EvidenceLevel` enum: `LEVEL_1A`, `LEVEL_1B`, `LEVEL_2A`, `LEVEL_2B`, `LEVEL_3`, `LEVEL_4`, `LEVEL_5`, `UNKNOWN`
- `DOC_TYPE_TO_EVIDENCE_LEVEL` dict: maps each `DocType` to its `EvidenceLevel`
- `EVIDENCE_BOOST_SCORE` dict: maps each `EvidenceLevel` to a float multiplier (1.0–1.35)
- `ChunkMetadata` dataclass with ALL the fields listed below
- `MetadataEnricher` class with `enrich(doc, source) -> dict`

**ChunkMetadata fields (all required):**
```
doc_id, chunk_id, chunk_type, source, doc_type, evidence_level,
title, year, journal, authors, doi, pmid,
specialty (List[str]), mesh_terms (List[str]), keywords (List[str]),
section_title, chunk_index, total_chunks,
india_relevant (bool), has_indian_data (bool), indian_source (bool),
has_table (bool), has_drug_dosing (bool), has_lab_values (bool)
```

**`ChunkMetadata` must have a `to_qdrant_payload()` method** that returns a flat dict suitable for Qdrant payload storage.

**`MetadataEnricher.enrich()` must detect:**
- `doc_type` from title/abstract keywords (see patterns in full guide)
- `specialty` from keyword matching (list: cardiology, endocrinology, pulmonology, neurology, infectious_disease, gastroenterology, oncology, nephrology)
- `india_relevant`: True if any India-related keywords found in text
- `has_indian_data`: True if study has Indian participants
- `indian_source`: True if source is `icmr`, `nmc_guideline`, `rssdi`, or journal matches Indian journal list
- `has_drug_dosing`: True if dosing patterns found (mg/kg, mg/day, etc.)
- `has_lab_values`: True if lab value patterns found

**Validation command:**
```bash
python -c "
from app.ingestion.metadata import MetadataEnricher, DocType, EvidenceLevel, ChunkMetadata
e = MetadataEnricher()
print('MetadataEnricher OK')

# Test doc type detection
from app.ingestion.parsers import ParsedDocument, SourceType, ParsedSection
doc = ParsedDocument(
    doc_id='test', source_type=SourceType.PUBMED_XML,
    title='A randomized controlled trial of metformin',
    abstract='We conducted a randomized controlled trial...',
    authors=[], year=2023, journal='NEJM', doi=None, pmid='12345',
    sections=[], mesh_terms=[], keywords=[], raw_text='', metadata={}
)
result = e.enrich(doc, 'pubmed')
assert result['doc_type'].value == 'rct', f'Expected rct, got {result[\"doc_type\"]}'
print('DocType detection OK:', result['doc_type'])
print('All metadata tests pass')
"
```

---

## Task 3 — Create Hierarchical Chunker

**File to create:** `app/ingestion/chunker.py`

**What this file must contain:**
- `Chunk` dataclass with fields: `chunk_id`, `doc_id`, `chunk_type`, `section_title`, `text`, `contextual_text`, `char_count`, `token_estimate`, `chunk_index`, `total_chunks`, `metadata`
- `HierarchicalChunker` class

**`HierarchicalChunker` constants:**
```python
TARGET_CHUNK_TOKENS = 350
MAX_CHUNK_TOKENS = 500
OVERLAP_TOKENS = 50
MIN_CHUNK_TOKENS = 80
```

**`HierarchicalChunker.chunk_document(doc, doc_metadata) -> List[Chunk]`:**

Must produce these chunk types in this order:
1. One `doc_summary` chunk per document: title + abstract + mesh_terms + keywords
2. For each section:
   a. Table chunks if section has tables (type: `table`)
   b. If section tokens ≤ 350: one paragraph chunk for the whole section
   c. If section tokens > 350: multiple sentence-window chunks with 50-token overlap

**`contextual_text` format (CRITICAL — must match exactly):**
```
Source: {source}
Document type: {doc_type}
Title: {doc_title}
Section: {section_title}

{chunk_text}
```

**`_sentence_window_split(text) -> List[str]`:**
- Must handle medical abbreviations without false splits: `et al.`, `Fig.`, `e.g.`, `i.e.`, `vs.`, `mg/dL`, `p.o.`, `i.v.`
- Strategy: replace abbreviations with placeholders, split on `(?<=[.!?])\s+(?=[A-Z])`, restore

**`_format_table_chunk(table: dict) -> str`:**
- Input: `{'caption': str, 'text': str}` or `{'caption': str, 'headers': list, 'rows': list}`
- If headers + rows: format as `"Header1: value | Header2: value"` per row
- Always include caption as first line
- Cap at 20 rows

**`chunk_document()` must fill `total_chunks` for all chunks after generation**

**Validation command:**
```bash
python -c "
from app.ingestion.chunker import HierarchicalChunker, Chunk
from app.ingestion.parsers import ParsedDocument, SourceType, ParsedSection

# Create a test document with multiple sections
doc = ParsedDocument(
    doc_id='test_001',
    source_type=SourceType.PUBMED_XML,
    title='Metformin for type 2 diabetes management',
    abstract='Metformin is the first-line treatment for type 2 diabetes mellitus. ' * 5,
    authors=['Smith J'], year=2023, journal='NEJM', doi=None, pmid='99999',
    sections=[
        ParsedSection(title='Methods', text='We enrolled 500 patients. ' * 30, section_index=0),
        ParsedSection(title='Results', text='Results showed improvement. ' * 25, section_index=1),
        ParsedSection(title='Conclusion', text='Metformin is effective.', section_index=2),
    ],
    mesh_terms=['Diabetes Mellitus, Type 2', 'Metformin'],
    keywords=['glycemic control'],
    raw_text='', metadata={}
)
metadata = {'source': 'pubmed', 'doc_type': 'rct', 'evidence_level': '1b',
            'specialty': ['endocrinology'], 'india_relevant': False,
            'has_indian_data': False, 'indian_source': False,
            'has_drug_dosing': True, 'has_lab_values': False}

chunker = HierarchicalChunker()
chunks = chunker.chunk_document(doc, metadata)

assert len(chunks) > 3, f'Expected >3 chunks, got {len(chunks)}'
assert chunks[0].chunk_type == 'doc_summary', 'First chunk must be doc_summary'
assert all(c.total_chunks == len(chunks) for c in chunks), 'total_chunks must be set'
assert all('Title:' in c.contextual_text for c in chunks), 'contextual_text must have Title prefix'
assert all(c.chunk_id.startswith('test_001') for c in chunks), 'chunk_id must include doc_id'

print(f'Generated {len(chunks)} chunks — OK')
print(f'Chunk types: {[c.chunk_type for c in chunks[:5]]}')
"
```

---

## Task 4 — Create Dual Embedder

**File to create:** `app/ingestion/embedder.py`

**What this file must contain:**
- `DualEmbedder` class with:
  - `__init__(dense_model_name)`: loads S-PubMedBERT, calls `.eval()`, moves to GPU if available
  - `embed_batch(texts, batch_size=32) -> np.ndarray`: uses `torch.inference_mode()`
  - `embed_query(query_text) -> np.ndarray`: single query embedding
  - `compute_sparse_vector(text) -> dict`: returns `{"indices": [...], "values": [...]}`
  - `_medical_tokenize(text) -> List[str]`: preserves compound terms as bigrams
  - `_term_to_index(term) -> int`: `hash(term) % 50000`
  - `_get_idf_weight(term) -> float`: length-based heuristic

**Medical compound terms to preserve in `_medical_tokenize`:**
```
"type 2 diabetes", "type 1 diabetes", "heart failure", "blood pressure",
"myocardial infarction", "atrial fibrillation", "blood glucose",
"hemoglobin a1c", "hba1c", "randomized controlled trial",
"systematic review", "meta analysis", "meta-analysis",
"insulin resistance", "glycemic control", "renal failure",
"coronary artery disease", "coronary heart disease"
```

**Both `embed_batch` and `embed_query` must:**
- Use `torch.inference_mode()` context manager
- Use `normalize_embeddings=True` (cosine = dot product)
- Return `np.ndarray`

**Validation command:**
```bash
python -c "
from app.ingestion.embedder import DualEmbedder
import numpy as np

e = DualEmbedder()
print('DualEmbedder loaded OK')

# Test single embed
v = e.embed_query('treatment for type 2 diabetes')
assert isinstance(v, np.ndarray), 'Must return ndarray'
assert v.shape == (768,), f'Expected (768,), got {v.shape}'
assert abs(np.linalg.norm(v) - 1.0) < 0.01, 'Vector must be normalized'
print('Single embed OK, shape:', v.shape)

# Test batch embed
texts = ['diabetes treatment', 'heart failure management', 'hypertension guidelines']
batch = e.embed_batch(texts, batch_size=2)
assert batch.shape == (3, 768), f'Expected (3, 768), got {batch.shape}'
print('Batch embed OK, shape:', batch.shape)

# Test sparse vector
sv = e.compute_sparse_vector('type 2 diabetes treatment metformin')
assert 'indices' in sv and 'values' in sv, 'Must return dict with indices and values'
assert len(sv['indices']) == len(sv['values']), 'Indices and values must match'
print('Sparse vector OK, terms:', len(sv['indices']))
"
```

---

## Task 5 — Create Qdrant Indexer

**File to create:** `app/ingestion/qdrant_indexer.py`

**What this file must contain:**
- `COLLECTION_NAME = "openinsight_v2"`
- `DENSE_DIM = 768`
- `QdrantIndexer` class with:
  - `__init__(qdrant_url)`: connects to Qdrant
  - `create_collection(recreate=False)`: creates collection with dense + sparse vectors + payload indexes
  - `upsert_chunks(chunks, dense_embeddings, sparse_vectors)`: upserts in batches of 100

**`create_collection` must create:**
- Named dense vector: `"dense"`, size=768, distance=COSINE, `on_disk=False`
- Named sparse vector: `"sparse"`, `on_disk=False`
- HNSW config: `m=16`, `ef_construct=128`
- Payload indexes for: `year` (INTEGER), `doc_type` (KEYWORD), `source` (KEYWORD), `evidence_level` (KEYWORD), `specialty` (KEYWORD), `india_relevant` (BOOL), `has_drug_dosing` (BOOL), `chunk_type` (KEYWORD), `pmid` (KEYWORD)

**`upsert_chunks` must:**
- Generate UUID from `chunk_id` using `uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id)`
- Use `PointStruct` with named vectors: `{"dense": [...], "sparse": SparseVector(...)}`
- Store in payload: all `ChunkMetadata.to_qdrant_payload()` fields + `raw_text` + `chunk_id` + `doc_id` + `chunk_type` + `section_title` + `chunk_index` + `total_chunks`
- Upsert in batches of 100 (not all at once)

**Validation command:**
```bash
python -c "
from app.ingestion.qdrant_indexer import QdrantIndexer, COLLECTION_NAME
import os

url = os.getenv('QDRANT_URL', 'http://localhost:6333')
indexer = QdrantIndexer(qdrant_url=url)
print('QdrantIndexer connected OK')

# Create collection (will fail gracefully if already exists)
indexer.create_collection(recreate=False)
info = indexer.client.get_collection(COLLECTION_NAME)
print('Collection exists:', COLLECTION_NAME)
print('Vectors config:', list(info.config.params.vectors.keys()) if hasattr(info.config.params.vectors, 'keys') else 'named')
"
```

---

## Task 6 — Create MongoDB Document Store

**File to create:** `app/ingestion/mongo_store.py`

**What this file must contain:**
- `MongoDocStore` class with:
  - `__init__(mongo_url, db_name)`
  - `async store_document(doc, enriched_metadata)`: upsert to `documents` collection, keyed on `doc_id`
  - `async store_chunks(chunks)`: bulk upsert to `chunks` collection
  - `async get_document(doc_id) -> Optional[dict]`
  - `async get_chunk(chunk_id) -> Optional[dict]`

**`store_document` must save:**
- `doc_id`, `title`, `abstract`, `authors`, `year`, `journal`, `doi`, `pmid`
- `sections` as list of `{title, text, index}`
- `mesh_terms`, `keywords`
- All fields from `enriched_metadata`
- `ingested_at` timestamp (ISO format)

**`store_chunks` must use `motor` (async pymongo) and `bulk_write` with `UpdateOne(..., upsert=True)` operations.**

**Validation command:**
```bash
python -c "
import asyncio
from app.ingestion.mongo_store import MongoDocStore

async def test():
    store = MongoDocStore()
    print('MongoDocStore connected OK')
    # Test get on non-existent doc
    result = await store.get_document('nonexistent_123')
    assert result is None, 'Should return None for missing doc'
    print('get_document None case OK')

asyncio.run(test())
"
```

---

## Task 7 — Create Ingestion Pipeline and CLI

**File to create:** `app/ingestion/pipeline.py`

**File to create:** `app/ingestion/run_ingestion.py`

**`IngestionPipeline` must:**
- Import all previous modules (parser, chunker, metadata, embedder, qdrant_indexer, mongo_store)
- `async ingest_directory(directory, source, recreate_index=False)`:
  1. Glob for `**/*.pdf` and `**/*.xml` in directory
  2. Process in batches of 10 documents
  3. Parse each document in a `ThreadPoolExecutor(max_workers=4)` (PDF parsing is blocking I/O)
  4. Enrich metadata for each parsed doc
  5. Chunk all documents in the batch
  6. Embed ALL chunks in the batch at once using `embed_batch()` — NOT one at a time
  7. Compute sparse vectors for all chunks
  8. Upsert to Qdrant via `upsert_chunks()`
  9. Store in MongoDB async
  10. Log progress with total chunk count
- `_infer_source_type(path, source) -> SourceType`: returns `PUBMED_XML` for `.xml`, `PDF_GUIDELINE` for icmr/nmc_guideline/rssdi sources, `PDF_PAPER` otherwise

**`run_ingestion.py` must:**
- Use `argparse` with `--dir`, `--source`, `--recreate` flags
- `source` choices: `pubmed`, `icmr`, `cochrane`, `nmc_guideline`, `rssdi`, `who`
- Call `asyncio.run(pipeline.ingest_directory(...))`
- Print final chunk count

**Validation command:**
```bash
# Dry-run validation (no actual files needed)
python -c "
from app.ingestion.pipeline import IngestionPipeline
print('IngestionPipeline import OK')
p = IngestionPipeline()
print('IngestionPipeline instantiated OK')
"

# Test CLI help
python -m app.ingestion.run_ingestion --help
```

---

## Task 8 — Create Query Understanding Layer

**File to create:** `app/search/query_understanding.py`

**What this file must contain:**
- `QueryIntent` enum: `DIAGNOSTIC`, `THERAPEUTIC`, `PROGNOSTIC`, `DRUG_INFO`, `GUIDELINE`, `GENERAL`
- `QueryAnalysis` dataclass: `original_query`, `intent`, `entities`, `rewritten_query`, `metadata_filters`, `use_hyde`, `expanded_terms`
- `QueryUnderstanding` class with:
  - `__init__()`: loads scispaCy `en_core_sci_md` model (with fallback if not installed)
  - `analyze(query) -> QueryAnalysis`
  - `_classify_intent(query_lower) -> QueryIntent`
  - `_extract_entities(query) -> dict`
  - `_infer_metadata_filters(query_lower, entities, intent) -> list`
  - `_expand_query(query_lower) -> List[str]`

**Intent classification rules (check in this order):**
1. DIAGNOSTIC: "what causes", "cause of", "differential", "how to diagnose", "symptoms of", "signs of", "what is", "aetiology"
2. THERAPEUTIC: "treatment", "treat", "therapy", "management", "drug of choice", "dose", "dosage", "first line", "medication for"
3. PROGNOSTIC: "prognosis", "outcome", "survival", "mortality", "risk of"
4. DRUG_INFO: "side effects", "adverse effects", "interactions", "contraindications", "mechanism of"
5. GUIDELINE: "guideline", "recommendation", "protocol", "standard of care", "icmr"
6. GENERAL: fallback

**Metadata filter inference rules:**
- "recent" / "latest" / "current" / "2024" / "2025" → `year >= 2020`
- "guideline" / "recommendation" / "protocol" → `doc_type in ["guideline", "systematic_review", "meta_analysis"]`
- "india" / "indian" / "indians" → `india_relevant = True`
- THERAPEUTIC intent + "dose" / "dosage" → `has_drug_dosing = True`

**`use_hyde` must be `True` only for `DIAGNOSTIC` and `PROGNOSTIC` intents.**

**Medical synonym table (for `_expand_query`):**
```python
MEDICAL_SYNONYMS = {
    "heart attack": ["myocardial infarction", "mi", "acute coronary syndrome"],
    "diabetes": ["diabetes mellitus", "dm", "type 2 diabetes", "t2dm"],
    "high blood pressure": ["hypertension", "htn"],
    "stroke": ["cerebrovascular accident", "cva"],
    "tb": ["tuberculosis", "mycobacterium tuberculosis"],
    "dengue": ["dengue fever", "dengue hemorrhagic fever"],
}
```

**Validation command:**
```bash
python -c "
from app.search.query_understanding import QueryUnderstanding, QueryIntent

qu = QueryUnderstanding()

# Test 1: therapeutic intent
a1 = qu.analyze('what is the treatment for type 2 diabetes?')
assert a1.intent == QueryIntent.THERAPEUTIC, f'Expected THERAPEUTIC, got {a1.intent}'
print('Test 1 THERAPEUTIC OK')

# Test 2: diagnostic
a2 = qu.analyze('what causes myocardial infarction?')
assert a2.intent == QueryIntent.DIAGNOSTIC, f'Expected DIAGNOSTIC, got {a2.intent}'
assert a2.use_hyde == True, 'DIAGNOSTIC must use HyDE'
print('Test 2 DIAGNOSTIC + HyDE OK')

# Test 3: India filter
a3 = qu.analyze('ICMR guideline for diabetes in India')
filters = a3.metadata_filters
india_filter = any('india_relevant' in str(f) for f in filters)
assert india_filter, 'India query must generate india_relevant filter'
print('Test 3 India filter OK')

# Test 4: guideline intent + year filter
a4 = qu.analyze('latest ICMR recommendation 2024')
assert a4.intent == QueryIntent.GUIDELINE
year_filter = any('year' in str(f) or 'gte' in str(f) for f in a4.metadata_filters)
assert year_filter, 'Latest query must have year filter'
print('Test 4 Guideline + year filter OK')

print('All QueryUnderstanding tests pass')
"
```

---

## Task 9 — Create Hybrid Retriever

**File to create:** `app/search/retriever.py`

**What this file must contain:**
- `RetrievedChunk` dataclass: `chunk_id`, `doc_id`, `score`, `text`, `contextual_text`, `metadata`, `retrieval_source`
- `HybridRetriever` class with:
  - `__init__(qdrant_url)`: connects to Qdrant, creates `DualEmbedder` instance
  - `async retrieve(query, query_analysis, top_k=50) -> Tuple[List[RetrievedChunk], List[RetrievedChunk]]`
  - `async _dense_search(embedding, filter, top_k) -> List[RetrievedChunk]`
  - `async _sparse_search(sparse_vector, filter, top_k) -> List[RetrievedChunk]`
  - `_build_filter(conditions) -> Optional[models.Filter]`
  - `_to_chunk(qdrant_result, source) -> RetrievedChunk`
  - `async _generate_hyde(query) -> Optional[str]`

**`retrieve()` must:**
1. If `query_analysis.use_hyde`: call `_generate_hyde(query)` with 15s timeout, use result as embed input
2. Add `expanded_terms` to sparse query string
3. Use `asyncio.gather` to compute dense embedding + sparse vector in parallel (use `loop.run_in_executor`)
4. Use `asyncio.gather` to run `_dense_search` + `_sparse_search` in parallel
5. Return `(dense_results, sparse_results)` tuple

**`_dense_search` and `_sparse_search` must:**
- Use `loop.run_in_executor(None, lambda: self.client.search(...))` to avoid blocking event loop
- Dense: `models.NamedVector(name="dense", vector=embedding.tolist())`
- Sparse: `models.NamedSparseVector(name="sparse", vector=models.SparseVector(indices=..., values=...))`

**`_generate_hyde` must:**
- POST to `{NVIDIA_NIM_URL}/v1/completions` with `max_tokens=200`, `temperature=0.1`
- Wrap in try/except — return `None` on any failure
- Timeout: 15 seconds

**Validation command:**
```bash
python -c "
from app.search.retriever import HybridRetriever, RetrievedChunk
print('HybridRetriever import OK')
r = HybridRetriever()
print('HybridRetriever connected OK')
"
```

---

## Task 10 — Create RRF Fusion

**File to create:** `app/search/fusion.py`

**What this file must contain:**
- `EVIDENCE_BOOST_SCORE` dict mapping evidence level strings to float multipliers
- `RECENCY_BOOST` dict: `{2025: 1.10, 2024: 1.08, 2023: 1.05, 2022: 1.03}`
- `reciprocal_rank_fusion(dense_results, sparse_results, k=60, top_n=20) -> List[RetrievedChunk]`

**`reciprocal_rank_fusion` algorithm:**
1. Build `{chunk_id: chunk}` lookup dict from all results (keep higher-score version on duplicate)
2. For each chunk at rank `r` in `dense_results`: `rrf_scores[chunk_id] += 1.0 / (k + r + 1)`
3. For each chunk at rank `r` in `sparse_results`: `rrf_scores[chunk_id] += 1.0 / (k + r + 1)`
4. For each chunk, multiply score by `EVIDENCE_BOOST_SCORE[evidence_level]`
5. For each chunk, multiply score by `RECENCY_BOOST[year]` if year in dict
6. Sort by final score descending, return top `top_n`
7. Update `chunk.score` to the final RRF score before returning

**Validation command:**
```bash
python -c "
from app.search.retriever import RetrievedChunk
from app.search.fusion import reciprocal_rank_fusion

# Create mock results
def make_chunk(chunk_id, score, evidence_level='unknown', year=2020):
    c = RetrievedChunk(
        chunk_id=chunk_id, doc_id='doc1', score=score,
        text='test', contextual_text='test',
        metadata={'evidence_level': evidence_level, 'year': year},
        retrieval_source='dense'
    )
    return c

dense = [make_chunk(f'c{i}', 1.0 - i*0.1) for i in range(10)]
sparse = [make_chunk(f'c{i+5}', 1.0 - i*0.1) for i in range(10)]

# RCT should be boosted
dense[0] = make_chunk('c_rct', 0.5, evidence_level='1b', year=2024)

result = reciprocal_rank_fusion(dense, sparse)
assert len(result) <= 20, 'Should return at most top_n results'
print(f'RRF returned {len(result)} results OK')
print(f'Top chunk: {result[0].chunk_id}, score: {result[0].score:.4f}')
"
```

---

## Task 11 — Create Cross-Encoder Reranker

**File to create:** `app/search/reranker.py`

**What this file must contain:**
- `CrossEncoderReranker` class with:
  - `MODEL_NAME = "BAAI/bge-reranker-base"` class constant
  - `__init__()`: loads tokenizer + model, calls `.eval()`, moves to GPU if available
  - `rerank(query, chunks, top_k=8) -> List[RetrievedChunk]`

**`rerank` must:**
1. Create `pairs = [[query, chunk.text[:512]] for chunk in chunks]`
2. Tokenize with `padding=True, truncation=True, max_length=512`
3. Run inference with `torch.inference_mode()`
4. Extract logits, squeeze to 1D, convert to numpy
5. Update `chunk.score = float(score)` for each chunk
6. Return `sorted(chunks, key=lambda c: c.score, reverse=True)[:top_k]`

**The model must be loaded at `__init__` time (not at first request). This is the expensive step — it should happen at startup.**

**Validation command:**
```bash
python -c "
from app.search.reranker import CrossEncoderReranker
from app.search.retriever import RetrievedChunk

reranker = CrossEncoderReranker()
print('CrossEncoderReranker loaded OK')

chunks = [
    RetrievedChunk('c1', 'd1', 0.5, 'Metformin reduces HbA1c in type 2 diabetes patients', '', {}, 'dense'),
    RetrievedChunk('c2', 'd2', 0.4, 'The weather in London is often rainy', '', {}, 'dense'),
    RetrievedChunk('c3', 'd3', 0.3, 'Insulin therapy for uncontrolled diabetes', '', {}, 'dense'),
]
query = 'treatment for type 2 diabetes'
ranked = reranker.rerank(query, chunks, top_k=3)

# Metformin chunk should rank highest for this query
assert ranked[0].chunk_id != 'c2', 'Weather chunk should not rank first for diabetes query'
assert ranked[0].chunk_id in ('c1', 'c3'), 'Medical chunk should rank first'
print(f'Reranker result order: {[c.chunk_id for c in ranked]}')
print('Reranker test OK')
"
```

---

## Task 12 — Create MMR Module

**File to create:** `app/search/mmr.py`

**What this file must contain:**
- `cosine_similarity(a, b) -> float`: returns float, handles zero-norm vectors
- `maximal_marginal_relevance(chunks, embedder, lambda_param=0.7, n_select=6) -> List[RetrievedChunk]`

**`maximal_marginal_relevance` must:**
1. Return `chunks` unchanged if `len(chunks) <= n_select`
2. Embed `[c.text[:400] for c in chunks]` using `embedder.embed_batch(texts, batch_size=16)`
3. Select first chunk as the one with highest `c.score`
4. Iteratively compute `mmr_score = lambda_param * chunk.score - (1 - lambda_param) * max_cosine_to_selected`
5. Select chunk with highest MMR score each iteration
6. Return selected chunks in order of selection

**Validation command:**
```bash
python -c "
from app.search.mmr import maximal_marginal_relevance
from app.search.retriever import RetrievedChunk
from app.ingestion.embedder import DualEmbedder

embedder = DualEmbedder()

# Create 8 chunks — two pairs should be near-duplicates
chunks = [
    RetrievedChunk('c1', 'd1', 0.9, 'Metformin reduces blood glucose in type 2 diabetes. HbA1c improved significantly.', '', {}, 'dense'),
    RetrievedChunk('c2', 'd1', 0.85, 'Metformin lowers blood sugar in type 2 diabetes. HbA1c was reduced.', '', {}, 'dense'),  # near-dup of c1
    RetrievedChunk('c3', 'd2', 0.8, 'Insulin therapy is used for uncontrolled hyperglycaemia', '', {}, 'dense'),
    RetrievedChunk('c4', 'd3', 0.7, 'SGLT2 inhibitors reduce cardiovascular risk in diabetics', '', {}, 'dense'),
    RetrievedChunk('c5', 'd4', 0.6, 'GLP-1 agonists promote weight loss and glycemic control', '', {}, 'dense'),
    RetrievedChunk('c6', 'd5', 0.5, 'Dietary intervention and exercise for diabetes management', '', {}, 'dense'),
    RetrievedChunk('c7', 'd6', 0.4, 'Sulfonylureas as second-line agents for type 2 diabetes', '', {}, 'dense'),
    RetrievedChunk('c8', 'd7', 0.3, 'Pioglitazone and thiazolidinediones in diabetes care', '', {}, 'dense'),
]

selected = maximal_marginal_relevance(chunks, embedder, lambda_param=0.7, n_select=6)
assert len(selected) == 6, f'Expected 6, got {len(selected)}'
# c1 and c2 are near-duplicates — both should NOT be selected
ids = [c.chunk_id for c in selected]
assert not ('c1' in ids and 'c2' in ids), 'Near-duplicate chunks c1 and c2 should not both be selected'
print('MMR selected:', ids)
print('MMR deduplication OK')
"
```

---

## Task 13 — Create Context Builder

**File to create:** `app/search/context_builder.py`

**What this file must contain:**
- `EVIDENCE_LEVEL_LABELS` dict mapping level codes to human-readable strings
- `assemble_context(chunks, max_tokens=3000) -> str`
- `build_citation_list(chunks) -> List[dict]`

**`assemble_context` format per chunk (must match exactly):**
```
[{i}] {title} ({year}, {journal})
Evidence: {evidence_level_label} | {doc_type_title_case}{india_flag}
{chunk_text}
```
- Chunks separated by `"\n---\n"`
- `india_flag` = `" 🇮🇳 India-relevant"` if `india_relevant=True`, else `""`
- `doc_type_title_case` = `doc_type.replace("_", " ").title()`
- Stop adding chunks when total character count exceeds `max_tokens * 4`

**`build_citation_list` must return a list of dicts with:**
- `number`, `doc_id`, `chunk_id`, `title`, `authors`, `year`, `journal`, `doi`, `pmid`
- `evidence_level`, `doc_type`, `india_relevant`
- `source_url`: `"https://pubmed.ncbi.nlm.nih.gov/{pmid}"` if pmid exists, else `""`

**Validation command:**
```bash
python -c "
from app.search.context_builder import assemble_context, build_citation_list
from app.search.retriever import RetrievedChunk

chunks = [
    RetrievedChunk('c1', 'd1', 0.9,
        'Metformin reduced HbA1c by 1.5% in this randomized trial.',
        '', {
            'title': 'Metformin for Type 2 Diabetes',
            'year': 2023, 'journal': 'NEJM',
            'evidence_level': '1b', 'doc_type': 'rct',
            'india_relevant': True, 'authors': ['Smith J'],
            'doi': '10.1056/test', 'pmid': '99999'
        }, 'dense'),
]

context = assemble_context(chunks, max_tokens=2000)
assert '[1]' in context, 'Must have citation [1]'
assert 'Metformin for Type 2 Diabetes' in context
assert 'India-relevant' in context, 'India flag must appear'
assert '---' in context
print('Context assembled OK')
print(context[:300])

citations = build_citation_list(chunks)
assert citations[0]['number'] == 1
assert citations[0]['source_url'] == 'https://pubmed.ncbi.nlm.nih.gov/99999'
print('Citations OK')
"
```

---

## Task 14 — Create Redis Cache

**File to create:** `app/search/cache.py`

**What this file must contain:**
- `SearchCache` class with:
  - `CACHE_VERSION = "v2"` class constant
  - `__init__(redis_url)`: creates async Redis client
  - `_make_key(operation, *components) -> str`: `"openinsight:{version}:{operation}:{sha256_hex[:16]}"`
  - `async get_search_result(query, filters) -> Optional[dict]`
  - `async set_search_result(query, filters, result, ttl=1800)`
  - `async get_reranked(query, chunk_ids) -> Optional[List[dict]]`
  - `async set_reranked(query, chunk_ids, reranked, ttl=3600)`
  - `async invalidate_all()`

**Key generation must use `hashlib.sha256` on `"|".join(str(c) for c in components)`, take first 16 chars of hex.**

**`get_reranked` key must use `sorted(chunk_ids)` to ensure order-independence.**

**Validation command:**
```bash
python -c "
import asyncio
from app.search.cache import SearchCache

async def test():
    cache = SearchCache()

    # Test set + get
    await cache.set_search_result('test query', [], {'answer': 'test', 'citations': []}, ttl=60)
    result = await cache.get_search_result('test query', [])
    assert result is not None, 'Cache miss after set'
    assert result['answer'] == 'test'
    print('set/get OK')

    # Test miss
    miss = await cache.get_search_result('unknown query xyz', [])
    assert miss is None
    print('Cache miss OK')

    # Test invalidate
    await cache.invalidate_all()
    miss2 = await cache.get_search_result('test query', [])
    assert miss2 is None, 'Should be None after invalidation'
    print('Invalidation OK')

asyncio.run(test())
"
```

---

## Task 15 — Wire Up Search Endpoint

**File to modify:** `app/api/search.py`

**Replace or update the existing `/search` endpoint with the full pipeline:**

```
POST /search
Request: { query: str, top_k_final: int = 6 }
Response: { answer: str, citations: List[dict], query_intent: str, chunks_retrieved: int, cache_hit: bool }
```

**Pipeline order (must follow this exactly):**
1. `QueryUnderstanding().analyze(query)`
2. `cache.get_search_result(query, metadata_filters)` → return early if hit
3. `await retriever.retrieve(query, analysis, top_k=50)` → `dense_results, sparse_results`
4. `reciprocal_rank_fusion(dense_results, sparse_results, k=60, top_n=20)` → `fused`
5. `cache.get_reranked(query, chunk_ids)` → skip reranker if hit
6. `reranker.rerank(query, fused, top_k=8)` → `reranked`
7. `cache.set_reranked(...)` after reranking
8. `maximal_marginal_relevance(reranked, embedder, lambda_param=0.7, n_select=6)` → `final_chunks`
9. `assemble_context(final_chunks, max_tokens=3000)`
10. `build_citation_list(final_chunks)`
11. LLM generation (existing NIM call, pass assembled context as system context)
12. `cache.set_search_result(...)` with full result
13. Return `SearchResponse`

**Singletons must be initialised in the FastAPI lifespan, not per request:**
```python
from contextlib import asynccontextmanager

_singletons = {}

@asynccontextmanager
async def lifespan(app):
    _singletons['query_understanding'] = QueryUnderstanding()
    _singletons['retriever'] = HybridRetriever()
    _singletons['reranker'] = CrossEncoderReranker()
    _singletons['cache'] = SearchCache()
    yield

app = FastAPI(lifespan=lifespan)
```

**Validation command (integration test — requires all services running):**
```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "what is the first-line treatment for type 2 diabetes in India?"}' \
  | python -m json.tool
```

**Expected response shape:**
```json
{
  "answer": "...",
  "citations": [{"number": 1, "title": "...", "pmid": "...", ...}],
  "query_intent": "therapeutic",
  "chunks_retrieved": 20,
  "cache_hit": false
}
```

---

## Final Integration Check

Run this full test after all 15 tasks are complete:

```bash
# 1. Start all services
docker-compose up -d

# 2. Wait for GROBID to be ready
curl -f http://localhost:8070/api/isalive

# 3. Ingest sample documents (put 5 PDFs in /tmp/test_docs first)
python -m app.ingestion.run_ingestion \
  --dir /tmp/test_docs \
  --source pubmed \
  --recreate

# 4. Verify chunk count (should be > num_docs * 5)
python -c "
from qdrant_client import QdrantClient
c = QdrantClient('http://localhost:6333')
info = c.get_collection('openinsight_v2')
print('Vectors count:', info.vectors_count)
assert info.vectors_count > 0, 'No vectors indexed!'
"

# 5. Run search endpoint test
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "treatment for hypertension"}'

# 6. Run same query again — should be cache hit
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "treatment for hypertension"}' \
  | python -c "import sys,json; d=json.load(sys.stdin); print('cache_hit:', d.get('cache_hit'))"
```
