# OpenInsight v2 — Architecture Overview
### For: Aditya Singh, SentArc Labs

---

## What Changed and Why

The v1 system had ~1,400 vectors from ~1,361 documents. That's barely one chunk per document — meaning 99% of the content inside every paper was never indexed, never retrieved, never used. The LLM was essentially answering from titles and abstracts only.

This document explains the redesigned architecture, the reasoning behind every major decision, and the performance targets.

---

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        DATA INGESTION PIPELINE                          │
│                                                                         │
│  PubMed XML / ICMR PDFs / Cochrane                                      │
│       │                                                                 │
│       ▼                                                                 │
│  ┌─────────────────┐    ┌──────────────────┐    ┌────────────────────┐  │
│  │ Source-Aware    │───▶│  Hierarchical    │───▶│  Metadata          │  │
│  │ Parser          │    │  Chunker         │    │  Enricher          │  │
│  │                 │    │                  │    │                    │  │
│  │ GROBID + XML    │    │ L0: doc_summary  │    │ DocType, Evidence  │  │
│  │ OCR fallback    │    │ L1: section      │    │ Specialty, India   │  │
│  └─────────────────┘    │ L2: paragraph    │    │ Level, MeSH terms  │  │
│                         └──────────────────┘    └────────────────────┘  │
│                                   │                      │              │
│                                   ▼                      ▼              │
│                    ┌──────────────────────┐   ┌────────────────────┐   │
│                    │  Dual Embedder       │   │  MongoDB           │   │
│                    │                      │   │                    │   │
│                    │  Dense:              │   │  Full documents    │   │
│                    │  S-PubMedBERT        │   │  Full chunk text   │   │
│                    │  (contextual prefix) │   │  Audit log         │   │
│                    │                      │   └────────────────────┘   │
│                    │  Sparse:             │                             │
│                    │  BM25 medical        │                             │
│                    └──────────┬───────────┘                             │
│                               │                                         │
│                               ▼                                         │
│                    ┌──────────────────────┐                             │
│                    │  Qdrant              │                             │
│                    │  openinsight_v2      │                             │
│                    │                      │                             │
│                    │  Dense vectors (768) │                             │
│                    │  Sparse vectors      │                             │
│                    │  Payload indexes     │                             │
│                    │  (~15–25K vectors)   │                             │
│                    └──────────────────────┘                             │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                     STANDARD SEARCH (QUERY PATH)                        │
│                                                                         │
│  Physician Query                                                         │
│       │                                                                 │
│       ▼                                                                 │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ Query Understanding                                              │   │
│  │                                                                  │   │
│  │  • Intent classification (diagnostic / therapeutic / guideline) │   │
│  │  • Entity extraction via scispaCy (disease, drug, symptom)      │   │
│  │  • Metadata filter inference (year, doc_type, india_relevant)   │   │
│  │  • Conditional HyDE rewrite (only for diagnostic/prognostic)    │   │
│  │  • Query expansion with medical synonyms                        │   │
│  └───────────────────────────────┬─────────────────────────────────┘   │
│                                  │                                      │
│                    ┌─────────────┴──────────────┐                      │
│                    │                            │                      │
│                    ▼                            ▼                      │
│          ┌──────────────────┐        ┌──────────────────┐              │
│          │  Dense Search    │        │  Sparse Search   │              │
│          │  (top-50)        │        │  BM25 (top-50)   │              │
│          │  asyncio.gather  │        │  asyncio.gather  │              │
│          └─────────┬────────┘        └────────┬─────────┘              │
│                    │                           │                       │
│                    └─────────────┬─────────────┘                       │
│                                  │                                      │
│                                  ▼                                      │
│                    ┌─────────────────────────────┐                     │
│                    │  RRF Fusion + Evidence Boost │                     │
│                    │                             │                     │
│                    │  Reciprocal Rank Fusion     │                     │
│                    │  +boost: RCT (1.25×)        │                     │
│                    │  +boost: Syst. Review (1.35×)│                    │
│                    │  +boost: Guideline (1.10×)  │                     │
│                    │  +boost: 2024+ year (1.08×) │                     │
│                    │                             │                     │
│                    │  Output: top-20             │                     │
│                    └──────────────┬──────────────┘                     │
│                                   │                                     │
│                    ┌──────────────▼──────────────┐                     │
│                    │  Redis Cache (reranker)      │◀── Cache hit?       │
│                    └──────────────┬──────────────┘    Skip reranker    │
│                                   │ Cache miss                         │
│                                   ▼                                     │
│                    ┌─────────────────────────────┐                     │
│                    │  Cross-Encoder Reranker      │                     │
│                    │  BAAI/bge-reranker-base      │                     │
│                    │  Input: top-20               │                     │
│                    │  Output: top-8               │                     │
│                    └──────────────┬──────────────┘                     │
│                                   │                                     │
│                                   ▼                                     │
│                    ┌─────────────────────────────┐                     │
│                    │  MMR Deduplication           │                     │
│                    │  λ=0.7                       │                     │
│                    │  Input: top-8                │                     │
│                    │  Output: top-6               │                     │
│                    │  (removes near-duplicates)   │                     │
│                    └──────────────┬──────────────┘                     │
│                                   │                                     │
│                                   ▼                                     │
│                    ┌─────────────────────────────┐                     │
│                    │  Context Assembly             │                     │
│                    │  6 chunks + citations         │                     │
│                    │  + evidence labels            │                     │
│                    └──────────────┬──────────────┘                     │
│                                   │                                     │
│                                   ▼                                     │
│                    ┌─────────────────────────────┐                     │
│                    │  Llama 3.1 70B (NVIDIA NIM)  │                     │
│                    │  Streamed response            │                     │
│                    └─────────────────────────────┘                     │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Key Design Decisions

### 1. Why Hierarchical Chunking?

**Old approach:** Split every document every N characters. One chunk per document for a 5-page paper.

**New approach:** Three levels:
- **L0 (doc_summary):** Title + abstract + MeSH terms. One per document. Always retrieved for document-level signal.
- **L1 (section):** Each section as one chunk if it's short enough. Preserves the semantic unit of "the Results section".
- **L2 (paragraph):** Sentence-window chunks with overlap for long sections.

Why this matters: a chunk that spans the last paragraph of Results and the first paragraph of Discussion contains two completely different concepts. The embedding model averages them — the resulting vector doesn't cleanly represent either. Splitting at section boundaries preserves the semantic signal.

**Target corpus size after re-ingestion: 15,000–25,000 vectors** (up from ~1,400).

---

### 2. Why Contextual Prefixes?

Every chunk gets embedded with this prefix:

```
Source: pubmed
Document type: rct
Title: Metformin for Type 2 Diabetes in Indian Adults
Section: Results

Metformin reduced HbA1c by 1.5% at 12 weeks...
```

Without the prefix, the embedding model sees an isolated paragraph: `"Metformin reduced HbA1c by 1.5% at 12 weeks..."`. It doesn't know what paper this is from, or that it's from the Results section of an RCT.

With the prefix, the model embeds the paragraph **in context** — the vector represents "this is a result from an RCT about Metformin for diabetes in India." When a physician asks "does metformin work for Indian diabetics?", the query embedding finds this chunk because both vectors are in the same semantic neighbourhood.

The **raw text** (without prefix) is still stored in MongoDB and in the Qdrant payload for display. Only the contextual version is used for embedding.

---

### 3. Why RRF Instead of Weighted Fusion?

Weighted fusion: `score = α * bm25_score + (1-α) * cosine_score`

The problem: BM25 scores range 0.1–12.0. Cosine similarity scores range 0.5–0.95. These are completely different scales and distributions. Tuning α becomes a brittle, corpus-dependent hack.

**Reciprocal Rank Fusion:** `score(doc) = Σ 1/(k + rank)`

RRF only uses *rank position*, not raw scores. A document ranked 1st by BM25 and 5th by dense gets the same treatment regardless of what its absolute scores were. This makes the fusion stable and tuning-free.

`k=60` is from the original 2009 paper and works well in practice. No need to tune it.

---

### 4. Why Evidence-Level Boosting?

After RRF, we multiply the score by a small factor based on the document's evidence level:
- Systematic review / meta-analysis: 1.35×
- RCT: 1.25×
- Guideline: 1.10×
- Case series: 1.00× (no boost)

This means a RCT that ranks 5th in raw retrieval can overtake a case report that ranks 2nd. For a clinical decision support system, this is exactly right — physicians should see the strongest evidence first.

---

### 5. Why Run the Reranker on Top-20, Not Top-50?

The cross-encoder (BAAI/bge-reranker-base) reads the full query and each candidate chunk **together** — it's O(n) expensive. Running it on 50 candidates takes ~3–4 seconds. Running it on 20 takes ~1–1.5 seconds.

The RRF fusion step reduces noise enough that top-20 post-fusion is already a high-quality candidate set. The reranker's job is fine-grained reordering of near-equal candidates — it doesn't need 50 items to do that well.

---

### 6. Why MMR?

Without MMR: the top 6 chunks might all come from the same paper. The LLM gets 6 variants of the same finding — it wastes the context window and produces a one-sided answer.

With MMR (λ=0.7): each selected chunk must be both relevant to the query AND different from the already-selected chunks. The λ=0.7 setting leans 70% toward relevance, 30% toward diversity — so we don't sacrifice quality, we just avoid redundancy.

Example without MMR: 5 of 6 chunks from the UKPDS trial about metformin.
Example with MMR: UKPDS (metformin), ACCORD trial (intensive control), ADVANCE trial (HbA1c targets), ICMR guideline, a meta-analysis, and a recent RCT on SGLT2i. The LLM now gives a balanced answer citing multiple lines of evidence.

---

### 7. Why Conditional HyDE?

HyDE (Hypothetical Document Embeddings) generates a fake answer paragraph and embeds that instead of the raw query. The idea: real relevant documents are more semantically similar to what an answer looks like than to the question itself.

Problem: HyDE adds a full LLM call (~1–2 seconds) before retrieval. This is expensive.

Solution: only use HyDE for intents where the query and answer are semantically distant:
- **Diagnostic:** "What causes X?" → answer is a paragraph about pathophysiology. Very different vocabulary from the question.
- **Prognostic:** "What is the survival rate of X?" → answer has survival statistics, Kaplan-Meier data.

Do NOT use HyDE for:
- **Drug info:** "Side effects of metformin" → the query already contains the right keywords. BM25 handles this well.
- **Guideline:** "ICMR recommendation for hypertension" → exact keyword match is better than a hypothetical guideline paragraph.

---

## Corpus Strategy

### Recommended sources for v2 ingestion (priority order)

| Source | Format | Content | India-relevance |
|---|---|---|---|
| ICMR guidelines | PDF | National protocols | ★★★★★ |
| NMC/MCI guidelines | PDF | National standards | ★★★★★ |
| RSSDI guidelines | PDF | Diabetes — Indian specific | ★★★★★ |
| Cardiological Society of India | PDF | Cardiac guidelines | ★★★★ |
| PubMed (Indian studies) | XML | RCTs, cohort studies with Indian data | ★★★★ |
| PubMed (global RCTs + systematic reviews) | XML | High evidence base | ★★★ |
| Cochrane Systematic Reviews | HTML/PDF | Gold standard evidence | ★★★ |
| WHO essential medicines guidelines | PDF | Relevant to Indian formulary | ★★★ |

### How to get PubMed data

Use the PubMed E-utilities API (free, no authentication needed):

```bash
# Example: fetch all RCTs on type 2 diabetes published 2019–2024
# Step 1: Search for PMIDs
curl "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term=type+2+diabetes[MeSH]+AND+randomized+controlled+trial[pt]&retmax=1000&datetype=pdat&mindate=2019&maxdate=2024&retmode=json" \
  > pmids.json

# Step 2: Fetch full XML for those PMIDs
python -c "
import json, requests, time
pmids = json.load(open('pmids.json'))['esearchresult']['idlist']
for i in range(0, len(pmids), 100):
    batch = ','.join(pmids[i:i+100])
    url = f'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id={batch}&rettype=xml&retmode=xml'
    r = requests.get(url)
    open(f'batch_{i}.xml', 'wb').write(r.content)
    time.sleep(0.5)  # rate limit: 2 req/sec without API key
"
```

Key PubMed search terms for Indian context:
- `india[affil]` — studies with Indian affiliates
- `[MeSH Major Topic]` for specific diseases
- `randomized controlled trial[pt]` for RCTs
- `systematic[sb]` for systematic reviews

---

## Performance Targets

| Metric | v1 (current) | v2 (target) |
|---|---|---|
| Total vectors | ~1,400 | 15,000–25,000 |
| Chunks per document | ~1 | 8–20 |
| First query latency | ~8–12s | <5s |
| Cached query latency | N/A | <0.5s |
| Retrieval recall (top-6) | Low (missing 99% of content) | High |
| Response citation quality | Titles only | Specific paragraphs with PMID links |
| Duplicate chunks in context | High | Zero (MMR) |

---

## Metadata Payload — Full Schema Reference

Every vector in Qdrant carries this payload. You can filter on any of these fields.

```json
{
  "doc_id": "pmid_38291047",
  "chunk_id": "pmid_38291047_s2_p3",
  "chunk_type": "paragraph",
  "source": "pubmed",
  "doc_type": "rct",
  "evidence_level": "1b",
  "evidence_boost": 1.25,
  "title": "Metformin versus glipizide in type 2 diabetes",
  "year": 2023,
  "journal": "NEJM",
  "doi": "10.1056/NEJMoa1234567",
  "pmid": "38291047",
  "authors": ["Shah A", "Patel B"],
  "specialty": ["endocrinology"],
  "mesh_terms": ["Diabetes Mellitus, Type 2", "Metformin", "Glipizide"],
  "keywords": ["glycemic control", "HbA1c", "cardiovascular risk"],
  "section_title": "Results",
  "chunk_index": 3,
  "total_chunks": 12,
  "india_relevant": true,
  "has_indian_data": true,
  "indian_source": false,
  "has_table": false,
  "has_drug_dosing": true,
  "has_lab_values": false,
  "raw_text": "Metformin reduced HbA1c by 1.5% compared to..."
}
```

---

## Evidence Level Reference

Used for boosting in RRF fusion and displayed in context assembly.

| Code | Label | Document Type | Boost |
|---|---|---|---|
| 1a | Systematic Review / Meta-Analysis | `systematic_review`, `meta_analysis` | 1.35× |
| 1b | Randomised Controlled Trial | `rct` | 1.25× |
| 2a | Systematic Review of Cohort Studies | — | 1.15× |
| 2b | Cohort Study | `cohort` | 1.10× |
| 3 | Case-Control Study | — | 1.05× |
| 4 | Case Series | `case_report` | 1.00× |
| 5 | Expert Opinion / Guideline | `guideline`, `review`, `editorial` | 1.10× |

Note: Guidelines get the same boost as cohort studies (1.10×) because for clinical decision support in India, national guidelines carry high practical value even though their evidence level is technically lower.

---

## What Comes After (DeepConsult Preview)

DeepConsult will be a separate mode that runs a multi-agent pipeline. Don't build this now — but keep the architecture in mind when designing the data layer, because DeepConsult reuses the same Qdrant index.

DeepConsult (future):
- Takes a complex clinical case (symptoms + history + labs)
- Decomposes into sub-queries (e.g. "differential diagnosis for these symptoms", "drug interactions for current medications", "relevant Indian epidemiology")
- Each sub-query runs through the standard search pipeline
- A synthesis agent aggregates all results into a structured clinical report
- Uses LightRAG or a lightweight knowledge graph for multi-hop reasoning (e.g. "what drug interactions exist between X and Y in a patient with renal impairment?")

The metadata fields `specialty`, `has_drug_dosing`, `has_lab_values`, and `india_relevant` that are being added now will be used by DeepConsult's routing logic to direct sub-queries to the right subset of the index.

---

## Quick Reference: New Module Map

```
app/
├── ingestion/
│   ├── parsers.py          GROBID + PubMed XML → ParsedDocument
│   ├── chunker.py          ParsedDocument → List[Chunk] (3-level hierarchy)
│   ├── metadata.py         Chunk → ChunkMetadata (evidence level, specialty, India flags)
│   ├── embedder.py         Text → dense (768-dim) + sparse (BM25) vectors
│   ├── qdrant_indexer.py   Chunks + embeddings → Qdrant (openinsight_v2 collection)
│   ├── mongo_store.py      Full docs + chunk text → MongoDB
│   ├── pipeline.py         Orchestrates all ingestion steps
│   └── run_ingestion.py    CLI: python -m app.ingestion.run_ingestion --dir ... --source ...
└── search/
    ├── query_understanding.py  Query → intent + entities + filters + HyDE flag
    ├── retriever.py            Qdrant dense + sparse parallel search
    ├── fusion.py               RRF + evidence boost → top-20
    ├── reranker.py             BAAI/bge-reranker-base → top-8
    ├── mmr.py                  MMR dedup → top-6
    ├── context_builder.py      Chunks → formatted context string + citation list
    └── cache.py                Redis cache for reranker + full results
```
