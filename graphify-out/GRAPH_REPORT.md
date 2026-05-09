# Graph Report - C:\Users\adisi\Downloads\openinsight  (2026-05-09)

## Corpus Check
- 81 files · ~144,841 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 916 nodes · 2120 edges · 66 communities detected
- Extraction: 50% EXTRACTED · 50% INFERRED · 0% AMBIGUOUS · INFERRED: 1064 edges (avg confidence: 0.61)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]
- [[_COMMUNITY_Community 9|Community 9]]
- [[_COMMUNITY_Community 10|Community 10]]
- [[_COMMUNITY_Community 11|Community 11]]
- [[_COMMUNITY_Community 12|Community 12]]
- [[_COMMUNITY_Community 13|Community 13]]
- [[_COMMUNITY_Community 14|Community 14]]
- [[_COMMUNITY_Community 15|Community 15]]
- [[_COMMUNITY_Community 16|Community 16]]
- [[_COMMUNITY_Community 17|Community 17]]
- [[_COMMUNITY_Community 18|Community 18]]
- [[_COMMUNITY_Community 19|Community 19]]
- [[_COMMUNITY_Community 20|Community 20]]
- [[_COMMUNITY_Community 21|Community 21]]
- [[_COMMUNITY_Community 22|Community 22]]
- [[_COMMUNITY_Community 23|Community 23]]
- [[_COMMUNITY_Community 24|Community 24]]
- [[_COMMUNITY_Community 25|Community 25]]
- [[_COMMUNITY_Community 26|Community 26]]
- [[_COMMUNITY_Community 27|Community 27]]
- [[_COMMUNITY_Community 28|Community 28]]
- [[_COMMUNITY_Community 29|Community 29]]
- [[_COMMUNITY_Community 30|Community 30]]
- [[_COMMUNITY_Community 31|Community 31]]
- [[_COMMUNITY_Community 32|Community 32]]
- [[_COMMUNITY_Community 33|Community 33]]
- [[_COMMUNITY_Community 34|Community 34]]
- [[_COMMUNITY_Community 35|Community 35]]
- [[_COMMUNITY_Community 36|Community 36]]
- [[_COMMUNITY_Community 37|Community 37]]
- [[_COMMUNITY_Community 38|Community 38]]
- [[_COMMUNITY_Community 39|Community 39]]
- [[_COMMUNITY_Community 40|Community 40]]
- [[_COMMUNITY_Community 41|Community 41]]
- [[_COMMUNITY_Community 42|Community 42]]
- [[_COMMUNITY_Community 43|Community 43]]
- [[_COMMUNITY_Community 44|Community 44]]
- [[_COMMUNITY_Community 45|Community 45]]
- [[_COMMUNITY_Community 46|Community 46]]
- [[_COMMUNITY_Community 47|Community 47]]
- [[_COMMUNITY_Community 48|Community 48]]
- [[_COMMUNITY_Community 49|Community 49]]
- [[_COMMUNITY_Community 50|Community 50]]
- [[_COMMUNITY_Community 51|Community 51]]
- [[_COMMUNITY_Community 52|Community 52]]
- [[_COMMUNITY_Community 53|Community 53]]
- [[_COMMUNITY_Community 54|Community 54]]
- [[_COMMUNITY_Community 55|Community 55]]
- [[_COMMUNITY_Community 56|Community 56]]
- [[_COMMUNITY_Community 57|Community 57]]
- [[_COMMUNITY_Community 58|Community 58]]
- [[_COMMUNITY_Community 59|Community 59]]
- [[_COMMUNITY_Community 60|Community 60]]
- [[_COMMUNITY_Community 61|Community 61]]
- [[_COMMUNITY_Community 62|Community 62]]
- [[_COMMUNITY_Community 63|Community 63]]
- [[_COMMUNITY_Community 64|Community 64]]
- [[_COMMUNITY_Community 65|Community 65]]

## God Nodes (most connected - your core abstractions)
1. `DocumentRecord` - 84 edges
2. `ConfidenceBreakdown` - 47 edges
3. `HallucinationResult` - 47 edges
4. `SafetyCheckResult` - 47 edges
5. `ValidationResult` - 45 edges
6. `SparseVector` - 42 edges
7. `BaseParser` - 39 edges
8. `IntentRouter` - 32 edges
9. `ChunkRecord` - 31 edges
10. `VectorPoint` - 29 edges

## Surprising Connections (you probably didn't know these)
- `Deduplication Engine Detects duplicate documents by DOI, normalised title, or c` --uses--> `DocumentRecord`  [INFERRED]
  C:\Users\adisi\Documents\openinsight\src\ingestion\deduplication.py → C:\Users\adisi\Downloads\openinsight\src\ingestion\document_db.py
- `Return SHA-256 hex digest of whitespace-normalised text.` --uses--> `DocumentRecord`  [INFERRED]
  C:\Users\adisi\Documents\openinsight\src\ingestion\deduplication.py → C:\Users\adisi\Downloads\openinsight\src\ingestion\document_db.py
- `Lowercase, strip punctuation/extra spaces for fuzzy title matching.` --uses--> `DocumentRecord`  [INFERRED]
  C:\Users\adisi\Documents\openinsight\src\ingestion\deduplication.py → C:\Users\adisi\Downloads\openinsight\src\ingestion\document_db.py
- `Simple character n-gram Jaccard similarity between two normalised titles.     R` --uses--> `DocumentRecord`  [INFERRED]
  C:\Users\adisi\Documents\openinsight\src\ingestion\deduplication.py → C:\Users\adisi\Downloads\openinsight\src\ingestion\document_db.py
- `Check whether an equivalent document already exists in MongoDB.      Checks (i` --uses--> `DocumentRecord`  [INFERRED]
  C:\Users\adisi\Documents\openinsight\src\ingestion\deduplication.py → C:\Users\adisi\Downloads\openinsight\src\ingestion\document_db.py

## Communities

### Community 0 - "Community 0"

Cohesion: 0.04
Nodes (75): BaseModel, Cache query embedding for faster repeated queries., Cache query embedding., SearchCache, assemble_context(), build_citation_list(), ContradictionDetector, ContradictionPair (+67 more)

### Community 1 - "Community 1"

Cohesion: 0.04
Nodes (105): check_citations(), CitationCheckResult, CitationIssue, Citation Checker Verifies that citations exist in MongoDB, are from trusted sou, An issue found with a citation., Result of citation validation., Validate citations for existence, trust level, evidence, and freshness.      A, _calculate_citation_score() (+97 more)

### Community 2 - "Community 2"

Cohesion: 0.04
Nodes (51): ABC, VectorStore, Enum, eq(), FilterCondition, FilterExpression, FilterOperator, from_conditions() (+43 more)

### Community 3 - "Community 3"

Cohesion: 0.05
Nodes (35): ChunkV3, HierarchicalChunkerV3, ParsedSectionLite, DocumentDeduplicator, Prevents duplicate documents from being re-ingested.          Checks:     1. doc, Check if document should be skipped or re-processed.                  Returns:, Check if chunk already exists., Compute hash of document content for change detection. (+27 more)

### Community 4 - "Community 4"

Cohesion: 0.05
Nodes (53): BaseParser, BaseParser, CDCParser, _fetch_cdc_api(), _fetch_pubmed_fallback(), CDC / NIH Guidelines Parser  Retrieves US Centers for Disease Control (CDC) an, Fetches CDC and NIH guideline documents.      Strategy:       1. CDC Media AP, Query CDC Public Health Media/Resource API. (+45 more)

### Community 5 - "Community 5"

Cohesion: 0.06
Nodes (35): ChunkRecord, get_db(), Document DB — MongoDB Stores raw + parsed documents before they go into the vec, A passage-level chunk, ready to be embedded and stored in Qdrant., Search pipeline v2 package., IngestionMonitor, Ingestion Monitoring & Analytics Tracks per-run and cumulative ingestion metric, Log a structured alert when a run fails or has high error rates.         Extend (+27 more)

### Community 6 - "Community 6"

Cohesion: 0.08
Nodes (21): chunk_text_v2(), _detect_section_header(), _estimate_tokens(), _merge_into_chunks(), Hierarchical Medical Text Chunker v2 Three-level chunking: section → semantic s, Sentence splitter respecting medical abbreviations., Merge segments into target-sized chunks with overlap., Main entry point. Hierarchical chunking:     1. Split into sections by header d (+13 more)

### Community 7 - "Community 7"

Cohesion: 0.06
Nodes (18): BaseSettings, Config, get_settings(), Settings, Get all existing doc_ids for a source type., get_nim_client(), NVIDIAClient, Direct NVIDIA NIM client using httpx.     Replaces OpenAI SDK for NVIDIA endpoin (+10 more)

### Community 8 - "Community 8"

Cohesion: 0.11
Nodes (16): _make_chunk(), _make_doc(), Tests for the data quality validation layer., TestFilterValidChunks, TestGarbleScore, TestValidateChunk, TestValidateDocument, filter_valid_chunks() (+8 more)

### Community 9 - "Community 9"

Cohesion: 0.09
Nodes (12): classify_content_type(), extract_entities(), infer_study_type(), _load_scispacy(), Medical Named Entity Recognition Extracts diseases, drugs, symptoms, dosages, c, Extract medical entities from text.     Returns dict with diseases, drugs, sympt, Classify chunk content type and return (content_type, weight).      Returns:, Infer study type and evidence level from text/title.     Returns (study_type, ev (+4 more)

### Community 10 - "Community 10"

Cohesion: 0.1
Nodes (16): compute_content_hash(), enrich_document_hashes(), is_duplicate(), _normalise_title(), Deduplication Engine Detects duplicate documents by DOI, normalised title, or c, Compute and set content_hash on a DocumentRecord before insertion.     Call thi, Return SHA-256 hex digest of whitespace-normalised text., Lowercase, strip punctuation/extra spaces for fuzzy title matching. (+8 more)

### Community 11 - "Community 11"

Cohesion: 0.25
Nodes (2): _parse_cron(), start_scheduler()

### Community 12 - "Community 12"

Cohesion: 0.25
Nodes (6): EvidenceBoost, get_boost(), Recency boost scores for ranking., Evidence level boost scores for ranking., RecencyBoost, reciprocal_rank_fusion()

### Community 13 - "Community 13"

Cohesion: 0.25
Nodes (4): ChunkDeduplicator, Check if chunk text is duplicate within current batch., Reset seen hashes for new document., Prevents duplicate chunks within a document.     Uses chunk text hash to detect

### Community 14 - "Community 14"

Cohesion: 0.38
Nodes (6): embed_query(), embed_texts(), get_embedder(), Embedding Generator Loads pritamdeka/S-PubMedBert-MS-MARCO once and reuses it., Embed a batch of texts. Returns list of float vectors., Embed a single query string.

### Community 15 - "Community 15"

Cohesion: 0.5
Nodes (2): _load_prompt(), Load a prompt from the prompts/ directory at project root.

### Community 16 - "Community 16"

Cohesion: 1.0
Nodes (0): 

### Community 17 - "Community 17"

Cohesion: 1.0
Nodes (0): 

### Community 18 - "Community 18"

Cohesion: 1.0
Nodes (0): 

### Community 19 - "Community 19"

Cohesion: 1.0
Nodes (0): 

### Community 20 - "Community 20"

Cohesion: 1.0
Nodes (0): 

### Community 21 - "Community 21"

Cohesion: 1.0
Nodes (0): 

### Community 22 - "Community 22"

Cohesion: 1.0
Nodes (0): 

### Community 23 - "Community 23"

Cohesion: 1.0
Nodes (0): 

### Community 24 - "Community 24"

Cohesion: 1.0
Nodes (0): 

### Community 25 - "Community 25"

Cohesion: 1.0
Nodes (0): 

### Community 26 - "Community 26"

Cohesion: 1.0
Nodes (0): 

### Community 27 - "Community 27"

Cohesion: 1.0
Nodes (1): Remove all v1-ingested ICMR documents and their chunks from MongoDB and Qdrant.

### Community 28 - "Community 28"

Cohesion: 1.0
Nodes (1): Mongo store for v2 ingestion artifacts.      - documents_v2: full parsed docum

### Community 29 - "Community 29"

Cohesion: 1.0
Nodes (1): Extract medical entities from text.     Returns dict with diseases, drugs, symp

### Community 30 - "Community 30"

Cohesion: 1.0
Nodes (1): Classify chunk content type and return (content_type, weight).      Returns:

### Community 31 - "Community 31"

Cohesion: 1.0
Nodes (1): Infer study type and evidence level from text/title.     Returns (study_type, e

### Community 32 - "Community 32"

Cohesion: 1.0
Nodes (1): Cross-Encoder Reranker Reranks retrieved chunks by true relevance to the query.

### Community 33 - "Community 33"

Cohesion: 1.0
Nodes (1): Rerank chunks by relevance to query using cross-encoder.      Args:         q

### Community 34 - "Community 34"

Cohesion: 1.0
Nodes (1): Query Rewriter Rewrites raw doctor queries into expanded medical queries before

### Community 35 - "Community 35"

Cohesion: 1.0
Nodes (1): Rewrite a raw doctor query into an expanded medical query.     Returns the orig

### Community 36 - "Community 36"

Cohesion: 1.0
Nodes (1): Medical Text Chunker Section-aware chunking — keeps tables, dosage info, and gu

### Community 37 - "Community 37"

Cohesion: 1.0
Nodes (1): Split text into overlapping chunks.     Tries to break at sentence boundaries,

### Community 38 - "Community 38"

Cohesion: 1.0
Nodes (1): Sentence splitter that respects common medical abbreviations.

### Community 39 - "Community 39"

Cohesion: 1.0
Nodes (1): Create the Qdrant collection with dense + sparse vectors if it doesn't exist.

### Community 40 - "Community 40"

Cohesion: 1.0
Nodes (1): Batch upsert embedded chunks into Qdrant.

### Community 41 - "Community 41"

Cohesion: 1.0
Nodes (1): Semantic search with optional source filter.

### Community 42 - "Community 42"

Cohesion: 1.0
Nodes (1): Build a simple sparse vector from text using term frequencies.     Maps term ha

### Community 43 - "Community 43"

Cohesion: 1.0
Nodes (1): Hybrid search — combine dense semantic search with sparse keyword search.     F

### Community 44 - "Community 44"

Cohesion: 1.0
Nodes (1): _RawArrayBSONDocument

### Community 45 - "Community 45"

Cohesion: 1.0
Nodes (1): _DocumentWithState

### Community 46 - "Community 46"

Cohesion: 1.0
Nodes (1): IndexedDocument

### Community 47 - "Community 47"

Cohesion: 1.0
Nodes (1): PDF

### Community 48 - "Community 48"

Cohesion: 1.0
Nodes (1): CSV

### Community 49 - "Community 49"

Cohesion: 1.0
Nodes (1): DOCX

### Community 50 - "Community 50"

Cohesion: 1.0
Nodes (1): URL

### Community 51 - "Community 51"

Cohesion: 1.0
Nodes (1): SentenceLevelPDF

### Community 52 - "Community 52"

Cohesion: 1.0
Nodes (1): SentenceLevelDOCX

### Community 53 - "Community 53"

Cohesion: 1.0
Nodes (1): Unstructured

### Community 54 - "Community 54"

Cohesion: 1.0
Nodes (1): InMemoryText

### Community 55 - "Community 55"

Cohesion: 1.0
Nodes (1): _HashedDocument

### Community 56 - "Community 56"

Cohesion: 1.0
Nodes (1): DocumentTooLarge

### Community 57 - "Community 57"

Cohesion: 1.0
Nodes (1): TestDocument

### Community 58 - "Community 58"

Cohesion: 1.0
Nodes (1): object_document

### Community 59 - "Community 59"

Cohesion: 1.0
Nodes (1): BitmapDocument

### Community 60 - "Community 60"

Cohesion: 1.0
Nodes (1): EditorDocumentBase

### Community 61 - "Community 61"

Cohesion: 1.0
Nodes (1): EditorDocument

### Community 62 - "Community 62"

Cohesion: 1.0
Nodes (1): CScintillaDocument

### Community 63 - "Community 63"

Cohesion: 1.0
Nodes (1): BrowserDocument

### Community 64 - "Community 64"

Cohesion: 1.0
Nodes (1): RegDocument

### Community 65 - "Community 65"

Cohesion: 1.0
Nodes (1): DebugDocumentText

## Knowledge Gaps
- **147 isolated node(s):** `Config`, `Evidence level boost scores for ranking.`, `Recency boost scores for ranking.`, `Prevents duplicate documents from being re-ingested.          Checks:     1. doc`, `Check if document should be skipped or re-processed.                  Returns:` (+142 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Community 16`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 17`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 18`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 19`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 20`** (1 nodes): `celery_app.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 21`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 22`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 23`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 24`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 25`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 26`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 27`** (1 nodes): `Remove all v1-ingested ICMR documents and their chunks from MongoDB and Qdrant.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 28`** (1 nodes): `Mongo store for v2 ingestion artifacts.      - documents_v2: full parsed docum`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 29`** (1 nodes): `Extract medical entities from text.     Returns dict with diseases, drugs, symp`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 30`** (1 nodes): `Classify chunk content type and return (content_type, weight).      Returns:`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 31`** (1 nodes): `Infer study type and evidence level from text/title.     Returns (study_type, e`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 32`** (1 nodes): `Cross-Encoder Reranker Reranks retrieved chunks by true relevance to the query.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 33`** (1 nodes): `Rerank chunks by relevance to query using cross-encoder.      Args:         q`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 34`** (1 nodes): `Query Rewriter Rewrites raw doctor queries into expanded medical queries before`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 35`** (1 nodes): `Rewrite a raw doctor query into an expanded medical query.     Returns the orig`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 36`** (1 nodes): `Medical Text Chunker Section-aware chunking — keeps tables, dosage info, and gu`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 37`** (1 nodes): `Split text into overlapping chunks.     Tries to break at sentence boundaries,`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 38`** (1 nodes): `Sentence splitter that respects common medical abbreviations.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 39`** (1 nodes): `Create the Qdrant collection with dense + sparse vectors if it doesn't exist.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 40`** (1 nodes): `Batch upsert embedded chunks into Qdrant.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 41`** (1 nodes): `Semantic search with optional source filter.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 42`** (1 nodes): `Build a simple sparse vector from text using term frequencies.     Maps term ha`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 43`** (1 nodes): `Hybrid search — combine dense semantic search with sparse keyword search.     F`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 44`** (1 nodes): `_RawArrayBSONDocument`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 45`** (1 nodes): `_DocumentWithState`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 46`** (1 nodes): `IndexedDocument`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 47`** (1 nodes): `PDF`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 48`** (1 nodes): `CSV`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 49`** (1 nodes): `DOCX`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 50`** (1 nodes): `URL`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 51`** (1 nodes): `SentenceLevelPDF`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 52`** (1 nodes): `SentenceLevelDOCX`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 53`** (1 nodes): `Unstructured`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 54`** (1 nodes): `InMemoryText`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 55`** (1 nodes): `_HashedDocument`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 56`** (1 nodes): `DocumentTooLarge`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 57`** (1 nodes): `TestDocument`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 58`** (1 nodes): `object_document`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 59`** (1 nodes): `BitmapDocument`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 60`** (1 nodes): `EditorDocumentBase`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 61`** (1 nodes): `EditorDocument`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 62`** (1 nodes): `CScintillaDocument`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 63`** (1 nodes): `BrowserDocument`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 64`** (1 nodes): `RegDocument`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 65`** (1 nodes): `DebugDocumentText`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.