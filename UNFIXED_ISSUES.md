# Unfixed Issues - System Review

> This document lists all issues identified in the original system review that have NOT been addressed by the requested fixes.

## Summary

The following fixes were requested:
- **Backend Architect**: Fix duplicate config, add input validation, request tracing
- **AI Engineer**: Fix parsing hardcoded, chunking (main), embedding sparse vocab, vector db, hallucination
- **Data Engineer**: Fix input sanitization on search

This document captures everything else that remains unfixed.

---

## Security Issues (Not Addressed)

### Critical

- **[Critical] No API Authentication** - All API endpoints are publicly accessible without any authentication mechanism. Any client can query the clinical decision support system without authorization.

- **[Critical] CORS Allows All Origins** - The API accepts requests from any origin (`allow_origins=["*"]` in `src/api/main.py` line 48-52). This allows cross-origin requests from any website, potentially exposing sensitive clinical data.

### High

- **[High] No Rate Limiting** - No rate limiting implemented on search endpoints. Vulnerable to abuse, denial-of-service attacks, and resource exhaustion.

---

## Backend Issues (Not Addressed)

### High

- **[High] No Request Tracing/Observability** - No distributed tracing, request logging, or observability middleware. Cannot track request flows across the system, making debugging and monitoring difficult. Requested as "request tracing" but not implemented.

- **[High] No Distributed Ingestion** - Sequential batch processing caps throughput at ~10 docs/sec. Cannot ingest >500K documents efficiently. Single point of failure: if ingestion worker crashes, entire process restarts.

- **[High] Error Recovery Not Implemented** - Failed documents are logged but not retried. No dead-letter queue for problematic files. No vector validation post-generation. No automatic retry for PDF → OCR fallback.

### Medium

- **[Medium] No Checkpoint/Resume for Ingestion** - Full restart required on failure. No way to resume from last successful batch.

- **[Medium] Thread Pool Limitation** - Only 4 workers (bottleneck on 8-core+ machines). In-memory accumulation of all chunks until batch complete.

- **[Medium] Single MongoDB Connection Per Batch** - Not pooled, causing connection overhead.

### Low

- **[Low] Duplicate Config Exists** - Deduplication settings appear twice in config.py (lines 108-112 and 114-118). Note: This was requested to be fixed but may not have been addressed yet.

---

## AI/ML Issues (Not Addressed)

### High

- **[High] Hardcoded Medical Compounds (Limited Coverage)** - Only 40 medical compounds hardcoded in `src/ingestion/embedder_v2.py` (lines 21-40). Missing many important clinical terms (e.g., "chronic kidney disease", "acute coronary syndrome", "dengue hemorrhagic fever").

- **[High] Limited Medical Synonyms** - Only 6 synonym mappings in `src/query/search/query_understanding.py` (lines 77-84). Query understanding fails for many medical abbreviations (e.g., "HTN", "DM", "CAD" not fully mapped).

- **[High] No Evidence Contradiction Detection** - RRF fusion treats conflicting evidence equally. Example: Study A says "metformin first-line", Study B says "insulin first-line" - both treated equally without flagging contradiction. Safety risk in clinical context.

### Medium

- **[Medium] Regex-Based Intent Classification** - Pattern matching in `query_understanding.py` is brittle. Limited synonym coverage means ~70-75% accuracy on novel queries. No confidence scoring for uncertain classifications.

- **[Medium] MMR Algorithm Limitations** - Lambda tuning is manual (0.7 may be suboptimal). Embedding-space diversity != semantic non-redundancy. Doesn't detect contradictions between studies.

- **[Medium] Reranker Bottleneck** - Single GPU inference with sequential processing. Top-20 truncation hides potentially better results. Fixed max_length=512 truncates longer clinical documents.

- **[Medium] Hardcoded Stopwords** - Only 46 stopwords in `src/ingestion/embedder_v2.py` (lines 42-94). Limited coverage for medical text processing.

- **[Medium] Hardcoded Abbreviations in Chunker** - Only 8 abbreviations handled in `src/ingestion/chunker_v3.py` (lines 38-47). Missing many medical abbreviations (e.g., "b.i.d.", "t.i.d.", "q.i.d.", "p.r.n.", "a.c.", "p.c.").

### Low

- **[Low] No Cross-lingual Support** - Medical Hindi/Tamil/Marathi queries unsupported. Cannot handle regional medical terminology.

- **[Low] No Query Confidence Scoring** - Can't flag uncertain intent classification or low-confidence entity extraction.

- **[Low] Sparse Vector Hash Collisions** - `hash(term) % 50000` causes estimated ~20K collisions per 1M unique terms. Weight merging mitigates but imperfect.

---

## Data Pipeline Issues (Not Addressed)

### High

- **[High] Cache Stampede Risk** - Concurrent identical queries cause multiple cache misses. All request threads hit vector DB simultaneously, causing latency spikes. Expected in ~5-10% of traffic.

### Medium

- **[Medium] Metadata Filter Explosion** - Dynamic filter construction could exceed Milvus limits. Large filter expressions slow parsing. Potential stack overflow on deeply nested OR/AND.

- **[Medium] Batch Processing Bottleneck** - Sequential batch processing. No async document parsing. ~10 docs/sec vs. 30-50 docs/sec potential.

- **[Medium] Query Performance Regression at Scale** - At 500K+ chunks, dense search latency grows exponentially (1200ms+). Reranker latency significant (500-800ms).

### Low

- **[Low] No Approximate Counting** - Missing optimization for result set size estimation in vector DB.

- **[Low] Query Result Caching Within Vector DB** - Only Redis-level caching, no Milvus-native result caching.

---

## Validation/Hallucination Issues (Not Addressed)

### Medium

- **[Medium] Hallucination Detection Limitations** - Uses simple pattern-based entity extraction in `src/query/validation/hallucination_detector.py`. Limited drug suffix detection may miss non-standard drug names. Similarity threshold of 0.45 may flag legitimate but novel medical terms as hallucinations.

- **[Medium] Citation Validation Gaps** - Fuzzy matching on citation text may miss properly cited content. No verification that cited information matches source context.

### Low

- **[Low] Confidence Score Accuracy** - Multi-factor confidence calculation may not reflect true answer reliability. No mechanism to learn from user feedback.

---

## Input Validation Status

The following was requested by Data Engineer but may not be fully addressed:

- **[Medium] Input Sanitization Inconsistency** - While `src/api/routes/search.py` has `sanitize_query()` and `validate_query_safety()`, other endpoints may lack similar validation. Filter value sanitization exists in `milvus_store.py` but query-level sanitization may be incomplete.

---

## Priority Recommendations

### Immediate (Critical)

1. Add API authentication
2. Restrict CORS to specific origins
3. Add request tracing/observability
4. Implement rate limiting

### Short-term (High)

5. Expand medical compounds vocabulary (target: 200+ terms)
6. Add more medical synonyms (target: 50+ mappings)
7. Implement evidence contradiction detection
8. Add distributed ingestion workers

### Medium-term

9. Implement cache stampede protection
10. Expand stopwords for medical text
11. Add cross-lingual NER support
12. Optimize reranker with batch processing

---

## File Reference Guide

| Issue | File Location |
|-------|---------------|
| CORS open | `src/api/main.py:48-52` |
| No request tracing | `src/api/main.py` |
| Duplicate config | `src/core/config.py:108-118` |
| Hardcoded compounds | `src/ingestion/embedder_v2.py:21-40` |
| Limited synonyms | `src/query/search/query_understanding.py:77-84` |
| No contradiction detection | `src/query/contradiction_detector.py` (exists but not integrated) |
| Cache stampede | `src/query/search/cache.py` |
| Input sanitization | `src/api/routes/search.py:33-76` |
| Hallucination detector | `src/query/validation/hallucination_detector.py` |

---

*Generated from system review analysis - May 2026*