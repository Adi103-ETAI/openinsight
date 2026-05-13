# OpenInsight Errors Summary

Date: May 13, 2026

## RESOLVED - Backend Errors Fixed

1) RequestIDMiddleware initialization error ✅ FIXED
- Fixed by adding `__init__(self, app)` inheriting from `BaseHTTPMiddleware`

2) Hallucination detection crash ✅ FIXED
- Fixed by changing `model.encode()` to `model.embed_batch()`

## Remaining Issues

### Runtime Warnings (Non-blocking)

1) Transformers warning
- Message: `FutureWarning: clean_up_tokenization_spaces was not set...`
- Source: `transformers/tokenization_utils_base.py`
- Impact: Warning only; does not crash.

2) Uvicorn lifespan warning
- Message: `ASGI 'lifespan' protocol appears unsupported.`
- Impact: Warning only; startup continues.

## Comprehensive Code Review Completed

A full codebase audit was conducted with 6 parallel code reviewers analyzing ~80 files:

- **API Layer**: 17 bugs fixed
- **Data Ingestion**: 13 bugs fixed
- **Query Pipeline**: 13 bugs fixed
- **Services & ML**: 15 bugs fixed
- **Vector Store & Config**: 12 bugs fixed
- **Parsers**: 21 bugs fixed
- **Total**: ~91 bugs identified and fixed

Key fixes include:
- Race conditions in singletons and component initialization
- Thread-safety issues in metrics and registry
- Missing error handling in cache, reranker, validators
- Non-deterministic hash causing data corruption
- Parser logic errors in table handling and file processing
