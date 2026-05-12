# Error Inventory

This list captures concrete runtime errors or likely runtime errors found via code inspection, plus references to general fix guidance.

## FIXED - Confirmed runtime error

### KeyError: 'document_stored' ✅ FIXED

- **Symptom:** `KeyError: 'document_stored'` during ingestion summary access.
- **Root cause:** Summary keys use `documents_stored` (plural), but a caller accesses `document_stored` (singular).
- **Evidence:** Summary definition in `src/ingestion/pipeline.py:315` defines `documents_stored`.
- **Fix:** Fixed by simplifying CLI - removed unsupported kwargs from run_ingestion.py

## FIXED - Likely runtime errors

### TypeError: unexpected keyword arguments to `ingest_directory` ✅ FIXED

- **Symptom:** `TypeError: ingest_directory() got an unexpected keyword argument 'skip_embed'` or `skip_index`.
- **Root cause:** `run_ingestion.py` passes `skip_embed`/`skip_index` to `ingest_directory`, but the method signature does not accept these kwargs.
- **Evidence:**
  - Call sites: `src/ingestion/run_ingestion.py:163`, `src/ingestion/run_ingestion.py:168`, `src/ingestion/run_ingestion.py:173`
  - Signature: `src/ingestion/pipeline.py:275`
- **Fix:** Removed the unsupported kwargs from CLI calls in run_ingestion.py

### ImportError: wrong module path for `get_nim_client` ✅ FIXED

- **Symptom:** `ImportError` when importing `src.api.routes.search`.
- **Root cause:** `src/api/routes/search.py` imports `get_nim_client` from `src.utils.llm_client`, but the file lives at `src/services/llm_client.py`.
- **Evidence:** `src/api/routes/search.py:19` and `src/services/llm_client.py:1`.
- **Fix:** Updated import to `from src.services.llm_client import get_nim_client`.

### NameError: missing `re` import ✅ FIXED

- **Symptom:** `NameError: name 're' is not defined` at module import.
- **Root cause:** `_DANGEROUS_PATTERNS = re.compile(...)` without importing `re`.
- **Evidence:** `src/api/routes/search.py:25`.
- **Fix:** Added `import re` to the file.

### TypeError: asyncio Lock used as callable ✅ FIXED

- **Symptom:** `TypeError: 'Lock' object is not callable`.
- **Root cause:** `async with self._lock():` uses a lock as a function.
- **Evidence:** `src/utils/metrics.py:106`.
- **Fix:** Changed to `async with self._lock:` (removed parentheses).

### TypeError: awaiting non-async Redis client creation ✅ FIXED

- **Symptom:** `TypeError: object Redis can't be used in 'await' expression`.
- **Root cause:** `aioredis.from_url` returns a client synchronously but is awaited.
- **Evidence:** `src/utils/metrics.py:215`.
- **Fix:** Removed `await` from `aioredis.from_url(...)`.

## NEW ISSUES FOUND (Agent Scans)

### 🔴 CRITICAL - RecursionError: Unbounded XML OCR Fallback ✅ FIXED

- **Symptom:** `RecursionError` when parsing `.xml` documents.
- **Root cause:** `_parse_with_ocr_fallback()` calls `_parse_with_retry()` which calls `_parse_with_ocr_fallback()` again for XML.
- **Evidence:** `src/ingestion/pipeline.py:157` and `src/ingestion/pipeline.py:242`.
- **Fix:** Changed XML parsing to use direct primary parser without recursion.

### 🔴 CRITICAL - ImportError: Wrong Test Import Names ✅ FIXED

- **Symptom:** pytest import error in `tests/test_answer_validation.py`.
- **Root cause:** Tests import `MedicalSafetyResult` and `check_medical_safety`, but `medical_safety.py` defines `SafetyCheckResult` and `check_safety`.
- **Evidence:**
  - `tests/test_answer_validation.py:22-25` - wrong imports
  - `tests/test_answer_validation.py:374` - enhance_response not imported
  - `tests/test_answer_validation.py:404` - SafetyCheckResult not imported
- **Fix:** Updated test imports to use correct names (SafetyCheckResult, check_safety, enhance_response).

### 🔴 CRITICAL - Year Hardcoded as 2025 (Wrong Boost for 2026) ✅ FIXED

- **Symptom:** Documents from 2026 get wrong boost (PRE_2022 instead of current year).
- **Root cause:** `YEAR_2025` hardcoded but today is 2026.
- **Evidence:** `src/constants/__init__.45-46`.
- **Fix:** Added YEAR_2026 with boost value 1.12.

### 🔴 CRITICAL - Wrong Collection Names in Monitoring ✅ FIXED

- **Symptom:** Storage stats always return zero.
- **Root cause:** Queries `documents` and `chunks` but actual collections are `documents_v2` and `chunks_v2`.
- **Evidence:** `src/ingestion/monitoring.py:93-103`.
- **Fix:** Updated collection names to v2 variants.

### 🟡 HIGH - Hash Algorithm Inconsistency in Deduplication ✅ FIXED

- **Symptom:** Deduplication never finds matches.
- **Root cause:** dedupe.py uses truncated hash (16 chars), deduplication.py uses full hash.
- **Evidence:** `src/ingestion/dedupe.py:73` vs `src/ingestion/deduplication.py:19`.
- **Fix:** Changed to use full hash (no truncation).

### 🟡 HIGH - Field Name Mismatch (token_count vs token_estimate) - NOT YET FIXED

- **Symptom:** Token counts may be wrong or missing.
- **Root cause:** document_db.py expects `token_count`, but doc_store.py stores `token_estimate`.
- **Evidence:** `src/ingestion/document_db.py:82` vs `src/data/mongo/doc_store.py:91`.
- **Fix:** Align field names.

### 🟡 HIGH - Content Hash Never Persisted - NOT YET FIXED

- **Symptom:** Deduplication check always fails.
- **Root cause:** Pipeline never populates `content_hash` field.
- **Evidence:** `src/ingestion/pipeline.py:516`.
- **Fix:** Store content_hash when saving documents.

### 🟡 MEDIUM - Missing Source Types in Validation - NOT YET FIXED

- **Symptom:** Documents from `nmc_guideline`, `rssdi`, `research` rejected.
- **Root cause:** validation.py only allows limited source types.
- **Evidence:** `src/ingestion/validation.py:15-18`.
- **Fix:** Add missing source types to VALID_SOURCE_TYPES.

### 🟡 MEDIUM - Duplicate get_embedder Functions - NOT YET FIXED

- **Symptom:** Unclear which embedder is used; potential type mismatch.
- **Root cause:** `get_embedder()` defined twice with different return types.
- **Evidence:** `src/ml/embedding/embedder.py:16-21` and `src/ml/embedding/embedder.py:209-220`.
- **Fix:** Rename second function or consolidate to single implementation.

### RecursionError: unbounded XML OCR fallback

- **Symptom:** `RecursionError` when parsing `.xml` documents.
- **Root cause:** `_parse_with_ocr_fallback()` calls `_parse_with_retry()` which calls `_parse_with_ocr_fallback()` again for XML.
- **Evidence:** `src/ingestion/pipeline.py:157` and `src/ingestion/pipeline.py:242`.
- **Fix:** Break the recursion; handle XML with a single parsing path or a base case.
- **Reference:** https://docs.python.org/3/library/exceptions.html#RecursionError

### RuntimeError: `asyncio.gather` used without a running loop

- **Symptom:** `RuntimeError: no running event loop`.
- **Root cause:** `asyncio.run(asyncio.gather(...))` is invalid; `gather` needs to run inside the loop.
- **Evidence:** `src/ingestion/tasks.py:175`.
- **Fix:** Wrap gather inside an async function and pass that to `asyncio.run`.
- **Reference:** https://docs.python.org/3/library/asyncio-task.html#asyncio.gather

### TypeError: asyncio Lock used as callable ✅ FIXED

- **Symptom:** `TypeError: 'Lock' object is not callable`.
- **Root cause:** `async with self._lock():` uses a lock as a function.
- **Evidence:** `src/utils/metrics.py:106`.
- **Fix:** Changed to `async with self._lock:` (removed parentheses).

### TypeError: awaiting non-async Redis client creation ✅ FIXED

- **Symptom:** `TypeError: object Redis can't be used in 'await' expression`.
- **Root cause:** `aioredis.from_url` returns a client synchronously but is awaited.
- **Evidence:** `src/utils/metrics.py:215`.
- **Fix:** Removed `await` from `aioredis.from_url(...)`.

## Type consistency risk (can become runtime error)

### `summary` dict mixes ints with strings

- **Symptom:** Consumers that assume numeric summary values may break or mis-compute.
- **Root cause:** `summary` is initialized with numeric values but later sets `summary["status"] = "already_completed"`.
- **Evidence:** `src/ingestion/pipeline.py:315` and `src/ingestion/pipeline.py:342`.
- **Fix:** Use a separate `status` field in a dedicated response object, or change the summary type to `dict[str, Any]` and adjust consumers.
- **Reference:** https://docs.python.org/3/library/typing.html

## Test failures (collection/import)

### ImportError: missing names in `medical_safety`

- **Symptom:** pytest import error in `tests/test_answer_validation.py`.
- **Root cause:** tests import `MedicalSafetyResult` and `check_medical_safety`, but `medical_safety.py` defines `SafetyCheckResult` and `check_safety`.
- **Evidence:** `tests/test_answer_validation.py:22` and `src/query/validation/medical_safety.py:24`.
- **Fix:** Update tests or export aliases in `medical_safety.py`.
- **Reference:** https://docs.pytest.org/en/stable/how-to/importpath.html

### NameError: `enhance_response` not imported

- **Symptom:** `NameError` during tests.
- **Root cause:** `enhance_response` used without import.
- **Evidence:** `tests/test_answer_validation.py:374`.
- **Fix:** Import `enhance_response` from its module.
- **Reference:** https://docs.python.org/3/tutorial/modules.html#modules

### NameError: `SafetyCheckResult` not imported

- **Symptom:** `NameError` during tests.
- **Root cause:** `SafetyCheckResult` referenced without import.
- **Evidence:** `tests/test_answer_validation.py:404`.
- **Fix:** Import `SafetyCheckResult` from `src.query.validation.medical_safety`.
- **Reference:** https://docs.python.org/3/tutorial/modules.html#modules

## Notes

- There is an existing backlog of system-review issues in `UNFIXED_ISSUES.md`. Those are not runtime exceptions but may represent important risks.
