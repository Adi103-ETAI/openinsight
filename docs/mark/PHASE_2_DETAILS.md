# PHASE 2: CACHING - Detailed Implementation Guide

## Overview
Phase 2 implements Redis caching to speed up queries by 80×. Medical queries often repeat, so we cache the results.

**Duration:** 12 hours  
**Goal:** Achieve <200ms response time for repeated queries (80× speedup)  
**Prerequisites:** Phase 1 must be complete

---

## Step 2.1: Create Redis Cache Module (2 hours)

### What problem does this solve?
- Queries take 4-8 seconds each
- Medical queries repeat frequently (users ask same questions multiple times)
- Solution: Cache responses in Redis for instant retrieval

### What needs to be created?

**File:** `src/cache/redis_client.py`

**Purpose:** Async Redis client for caching

**Features it should have:**
1. AsyncRedis connection with connection pooling
2. Methods for basic cache operations:
   - `get(key)` - Retrieve value from cache
   - `set(key, value, ttl)` - Store value with time-to-live
   - `delete(key)` - Remove key from cache
   - `clear_prefix(prefix)` - Clear all keys matching pattern (e.g., "answer:*")
3. Automatic JSON serialization/deserialization
4. Error handling with graceful fallback (if Redis unavailable, cache is disabled)
5. Statistics tracking:
   - Cache hits (how many successful retrievals)
   - Cache misses (how many not in cache)
   - Errors count
   - Hit rate percentage

**Configuration needed:**
- Redis URL from `.env` (e.g., `redis://localhost:6379`)
- Redis database number (e.g., 0 for queries, 1 for cache)
- Max connections in pool
- Connection timeout

**Usage example:**
```python
from src.cache.redis_client import redis_cache

# Initialize
await redis_cache.connect()

# Store data
await redis_cache.set("query:abc123", {"answer": "result"}, ttl=3600)

# Retrieve data
value = await redis_cache.get("query:abc123")

# Get statistics
stats = await redis_cache.get_stats()  # {"hits": 10, "misses": 2, "hit_rate": 83.3%}
```

---

## Step 2.2: Create Cache Decorators (2 hours)

### What problem does this solve?
- Want to cache different functions without rewriting code
- Solution: Create decorators that add caching transparently

### What needs to be created?

**File:** `src/cache/decorators.py`

**Purpose:** Decorators for transparent caching

**Decorators needed:**

1. **@cache_query_rewrite(ttl=86400)**
   - What it caches: Query rewrites (e.g., "drug resistant TB" → "tuberculosis resistance antimicrobial")
   - TTL: 24 hours (queries don't change often)
   - Saves: ~1 second per request

2. **@cache_embedding(ttl=604800)**
   - What it caches: Query embeddings (768-dimensional vectors)
   - TTL: 7 days
   - Saves: ~100ms per request

3. **@cache_search(ttl=43200)**
   - What it caches: Vector search results (list of matching chunks)
   - TTL: 12 hours
   - Saves: ~500ms per request

4. **@cache_response(ttl=604800)**
   - What it caches: Full API response (answer + citations)
   - TTL: 7 days
   - Saves: 4-8 seconds per request ← BIGGEST IMPACT!

**How they should work:**
1. Generate cache key from function arguments (hash of query/params)
2. Check if key exists in cache
3. If yes (hit): Return cached value instantly
4. If no (miss): Call the function, store result, return it
5. Handle errors gracefully (fall through to function if cache fails)

**Example usage:**
```python
@cache_response(ttl=604800)
async def query_endpoint(request: QueryRequest) -> QueryResponse:
    # First call: executes full pipeline, caches result
    # Subsequent calls with same query: returns from cache in <200ms
    return response
```

---

## Step 2.3: Integrate Caching into Query Pipeline (3 hours)

### What needs to be done?

**File to modify:** `src/api/routes/query.py` (or create `query_with_caching.py`)

**Changes:**

1. **Import cache client:**
   ```python
   from src.cache.redis_client import redis_cache
   ```

2. **Modify query endpoint function:**
   - Add cache check at the beginning
   - Generate cache key from query
   - If found in cache: return cached response
   - If not found: run full pipeline as normal
   - Store result in cache before returning

3. **Query flow with caching:**
   ```
   User Query → Check Cache → Hit? → Return (200ms) 
                    ↓ Miss
              Run Full Pipeline (4-8s)
                    ↓
              Store in Cache
                    ↓
              Return Response
   ```

4. **Cache key generation:**
   - Use MD5 hash of query string
   - Example: `answer:{md5('Amikacin resistance')} → answer:a1b2c3d4...`
   - Prevents duplicate long keys

**Testing needed:**
```bash
# First query (cache miss, slow)
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Amikacin resistance", "top_k": 8}'
# Expected: 4-8 seconds

# Second query (cache hit, fast!)
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Amikacin resistance", "top_k": 8}'
# Expected: <200ms ⚡
```

---

## Step 2.4: Add Cache Lifecycle Management (1.5 hours)

### What needs to be done?

**File to modify:** `src/api/main.py`

**Changes:**

1. **Import cache client:**
   ```python
   from src.cache.redis_client import redis_cache
   ```

2. **Add startup event:**
   ```python
   @app.on_event("startup")
   async def startup_event():
       await redis_cache.connect()
   ```

3. **Add shutdown event:**
   ```python
   @app.on_event("shutdown")
   async def shutdown_event():
       await redis_cache.disconnect()
   ```

**Purpose:**
- Initialize Redis connection when backend starts
- Clean up connection when backend shuts down
- Automatically handle reconnection if connection drops

---

## Step 2.5: Add Cache Management Endpoints (2 hours)

### What endpoints to add?

**File:** `src/api/routes/query_caching.py` (or add to existing query router)

**Endpoints:**

1. **GET /query/cache/stats**
   - Returns cache statistics
   - Response includes:
     - `hits`: Number of cache hits
     - `misses`: Number of cache misses
     - `hit_rate_percent`: Hit rate (0-100%)
     - `redis_memory_mb`: Memory used by Redis
     - `redis_status`: "connected" or "error"

2. **POST /query/cache/clear**
   - Clear all cache (optional prefix parameter)
   - Query: `/query/cache/clear?prefix=answer` (clear responses only)
   - Query: `/query/cache/clear?prefix=rewrite` (clear rewrites only)
   - No query: Clear entire cache

3. **POST /query/cache/enable**
   - Re-enable caching (useful for debugging)

4. **POST /query/cache/disable**
   - Disable caching temporarily (useful for debugging)

**Usage:**
```bash
# View cache stats
curl http://localhost:8000/query/cache/stats
# Expected: {"hits": 42, "misses": 8, "hit_rate_percent": 84.0, ...}

# Clear all cache
curl -X POST http://localhost:8000/query/cache/clear

# Clear only response cache
curl -X POST "http://localhost:8000/query/cache/clear?prefix=answer"
```

---

## Step 2.6: Add Monitoring & Logging (1.5 hours)

### What logging to add?

**In cache operations:**
- Log cache hits: `💚 Cache HIT: {key}`
- Log cache misses: `💔 Cache MISS: {key}`
- Log storage: `💾 Cached: {key} (TTL: {ttl}s)`
- Log errors: `❌ Cache error: {error}`

**In query endpoint:**
- Log response times
- Log whether result came from cache or pipeline
- Log cache statistics periodically

**Dashboard/metrics needed:**
- Cache hit/miss rate over time
- Average response times (cache hits vs misses)
- Redis memory usage trend
- Most frequently cached queries

---

## Step 2.7: Integration into main.py (1 hour)

### Changes to make:

**File:** `src/api/main.py`

**Add these imports:**
```python
from src.cache.redis_client import redis_cache
from src.api.routes import query_caching  # or query_with_caching
```

**Update FastAPI app setup:**
```python
app = FastAPI(
    title="OpenInsight API",
    version="0.2.0",  # Version bump
    # ... other config
)

# Add startup/shutdown events for cache
@app.on_event("startup")
async def startup_event():
    await redis_cache.connect()

@app.on_event("shutdown")
async def shutdown_event():
    await redis_cache.disconnect()

# Register query router with caching
app.include_router(query_caching.router, prefix="/query", tags=["Query"])
```

---

## ✅ Phase 2 Complete Checklist

- [ ] Created `src/cache/redis_client.py` with AsyncRedis client
- [ ] Created `src/cache/decorators.py` with cache decorators
- [ ] Created/modified `src/api/routes/query_caching.py` with caching logic
- [ ] Updated `src/api/main.py` with cache lifecycle
- [ ] Redis connects successfully on startup
- [ ] First query caches correctly
- [ ] Second identical query returns <200ms ⚡
- [ ] `GET /query/cache/stats` returns statistics
- [ ] `POST /query/cache/clear` works
- [ ] Cache hit rate ≥60%
- [ ] No performance degradation for cache misses
- [ ] Graceful fallback if Redis unavailable
- [ ] Logging shows cache operations
- [ ] All endpoints tested and working

---

## Performance Validation

### Expected Results:

**Without cache (Phase 1):**
```
Query 1: 4.2 seconds
Query 2: 4.1 seconds
Query 3: 4.3 seconds
Average: 4.2 seconds per query
Throughput: ~1 query/second
```

**With cache (Phase 2):**
```
Query 1 (miss):  4.2 seconds (stored in cache)
Query 2 (hit):   0.15 seconds ⚡ (28× faster!)
Query 3 (hit):   0.12 seconds ⚡ (35× faster!)
Average (80% hit rate): 1.5 seconds
Throughput: 10-20 queries/second
```

### How to validate:

```bash
# Run 10 identical queries and measure time
for i in {1..10}; do
  time curl -X POST http://localhost:8000/query \
    -H "Content-Type: application/json" \
    -d '{"query": "Amikacin resistance", "top_k": 8}'
  sleep 0.5
done

# First should be slow (~4-8s), rest should be fast (~0.2s)
# Check cache stats
curl http://localhost:8000/query/cache/stats | jq .
# Expected: hit_rate_percent > 80
```

---

## Troubleshooting Phase 2

### Redis not connecting?
- Check Redis is running: `redis-cli ping`
- Check URL in `.env`: `REDIS_URL=redis://localhost:6379`
- Restart Redis: `docker-compose restart redis`

### Cache not working?
- Check logs for connection errors
- Verify `/query/cache/stats` shows "connected"
- Check cache is enabled: `USE_CACHE=true` in `.env`
- Clear cache: `POST /query/cache/clear`

### Hit rate too low?
- Ensure users are making repeated queries
- Check cache TTL (might be expiring too fast)
- Verify cache key generation logic

### Performance got worse?
- Redis might be down (falls back to no-cache pipeline)
- Check Redis memory usage
- Make sure models still preload at startup (Phase 1)

---

## Expected Results After Phase 2

✅ Cache module working  
✅ Decorators functioning  
✅ Query endpoint caching responses  
✅ Cache statistics available  
✅ Repeated queries: <200ms ⚡  
✅ Hit rate: 60-80%  
✅ No performance penalty for cache misses  
✅ Graceful fallback if Redis unavailable  
✅ Production ready!  

**Total speedup: 8-80× depending on query patterns**
