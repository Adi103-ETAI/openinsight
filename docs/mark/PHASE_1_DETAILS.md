# PHASE 1: RECOVERY - Detailed Implementation Guide

## Overview
Phase 1 fixes the broken OpenInsight system after the v3 ingestion pipeline upgrade. All changes are made to the **backend** (`/workspaces/openinsight`).

**Duration:** 4 hours  
**Goal:** Get system running and responding to queries

---

## Step 1.1: Fix Environment Variables (5 minutes)

### What's wrong?
The system is missing critical API keys required by the v3 ingestion pipeline.

### What to do?
Edit `/workspaces/openinsight/.env` file:

```bash
# Set these API keys (currently missing or wrong):
NVIDIA_NIM_API_KEY=nvapi-YOUR_ACTUAL_KEY_HERE
NCBI_API_KEY=YOUR_NCBI_KEY_HERE
NCBI_EMAIL=your.email@example.com
```

### Where to get the keys?
- **NVIDIA_NIM_API_KEY**: https://build.nvidia.com/ (free tier available)
- **NCBI_API_KEY**: https://www.ncbi.nlm.nih.gov/account/ (free registration)
- **NCBI_EMAIL**: Your email address for the NCBI account

### How to verify?
```bash
# These should NOT be empty
grep "NVIDIA_NIM_API_KEY" /workspaces/openinsight/.env | grep -v "YOUR_"
grep "NCBI_API_KEY" /workspaces/openinsight/.env | grep -v "YOUR_"
```

---

## Step 1.2: Verify All Services are Running (5 minutes)

### What services need to run?
1. **MongoDB** - Document database
2. **Qdrant** - Vector database
3. **Redis** - Cache (optional but recommended)

### How to check?

**MongoDB:**
```bash
mongo mongodb://localhost:27017/test --eval "db.adminCommand('ping')"
# Expected output: { "ok" : 1 }
```

**Qdrant:**
```bash
curl http://localhost:6333/health
# Expected: HTTP 200 with status "ok"
```

**Redis:**
```bash
redis-cli ping
# Expected: PONG
```

### If services are not running?
```bash
# Restart Docker containers
docker-compose restart mongodb qdrant redis

# Or manually start them
docker run -d -p 27017:27017 mongo:7
docker run -d -p 6333:6333 qdrant/qdrant
docker run -d -p 6379:6379 redis:7-alpine
```

---

## Step 1.3: Add Model Warmup at Startup (1 hour)

### What problem does this solve?
- First query takes 9-13 seconds (models load on demand)
- Solution: Preload models when backend starts
- Result: First query responds in <1 second

### What needs to be created?

**File:** `src/core/startup.py`

**Purpose:** Create a FastAPI lifespan context manager that preloads embedding and reranker models on app startup.

**What it should do:**
1. When app starts:
   - Load S-PubMedBert embedding model
   - Load BGE reranker model
   - Log success messages
2. When app shuts down:
   - Clean up resources

**How to integrate into main.py:**
1. Import the lifespan function
2. Pass it to FastAPI() app creation
3. Restart backend

---

## Step 1.4: Add Health Check Endpoints (30 minutes)

### What problem does this solve?
- No way to verify all services are working
- Solution: Add health check endpoints
- Result: Can test MongoDB, Qdrant, Redis, NIM API, models

### What needs to be created?

**File:** `src/api/routes/health.py`

**Purpose:** Create comprehensive health check endpoints

**Endpoints needed:**
1. `GET /health` - Quick health check
   - Returns: `{"status": "ok", "service": "openinsight-api"}`

2. `GET /health/detailed` - Full service validation
   - Tests MongoDB connectivity
   - Tests Qdrant connectivity
   - Tests Redis connectivity
   - Tests NVIDIA NIM API reachability
   - Tests if models can be loaded
   - Returns: All results in JSON

**How to integrate:**
1. Create new file `src/api/routes/health.py`
2. Create router with above endpoints
3. Register in `src/api/main.py` with `app.include_router()`
4. Test: `curl http://localhost:8000/health/detailed`

---

## Step 1.5: Handle Database Schema Migration (1.5 hours)

### What changed in v3?
The ingestion pipeline v3 added 25+ new fields to the data models:
- ChunkRecord now has: quality_score, dosages, contraindications, etc.
- DocumentRecord now has: content_hash, is_duplicate, duplicate_of, etc.

### What problem could this cause?
- Old documents in MongoDB missing new fields
- Queries might fail trying to access missing fields

### What to do?
1. **Option A (Recommended):** Let Pydantic handle defaults
   - Old documents will use default values for new fields
   - No migration needed
   - New documents will have all v3 fields

2. **Option B (Better quality):** Re-ingest documents
   - Run ingestion pipeline on existing documents
   - New fields will be populated correctly
   - Better quality scores and entity extraction

3. **Testing needed:**
   ```bash
   # Verify MongoDB can still query chunks
   db.chunks.find({}).limit(1)
   
   # Try a sample query
   curl -X POST http://localhost:8000/query \
     -H "Content-Type: application/json" \
     -d '{"query": "test", "top_k": 8}'
   ```

---

## ✅ Phase 1 Complete Checklist

- [ ] `.env` file has all three API keys (no placeholder values)
- [ ] `mongo` command connects to MongoDB
- [ ] `curl http://localhost:6333/health` returns 200
- [ ] `redis-cli ping` returns PONG
- [ ] Created `src/core/startup.py` for model warmup
- [ ] Created `src/api/routes/health.py` for health checks
- [ ] Updated `src/api/main.py` to use lifespan and register health routes
- [ ] Backend starts without errors
- [ ] `/health` endpoint returns `{"status":"ok"}`
- [ ] `/health/detailed` shows all services connected
- [ ] Query endpoint returns results in 4-8 seconds
- [ ] No 500 errors in logs
- [ ] First query responds in <1 second (models preloaded)

---

## Troubleshooting Phase 1

### Error: "NVIDIA_NIM_API_KEY not set"
**Solution:** Set the key in `.env` file

### Error: "Cannot connect to MongoDB"
**Solution:** Start MongoDB container or verify connection string

### Error: "Models failed to load at startup"
**Solution:** Check disk space, RAM available, and internet connection for model downloads

### Error: "502 Bad Gateway" from frontend
**Solution:** Backend is not running or not accessible. Check port 8000 is open.

---

## Expected Results After Phase 1

✅ System is running  
✅ All services connected  
✅ Health checks passing  
✅ Query endpoint responding  
✅ First request: <1 second (models preloaded)  
✅ Normal queries: 4-8 seconds  
✅ No errors in logs  

**Proceed to Phase 2 for caching optimization →**
