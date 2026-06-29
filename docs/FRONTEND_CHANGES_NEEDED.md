# Frontend Changes Needed

Living document tracking what the frontend (`openinsight-ui` repo, `nextjs-ui` branch) needs to do to stay in sync with backend changes. Updated after each backend phase.

**Last updated**: After Phase 0 (commit `3330693` on `phase0-stabilize` branch, PR #10)
**Frontend branch to work on**: `phase0-frontend-cleanup` (create off `nextjs-ui`)

---

## Phase 0 Backend — What Changed That Affects Frontend

### P0.1 — `/deep-insights` endpoint is now functional

**Before**: `POST /deep-insights` returned 500 on every call (orchestrator signature mismatch + missing response fields).

**After**: Endpoint works. Response shape (Pydantic model `DeepInsightsResponse`):

```typescript
interface DeepInsightsResponse {
  answer: string;
  sections: Record<string, string>;        // named sections: diagnosis, treatment, dosage, etc.
  citations: Citation[];
  sub_queries: Array<{                      // structured sub-query results
    id: string;
    query: string;
    focus: string;                          // "drug_interactions" | "dosage" | etc.
    chunks_retrieved: number;
    error: string | null;
  }>;
  contradictions: Array<{
    type: string;
    evidence: string;
    chunk_a_title: string;
    chunk_b_title: string;
  }>;
  confidence: number;                       // 0.0–1.0
  complexity_detected: string;              // "simple" | "medium" | "complex"
  processing_time_ms: number;
  mode: "deep_insights";
}
```

**Frontend action**: Wire up a "Deep Insights" toggle/button in `IndexView.tsx`. When toggled, POST to `/deep-insights` instead of `/search`. Render:
- Sub-queries as an expandable "How we approached this" panel
- Contradictions as warning callouts (⚠ icon, amber color)
- Confidence as a badge (high/medium/low based on score thresholds)
- Processing time as "(took 4.2s)" subtle text

### P0.2 — `/search/document` endpoint is now functional

**Before**: 500 on every successful PDF/DOCX request (missing `Response` import).

**After**: Endpoint works. `POST /search/document` with body:
```json
{
  "query": "metformin dosing in renal impairment",
  "format": "pdf",                          // "pdf" | "docx"
  "title": "Optional custom title"
}
```
Returns binary file with appropriate `Content-Type` and `Content-Disposition: attachment; filename="..."` headers.

**Frontend action**: Add a "Download as PDF" / "Download as DOCX" button in `AnswerCard.tsx` action toolbar. On click, POST to `/api/search/document` (proxied) or directly to `${API_BASE}/search/document`, trigger browser download from the response blob.

### P0.3 — Scraper framework added (no API surface change yet)

Backend-only infrastructure. **No frontend action needed** — but note that Phase 1 will add new `source_type` values (see below).

### P0.4 — Expanded `INDIAN_JOURNALS` list

`indian_source: true` flag now correctly set on 30+ Indian journals (was 5). Affects retrieval relevance ranking — Indian-context queries should now return more India-specific results.

**Frontend action**: None directly. But verify that `SourceType` in `src/types/api.ts` and the color map in `src/lib/sources.ts` are ready for the new types coming in Phase 1 (below).

### P0.5 — `requirements.txt` changes

Backend-only. No frontend impact.

### P0.6 — `mypy.ini` added

Backend-only. No frontend impact.

### P0.7 — GROBID version bumped 0.8.0 → 0.9.0

Backend-only. No frontend impact.

---

## Phase 1 Backend (in progress) — Indian Journals

### P1.1 — New `source_type` values

The frontend `SourceType` union currently is:
```typescript
export type SourceType = "icmr" | "pubmed" | "cochrane" | "who" | "cdc" | "statpearls" | "nmc";
```

Phase 1 will add these source types (and Phase 2-5 will add more):
```typescript
// Phase 1 — Indian journals
"ijmr" | "nmji" | "japi" | "jima" | "indmed" | "medknow" | "pmc_india"

// Phase 2 — Foundational (open-access)
"statpearls"            // already in list, but will be expanded
"ncbi_bookshelf"        // NEW
"nmc_curriculum"        // NEW
"ntep" | "nvbdcp"       // NEW (govt manuals — cross Layer 1+3)

// Phase 3 — Drug/regulatory
"nfi" | "cdsco" | "ctri" | "pvpi" | "ipc"  // NEW

// Phase 4 — Specialty guidelines
"rssdi" | "csi"         // NEW

// Phase 5 — Epidemiology
"nfhs" | "ncdir"        // NEW
```

**Frontend action** (do this when Phase 1 lands):
1. Update `src/types/api.ts` `SourceType` to include all new types
2. Update `src/lib/sources.ts` — add `{ label, colorClass }` for each new source
3. Update `tailwind.config.ts` — add new `source-*` color entries (lines 69-77)
4. Update `src/index.css` — add CSS variable declarations for new source colors
5. Update `CitationCard.tsx` — ensure source badge renders correctly for new types
6. Add filter UI in `SourcesPanel.tsx` to filter citations by source type

### P1.2 — Per-citation `evidence_level` field

Backend `SearchResponse.citations[]` already returns `evidence_level` but frontend discards it. Add to `Citation` interface:
```typescript
interface Citation {
  index: number;
  title: string;
  source_type: SourceType;
  chunk_text: string;
  score: number;
  mongo_id: string;
  evidence_level?: string;    // NEW — "systematic_review" | "meta_analysis" | "rct" | "guideline" | "expert_consensus" | "case_series" | "textbook" | "regulatory"
  trust_tier?: number;        // NEW — 1 (highest) to 5 (lowest)
  indian_source?: boolean;    // NEW — true if from Indian journal/source
  also_indexed_in?: string[]; // NEW — other sources where this doc appears (e.g., ["pubmed", "indmed"])
}
```

**Frontend action**:
1. Extend `Citation` interface in `src/types/api.ts`
2. Show evidence level as a colored chip on `CitationCard` (e.g., RCT = green, guideline = blue, case_series = grey)
3. Show 🇮🇳 flag (or "Indian source" text) when `indian_source === true`
4. Show "Also in: PubMed, IndMED" subtle text when `also_indexed_in` is non-empty

### P1.3 — Per-journal trust tier in citation metadata

Citations from IJMR/NMJI (Tier 1) should visually rank above citations from IndMED-only journals (Tier 4). Backend now sets `trust_tier` on chunks.

**Frontend action**: Sort citations by `trust_tier` ascending (1 = highest), then by `score` descending. Show tier as a small badge or use it to set visual hierarchy (Tier 1 cards get a subtle highlight).

---

## Phase 2 Backend (upcoming) — Foundational Open-Access

### P2.1 — StatPearls + NCBI Bookshelf expansion

No new API surface, but retrieval quality for foundational questions improves. **No frontend action needed** beyond the source type additions above.

### P2.2 — NMC curriculum source

New source type `nmc_curriculum`. Frontend should label as "NMC Curriculum" with a distinct color (suggest navy blue — government/educational).

---

## Phase 3 Backend (upcoming) — Drug & Regulatory

### P3.1 — New `/drug-interactions` endpoint

```
POST /drug-interactions
{
  "drugs": ["metformin", "glimepiride", "atorvastatin"]
}
→
{
  "interactions": [
    {
      "drug_a": "metformin",
      "drug_b": "glimepiride",
      "severity": "moderate" | "major" | "minor" | "contraindicated",
      "mechanism": "Additive blood glucose lowering",
      "clinical_note": "Monitor blood glucose; dose adjustment may be needed.",
      "source": "nfi"
    }
  ]
}
```

**Frontend action**:
1. Build a "Drug Interaction Checker" tool — accessible from the sidebar or a new `/tools/ddi` route
2. UI: multi-select drug input (with autocomplete from a drug list endpoint we'll add)
3. Render interactions as a table or matrix (drugs on both axes, cells colored by severity)
4. Each interaction card shows: drugs involved, severity badge, mechanism, clinical note, source citation

### P3.2 — `/search` fast-path for drug queries

When intent is `DRUG_INFO`, backend bypasses RAG and returns NFI monograph directly. Response shape is unchanged (still `SearchResponse`), but:
- `evidence_distribution` will show `{"nfi": 1}` instead of mixed sources
- `confidence_score` will typically be higher (structured lookup vs. RAG synthesis)
- `chunks_retrieved` will be lower (1 monograph vs. 6 chunks)

**Frontend action**: When `evidence_distribution` is dominated by a single source (e.g., `nfi` > 80%), show a "Structured drug lookup" badge instead of the normal "AI-synthesized" badge. Differentiates the two response modes for the user.

### P3.3 — NFI / CDSCO / CTRI / PvPI source types

Same as P1.1 — extend `SourceType` and `sources.ts`.

---

## Phase 4 Backend (upcoming) — Specialty Guidelines

### P4.1 — RSSDI / CSI / specialty society guidelines

New source types `rssdi`, `csi`, etc. Frontend: extend source type list.

### P4.2 — Guideline-vs-journal retrieval boost

For therapeutic queries, guidelines will rank higher. **No frontend action** — retrieval ranking is backend-only.

---

## Phase 5 Backend (upcoming) — Epidemiology

### P5.1 — New `/epidemiology` endpoint

```
GET /epidemiology?state=Maharashtra&indicator=anemia_prevalence
→
{
  "state": "Maharashtra",
  "indicator": "anemia_prevalence",
  "value": 53.2,
  "unit": "%",
  "population": "Women 15-49 years",
  "source": "NFHS-5 (2019-21)",
  "citation_url": "https://nhfsindia.org/mh_report.pdf",
  "confidence": "high"
}
```

**Frontend action**:
1. Build a "State Health Stats" tool — accessible from sidebar
2. UI: select state (dropdown of 28 states + 8 UTs), select indicator (anemia, stunting, vaccination, NCD risk factors, etc.)
3. Render the stat prominently with the source citation
4. Could be embedded in chat responses too — when an epi query returns, surface the structured stat above the prose answer

---

## Cross-Phase Frontend Cleanup (do once, early)

These are existing frontend issues surfaced during the audit, not tied to a specific backend phase. Tackle them in a `phase0-frontend-cleanup` branch off `nextjs-ui`.

### F0.1 — Surface safety signals the backend already returns

**This is the #1 frontend gap.** Backend `SearchResponse` returns 8 fields that frontend silently discards:

```typescript
// CURRENT frontend QueryResponse (src/types/api.ts)
interface QueryResponse {
  answer: string;
  citations: Citation[];
  query: string;
  model: string;              // hardcoded "nim" — should come from response
  chunks_retrieved: number;
  mode: string;               // hardcoded "standard" — should come from response
}

// FULL backend response (what frontend SHOULD capture)
interface QueryResponse {
  answer: string;
  citations: Citation[];
  query: string;
  model: string;              // from response
  chunks_retrieved: number;
  mode: string;               // from response — "standard" | "deep_insights"
  // MISSING from frontend:
  query_intent?: string;      // "diagnostic" | "therapeutic" | "drug_info" | "guideline" | "epidemiology"
  cache_hit?: boolean;        // show "cached" badge when true
  confidence_score?: number;  // 0.0–1.0 — render as badge (high/medium/low)
  recommendation?: string;    // "OK" | "NEEDS_REVIEW" | "UNSAFE" — drives UI emphasis
  unverified_claims?: string[];  // claims the validator couldn't verify — show as warnings
  safety_warnings?: Array<{
    type: string;             // "dosage" | "contraindication" | "pregnancy" | "pediatric"
    severity: "info" | "warning" | "danger";
    message: string;
  }>;
  evidence_distribution?: Record<string, number>;  // {"icmr": 3, "pubmed": 2, "nfi": 1}
  is_safe?: boolean;          // if false, show prominent warning banner
  needs_disclaimer?: boolean; // if true, show "Clinical decision support — verify with your judgment" banner
  confidence_breakdown?: {
    retrieval_confidence: number;
    citation_confidence: number;
    hallucination_score: number;  // lower is better
    safety_score: number;
  };
}
```

**Action**: Extend `QueryResponse` interface, then build UI components:
- `SafetyBanner.tsx` — red/amber banner when `is_safe === false` or `safety_warnings.length > 0`
- `ConfidenceBadge.tsx` — colored badge based on `confidence_score` (≥0.8 green, 0.5-0.8 amber, <0.5 red)
- `EvidenceDistribution.tsx` — horizontal bar chart showing source breakdown
- `UnverifiedClaims.tsx` — list of claims flagged by hallucination detector
- `DisclaimerBanner.tsx` — when `needs_disclaimer === true`

These are **clinical safety features** — the highest priority frontend work.

### F0.2 — Delete Vite leftovers

Per the frontend's own `implementation_plan.md` (which was never completed):
- `vite.config.ts`
- `index.html`
- `src/main.tsx`
- `src/App.tsx`
- `src/vite-env.d.ts`
- `src/App.css`
- `tsconfig.app.json`
- `tsconfig.node.json`

Pick one lockfile (`bun.lock` OR `package-lock.json`) and delete the others.

### F0.3 — Turn on TypeScript strict mode

`tsconfig.json` currently has `"strict": false` and `"strictNullChecks": false`. Flip both to `true` and fix the resulting errors. This is the single biggest code-quality improvement available.

### F0.4 — Wire `ApiTab.tsx` into SettingsView

`src/components/settings/ApiTab.tsx` (157 LOC) is fully built but never imported. Either:
- Add it to `SettingsView.tsx` `TABS` array, OR
- Delete it if not needed

### F0.5 — Wire `/api/chat` SSE route or delete it

`src/app/api/chat/route.ts` is a mock that emits `chunk-1...chunk-5`. Either:
- Wire it to real backend streaming (backend needs to add SSE support — separate task), OR
- Delete it and the `IndexView.tsx` code that references streaming

### F0.6 — Auth layer (no backend blocker, but design now)

Frontend has zero auth today. Backend's `vault_store._get_user_id()` reads `X-User-ID` header, defaults to `"default_user"`.

**Plan**: When ready, frontend should:
1. Add login page (email + NMC number — backend will validate)
2. Store session in HTTP-only cookie (set by backend `/auth/login` endpoint — needs to be built)
3. Pass `X-User-ID` header on every `/api/search` and `/api/vault` call
4. Sync with Supabase auth from the marketing website (single sign-on)

Backend endpoints needed (build in Phase 6 or when frontend is ready):
- `POST /auth/login` — issue session
- `POST /auth/logout` — revoke
- `GET /auth/me` — current user info
- Middleware to validate session on protected routes

### F0.7 — Replace localStorage vault with backend `/vault` API

Frontend vault is currently localStorage-only. Backend has full MongoDB CRUD at `/vault/items` and `/vault/collections`. Replace `useVault` hook to call backend API. Keep localStorage as offline fallback if desired.

### F0.8 — Fix `IndexView.tsx` hardcoded values

Lines 79-80: `model: "nim"` and `mode: "standard"` are hardcoded. Should come from the backend response.

### F0.9 — Fix inconsistent `API_BASE`

`IndexView.tsx:16` uses `process.env.NEXT_PUBLIC_API_BASE_URL || "/api"` (bypasses proxy).
`api/search/route.ts:4-8` uses `OPENINSIGHT_API_BASE_URL || NEXT_PUBLIC_API_BASE_URL || "http://127.0.0.1:8000"`.

Two sources of truth. Pick one (recommend: always go through the Next.js API proxy at `/api/search` — it can inject `X-User-ID` and other headers, and keeps the backend URL secret).

### F0.10 — Document env vars in README

`OPENINSIGHT_API_BASE_URL` and `NEXT_PUBLIC_API_BASE_URL` are used but never documented. Add a `.env.example` to the frontend repo.

### F0.11 — Remove unused dependencies

`react-query` (provider mounted, zero queries), `react-hook-form` + `zod` (only shadcn wrapper), `recharts` (never imported), `next-themes` (only sonner uses it, no provider mounted). Either use them or delete them.

---

## Cross-Phase Website Cleanup (separate repo)

The marketing website (`openinsight-web`) has its own issues. Tackle in a separate `phase0-website-cleanup` branch off `web-insight-3`. See backend `docs/WEBSITE_CHANGES_NEEDED.md` (to be created when website work begins).

Key items:
1. Replace fictional testimonials with real or labeled-illustrative
2. Remove fabricated stats (10K+ queries, 500+ guidelines, etc.) until real numbers exist
3. Wire `CookieConsent` to actually gate PostHog/Clarity (DPDP Act compliance)
4. Centralize `app.openinsight.in` placeholder URL in `lib/config.ts`
5. Fix future-dated Privacy/Terms ("June 2026")

---

## How to use this document

When starting frontend work:
1. Read this file end-to-end
2. Check which backend phase has been merged to `restruct`
3. Pick items from the matching section + the Cross-Phase Cleanup section
4. Create a `phaseN-frontend-<topic>` branch off `nextjs-ui`
5. Update this file's "Last updated" line when backend ships new changes

This file lives in the backend repo at `docs/FRONTEND_CHANGES_NEEDED.md` so backend devs remember to update it. Frontend devs should `git fetch` and read the latest version before starting any frontend work.
