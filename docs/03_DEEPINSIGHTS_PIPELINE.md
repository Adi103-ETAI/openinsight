# DeepInsights Pipeline

## When to Use DeepInsights

Use DeepInsights for complex clinical queries that require:
- Drug interaction checks (e.g., "Can I give metformin with ACE inhibitors?")
- Differential diagnosis (e.g., "What could cause this presentation?")
- Protocol conflicts (e.g., "ICMR vs WHO guidelines for dengue")
- Multi-condition management (e.g., "DM with HTN and CKD")

---

## DeepInsights Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              REQUEST                                        │
│          { "query": "treatment for diabetes with hypertension..." }         │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         INTENT ROUTER                                       │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Complex Query Detection (Rule-based + Entity Count)                │    │
│  │                                                                     │    │
│  │  Complex Patterns:                                                  │    │
│  │    - "vs / versus" (comparisons)                                    │    │
│  │    - "interaction" (drug interactions)                              │    │
│  │    - Multi-condition ("X and Y and Z")                              │    │
│  │    - "contraindicated"                                              │    │
│  │    - "differential"                                                 │    │
│  │                                                                     │    │
│  │  Output:                                                            │    │
│  │    - complexity: SIMPLE / MEDIUM / COMPLEX                          │    │
│  │    - confidence: 0.0-1.0                                            │    │
│  │    - detected_intent: therapeutic/diagnostic/etc.                   │    │
│  │    - sub_query_types: [treatment, dosage, interactions, etc.]       │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
                 [SIMPLE]                       [COMPLEX]
               Use standard                      Continue
                /search
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       QUERY DECOMPOSER                                      │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  LLM-based Decomposition                                            │    │
│  │                                                                     │    │
│  │  Input: "treatment for diabetes with hypertension and CKD"          │    │
│  │                                                                     │    │
│  │  Output: 3-6 Sub-queries:                                           │    │
│  │    - q1: "diabetes treatment options" (focus: treatment)            │    │
│  │    - q2: "hypertension medication dosage" (focus: dosage)           │    │
│  │    - q3: "drug interactions diabetes hypertension" (focus: inter)   │    │
│  │    - q4: "CKD contraindications diabetes drugs" (focus: contra)     │    │
│  │    - q5: "ICMR guidelines diabetes hypertension" (focus: guide)     │    │
│  │    - q6: "diabetes CKD management protocols" (focus: protocol)      │    │
│  │                                                                     │    │
│  │  Fallback: Rule-based if LLM fails                                  │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      PARALLEL RETRIEVAL                                     │
│                                                                             │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐      │
│   │ Sub-q 1  │  │ Sub-q 2  │  │ Sub-q 3  │  │ Sub-q 4  │  │ Sub-q 5  │      │
│   │          │  │          │  │          │  │          │  │          │      │
│   │  Dense   │  │  Dense   │  │  Dense   │  │  Dense   │  │  Dense   │      │
│   │  +       │  │  +       │  │  +       │  │  +       │  │  +       │      │
│   │  Sparse  │  │  Sparse  │  │  Sparse  │  │  Sparse  │  │  Sparse  │      │
│   └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘      │
│        │             │             │             │             │            │
│        └─────────────┴─────────────┴─────────────┴─────────────┘            │
│                              │                                              │
│                              ▼                                              │
│              ┌───────────────────────────────────────┐                      │
│              │  All chunks combined (with dedup)     │                      │
│              │  Total: sub_queries × top_k           │                      │
│              └───────────────────────────────────────┘                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    CONTRADICTION DETECTION                                  │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Check for conflicting evidence                                     │    │
│  │                                                                     │    │
│  │  Method 1: NLI Model (future)                                       │    │
│  │  Method 2: Keyword-based (current)                                  │    │
│  │    - "improve" vs "worsen"                                          │    │
│  │    - "recommended" vs "not recommended"                             │    │
│  │    - "effective" vs "ineffective"                                   │    │
│  │    - dosage conflicts                                               │    │
│  │                                                                     │    │
│  │  Output: List of contradiction pairs                                │    │
│  │    - type: treatment_conflict/dosage_conflict/outcome_conflict      │    │
│  │    - evidence: conflicting keywords                                 │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        ANSWER SYNTHESIS                                     │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  LLM Synthesis Prompt                                               │    │
│  │                                                                     │    │
│  │  "Synthesize the following evidence into a comprehensive answer:    │    │
│  │                                                                     │    │
│  │   Original Query: {query}                                           │    │
│  │   Synthesis Guidance: {synthesis_prompt}                            │    │
│  │   Evidence: {all_chunks}                                            │    │
│  │                                                                     │    │
│  │   Include:                                                          │    │
│  │   - Key findings from each sub-query                                │    │
│  │   - Recommendations with citations                                  │    │
│  │   - Warnings about contradictions                                   │    │
│  │   - Confidence assessment                                           │    │
│  │  "                                                                  │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           RESPONSE                                          │
│  {                                                                          │
│    "answer": "...",                                                         │
│    "sections": {                                                            │
│      "summary": "...",                                                      │
│      "diabetes_control": "...",                                             │
│      "hypertension_management": "...",                                      │
│      "kidney_considerations": "..."                                         │
│    },                                                                       │
│    "citations": [...],                                                      │
│    "sub_queries": [                                                         │
│      {"id": "q1", "focus": "treatment", "chunks_retrieved": 8},             │
│      {"id": "q2", "focus": "dosage", "chunks_retrieved": 6},                │
│      ...                                                                    │
│    ],                                                                       │
│    "contradictions": [                                                      │
│      {"type": "dosage_conflict", "evidence": "500mg vs 1000mg"}             │
│    ],                                                                       │
│    "confidence": 0.78,                                                      │
│    "complexity_detected": "complex",                                        │
│    "processing_time_ms": 4500                                               │
│  }                                                                          │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Intent Router Logic

```python
COMPLEX_PATTERNS = {
    # Comparisons
    r"\bvs\b|\bversus\b|\bcompared to\b",
    
    # Drug interactions
    r"\binteract(?:ion|ing|s)?\b",
    r"\bwith\b.*\b(medication|drug|pill)\b",
    
    # Multi-condition
    r"\b(and|with)\b.*\b(diabetes|hypertension|ckd|copd|chf)\b.*\b(and|with)\b",
    
    # Contraindications
    r"\bcontraindicat(?:ed|ion|ions)\b",
    
    # Differential
    r"\bdifferential\b",
}
```

### Complexity Calculation
- 2+ patterns → COMPLEX (95% confidence)
- 1 pattern + 3+ entities → COMPLEX (90%)
- 1 pattern + 2 entities → MEDIUM (70%)
- 4+ entities → COMPLEX (85%)
- Default → SIMPLE (75%)

---

## Configuration

```python
DEEP_INSIGHTS_ENABLED = true
DEEP_INSIGHTS_MAX_SUB_QUERIES = 6
DEEP_INSIGHTS_SUB_QUERY_TOP_K = 8
DEEP_INSIGHTS_TIMEOUT = 60

CONTRADICTION_DETECTION = true
CONTRADICTION_MIN_CHUNKS = 3
```

---

## Performance

| Stage | Typical Time |
|-------|--------------|
| Intent routing | 5-10ms |
| Query decomposition | 200-500ms |
| Parallel retrieval | 800-1500ms |
| Contradiction detection | 100-200ms |
| Answer synthesis | 1500-2500ms |
| **Total** | **~4-5 seconds** |