# Medical Knowledge Hardcoded in Codebase

This document lists all domain-specific medical knowledge that is hardcoded in the codebase. These are **intentional design decisions** - they're medical rules that should not be externalized to config files.

---

## 1. Intent Detection Patterns

**File:** `src/query/search/query_understanding.py`

### DIAGNOSTIC_PATTERNS
```python
DIAGNOSTIC_PATTERNS = [
    "what causes", "cause of", "differential", "how to diagnose",
    "symptoms of", "signs of", "what is", "aetiology",
]
```

### THERAPEUTIC_PATTERNS
```python
THERAPEUTIC_PATTERNS = [
    "treatment", "treat", "therapy", "management", "drug of choice",
    "dose", "dosage", "first line", "medication for",
]
```

### PROGNOSTIC_PATTERNS
```python
PROGNOSTIC_PATTERNS = [
    "prognosis", "outcome", "survival", "mortality", "risk of",
]
```

### DRUG_INFO_PATTERNS
```python
DRUG_INFO_PATTERNS = [
    "side effects", "adverse effects", "interactions", "contraindications",
    "mechanism of",
]
```

### GUIDELINE_PATTERNS
```python
GUIDELINE_PATTERNS = [
    "guideline", "recommendation", "protocol", "standard of care", "icmr",
]
```

### MEDICAL_SYNONYMS
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

---

## 2. Complex Query Detection Patterns

**File:** `src/query/agents/intent_router.py`

### COMPLEX_PATTERNS
```python
COMPLEX_PATTERNS = {
    # Comparison queries
    r"\bvs\b|\bversus\b|\bcompared to\b",
    
    # Drug interactions
    r"\binteract(?:ion|ing|s)?\b",
    r"\bwith\b.*\b(medication|drug|pill)\b",
    
    # Multi-condition
    r"\b(and|with)\b.*\b(diabetes|hypertension|ckd|copd|chf)\b.*\b(and|with)\b",
    
    # Contraindications
    r"\bcontraindicat(?:ed|ion|ions)\b",
    
    # Differential diagnosis
    r"\bdifferential\b",
    r"\bwhat could cause\b",
    
    # Protocol conflicts
    r"\bprotocol\b.*\bversus\b",
    r"\bguideline\b.*\bconflic",
}
```

### SUB_QUERY_TEMPLATES
```python
SUB_QUERY_TEMPLATES = {
    "diagnostic": ["diagnosis", "symptoms", "differential", "etiology"],
    "therapeutic": ["treatment", "drug_of_choice", "dosage", "regimen"],
    "drug_info": ["interactions", "contraindications", "side_effects", "mechanism"],
    "prognostic": ["prognosis", "outcome", "mortality", "survival"],
    "guideline": ["guidelines", "recommendations", "protocols", "standards"],
    "comparative": ["comparison", "versus", "which_is_better", "efficacy"],
}
```

---

## 3. Medical Safety Patterns

**File:** `src/query/validation/medical_safety.py`

### TREATMENT_PATTERNS
```python
TREATMENT_PATTERNS = [
    ("prescribe", "Prescription mentioned"),
    ("recommend.*treatment", "Treatment recommended"),
    ("first.*line.*therapy", "First-line therapy mentioned"),
    ...
]
```

### DOSAGE_PATTERNS
```python
DOSAGE_PATTERNS = [
    ("\d+\s*mg", "Dosage amount detected"),
    ("\d+\s*ml", "Volume dosage detected"),
    ("\d+\s*units", "Unit dosage detected"),
    ...
]
```

### CONTRAINDICATION_PATTERNS
```python
CONTRAINDICATION_PATTERNS = [
    ("contraindicated", "Contraindication mentioned"),
    ("should not be used", "Negative recommendation"),
    ("avoid.*patients", "Patient population warning"),
    ...
]
```

### INTERACTION_PATTERNS
```python
INTERACTION_PATTERNS = [
    ("interact", "Drug interaction mentioned"),
    ("may affect", "Potential interaction"),
    ("combined with", "Combination therapy"),
    ...
]
```

---

## 4. Chunking Abbreviations

**File:** `src/ingestion/chunker_v3.py`

### _ABBREVIATIONS
```python
_ABBREVIATIONS = {
    "et al.": "et al<PERIOD>",
    "fig.": "fig<PERIOD>",
    "e.g.": "e<PERIOD>g<PERIOD>",
    "i.e.": "i<PERIOD>e<PERIOD>",
    "vs.": "vs<PERIOD>",
    "mg/dl": "mg<SLASH>dl",
    "p.o.": "p<PERIOD>o<PERIOD>",
    "i.v.": "i<PERIOD>v<PERIOD>",
    "b.d.": "b<PERIOD>d<PERIOD>",
    "t.d.s.": "t<PERIOD>d<PERIOD>s<PERIOD>",
}
```

---

## 5. Evidence Level Detection Patterns

**File:** `src/ingestion/metadata_v2.py`

### RCT_TITLE_PATTERNS
```python
RCT_TITLE_PATTERNS = [
    "randomized controlled trial",
    "randomised controlled trial",
    "rct",
    "clinical trial",
    ...
]
```

### SYSTEMATIC_REVIEW_PATTERNS
```python
SYSTEMATIC_REVIEW_PATTERNS = [
    "systematic review",
    "meta-analysis",
    "meta analysis",
    ...
]
```

### GUIDELINE_PATTERNS
```python
GUIDELINE_PATTERNS = [
    "guideline",
    "clinical practice guideline",
    "who guideline",
    "icmr guideline",
    ...
]
```

---

## 6. Quality Scoring Patterns

**File:** `src/ingestion/quality.py`

### _HIGH_VALUE_PATTERNS
```python
_HIGH_VALUE_PATTERNS = [
    r"\brandomized\b",
    r"\bcontrolled\b",
    r"\bmeta[-\s]analysis\b",
    r"\bsystematic review\b",
    r"\bprospective\b",
    r"\bcohort\b",
    ...
]
```

### _LOW_VALUE_PATTERNS
```python
_LOW_VALUE_PATTERNS = [
    r"\bopinion\b",
    r"\bcase report\b",
    r"\beditorial\b",
    r"\bletter\b",
    ...
]
```

---

## 7. NER Entity Patterns

**File:** `src/ingestion/ner.py`

### DRUG_PATTERNS
```python
DRUG_PATTERNS = [
    r"\b(?:doxycycline|azithromycin|rifampicin|isoniazid|pyrazinamide|ethambutol|"
    r"streptomycin|amoxicillin|ciprofloxacin|metronidazole|fluconazole|amphotericin|"
    ...
]
```

### DISEASE_PATTERNS
```python
DISEASE_PATTERNS = [
    r"\b(?:tuberculosis|malaria|dengue|typhoid|leptospirosis|"
    r"diabetes|hypertension|copd|asthma|chf|cad|"
    ...
]
```

---

## Why These Are Hardcoded

1. **Medical Domain Knowledge**: These are clinical rules derived from medical practice
2. **Performance**: Regex patterns are faster than external lookups
3. **Type Safety**: Hardcoded enums provide compile-time checking
4. **Maintainability**: Changes to medical rules require code review

## Future Considerations

- Consider moving patterns to a medical knowledge base if they need frequent updates
- Patterns may need localization for Indian regional languages
- Evidence level patterns may need updates as medical literature evolves