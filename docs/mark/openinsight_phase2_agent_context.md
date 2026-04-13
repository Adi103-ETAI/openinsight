# OpenInsight — Agent Build Context
## Phase 2: Ingestion Pipeline v2

You are building **OpenInsight** — an AI clinical decision support platform for Indian physicians by SentArc Labs, Pune. The product was previously called OpenInsight. All code references of `openinsight` in file paths and imports should be change to `openinsight` — do not rename anything.

Phase 1 is complete. This document covers Phase 2 entirely. Read every section before writing any code.

---

## 1. What Phase 2 Delivers

A complete rebuild of the data ingestion pipeline. The v1 pipeline used flat 512-word chunking, filename-derived metadata, and no content classification. v2 replaces all of that with:

1. **GROBID integration** — structured parsing for research PDFs (extracts sections, titles, abstracts properly)
2. **Hierarchical chunking** — section → semantic → sentence fallback, 250-350 tokens
3. **Rich metadata schema** — 15 structured fields per chunk including disease, drugs, study type, evidence level
4. **Content type classifier** — clinical / preclinical / background / noise
5. **OCR fallback** — Tesseract for scanned PDFs that pdfplumber can't read
6. **Re-ingestion script** — clear old data, re-ingest everything with new pipeline

---

## 2. Current Project Structure

```
openinsight/                        ← root (folder may still be named openinsight)
├── prompts/
│   ├── system.md                   ← EXISTS
│   └── query_rewrite.md            ← EXISTS
├── src/
│   ├── core/
│   │   └── config.py               ← EXISTS
│   ├── ingestion/
│   │   ├── __init__.py             ← EXISTS
│   │   ├── document_db.py          ← EXISTS — MongoDB client, DocumentRecord, ChunkRecord
│   │   ├── vector_db.py            ← EXISTS — Qdrant hybrid search
│   │   ├── embeddings.py           ← EXISTS — embed_texts(), embed_query()
│   │   └── parsers/
│   │       ├── __init__.py         ← EXISTS
│   │       ├── base.py             ← EXISTS — BaseParser abstract class
│   │       ├── icmr.py             ← EXISTS — pdfplumber PDF parser
│   │       └── pubmed.py           ← EXISTS — NCBI Entrez API parser
│   ├── query/
│   │   ├── prompts.py              ← EXISTS
│   │   ├── rewriter.py             ← EXISTS — query rewriting
│   │   ├── reranker.py             ← EXISTS — cross-encoder reranker
│   │   └── standard.py            ← EXISTS — full query pipeline
│   └── api/
│       ├── main.py                 ← EXISTS
│       └── routes/
│           └── query.py            ← EXISTS
├── scripts/
│   ├── seed_icmr.py                ← EXISTS
│   └── seed_pubmed.py              ← EXISTS
├── data/
│   └── raw/
│       └── icmr/                   ← ICMR PDFs live here
└── utils/
    └── chunker.py                  ← EXISTS — sentence-boundary chunker
```

---

## 3. Existing Code You Must Know

### `src/ingestion/document_db.py` — Current Models

```python
class DocumentRecord(BaseModel):
    source_type: str          # "icmr" | "pubmed" | "nmc" | "state_guideline"
    title: str
    content: str
    url: Optional[str] = None
    doi: Optional[str] = None
    published_date: Optional[str] = None
    condition_tags: list[str] = Field(default_factory=list)
    specialty_tags: list[str] = Field(default_factory=list)
    ingested_at: datetime = Field(default_factory=datetime.utcnow)

class ChunkRecord(BaseModel):
    document_id: str
    source_type: str
    title: str
    chunk_text: str
    chunk_index: int
    condition_tags: list[str] = Field(default_factory=list)
    specialty_tags: list[str] = Field(default_factory=list)
    char_count: int = 0
    embedded: bool = False
    embedded_at: Optional[datetime] = None
```

### `src/ingestion/parsers/base.py`

```python
class BaseParser(ABC):
    @property
    @abstractmethod
    def source_type(self) -> str: pass

    @abstractmethod
    def parse(self) -> list[DocumentRecord]: pass
```

### `src/core/config.py` — Current Settings

```python
class Settings(BaseSettings):
    nvidia_nim_api_key: str
    nvidia_nim_base_url: str = "https://integrate.api.nvidia.com/v1"
    nim_model: str = "meta/llama-3.1-70b-instruct"
    mongodb_url: str = "mongodb://localhost:27017"
    mongodb_db: str = "openinsight"
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "openinsight_chunks"
    redis_url: str = "redis://localhost:6379"
    ncbi_api_key: str = ""
    ncbi_email: str = "adi.singh1426@gmail.com"
    embedding_model: str = "pritamdeka/S-PubMedBert-MS-MARCO"
    embedding_dim: int = 768
    nim_temperature: float = 0.1
    nim_max_tokens: int = 1024
    retrieval_top_k: int = 8
    reranker_top_n: int = 8
```

---

## 4. New Dependencies To Install

Run these before building anything:

```bash
pip install pytesseract pillow scispacy --break-system-packages 2>/dev/null || pip install pytesseract pillow scispacy

# Download scispaCy medical NER model
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.3/en_core_sci_sm-0.5.3.tar.gz

# Install Tesseract OCR system package
sudo apt-get install -y tesseract-ocr tesseract-ocr-eng 2>/dev/null || echo "apt not available"

# GROBID runs as a Docker container - add to docker-compose.yml (see Step 2.1)
```

Add to `requirements.txt`:
```
pytesseract==0.3.13
Pillow>=10.0.0
scispacy==0.5.3
```

---

## 5. What To Build — Step by Step

---

### Step 2.1 — GROBID Docker Service

GROBID is a Java-based PDF parser that extracts structured sections from research papers. It runs as a REST server.

**Add to `docker-compose.yml`** (alongside existing mongodb, qdrant, redis services):

```yaml
  grobid:
    image: lfoppiano/grobid:0.8.0
    container_name: openinsight_grobid
    ports:
      - "8070:8070"
    restart: unless-stopped
    environment:
      - JAVA_OPTS=-Xmx4g
```

Start it:
```bash
docker compose up -d grobid
```

Wait 60 seconds for GROBID to initialise, then verify:
```bash
curl http://localhost:8070/api/isalive
```

Should return `true`.

**Add GROBID URL to `src/core/config.py`** inside Settings class:
```python
grobid_url: str = "http://localhost:8070"
```

---

### Step 2.2 — Updated Document and Chunk Models

The existing `DocumentRecord` and `ChunkRecord` models need new fields for v2 metadata. 

**Modify `src/ingestion/document_db.py`** — replace the two model classes with these expanded versions. Keep all existing fields, only add new ones:

```python
class DocumentRecord(BaseModel):
    # Existing fields — keep all
    source_type: str
    title: str
    content: str
    url: Optional[str] = None
    doi: Optional[str] = None
    published_date: Optional[str] = None
    condition_tags: list[str] = Field(default_factory=list)
    specialty_tags: list[str] = Field(default_factory=list)
    ingested_at: datetime = Field(default_factory=datetime.utcnow)

    # New v2 fields
    year: Optional[int] = None
    journal: Optional[str] = None
    study_type: Optional[str] = None      # "rct" | "meta_analysis" | "review" | "case_report" | "guideline" | "unknown"
    population: Optional[str] = None      # "adult" | "pediatric" | "maternal" | "general"
    evidence_level: int = 5               # 1=RCT/meta → 5=expert opinion/unknown
    is_india_specific: bool = False
    parser_version: str = "v1"            # "v1" | "v2" to track which pipeline ingested it
    total_chunks: int = 0


class ChunkRecord(BaseModel):
    # Existing fields — keep all
    document_id: str
    source_type: str
    title: str
    chunk_text: str
    chunk_index: int
    condition_tags: list[str] = Field(default_factory=list)
    specialty_tags: list[str] = Field(default_factory=list)
    char_count: int = 0
    embedded: bool = False
    embedded_at: Optional[datetime] = None

    # New v2 fields
    section: Optional[str] = None          # "treatment" | "diagnosis" | "dosage" | "background" | "abstract" etc
    diseases: list[str] = Field(default_factory=list)
    drugs: list[str] = Field(default_factory=list)
    symptoms: list[str] = Field(default_factory=list)
    content_type: str = "unknown"          # "clinical" | "preclinical" | "background" | "noise"
    content_weight: float = 1.0            # clinical=1.5, preclinical=1.0, background=0.7, noise=0.1
    is_india_specific: bool = False
    evidence_level: int = 5
    page_number: Optional[int] = None
    token_count: int = 0
    parser_version: str = "v1"
```

---

### Step 2.3 — Medical NER Extractor

Creates a utility that extracts diseases, drugs, and symptoms from text using scispaCy.

**Create `src/ingestion/ner.py`**:

```python
"""
Medical Named Entity Recognition
Extracts diseases, drugs, symptoms from chunk text using scispaCy.
Falls back to rule-based extraction if model unavailable.
"""
from functools import lru_cache
from loguru import logger
from typing import Optional
import re


DRUG_PATTERNS = [
    r'\b(?:doxycycline|azithromycin|rifampicin|isoniazid|pyrazinamide|ethambutol|'
    r'streptomycin|amoxicillin|ciprofloxacin|metronidazole|fluconazole|amphotericin|'
    r'artemisinin|chloroquine|primaquine|oseltamivir|acyclovir|cotrimoxazole|'
    r'paracetamol|ibuprofen|aspirin|metformin|insulin|amlodipine|atenolol|'
    r'enalapril|losartan|furosemide|spironolactone|digoxin|warfarin|heparin)\b',
]

DISEASE_PATTERNS = [
    r'\b(?:tuberculosis|TB|malaria|dengue|typhoid|leptospirosis|scrub typhus|'
    r'rickettsial|COVID-19|SARS-CoV-2|pneumonia|sepsis|meningitis|encephalitis|'
    r'hepatitis|cirrhosis|diabetes|hypertension|heart failure|myocardial infarction|'
    r'stroke|asthma|COPD|chronic kidney disease|CKD|anaemia|anemia|cholera|'
    r'chikungunya|Japanese encephalitis|rabies|snakebite|mucormycosis)\b',
]


@lru_cache(maxsize=1)
def _load_scispacy():
    try:
        import spacy
        nlp = spacy.load("en_core_sci_sm")
        logger.info("scispaCy model loaded successfully")
        return nlp
    except Exception as e:
        logger.warning(f"scispaCy model not available, using rule-based NER: {e}")
        return None


def extract_entities(text: str) -> dict:
    """
    Extract medical entities from text.
    Returns dict with diseases, drugs, symptoms lists.
    """
    diseases = []
    drugs = []
    symptoms = []

    # Rule-based extraction (always runs)
    text_lower = text.lower()
    for pattern in DRUG_PATTERNS:
        matches = re.findall(pattern, text, re.IGNORECASE)
        drugs.extend([m.lower() for m in matches])

    for pattern in DISEASE_PATTERNS:
        matches = re.findall(pattern, text, re.IGNORECASE)
        diseases.extend([m.lower() for m in matches])

    # scispaCy NER (runs if model available)
    nlp = _load_scispacy()
    if nlp:
        try:
            doc = nlp(text[:1000])  # limit to 1000 chars for speed
            for ent in doc.ents:
                label = ent.label_.upper()
                val = ent.text.lower().strip()
                if len(val) < 3:
                    continue
                if label in ("DISEASE", "DISORDER", "SYNDROME"):
                    diseases.append(val)
                elif label in ("CHEMICAL", "DRUG", "MEDICATION"):
                    drugs.append(val)
                elif label in ("SIGN_SYMPTOM", "SYMPTOM"):
                    symptoms.append(val)
        except Exception as e:
            logger.debug(f"scispaCy extraction error: {e}")

    # Deduplicate
    return {
        "diseases": list(set(diseases))[:10],
        "drugs": list(set(drugs))[:10],
        "symptoms": list(set(symptoms))[:10],
    }


def classify_content_type(text: str, section: Optional[str] = None) -> tuple[str, float]:
    """
    Classify chunk content type and return (content_type, weight).
    
    Returns:
        content_type: "clinical" | "preclinical" | "background" | "noise"
        weight: 1.5 for clinical, 1.0 for preclinical, 0.7 for background, 0.1 for noise
    """
    text_lower = text.lower()

    # Noise patterns — administrative boilerplate
    noise_patterns = [
        r'\breferences?\b', r'\backnowledgements?\b', r'\bforeword\b',
        r'\bcommittee members?\b', r'\btable of contents\b', r'\bindex\b',
        r'\bcopyright\b', r'\ball rights reserved\b', r'\bissn\b', r'\bdoi:\s*10\.',
        r'\bfigure \d+\b', r'\btable \d+\b', r'\bappendix\b',
    ]
    noise_count = sum(1 for p in noise_patterns if re.search(p, text_lower))
    if noise_count >= 2 or len(text.strip()) < 80:
        return "noise", 0.1

    # Check section label
    if section:
        section_lower = section.lower()
        if any(w in section_lower for w in ["reference", "acknowledgement", "foreword", "appendix", "index"]):
            return "noise", 0.1
        if any(w in section_lower for w in ["abstract", "introduction", "background", "history", "epidemiology"]):
            return "background", 0.7
        if any(w in section_lower for w in ["method", "animal", "in vitro", "mouse", "rat model"]):
            return "preclinical", 1.0
        if any(w in section_lower for w in ["treatment", "management", "dosage", "dose", "therapy",
                                              "diagnosis", "protocol", "guideline", "recommendation",
                                              "drug", "antibiotic", "clinical", "patient"]):
            return "clinical", 1.5

    # Content-based classification
    clinical_signals = [
        r'\b(?:dosage|dose|mg|mcg|kg|treatment|therapy|antibiotic|drug|prescri)\b',
        r'\b(?:patient|physician|doctor|hospital|clinic|ward)\b',
        r'\b(?:diagnosis|diagnostic|symptom|sign|fever|pain|infection)\b',
        r'\b(?:guideline|protocol|recommendation|management|should|must)\b',
        r'\b(?:oral|intravenous|IV|IM|SC|subcutaneous|intramuscular)\b',
    ]
    clinical_count = sum(1 for p in clinical_signals if re.search(p, text_lower))

    preclinical_signals = [
        r'\b(?:mouse|rat|animal|in vitro|cell line|assay|mechanism|pathway)\b',
    ]
    preclinical_count = sum(1 for p in preclinical_signals if re.search(p, text_lower))

    background_signals = [
        r'\b(?:history|historical|was first|reported in|century|epidemic|endemic)\b',
        r'\b(?:introduction|overview|background|etiology|epidemiology|prevalence)\b',
    ]
    background_count = sum(1 for p in background_signals if re.search(p, text_lower))

    if preclinical_count >= 2:
        return "preclinical", 1.0
    if clinical_count >= 3:
        return "clinical", 1.5
    if clinical_count >= 1:
        return "clinical", 1.5
    if background_count >= 2:
        return "background", 0.7

    return "background", 0.7


def infer_study_type(text: str, title: str = "") -> tuple[str, int]:
    """
    Infer study type and evidence level from text/title.
    Returns (study_type, evidence_level) where level 1=highest, 5=lowest.
    """
    combined = (title + " " + text[:500]).lower()

    if re.search(r'\b(?:meta.analysis|systematic review|cochrane)\b', combined):
        return "meta_analysis", 1
    if re.search(r'\b(?:randomized|randomised|RCT|clinical trial|controlled trial)\b', combined):
        return "rct", 1
    if re.search(r'\b(?:cohort study|case.control|prospective|retrospective study)\b', combined):
        return "observational", 2
    if re.search(r'\b(?:guideline|recommendation|protocol|ICMR|WHO|NMC|MoHFW)\b', combined):
        return "guideline", 2
    if re.search(r'\b(?:review article|narrative review|literature review)\b', combined):
        return "review", 3
    if re.search(r'\b(?:case report|case series|case study)\b', combined):
        return "case_report", 4

    return "unknown", 5
```

---

### Step 2.4 — Hierarchical Chunker

Replaces the flat sentence-boundary chunker with a three-level hierarchy.

**Create `src/utils/chunker_v2.py`** (keep old `chunker.py` — do not delete it):

```python
"""
Hierarchical Medical Text Chunker v2
Three-level chunking: section → semantic segments → sentence fallback.
Target: 250-350 tokens, 50-75 token overlap.
"""
import re
from dataclasses import dataclass, field
from typing import Optional
from loguru import logger


TARGET_TOKENS = 300       # target chunk size in tokens (approx words * 1.3)
MAX_TOKENS = 400          # hard max before forced split
OVERLAP_TOKENS = 60       # overlap between chunks
MIN_CHUNK_CHARS = 100     # skip chunks shorter than this


# Section header patterns — ordered by specificity
SECTION_PATTERNS = [
    # Clinical sections (high value)
    r'^(?:\d+\.?\d*\s+)?(?:treatment|management|therapy|therapeutic|dosage|dose|drug|antibiotic|'
    r'medication|prescription|clinical|diagnosis|diagnostic|investigation|laboratory|'
    r'guideline|recommendation|protocol|procedure|intervention|prevention|prophylaxis)',
    # Abstract sections
    r'^(?:abstract|summary|conclusion|result|finding|outcome|discussion)',
    # Background sections (lower value)
    r'^(?:introduction|background|overview|history|epidemiology|etiology|pathophysiology|'
    r'method|material|appendix|reference|acknowledgement|foreword|index)',
]


@dataclass
class TextChunk:
    text: str
    chunk_index: int
    char_count: int
    token_count: int
    section: Optional[str] = None
    page_number: Optional[int] = None
    level: str = "paragraph"   # "section" | "paragraph" | "sentence"


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: words * 1.3"""
    return int(len(text.split()) * 1.3)


def _detect_section_header(line: str) -> Optional[str]:
    """Return section name if line looks like a section header, else None."""
    line = line.strip()
    if not line or len(line) > 120:
        return None
    # Short line, possibly title-cased or numbered
    if len(line) < 80 and (line.isupper() or re.match(r'^\d+\.?\d*\s+\w', line) or
                            re.match(r'^[A-Z][a-z]+ [A-Z]', line)):
        for pattern in SECTION_PATTERNS:
            if re.match(pattern, line, re.IGNORECASE):
                return line
    return None


def _split_into_sections(text: str) -> list[tuple[Optional[str], str]]:
    """
    Split text into (section_header, section_content) pairs.
    Returns list of tuples.
    """
    lines = text.splitlines()
    sections = []
    current_header = None
    current_lines = []

    for line in lines:
        header = _detect_section_header(line)
        if header and current_lines:
            content = "\n".join(current_lines).strip()
            if content:
                sections.append((current_header, content))
            current_header = header
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        content = "\n".join(current_lines).strip()
        if content:
            sections.append((current_header, content))

    return sections if sections else [(None, text)]


def _split_by_paragraphs(text: str) -> list[str]:
    """Split text into paragraphs by double newline or blank line."""
    paragraphs = re.split(r'\n\s*\n', text)
    return [p.strip() for p in paragraphs if p.strip()]


def _split_sentences(text: str) -> list[str]:
    """Sentence splitter respecting medical abbreviations."""
    abbrevs = ["Dr", "Mr", "Mrs", "Ms", "Prof", "vs", "etc", "approx",
               "mg", "kg", "mcg", "mL", "IV", "IM", "SC", "BD", "TDS",
               "OD", "SOS", "tab", "cap", "inj", "soln", "approx", "Fig"]
    protected = text
    for abbrev in abbrevs:
        protected = protected.replace(f"{abbrev}.", f"{abbrev}<DOT>")
    parts = re.split(r'(?<=[.!?])\s+', protected)
    return [p.replace("<DOT>", ".").strip() for p in parts if p.strip()]


def _merge_into_chunks(
    segments: list[str],
    section: Optional[str],
    start_index: int,
) -> list[TextChunk]:
    """
    Merge segments into target-sized chunks with overlap.
    """
    chunks = []
    current_words = []
    current_tokens = 0
    idx = start_index

    # Overlap buffer: last N words from previous chunk
    overlap_buffer = []

    for segment in segments:
        segment_tokens = _estimate_tokens(segment)
        segment_words = segment.split()

        # If single segment already exceeds max, split by sentences
        if segment_tokens > MAX_TOKENS:
            sentences = _split_sentences(segment)
            for sentence in sentences:
                s_tokens = _estimate_tokens(sentence)
                s_words = sentence.split()

                if current_tokens + s_tokens > TARGET_TOKENS and current_words:
                    chunk_text = " ".join(current_words)
                    if len(chunk_text) >= MIN_CHUNK_CHARS:
                        chunks.append(TextChunk(
                            text=chunk_text,
                            chunk_index=idx,
                            char_count=len(chunk_text),
                            token_count=current_tokens,
                            section=section,
                            level="sentence",
                        ))
                        idx += 1
                    # Overlap
                    overlap_words = current_words[-OVERLAP_TOKENS:] if len(current_words) > OVERLAP_TOKENS else current_words[:]
                    current_words = overlap_buffer + overlap_words + s_words
                    current_tokens = _estimate_tokens(" ".join(current_words))
                    overlap_buffer = []
                else:
                    current_words.extend(s_words)
                    current_tokens += s_tokens
        else:
            if current_tokens + segment_tokens > TARGET_TOKENS and current_words:
                chunk_text = " ".join(current_words)
                if len(chunk_text) >= MIN_CHUNK_CHARS:
                    chunks.append(TextChunk(
                        text=chunk_text,
                        chunk_index=idx,
                        char_count=len(chunk_text),
                        token_count=current_tokens,
                        section=section,
                        level="paragraph",
                    ))
                    idx += 1
                overlap_words = current_words[-OVERLAP_TOKENS:] if len(current_words) > OVERLAP_TOKENS else current_words[:]
                current_words = overlap_words + segment_words
                current_tokens = _estimate_tokens(" ".join(current_words))
            else:
                current_words.extend(segment_words)
                current_tokens += segment_tokens

    # Flush remaining
    if current_words:
        chunk_text = " ".join(current_words)
        if len(chunk_text) >= MIN_CHUNK_CHARS:
            chunks.append(TextChunk(
                text=chunk_text,
                chunk_index=idx,
                char_count=len(chunk_text),
                token_count=current_tokens,
                section=section,
                level="paragraph",
            ))

    return chunks


def chunk_text_v2(text: str) -> list[TextChunk]:
    """
    Main entry point. Hierarchical chunking:
    1. Split into sections by header detection
    2. Split each section into paragraphs
    3. Merge paragraphs into target-sized chunks with overlap
    4. Sentence fallback for oversized segments
    """
    if not text or not text.strip():
        return []

    sections = _split_into_sections(text)
    all_chunks = []
    global_idx = 0

    for section_header, section_content in sections:
        paragraphs = _split_by_paragraphs(section_content)
        if not paragraphs:
            continue
        section_chunks = _merge_into_chunks(paragraphs, section_header, global_idx)
        all_chunks.extend(section_chunks)
        global_idx += len(section_chunks)

    logger.debug(f"Chunked {len(text)} chars into {len(all_chunks)} chunks")
    return all_chunks
```

---

### Step 2.5 — GROBID Parser for Research PDFs

**Create `src/ingestion/parsers/grobid.py`**:

```python
"""
GROBID Parser
Uses GROBID REST API to extract structured content from research PDFs.
Extracts: title, abstract, sections, full text with section labels.
Falls back to pdfplumber if GROBID is unavailable.
"""
import requests
from pathlib import Path
from loguru import logger
from bs4 import BeautifulSoup
from typing import Optional

from src.ingestion.document_db import DocumentRecord
from src.ingestion.parsers.base import BaseParser
from src.ingestion.ner import infer_study_type
from src.core.config import get_settings

settings = get_settings()


class GROBIDParser(BaseParser):
    """
    Parser for research PDFs using GROBID.
    Best for: PubMed full-text PDFs, journal articles, research papers.
    Not ideal for: government guidelines, policy documents (use ICMRParser for those).
    """

    def __init__(self, file_path: str | Path, source_type: str = "research"):
        self.file_path = Path(file_path)
        self._source_type = source_type

    @property
    def source_type(self) -> str:
        return self._source_type

    def _call_grobid(self) -> Optional[str]:
        """Send PDF to GROBID and return TEI XML response."""
        url = f"{settings.grobid_url}/api/processFulltextDocument"
        try:
            with open(self.file_path, "rb") as f:
                response = requests.post(
                    url,
                    files={"input": f},
                    data={"consolidateHeader": "1", "consolidateCitations": "0"},
                    timeout=120,
                )
            if response.status_code == 200:
                return response.text
            else:
                logger.warning(f"GROBID returned {response.status_code} for {self.file_path.name}")
                return None
        except Exception as e:
            logger.warning(f"GROBID call failed for {self.file_path.name}: {e}")
            return None

    def _parse_tei(self, tei_xml: str) -> dict:
        """Parse TEI XML from GROBID into structured dict."""
        soup = BeautifulSoup(tei_xml, "xml")
        result = {
            "title": "",
            "abstract": "",
            "sections": [],
            "year": None,
            "journal": "",
        }

        # Title
        title_tag = soup.find("titleStmt")
        if title_tag:
            result["title"] = title_tag.get_text(separator=" ").strip()

        # Abstract
        abstract_tag = soup.find("abstract")
        if abstract_tag:
            result["abstract"] = abstract_tag.get_text(separator=" ").strip()

        # Year
        date_tag = soup.find("date", {"type": "published"})
        if date_tag and date_tag.get("when"):
            try:
                result["year"] = int(date_tag["when"][:4])
            except Exception:
                pass

        # Journal
        journal_tag = soup.find("title", {"level": "j"})
        if journal_tag:
            result["journal"] = journal_tag.get_text().strip()

        # Body sections
        body = soup.find("body")
        if body:
            for div in body.find_all("div"):
                head = div.find("head")
                section_title = head.get_text().strip() if head else None
                paragraphs = div.find_all("p")
                section_text = " ".join(p.get_text(separator=" ").strip() for p in paragraphs)
                if section_text.strip():
                    result["sections"].append({
                        "title": section_title,
                        "text": section_text,
                    })

        return result

    def parse(self) -> list[DocumentRecord]:
        if not self.file_path.exists():
            logger.error(f"File not found: {self.file_path}")
            return []

        logger.info(f"Parsing with GROBID: {self.file_path.name}")
        tei_xml = self._call_grobid()

        if not tei_xml:
            logger.warning(f"GROBID failed, falling back to pdfplumber for: {self.file_path.name}")
            from src.ingestion.parsers.icmr import ICMRParser
            fallback = ICMRParser(self.file_path)
            return fallback.parse()

        parsed = self._parse_tei(tei_xml)

        # Build full content: abstract + all section texts
        content_parts = []
        if parsed["abstract"]:
            content_parts.append(f"Abstract\n{parsed['abstract']}")
        for section in parsed["sections"]:
            if section["title"]:
                content_parts.append(f"{section['title']}\n{section['text']}")
            else:
                content_parts.append(section["text"])

        full_content = "\n\n".join(content_parts).strip()
        if not full_content:
            logger.warning(f"GROBID extracted no content from {self.file_path.name}")
            return []

        title = parsed["title"] or self.file_path.stem.replace("_", " ")
        study_type, evidence_level = infer_study_type(full_content[:500], title)

        doc = DocumentRecord(
            source_type=self.source_type,
            title=title,
            content=full_content,
            url=str(self.file_path.resolve()),
            published_date=str(parsed["year"]) if parsed["year"] else None,
            year=parsed["year"],
            journal=parsed["journal"],
            study_type=study_type,
            evidence_level=evidence_level,
            is_india_specific=False,
            parser_version="v2",
        )
        logger.info(f"GROBID parsed: {title[:60]} — {len(full_content)} chars")
        return [doc]
```

---

### Step 2.6 — OCR Parser for Scanned PDFs

**Create `src/ingestion/parsers/ocr.py`**:

```python
"""
OCR Parser
Uses Tesseract to extract text from scanned PDFs that pdfplumber cannot read.
Auto-detects scanned PDFs by checking if pdfplumber returns empty text.
"""
import re
from pathlib import Path
from loguru import logger

from src.ingestion.document_db import DocumentRecord
from src.ingestion.parsers.base import BaseParser
from src.ingestion.ner import infer_study_type


class OCRParser(BaseParser):
    """
    OCR fallback parser for scanned PDFs.
    Converts each page to image, runs Tesseract OCR, combines results.
    """

    def __init__(self, file_path: str | Path, source_type: str = "icmr"):
        self.file_path = Path(file_path)
        self._source_type = source_type

    @property
    def source_type(self) -> str:
        return self._source_type

    @staticmethod
    def is_scanned(file_path: Path) -> bool:
        """Check if a PDF is scanned (pdfplumber returns empty text on first pages)."""
        try:
            import pdfplumber
            with pdfplumber.open(file_path) as pdf:
                text_found = 0
                for page in pdf.pages[:5]:
                    text = page.extract_text() or ""
                    if len(text.strip()) > 50:
                        text_found += 1
                return text_found == 0
        except Exception:
            return False

    def parse(self) -> list[DocumentRecord]:
        if not self.file_path.exists():
            logger.error(f"File not found: {self.file_path}")
            return []

        try:
            import pytesseract
            from PIL import Image
            import pdfplumber
        except ImportError as e:
            logger.error(f"OCR dependencies not installed: {e}")
            return []

        logger.info(f"Running OCR on scanned PDF: {self.file_path.name}")

        try:
            with pdfplumber.open(self.file_path) as pdf:
                page_texts = []
                for i, page in enumerate(pdf.pages):
                    try:
                        # Convert page to image
                        img = page.to_image(resolution=200).original
                        text = pytesseract.image_to_string(img, lang="eng")
                        text = re.sub(r'\s+', ' ', text).strip()
                        if len(text) > 30:
                            page_texts.append(text)
                    except Exception as e:
                        logger.debug(f"OCR failed on page {i}: {e}")

            if not page_texts:
                logger.error(f"OCR extracted no text from {self.file_path.name}")
                return []

            full_text = "\n\n".join(page_texts)
            title = self.file_path.stem.replace("_", " ")
            study_type, evidence_level = infer_study_type(full_text[:500], title)

            doc = DocumentRecord(
                source_type=self.source_type,
                title=title,
                content=full_text,
                url=str(self.file_path.resolve()),
                study_type=study_type,
                evidence_level=evidence_level,
                is_india_specific=True,
                parser_version="v2",
            )
            logger.info(f"OCR complete: {title[:60]} — {len(full_text)} chars from {len(page_texts)} pages")
            return [doc]

        except Exception as e:
            logger.error(f"OCR parsing failed for {self.file_path.name}: {e}")
            return []
```

---

### Step 2.7 — Updated Ingestion Pipeline v2

**Create `src/ingestion/pipeline_v2.py`** (keep `pipeline.py` — do not delete):

```python
"""
Ingestion Pipeline v2
Uses hierarchical chunking, metadata extraction, NER, content classification.
Replaces pipeline.py for new ingestion. Old pipeline.py still works for compatibility.
"""
from datetime import datetime
from uuid import uuid4

from loguru import logger
from qdrant_client.models import PointStruct

from src.ingestion.document_db import ChunkRecord, DocumentRecord, get_db
from src.ingestion.embeddings import embed_texts
from src.ingestion.ner import extract_entities, classify_content_type, infer_study_type
from src.ingestion.vector_db import ensure_collection, upsert_chunks
from src.utils.chunker_v2 import chunk_text_v2


async def run_pipeline_v2(documents: list[DocumentRecord]) -> dict:
    """
    v2 ingestion pipeline with hierarchical chunking, NER, content classification.
    """
    summary = {
        "documents_stored": 0,
        "chunks_created": 0,
        "chunks_embedded": 0,
        "chunks_skipped_noise": 0,
    }

    if not documents:
        logger.info("No documents provided to v2 pipeline")
        return summary

    db = get_db()
    documents_col = db["documents"]
    chunks_col = db["chunks"]
    qdrant_ready = False

    for document in documents:
        logger.info(f"[v2] Processing: {document.title[:60]} ({document.source_type})")

        # Store document
        doc_dict = document.model_dump()
        insert_result = await documents_col.insert_one(doc_dict)
        document_id = str(insert_result.inserted_id)
        summary["documents_stored"] += 1

        # Hierarchical chunking
        text_chunks = chunk_text_v2(document.content)
        logger.info(f"[v2] Created {len(text_chunks)} chunks for: {document.title[:60]}")

        if not text_chunks:
            continue

        # Build ChunkRecord objects with v2 metadata
        chunk_records = []
        for text_chunk in text_chunks:
            # NER extraction
            entities = extract_entities(text_chunk.text)

            # Content classification
            content_type, weight = classify_content_type(
                text_chunk.text, text_chunk.section
            )

            # Skip noise chunks
            if content_type == "noise":
                summary["chunks_skipped_noise"] += 1
                continue

            # India-specific detection
            is_india = (
                document.is_india_specific or
                document.source_type in ("icmr", "nmc", "mohfw", "state_guideline") or
                any(w in text_chunk.text.lower() for w in ["india", "indian", "icmr", "nmc", "aiims"])
            )

            chunk_record = ChunkRecord(
                document_id=document_id,
                source_type=document.source_type,
                title=document.title,
                chunk_text=text_chunk.text,
                chunk_index=text_chunk.chunk_index,
                condition_tags=document.condition_tags,
                specialty_tags=document.specialty_tags,
                char_count=text_chunk.char_count,
                section=text_chunk.section,
                diseases=entities["diseases"],
                drugs=entities["drugs"],
                symptoms=entities["symptoms"],
                content_type=content_type,
                content_weight=weight,
                is_india_specific=is_india,
                evidence_level=document.evidence_level,
                token_count=text_chunk.token_count,
                parser_version="v2",
            )
            chunk_records.append(chunk_record)

        if not chunk_records:
            continue

        # Store chunks in MongoDB
        chunk_payloads = [c.model_dump() for c in chunk_records]
        insert_many_result = await chunks_col.insert_many(chunk_payloads)
        chunk_mongo_ids = [str(cid) for cid in insert_many_result.inserted_ids]
        summary["chunks_created"] += len(chunk_mongo_ids)

        # Embed and store in Qdrant
        try:
            texts = [c.chunk_text for c in chunk_records]
            embeddings = embed_texts(texts)

            if not qdrant_ready:
                ensure_collection()
                qdrant_ready = True

            points = []
            for chunk, vector, mongo_id in zip(chunk_records, embeddings, chunk_mongo_ids):
                points.append(PointStruct(
                    id=str(uuid4()),
                    vector=vector,
                    payload={
                        "mongo_id": mongo_id,
                        "source_type": chunk.source_type,
                        "title": chunk.title,
                        "condition_tags": chunk.condition_tags,
                        "chunk_text": chunk.chunk_text,
                        "section": chunk.section,
                        "diseases": chunk.diseases,
                        "drugs": chunk.drugs,
                        "content_type": chunk.content_type,
                        "content_weight": chunk.content_weight,
                        "is_india_specific": chunk.is_india_specific,
                        "evidence_level": chunk.evidence_level,
                        "parser_version": "v2",
                    },
                ))

            for start in range(0, len(points), 100):
                batch = points[start:start + 100]
                upsert_chunks(batch)

            embedded_at = datetime.utcnow()
            await chunks_col.update_many(
                {"_id": {"$in": insert_many_result.inserted_ids}},
                {"$set": {"embedded": True, "embedded_at": embedded_at}},
            )

            summary["chunks_embedded"] += len(points)
            logger.info(f"[v2] Embedded {len(points)} chunks for: {document.title[:60]}")

        except Exception as e:
            logger.error(f"[v2] Embedding failed for '{document.title}': {e}")

    # Update document total_chunks count
    logger.info(f"[v2] Pipeline complete: {summary}")
    return summary
```

---

### Step 2.8 — Re-ingestion Script

**Create `scripts/reingest_v2.py`**:

```python
"""
Re-ingestion Script v2
Clears old v1 data, re-ingests all ICMR PDFs using the v2 pipeline.
Run this once after Phase 2 is complete.
"""
import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from loguru import logger
from src.ingestion.parsers.icmr import ICMRParser
from src.ingestion.parsers.ocr import OCRParser, is_scanned_check
from src.ingestion.pipeline_v2 import run_pipeline_v2


async def clear_v1_data():
    """Remove all v1-ingested ICMR documents and their chunks from MongoDB and Qdrant."""
    from src.ingestion.document_db import get_db
    from src.ingestion.vector_db import get_qdrant
    from src.core.config import get_settings

    settings = get_settings()
    db = get_db()

    logger.info("Clearing v1 ICMR data from MongoDB...")
    result = await db["documents"].delete_many({
        "source_type": "icmr",
        "$or": [
            {"parser_version": "v1"},
            {"parser_version": {"$exists": False}},
        ]
    })
    logger.info(f"Deleted {result.deleted_count} v1 documents")

    chunk_result = await db["chunks"].delete_many({
        "source_type": "icmr",
        "$or": [
            {"parser_version": "v1"},
            {"parser_version": {"$exists": False}},
        ]
    })
    logger.info(f"Deleted {chunk_result.deleted_count} v1 chunks")

    # Clear Qdrant collection entirely and recreate
    client = get_qdrant()
    client.delete_collection(settings.qdrant_collection)
    logger.info("Qdrant collection cleared")


async def main():
    icmr_dir = Path("data/raw/icmr")
    if not icmr_dir.exists():
        print("data/raw/icmr/ not found. Add ICMR PDFs first.")
        return

    pdf_files = sorted(icmr_dir.glob("*.pdf"))
    if not pdf_files:
        print("No PDFs found in data/raw/icmr/")
        return

    print(f"Found {len(pdf_files)} PDFs")
    print("Clearing v1 data...")
    await clear_v1_data()

    all_documents = []
    parsed = 0
    ocr_used = 0
    failed = 0

    for pdf_path in pdf_files:
        # Try pdfplumber first, fall back to OCR for scanned PDFs
        parser = ICMRParser(pdf_path)
        docs = parser.parse()

        if not docs or not docs[0].content.strip():
            logger.warning(f"pdfplumber got no text, trying OCR: {pdf_path.name}")
            parser = OCRParser(pdf_path, source_type="icmr")
            docs = parser.parse()
            if docs:
                ocr_used += 1

        if docs:
            # Mark as India-specific for ICMR sources
            for doc in docs:
                doc.is_india_specific = True
                doc.parser_version = "v2"
            all_documents.extend(docs)
            parsed += 1
        else:
            logger.error(f"Failed to parse: {pdf_path.name}")
            failed += 1

    print(f"\nParsing complete:")
    print(f"  Parsed: {parsed}")
    print(f"  OCR used: {ocr_used}")
    print(f"  Failed: {failed}")
    print(f"\nStarting v2 pipeline for {len(all_documents)} documents...")

    summary = await run_pipeline_v2(all_documents)

    print(f"\nRe-ingestion complete:")
    print(f"  Documents stored: {summary['documents_stored']}")
    print(f"  Chunks created:   {summary['chunks_created']}")
    print(f"  Chunks embedded:  {summary['chunks_embedded']}")
    print(f"  Noise skipped:    {summary['chunks_skipped_noise']}")


if __name__ == "__main__":
    asyncio.run(main())
```

---

## 6. Verification — Run In Order

```bash
# Step 1 — GROBID running
curl http://localhost:8070/api/isalive

# Step 2 — NER works
python -c "
from src.ingestion.ner import extract_entities, classify_content_type
text = 'Doxycycline 200mg/day for 7 days is recommended for treatment of scrub typhus.'
entities = extract_entities(text)
content_type, weight = classify_content_type(text)
print('Entities:', entities)
print('Content type:', content_type, '| Weight:', weight)
"

# Step 3 — v2 chunker works
python -c "
from src.utils.chunker_v2 import chunk_text_v2
text = '''Treatment Guidelines
Doxycycline is the drug of choice. Give 200mg daily in two divided doses for 7 days.

Background
Scrub typhus was first described in Japan in 1810.

References
1. WHO Guidelines 2020
2. ICMR Task Force Report'''
chunks = chunk_text_v2(text)
for c in chunks:
    print(f'Section: {c.section} | Type level: {c.level} | Tokens: {c.token_count}')
    print(f'  Text: {c.text[:80]}...')
    print()
"

# Step 4 — Full re-ingestion
python scripts/reingest_v2.py

# Step 5 — Verify counts
python -c "
import asyncio
from src.ingestion.document_db import get_db
async def check():
    db = get_db()
    total_docs = await db['documents'].count_documents({})
    v2_docs = await db['documents'].count_documents({'parser_version': 'v2'})
    clinical = await db['chunks'].count_documents({'content_type': 'clinical'})
    noise = await db['chunks'].count_documents({'content_type': 'noise'})
    total_chunks = await db['chunks'].count_documents({})
    print(f'Total documents: {total_docs} (v2: {v2_docs})')
    print(f'Total chunks: {total_chunks}')
    print(f'Clinical chunks: {clinical}')
    print(f'Noise chunks (skipped): {noise}')
asyncio.run(check())
"

# Step 6 — Test query quality improvement
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "what are hospital infection prevention protocols", "top_k": 5}'
```

The hospital infection query should now return clinical protocols — not forewords or committee acknowledgements. If clinical chunks are being returned and noise is absent from citations, Phase 2 is working.

---

## 7. Rules

- Never delete `pipeline.py`, `chunker.py`, or any v1 file — keep them for compatibility
- All new files use `pipeline_v2.py`, `chunker_v2.py` naming
- Absolute imports only (`src.*`)
- Never hardcode model names, URLs, API keys — always `get_settings()`
- `loguru` throughout — never `print()` inside `src/`
- If GROBID is down, parsers must fall back gracefully — never crash
- If scispaCy model is missing, NER must fall back to rule-based — never crash
- Run verification steps in order — do not skip

---

*OpenInsight Phase 2 — SentArc Labs | Director: Aditya Singh | adi.singh1426@gmail.com*
