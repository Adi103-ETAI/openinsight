# OpenInsight v2 — Full Implementation Guide
> Every step, every detail, every code snippet. Read this entirely before writing any code.

---

## Table of Contents
1. [Project Context](#1-project-context)
2. [Current State Diagnosis](#2-current-state-diagnosis)
3. [Phase A — Data Ingestion Layer](#3-phase-a--data-ingestion-layer)
   - [A1 — Source-Aware Parser](#a1--source-aware-parser)
   - [A2 — Hierarchical Chunker](#a2--hierarchical-chunker)
   - [A3 — Metadata Schema](#a3--metadata-schema)
   - [A4 — Dual Embedding](#a4--dual-embedding)
   - [A5 — Qdrant Indexing](#a5--qdrant-indexing)
   - [A6 — MongoDB Full-Doc Store](#a6--mongodb-full-doc-store)
   - [A7 — Batch Ingestion Pipeline](#a7--batch-ingestion-pipeline)
4. [Phase B — Standard Search (Query Path)](#4-phase-b--standard-search-query-path)
   - [B1 — Query Understanding Layer](#b1--query-understanding-layer)
   - [B2 — Parallel Hybrid Retrieval](#b2--parallel-hybrid-retrieval)
   - [B3 — RRF Fusion + Metadata Boost](#b3--rrf-fusion--metadata-boost)
   - [B4 — Two-Stage Reranking](#b4--two-stage-reranking)
   - [B5 — MMR Deduplication](#b5--mmr-deduplication)
   - [B6 — Context Assembly](#b6--context-assembly)
   - [B7 — Redis Caching](#b7--redis-caching)
5. [Speed Optimisations](#5-speed-optimisations)
6. [File and Folder Structure](#6-file-and-folder-structure)
7. [Environment Variables](#7-environment-variables)
8. [Docker Compose Changes](#8-docker-compose-changes)
9. [Implementation Order](#9-implementation-order)
10. [Testing Checklist](#10-testing-checklist)

---

## 1. Project Context

**What OpenInsight is:** A clinical decision support platform for Indian physicians. Doctors ask clinical questions and get grounded, evidence-based answers with citations from PubMed, ICMR guidelines, and Cochrane reviews.

**Stack:**
- Backend: FastAPI + Python
- Vector DB: Qdrant (hybrid search enabled)
- LLM: Llama 3.1 70B via NVIDIA NIM
- Embeddings: `pritamdeka/S-PubMedBert-MS-MARCO`
- Full-doc store: MongoDB (Motor async driver)
- Cache: Redis
- Dev environment: GitHub Codespaces + Docker Compose

**Current corpus:** ~1,361 docs, ~1,402 vectors (barely chunked — this is the core problem).

**Target after rebuild:** ~15,000–25,000 high-quality, contextually-embedded chunks from the same documents.

---

## 2. Current State Diagnosis

### What is broken and why

| Problem | Root Cause | Impact |
|---|---|---|
| ~1 chunk per doc | No real chunking logic | Terrible recall — 99% of content is never retrieved |
| Flat text embedding | Raw chunk text embedded without context | Model doesn't know what paper a paragraph belongs to |
| Cross-encoder on 50+ candidates | No fusion step to reduce candidates first | Slow reranking (~3–5 seconds) |
| Sequential retrieval | Dense then sparse, not parallel | Extra latency for no reason |
| No MMR | Duplicate chunks sent to LLM | Context window wasted on same paragraph variants |
| No metadata filtering | All docs searched regardless of relevance signals | Noise in results |
| No evidence-level boosting | RCT ranked same as case report | Clinical relevance suffers |
| Tables embedded as raw text | No structured extraction | Table data is unretrievable noise |

---

## 3. Phase A — Data Ingestion Layer

### A1 — Source-Aware Parser

**Location:** `app/ingestion/parsers.py`

The parser must route each document to the correct parsing strategy based on its source type. Do not treat all documents the same.

```python
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import xml.etree.ElementTree as ET

class SourceType(Enum):
    PUBMED_XML = "pubmed_xml"
    PDF_GUIDELINE = "pdf_guideline"
    PDF_PAPER = "pdf_paper"
    COCHRANE_HTML = "cochrane_html"
    NMC_PDF = "nmc_pdf"

@dataclass
class ParsedSection:
    title: str              # e.g. "Results", "Methods", "Discussion"
    text: str               # full section text (before chunking)
    section_index: int      # position in document
    has_table: bool = False
    tables: List[Dict] = field(default_factory=list)

@dataclass
class ParsedDocument:
    doc_id: str             # PMID, DOI, or filename hash
    source_type: SourceType
    title: str
    abstract: str
    authors: List[str]
    year: int
    journal: str
    doi: Optional[str]
    pmid: Optional[str]
    sections: List[ParsedSection]
    mesh_terms: List[str]
    keywords: List[str]
    raw_text: str           # full text for BM25 fallback
    metadata: Dict[str, Any] = field(default_factory=dict)


class DocumentParser:
    def parse(self, path: str, source_type: SourceType) -> ParsedDocument:
        if source_type == SourceType.PUBMED_XML:
            return self._parse_pubmed_xml(path)
        elif source_type in (SourceType.PDF_GUIDELINE, SourceType.PDF_PAPER, SourceType.NMC_PDF):
            return self._parse_grobid_pdf(path, source_type)
        elif source_type == SourceType.COCHRANE_HTML:
            return self._parse_cochrane_html(path)
        else:
            raise ValueError(f"Unknown source type: {source_type}")

    def _parse_pubmed_xml(self, path: str) -> ParsedDocument:
        """
        Parse PubMed XML format. Handles both single-article and
        PubmedArticleSet batch exports.
        PubMed XML structure:
          PubmedArticle > MedlineCitation > Article > ArticleTitle
                                                     > Abstract > AbstractText
                                                     > AuthorList
          PubmedArticle > MedlineCitation > MeshHeadingList
          PubmedArticle > PubmedData > ArticleIdList
        """
        tree = ET.parse(path)
        root = tree.getroot()
        # Handle both PubmedArticleSet and single PubmedArticle
        articles = root.findall('.//PubmedArticle')
        if not articles:
            articles = [root]

        results = []
        for article in articles:
            citation = article.find('MedlineCitation')
            art = citation.find('Article')

            title = art.findtext('ArticleTitle', '').strip()

            # Abstract may have multiple structured sections (Background, Methods, etc.)
            abstract_texts = art.findall('.//AbstractText')
            abstract_parts = []
            abstract_sections = []
            for ab in abstract_texts:
                label = ab.get('Label', '')
                text = ab.text or ''
                if label:
                    abstract_parts.append(f"{label}: {text}")
                    abstract_sections.append(ParsedSection(
                        title=label,
                        text=text,
                        section_index=len(abstract_sections)
                    ))
                else:
                    abstract_parts.append(text)
            abstract = '\n'.join(abstract_parts)

            # Authors
            authors = []
            for author in art.findall('.//Author'):
                last = author.findtext('LastName', '')
                fore = author.findtext('ForeName', '')
                if last:
                    authors.append(f"{last} {fore}".strip())

            # Year
            year_el = citation.find('.//PubDate/Year')
            year = int(year_el.text) if year_el is not None and year_el.text else 0

            # Journal
            journal = art.findtext('.//Journal/Title', '') or art.findtext('.//Journal/ISOAbbreviation', '')

            # IDs
            pmid = citation.findtext('PMID', '')
            doi = ''
            for id_el in article.findall('.//ArticleId'):
                if id_el.get('IdType') == 'doi':
                    doi = id_el.text or ''

            # MeSH terms
            mesh_terms = []
            for mesh in citation.findall('.//MeshHeading'):
                descriptor = mesh.findtext('DescriptorName', '')
                if descriptor:
                    mesh_terms.append(descriptor)

            # Keywords
            keywords = [kw.text for kw in citation.findall('.//Keyword') if kw.text]

            results.append(ParsedDocument(
                doc_id=f"pmid_{pmid}" if pmid else f"doi_{doi.replace('/', '_')}",
                source_type=SourceType.PUBMED_XML,
                title=title,
                abstract=abstract,
                authors=authors,
                year=year,
                journal=journal,
                doi=doi,
                pmid=pmid,
                sections=abstract_sections,
                mesh_terms=mesh_terms,
                keywords=keywords,
                raw_text=f"{title}\n{abstract}"
            ))

        return results[0] if len(results) == 1 else results

    def _parse_grobid_pdf(self, path: str, source_type: SourceType) -> ParsedDocument:
        """
        Use GROBID REST API to extract structured XML from PDF.
        GROBID must be running (docker-compose service).
        Endpoint: POST /api/processFulltextDocument
        Returns TEI XML with <div> sections.
        """
        import httpx
        import hashlib

        with open(path, 'rb') as f:
            pdf_bytes = f.read()

        # Call GROBID
        grobid_url = "http://grobid:8070/api/processFulltextDocument"
        response = httpx.post(
            grobid_url,
            files={"input": (path, pdf_bytes, "application/pdf")},
            data={"consolidateHeader": "1", "consolidateCitations": "0"},
            timeout=60.0
        )
        response.raise_for_status()
        tei_xml = response.text

        return self._parse_tei_xml(tei_xml, path, source_type)

    def _parse_tei_xml(self, tei_xml: str, original_path: str, source_type: SourceType) -> ParsedDocument:
        """Parse GROBID TEI XML output into ParsedDocument."""
        import hashlib
        from lxml import etree

        ns = {'tei': 'http://www.tei-c.org/ns/1.0'}
        root = etree.fromstring(tei_xml.encode())

        # Title
        title_el = root.find('.//tei:titleStmt/tei:title', ns)
        title = title_el.text.strip() if title_el is not None and title_el.text else ''

        # Abstract
        abstract_el = root.find('.//tei:abstract', ns)
        abstract = ' '.join(abstract_el.itertext()).strip() if abstract_el is not None else ''

        # Authors
        authors = []
        for pers in root.findall('.//tei:author/tei:persName', ns):
            forename = pers.findtext('tei:forename', '', ns)
            surname = pers.findtext('tei:surname', '', ns)
            if surname:
                authors.append(f"{surname} {forename}".strip())

        # Year
        date_el = root.find('.//tei:publicationStmt/tei:date[@type="published"]', ns)
        year = 0
        if date_el is not None:
            when = date_el.get('when', '')
            if when and len(when) >= 4:
                try:
                    year = int(when[:4])
                except ValueError:
                    pass

        # Journal
        journal_el = root.find('.//tei:monogr/tei:title[@level="j"]', ns)
        journal = journal_el.text.strip() if journal_el is not None and journal_el.text else ''

        # Sections from body
        sections = []
        body = root.find('.//tei:body', ns)
        if body is not None:
            for idx, div in enumerate(body.findall('.//tei:div', ns)):
                head_el = div.find('tei:head', ns)
                section_title = head_el.text.strip() if head_el is not None and head_el.text else f"Section {idx+1}"
                paragraphs = div.findall('tei:p', ns)
                section_text = '\n'.join(' '.join(p.itertext()).strip() for p in paragraphs)
                # Detect tables
                tables_raw = div.findall('.//tei:figure[@type="table"]', ns)
                tables = []
                for t in tables_raw:
                    caption = ' '.join(t.find('tei:head', ns).itertext()) if t.find('tei:head', ns) is not None else ''
                    table_text = ' '.join(t.itertext())
                    tables.append({'caption': caption, 'text': table_text})

                if section_text.strip():
                    sections.append(ParsedSection(
                        title=section_title,
                        text=section_text,
                        section_index=idx,
                        has_table=len(tables) > 0,
                        tables=tables
                    ))

        doc_id = hashlib.md5(original_path.encode()).hexdigest()[:12]
        raw_text = f"{title}\n{abstract}\n" + "\n".join(s.text for s in sections)

        return ParsedDocument(
            doc_id=f"pdf_{doc_id}",
            source_type=source_type,
            title=title,
            abstract=abstract,
            authors=authors,
            year=year,
            journal=journal,
            doi=None,
            pmid=None,
            sections=sections,
            mesh_terms=[],
            keywords=[],
            raw_text=raw_text
        )
```

**OCR fallback:** If GROBID returns empty sections (scanned PDF), fall back to pytesseract:

```python
def _ocr_fallback(self, path: str) -> str:
    """Fallback OCR using pytesseract when GROBID fails on scanned PDFs."""
    from pdf2image import convert_from_path
    import pytesseract
    pages = convert_from_path(path, dpi=200)
    return '\n'.join(pytesseract.image_to_string(page) for page in pages)
```

---

### A2 — Hierarchical Chunker

**Location:** `app/ingestion/chunker.py`

This is the most critical component. The goal is three-level hierarchy with upward context propagation.

```python
from dataclasses import dataclass, field
from typing import List, Optional
import re

@dataclass
class Chunk:
    chunk_id: str           # "{doc_id}_s{section_idx}_p{para_idx}"
    doc_id: str
    chunk_type: str         # "abstract" | "paragraph" | "table" | "fact" | "doc_summary"
    section_title: str
    text: str               # raw text (stored in MongoDB)
    contextual_text: str    # prefixed text (what gets embedded in Qdrant)
    char_count: int
    token_estimate: int     # char_count // 4 (rough estimate)
    chunk_index: int        # position within document
    total_chunks: int       # total chunks for this doc (filled after all chunks generated)
    metadata: dict = field(default_factory=dict)


class HierarchicalChunker:
    """
    Three levels:
      L0 — doc_summary: title + abstract + MeSH (1 chunk per doc, always)
      L1 — section: one chunk per major section if section is short (<600 tokens)
      L2 — paragraph: sentence-window chunks from long sections
    
    Plus special types: table chunks, atomic fact chunks.
    
    Contextual prefix format for ALL chunks:
      "Source: {source}\nType: {doc_type}\nTitle: {title}\nSection: {section_title}\n\n{chunk_text}"
    
    This prefix is embedded but NOT stored in the text field (store raw text only).
    The prefix dramatically improves retrieval because the embedding captures
    what the paragraph means in context of its source paper.
    """

    TARGET_CHUNK_TOKENS = 350      # target size for paragraph chunks
    MAX_CHUNK_TOKENS = 500         # hard max before forced split
    OVERLAP_TOKENS = 50            # overlap between consecutive chunks
    MIN_CHUNK_TOKENS = 80          # ignore chunks smaller than this (noise)

    def chunk_document(self, doc: 'ParsedDocument', doc_metadata: dict) -> List[Chunk]:
        all_chunks = []
        chunk_counter = 0

        source = doc_metadata.get('source', 'unknown')
        doc_type = doc_metadata.get('doc_type', 'unknown')
        title = doc.title

        def make_contextual(section_title: str, text: str) -> str:
            return (
                f"Source: {source}\n"
                f"Document type: {doc_type}\n"
                f"Title: {title}\n"
                f"Section: {section_title}\n\n"
                f"{text}"
            )

        # ── L0: Document summary chunk ──────────────────────────────────────
        summary_text = f"{title}\n\n{doc.abstract}"
        if doc.mesh_terms:
            summary_text += f"\n\nMeSH terms: {', '.join(doc.mesh_terms[:15])}"
        if doc.keywords:
            summary_text += f"\nKeywords: {', '.join(doc.keywords[:10])}"

        all_chunks.append(Chunk(
            chunk_id=f"{doc.doc_id}_summary",
            doc_id=doc.doc_id,
            chunk_type="doc_summary",
            section_title="Document Summary",
            text=summary_text,
            contextual_text=make_contextual("Document Summary", summary_text),
            char_count=len(summary_text),
            token_estimate=len(summary_text) // 4,
            chunk_index=chunk_counter,
            total_chunks=0,  # filled later
            metadata=doc_metadata
        ))
        chunk_counter += 1

        # ── L1/L2: Sections and paragraphs ──────────────────────────────────
        for section in doc.sections:
            # Handle tables first — extract them as special chunks
            for table_idx, table in enumerate(section.tables):
                table_text = self._format_table_chunk(table)
                if len(table_text) // 4 >= self.MIN_CHUNK_TOKENS:
                    all_chunks.append(Chunk(
                        chunk_id=f"{doc.doc_id}_s{section.section_index}_t{table_idx}",
                        doc_id=doc.doc_id,
                        chunk_type="table",
                        section_title=section.title,
                        text=table_text,
                        contextual_text=make_contextual(section.title, table_text),
                        char_count=len(table_text),
                        token_estimate=len(table_text) // 4,
                        chunk_index=chunk_counter,
                        total_chunks=0,
                        metadata=doc_metadata
                    ))
                    chunk_counter += 1

            section_tokens = len(section.text) // 4

            # Short section — keep as one chunk (L1)
            if section_tokens <= self.TARGET_CHUNK_TOKENS:
                if section_tokens >= self.MIN_CHUNK_TOKENS:
                    all_chunks.append(Chunk(
                        chunk_id=f"{doc.doc_id}_s{section.section_index}_p0",
                        doc_id=doc.doc_id,
                        chunk_type="paragraph",
                        section_title=section.title,
                        text=section.text,
                        contextual_text=make_contextual(section.title, section.text),
                        char_count=len(section.text),
                        token_estimate=section_tokens,
                        chunk_index=chunk_counter,
                        total_chunks=0,
                        metadata=doc_metadata
                    ))
                    chunk_counter += 1

            # Long section — split into sentence-window paragraph chunks (L2)
            else:
                para_chunks = self._sentence_window_split(section.text)
                for p_idx, para_text in enumerate(para_chunks):
                    if len(para_text) // 4 < self.MIN_CHUNK_TOKENS:
                        continue
                    all_chunks.append(Chunk(
                        chunk_id=f"{doc.doc_id}_s{section.section_index}_p{p_idx}",
                        doc_id=doc.doc_id,
                        chunk_type="paragraph",
                        section_title=section.title,
                        text=para_text,
                        contextual_text=make_contextual(section.title, para_text),
                        char_count=len(para_text),
                        token_estimate=len(para_text) // 4,
                        chunk_index=chunk_counter,
                        total_chunks=0,
                        metadata=doc_metadata
                    ))
                    chunk_counter += 1

        # Fill total_chunks now that we know the count
        total = len(all_chunks)
        for chunk in all_chunks:
            chunk.total_chunks = total

        return all_chunks

    def _sentence_window_split(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks by sentence boundaries.
        Target: TARGET_CHUNK_TOKENS tokens per chunk.
        Overlap: OVERLAP_TOKENS tokens (carry last N tokens into next chunk).
        
        Strategy:
          1. Split into sentences using a regex (handles abbreviations poorly
             but good enough for medical text; replace with spaCy sentencizer
             for production).
          2. Greedily add sentences until approaching target token count.
          3. Start new chunk, carrying over overlap sentences.
        """
        sentences = self._split_sentences(text)
        if not sentences:
            return []

        chunks = []
        current_sentences = []
        current_tokens = 0

        for sent in sentences:
            sent_tokens = len(sent) // 4

            # If adding this sentence would exceed max, flush current chunk
            if current_tokens + sent_tokens > self.MAX_CHUNK_TOKENS and current_sentences:
                chunk_text = ' '.join(current_sentences)
                chunks.append(chunk_text)

                # Carry overlap: roll back sentences until we have ~OVERLAP_TOKENS
                overlap_sentences = []
                overlap_tokens = 0
                for s in reversed(current_sentences):
                    s_tok = len(s) // 4
                    if overlap_tokens + s_tok > self.OVERLAP_TOKENS:
                        break
                    overlap_sentences.insert(0, s)
                    overlap_tokens += s_tok

                current_sentences = overlap_sentences
                current_tokens = overlap_tokens

            current_sentences.append(sent)
            current_tokens += sent_tokens

        # Don't forget the last chunk
        if current_sentences:
            chunk_text = ' '.join(current_sentences)
            if len(chunk_text) // 4 >= self.MIN_CHUNK_TOKENS:
                chunks.append(chunk_text)

        return chunks

    def _split_sentences(self, text: str) -> List[str]:
        """
        Sentence splitter that handles common medical abbreviations.
        Abbreviations to NOT split on: et al., Fig., e.g., i.e., vs., Dr., et al.
        Falls back to period + space + uppercase heuristic.
        """
        # Protect known abbreviations with placeholder
        abbreviations = [
            'et al.', 'Fig.', 'Figs.', 'e.g.', 'i.e.', 'vs.',
            'Dr.', 'Prof.', 'approx.', 'min.', 'max.', 'p.o.',
            'i.v.', 's.c.', 'b.i.d.', 't.i.d.', 'q.d.', 'mg/dL',
            'No.', 'Vol.', 'pp.', 'ed.'
        ]
        protected = text
        placeholders = {}
        for i, abbr in enumerate(abbreviations):
            placeholder = f"__ABBR{i}__"
            placeholders[placeholder] = abbr
            protected = protected.replace(abbr, placeholder)

        # Split on sentence boundaries
        pattern = r'(?<=[.!?])\s+(?=[A-Z])'
        raw_sentences = re.split(pattern, protected)

        # Restore abbreviations
        sentences = []
        for sent in raw_sentences:
            for placeholder, abbr in placeholders.items():
                sent = sent.replace(placeholder, abbr)
            sent = sent.strip()
            if sent:
                sentences.append(sent)

        return sentences

    def _format_table_chunk(self, table: dict) -> str:
        """
        Format a table dict into a searchable text chunk.
        Tables are not embedded as raw ASCII — they're formatted as
        a prose-readable description.
        
        Input: {'caption': '...', 'text': '...', 'headers': [...], 'rows': [[...]]}
        Output: Structured text with caption + flattened key-value pairs
        """
        parts = []
        if table.get('caption'):
            parts.append(f"Table: {table['caption']}")
        if table.get('headers') and table.get('rows'):
            # Structured table format
            headers = table['headers']
            for row in table['rows'][:20]:  # cap at 20 rows
                row_parts = []
                for h, v in zip(headers, row):
                    if v:
                        row_parts.append(f"{h}: {v}")
                if row_parts:
                    parts.append(' | '.join(row_parts))
        elif table.get('text'):
            # Unstructured — use raw text but truncate
            parts.append(table['text'][:1000])

        return '\n'.join(parts)
```

---

### A3 — Metadata Schema

**Location:** `app/ingestion/metadata.py`

Every chunk stored in Qdrant has a payload with this schema. This powers metadata filter inference in the query path.

```python
from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum


class DocType(str, Enum):
    RCT = "rct"
    SYSTEMATIC_REVIEW = "systematic_review"
    META_ANALYSIS = "meta_analysis"
    GUIDELINE = "guideline"
    REVIEW = "review"
    CASE_REPORT = "case_report"
    EDITORIAL = "editorial"
    COHORT = "cohort"
    UNKNOWN = "unknown"


class EvidenceLevel(str, Enum):
    # Oxford Centre for Evidence-Based Medicine levels
    LEVEL_1A = "1a"   # systematic review of RCTs
    LEVEL_1B = "1b"   # individual RCT
    LEVEL_2A = "2a"   # systematic review of cohort studies
    LEVEL_2B = "2b"   # individual cohort study
    LEVEL_3 = "3"     # case-control study
    LEVEL_4 = "4"     # case series
    LEVEL_5 = "5"     # expert opinion / guideline
    UNKNOWN = "unknown"


# Evidence level lookup — inferred from doc_type
DOC_TYPE_TO_EVIDENCE_LEVEL = {
    DocType.META_ANALYSIS: EvidenceLevel.LEVEL_1A,
    DocType.SYSTEMATIC_REVIEW: EvidenceLevel.LEVEL_1A,
    DocType.RCT: EvidenceLevel.LEVEL_1B,
    DocType.COHORT: EvidenceLevel.LEVEL_2B,
    DocType.GUIDELINE: EvidenceLevel.LEVEL_5,
    DocType.REVIEW: EvidenceLevel.LEVEL_5,
    DocType.CASE_REPORT: EvidenceLevel.LEVEL_4,
    DocType.EDITORIAL: EvidenceLevel.LEVEL_5,
    DocType.UNKNOWN: EvidenceLevel.UNKNOWN,
}

# Evidence level score for boosting (higher = more credible)
EVIDENCE_BOOST_SCORE = {
    EvidenceLevel.LEVEL_1A: 1.35,
    EvidenceLevel.LEVEL_1B: 1.25,
    EvidenceLevel.LEVEL_2A: 1.15,
    EvidenceLevel.LEVEL_2B: 1.10,
    EvidenceLevel.LEVEL_3: 1.05,
    EvidenceLevel.LEVEL_4: 1.00,
    EvidenceLevel.LEVEL_5: 1.10,  # guidelines still boosted for clinical use
    EvidenceLevel.UNKNOWN: 1.00,
}


@dataclass
class ChunkMetadata:
    # Identity
    doc_id: str
    chunk_id: str
    chunk_type: str         # paragraph | table | doc_summary | fact

    # Source info
    source: str             # pubmed | icmr | cochrane | nmc_guideline | who_guideline | rssdi
    doc_type: DocType
    evidence_level: EvidenceLevel

    # Publication details
    title: str
    year: int
    journal: str
    authors: List[str]
    doi: Optional[str]
    pmid: Optional[str]

    # Medical classification
    specialty: List[str]    # cardiology | endocrinology | pulmonology | etc.
    mesh_terms: List[str]
    keywords: List[str]

    # Chunk position
    section_title: str
    chunk_index: int
    total_chunks: int

    # India-specific flags (critical for OpenInsight's use case)
    india_relevant: bool = False
    has_indian_data: bool = False  # study has Indian participants
    indian_source: bool = False    # ICMR, NMC, RSSDI, JAPI, NMJ

    # Content flags
    has_table: bool = False
    has_drug_dosing: bool = False  # contains dosing information
    has_lab_values: bool = False   # contains reference ranges

    def to_qdrant_payload(self) -> dict:
        """Serialize to flat dict for Qdrant payload storage."""
        return {
            "doc_id": self.doc_id,
            "chunk_id": self.chunk_id,
            "chunk_type": self.chunk_type,
            "source": self.source,
            "doc_type": self.doc_type.value,
            "evidence_level": self.evidence_level.value,
            "evidence_boost": EVIDENCE_BOOST_SCORE[self.evidence_level],
            "title": self.title,
            "year": self.year,
            "journal": self.journal,
            "doi": self.doi or "",
            "pmid": self.pmid or "",
            "specialty": self.specialty,
            "mesh_terms": self.mesh_terms,
            "keywords": self.keywords,
            "section_title": self.section_title,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "india_relevant": self.india_relevant,
            "has_indian_data": self.has_indian_data,
            "indian_source": self.indian_source,
            "has_table": self.has_table,
            "has_drug_dosing": self.has_drug_dosing,
            "has_lab_values": self.has_lab_values,
        }


class MetadataEnricher:
    """
    Enriches parsed documents with inferred metadata:
    - doc_type classification from title/journal
    - specialty detection via keyword matching
    - india_relevant flags
    - drug dosing / lab value detection
    """

    INDIA_KEYWORDS = [
        "india", "indian", "icmr", "nmc", "aiims", "pgimer", "jipmer",
        "mumbai", "delhi", "chennai", "kolkata", "bangalore", "hyderabad",
        "rssdi", "cardiological society of india", "japi", "nmj", "ijmr"
    ]

    SPECIALTY_KEYWORDS = {
        "cardiology": ["cardiac", "heart", "coronary", "myocardial", "arrhythmia",
                       "atrial fibrillation", "heart failure", "hypertension", "ecg"],
        "endocrinology": ["diabetes", "insulin", "thyroid", "metformin", "hba1c",
                          "glucose", "pancreas", "adrenal", "hormone"],
        "pulmonology": ["lung", "respiratory", "asthma", "copd", "pneumonia",
                        "tuberculosis", "tb", "bronchitis", "spirometry"],
        "neurology": ["stroke", "seizure", "epilepsy", "parkinson", "alzheimer",
                      "dementia", "neuropathy", "brain", "neurological"],
        "infectious_disease": ["infection", "antibiotic", "antimicrobial", "viral",
                               "bacterial", "sepsis", "hiv", "covid", "malaria",
                               "dengue", "typhoid", "rickettsial"],
        "gastroenterology": ["liver", "hepatitis", "cirrhosis", "gastric", "peptic",
                             "ibd", "crohn", "colitis", "bowel"],
        "oncology": ["cancer", "tumor", "malignant", "chemotherapy", "oncology",
                     "carcinoma", "lymphoma", "leukemia"],
        "nephrology": ["kidney", "renal", "creatinine", "dialysis", "gfr", "nephritis"],
    }

    INDIAN_JOURNALS = [
        "japi", "journal of association of physicians of india",
        "national medical journal", "nmj", "indian journal of medical research",
        "ijmr", "indian heart journal", "journal of indian medical association"
    ]

    RCT_TITLE_PATTERNS = [
        "randomized", "randomised", "randomization", "randomisation",
        "controlled trial", "rct"
    ]

    SYSTEMATIC_REVIEW_PATTERNS = [
        "systematic review", "meta-analysis", "meta analysis", "cochrane"
    ]

    def enrich(self, doc: 'ParsedDocument', source: str) -> dict:
        """Return enriched metadata dict for use in ChunkMetadata."""
        full_text_lower = (
            doc.title + " " + doc.abstract + " " +
            " ".join(s.text for s in doc.sections)
        ).lower()

        doc_type = self._infer_doc_type(doc, full_text_lower)
        evidence_level = DOC_TYPE_TO_EVIDENCE_LEVEL[doc_type]
        specialty = self._detect_specialties(full_text_lower)
        india_relevant, has_indian_data, indian_source = self._detect_india_relevance(
            doc, full_text_lower, source
        )

        return {
            "source": source,
            "doc_type": doc_type,
            "evidence_level": evidence_level,
            "specialty": specialty,
            "india_relevant": india_relevant,
            "has_indian_data": has_indian_data,
            "indian_source": indian_source,
            "has_drug_dosing": self._detect_drug_dosing(full_text_lower),
            "has_lab_values": self._detect_lab_values(full_text_lower),
        }

    def _infer_doc_type(self, doc: 'ParsedDocument', text_lower: str) -> DocType:
        if any(p in text_lower for p in self.SYSTEMATIC_REVIEW_PATTERNS):
            if "meta-analysis" in text_lower or "meta analysis" in text_lower:
                return DocType.META_ANALYSIS
            return DocType.SYSTEMATIC_REVIEW
        if any(p in text_lower for p in self.RCT_TITLE_PATTERNS):
            return DocType.RCT
        if doc.source_type.value in ("pdf_guideline", "nmc_pdf"):
            return DocType.GUIDELINE
        if "guideline" in text_lower or "recommendation" in text_lower:
            return DocType.GUIDELINE
        if "review" in doc.title.lower():
            return DocType.REVIEW
        return DocType.UNKNOWN

    def _detect_specialties(self, text_lower: str) -> List[str]:
        found = []
        for specialty, keywords in self.SPECIALTY_KEYWORDS.items():
            if any(kw in text_lower for kw in keywords):
                found.append(specialty)
        return found[:5]  # max 5 specialties per document

    def _detect_india_relevance(self, doc, text_lower: str, source: str):
        india_relevant = any(kw in text_lower for kw in self.INDIA_KEYWORDS)
        has_indian_data = any(kw in text_lower for kw in [
            "india", "indian", "delhi", "mumbai", "chennai", "kolkata"
        ])
        indian_source = (
            source in ("icmr", "nmc_guideline", "rssdi") or
            any(j in doc.journal.lower() for j in self.INDIAN_JOURNALS)
        )
        if indian_source:
            india_relevant = True
        return india_relevant, has_indian_data, indian_source

    def _detect_drug_dosing(self, text_lower: str) -> bool:
        dosing_patterns = ["mg/kg", "mg/day", "mg twice", "mg once", "units/kg",
                           "dosage", "dose of", "starting dose", "maximum dose"]
        return any(p in text_lower for p in dosing_patterns)

    def _detect_lab_values(self, text_lower: str) -> bool:
        lab_patterns = ["reference range", "normal range", "hba1c", "serum creatinine",
                        "hemoglobin", "platelet count", "wbc count", "blood glucose",
                        "triglycerides", "ldl", "hdl"]
        return any(p in text_lower for p in lab_patterns)
```

---

### A4 — Dual Embedding

**Location:** `app/ingestion/embedder.py`

```python
import torch
from sentence_transformers import SentenceTransformer
from typing import List, Tuple
import numpy as np


class DualEmbedder:
    """
    Generates both dense and sparse embeddings for each chunk.
    
    Dense: S-PubMedBERT via sentence-transformers (contextual_text as input)
    Sparse: BM25-style sparse vector using medical-aware tokenization
    
    CRITICAL: Always embed contextual_text (with prefix), not raw text.
    Store raw text in MongoDB but embed contextual_text in Qdrant.
    """

    def __init__(self, dense_model_name: str = "pritamdeka/S-PubMedBert-MS-MARCO"):
        self.dense_model = SentenceTransformer(dense_model_name)
        self.dense_model.eval()  # disable dropout
        # Move to GPU if available
        if torch.cuda.is_available():
            self.dense_model = self.dense_model.cuda()

    def embed_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Batch embed texts. Use torch.inference_mode() for speed.
        Always pass contextual_text here, not raw text.
        """
        with torch.inference_mode():
            embeddings = self.dense_model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=True,
                normalize_embeddings=True,  # cosine similarity = dot product
                convert_to_numpy=True,
            )
        return embeddings

    def embed_query(self, query_text: str) -> np.ndarray:
        """
        Embed a single query. No prefix needed — S-PubMedBERT handles this.
        Use torch.inference_mode() for speed.
        """
        with torch.inference_mode():
            embedding = self.dense_model.encode(
                query_text,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
        return embedding

    def compute_sparse_vector(self, text: str) -> dict:
        """
        Compute a sparse BM25-style vector with medical-aware tokenization.
        Returns: {term_index: tf_idf_weight} — sparse representation
        for Qdrant's sparse vector format.
        
        Medical compound terms are preserved as bigrams (not split).
        Term frequency is normalized by document length.
        IDF is approximated using a pre-built medical vocabulary.
        
        For production: Replace this with SPLADE model inference
        (naver/splade-cocondenser-ensemble) for higher quality sparse vectors.
        Current BM25 implementation is a good first step.
        """
        tokens = self._medical_tokenize(text)
        if not tokens:
            return {"indices": [], "values": []}

        # Term frequency
        from collections import Counter
        tf = Counter(tokens)
        total_tokens = len(tokens)

        # Vocabulary lookup (load from file or use in-memory hash)
        indices = []
        values = []
        for term, count in tf.items():
            term_idx = self._term_to_index(term)
            tf_norm = count / total_tokens
            idf_weight = self._get_idf_weight(term)
            weight = tf_norm * idf_weight
            if weight > 0.001:  # prune very small weights
                indices.append(term_idx)
                values.append(float(weight))

        return {"indices": indices, "values": values}

    def _medical_tokenize(self, text: str) -> List[str]:
        """
        Tokenize medical text, preserving compound terms as single tokens.
        
        Rules:
        1. Lowercase
        2. Extract known medical bigrams before splitting
        3. Split remaining text on whitespace + punctuation
        4. Remove stopwords (but NOT medical stopwords like 'no', 'not', 'without')
        """
        MEDICAL_COMPOUNDS = [
            "type 2 diabetes", "type 1 diabetes", "heart failure", "blood pressure",
            "myocardial infarction", "atrial fibrillation", "blood glucose",
            "hemoglobin a1c", "hba1c", "body mass index", "bmi", "ace inhibitor",
            "angiotensin converting enzyme", "randomized controlled trial",
            "systematic review", "meta analysis", "meta-analysis",
            "insulin resistance", "glycemic control", "renal failure",
            "coronary artery disease", "coronary heart disease",
        ]
        text_lower = text.lower()
        tokens = []

        # Extract compound terms first (replace with placeholder)
        for compound in MEDICAL_COMPOUNDS:
            token_version = compound.replace(' ', '_')
            if compound in text_lower:
                text_lower = text_lower.replace(compound, f" {token_version} ")
                tokens.append(token_version)

        # Tokenize remainder
        import re
        words = re.findall(r'\b[a-z][a-z0-9\-]{2,}\b', text_lower)

        # Stopwords to remove (preserve medical negations)
        STOPWORDS = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
            'for', 'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were',
            'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'may', 'might', 'shall',
            'this', 'that', 'these', 'those', 'it', 'its', 'we', 'our',
            'they', 'their', 'as', 'if', 'than', 'more', 'most', 'such',
        }
        # NOTE: do NOT include 'no', 'not', 'without', 'versus' — medically significant

        words = [w for w in words if w not in STOPWORDS]
        tokens.extend(words)
        return tokens

    def _term_to_index(self, term: str) -> int:
        """Hash term to a vocabulary index (0 to VOCAB_SIZE-1)."""
        VOCAB_SIZE = 50000
        return hash(term) % VOCAB_SIZE

    def _get_idf_weight(self, term: str) -> float:
        """
        Approximate IDF weight.
        For production: load a pre-computed IDF table from the corpus.
        For now: use a simple heuristic based on term length and type.
        """
        # Medical terms (longer, specific) get higher weight
        if len(term) > 8:
            return 3.0
        elif len(term) > 5:
            return 2.0
        else:
            return 1.0
```

---

### A5 — Qdrant Indexing

**Location:** `app/ingestion/qdrant_indexer.py`

```python
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import (
    VectorParams, Distance, SparseVectorParams, SparseIndexParams,
    PointStruct, NamedVector, NamedSparseVector, SparseVector,
    PayloadSchemaType
)
import uuid
from typing import List


COLLECTION_NAME = "openinsight_v2"
DENSE_DIM = 768  # S-PubMedBERT output dimension


class QdrantIndexer:
    def __init__(self, qdrant_url: str = "http://qdrant:6333"):
        self.client = QdrantClient(url=qdrant_url)

    def create_collection(self, recreate: bool = False):
        """
        Create Qdrant collection with both dense and sparse vector configs.
        
        IMPORTANT: Call this ONCE before ingestion. If recreate=True, drops
        and recreates (destroys all existing data — use with care).
        
        Named vectors:
          "dense"  — S-PubMedBERT 768-dim, cosine similarity
          "sparse" — BM25/SPLADE, dot product
        
        Payload indexes are created for fast metadata filtering.
        """
        if recreate:
            try:
                self.client.delete_collection(COLLECTION_NAME)
            except Exception:
                pass

        self.client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={
                "dense": VectorParams(
                    size=DENSE_DIM,
                    distance=Distance.COSINE,
                    on_disk=False,  # keep in RAM for speed
                )
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(
                    index=SparseIndexParams(on_disk=False)
                )
            },
            # HNSW config — tune for recall vs speed
            hnsw_config=models.HnswConfigDiff(
                m=16,            # higher = better recall, more RAM
                ef_construct=128  # higher = better index quality, slower build
            ),
            optimizers_config=models.OptimizersConfigDiff(
                default_segment_number=4  # parallel segments for speed
            )
        )

        # Create payload indexes for fast metadata filtering
        # These are CRITICAL for the metadata filter inference in the query path
        payload_indexes = [
            ("year", PayloadSchemaType.INTEGER),
            ("doc_type", PayloadSchemaType.KEYWORD),
            ("source", PayloadSchemaType.KEYWORD),
            ("evidence_level", PayloadSchemaType.KEYWORD),
            ("specialty", PayloadSchemaType.KEYWORD),  # array field
            ("india_relevant", PayloadSchemaType.BOOL),
            ("has_drug_dosing", PayloadSchemaType.BOOL),
            ("chunk_type", PayloadSchemaType.KEYWORD),
            ("pmid", PayloadSchemaType.KEYWORD),
        ]
        for field_name, schema_type in payload_indexes:
            self.client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name=field_name,
                field_schema=schema_type
            )

        print(f"Collection '{COLLECTION_NAME}' created with dense + sparse vectors.")
        print(f"Payload indexes created for: {[f for f, _ in payload_indexes]}")

    def upsert_chunks(self, chunks: List['Chunk'], dense_embeddings, sparse_vectors: List[dict]):
        """
        Upsert a batch of chunks with their embeddings to Qdrant.
        
        chunks: List[Chunk] dataclass instances
        dense_embeddings: np.ndarray of shape (len(chunks), 768)
        sparse_vectors: List of {indices: [...], values: [...]} dicts
        """
        assert len(chunks) == len(dense_embeddings) == len(sparse_vectors)

        points = []
        for chunk, dense_emb, sparse_vec in zip(chunks, dense_embeddings, sparse_vectors):
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.chunk_id))

            point = PointStruct(
                id=point_id,
                vector={
                    "dense": dense_emb.tolist(),
                    "sparse": SparseVector(
                        indices=sparse_vec["indices"],
                        values=sparse_vec["values"]
                    )
                },
                payload={
                    **chunk.metadata,   # ChunkMetadata.to_qdrant_payload() dict
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "chunk_type": chunk.chunk_type,
                    "section_title": chunk.section_title,
                    "raw_text": chunk.text,  # store raw text in payload for retrieval
                    "chunk_index": chunk.chunk_index,
                    "total_chunks": chunk.total_chunks,
                }
            )
            points.append(point)

        # Upsert in batches of 100
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            self.client.upsert(collection_name=COLLECTION_NAME, points=batch)

        return len(points)
```

---

### A6 — MongoDB Full-Doc Store

**Location:** `app/ingestion/mongo_store.py`

```python
from motor.motor_asyncio import AsyncIOMotorClient
from typing import Optional


class MongoDocStore:
    """
    Stores full parsed documents and chunk-to-doc linking.
    Qdrant stores embeddings + payload snippets.
    MongoDB stores the full text for display, citation building, and audit.
    
    Collections:
      documents   — one doc per ParsedDocument
      chunks      — one doc per Chunk (links to document)
      ingestion_log — audit trail of ingestion runs
    """

    def __init__(self, mongo_url: str = "mongodb://mongodb:27017", db_name: str = "openinsight"):
        self.client = AsyncIOMotorClient(mongo_url)
        self.db = self.client[db_name]

    async def store_document(self, doc: 'ParsedDocument', enriched_metadata: dict):
        await self.db.documents.update_one(
            {"doc_id": doc.doc_id},
            {"$set": {
                "doc_id": doc.doc_id,
                "title": doc.title,
                "abstract": doc.abstract,
                "authors": doc.authors,
                "year": doc.year,
                "journal": doc.journal,
                "doi": doc.doi,
                "pmid": doc.pmid,
                "sections": [
                    {"title": s.title, "text": s.text, "index": s.section_index}
                    for s in doc.sections
                ],
                "mesh_terms": doc.mesh_terms,
                "keywords": doc.keywords,
                **enriched_metadata,
                "ingested_at": __import__('datetime').datetime.utcnow().isoformat(),
            }},
            upsert=True
        )

    async def store_chunks(self, chunks: List['Chunk']):
        """Bulk upsert chunks for linking and display."""
        ops = []
        from pymongo import UpdateOne
        for chunk in chunks:
            ops.append(UpdateOne(
                {"chunk_id": chunk.chunk_id},
                {"$set": {
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "chunk_type": chunk.chunk_type,
                    "section_title": chunk.section_title,
                    "text": chunk.text,
                    "char_count": chunk.char_count,
                    "token_estimate": chunk.token_estimate,
                    "chunk_index": chunk.chunk_index,
                    "total_chunks": chunk.total_chunks,
                    "metadata": chunk.metadata,
                }},
                upsert=True
            ))
        if ops:
            await self.db.chunks.bulk_write(ops)

    async def get_document(self, doc_id: str) -> Optional[dict]:
        return await self.db.documents.find_one({"doc_id": doc_id}, {"_id": 0})

    async def get_chunk(self, chunk_id: str) -> Optional[dict]:
        return await self.db.chunks.find_one({"chunk_id": chunk_id}, {"_id": 0})
```

---

### A7 — Batch Ingestion Pipeline

**Location:** `app/ingestion/pipeline.py`

This is the entry point that orchestrates all ingestion steps.

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)


class IngestionPipeline:
    def __init__(self):
        self.parser = DocumentParser()
        self.chunker = HierarchicalChunker()
        self.enricher = MetadataEnricher()
        self.embedder = DualEmbedder()
        self.indexer = QdrantIndexer()
        self.mongo = MongoDocStore()

    async def ingest_directory(self, directory: str, source: str, recreate_index: bool = False):
        """
        Main entry point. Ingest all documents in a directory.
        
        directory: path to directory containing PDFs or XML files
        source: one of "pubmed", "icmr", "cochrane", "nmc_guideline", "rssdi"
        recreate_index: WARNING — drops and rebuilds Qdrant collection
        """
        if recreate_index:
            self.indexer.create_collection(recreate=True)

        paths = list(Path(directory).glob("**/*.pdf")) + list(Path(directory).glob("**/*.xml"))
        logger.info(f"Found {len(paths)} documents in {directory}")

        # Process in batches of 10 documents
        BATCH_SIZE = 10
        total_chunks = 0

        for batch_start in range(0, len(paths), BATCH_SIZE):
            batch_paths = paths[batch_start:batch_start + BATCH_SIZE]
            logger.info(f"Processing batch {batch_start//BATCH_SIZE + 1}/{len(paths)//BATCH_SIZE + 1}")

            # Parse documents (can be parallelized with ThreadPoolExecutor for I/O)
            parsed_docs = []
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = []
                for path in batch_paths:
                    source_type = self._infer_source_type(path, source)
                    futures.append(executor.submit(self.parser.parse, str(path), source_type))

                for future in futures:
                    try:
                        doc = future.result(timeout=120)
                        if doc:
                            parsed_docs.append(doc)
                    except Exception as e:
                        logger.error(f"Parse failed: {e}")
                        continue

            # Enrich metadata
            enriched = []
            for doc in parsed_docs:
                meta = self.enricher.enrich(doc, source)
                enriched.append((doc, meta))

            # Chunk all documents in batch
            all_chunks = []
            for doc, meta in enriched:
                try:
                    chunks = self.chunker.chunk_document(doc, meta)
                    all_chunks.extend(chunks)
                except Exception as e:
                    logger.error(f"Chunking failed for {doc.doc_id}: {e}")
                    continue

            if not all_chunks:
                continue

            # Embed all chunks in this batch at once
            # CRITICAL: embed contextual_text, not raw text
            contextual_texts = [c.contextual_text for c in all_chunks]
            logger.info(f"Embedding {len(all_chunks)} chunks...")
            dense_embeddings = self.embedder.embed_batch(contextual_texts, batch_size=32)

            # Compute sparse vectors
            sparse_vectors = [
                self.embedder.compute_sparse_vector(c.contextual_text)
                for c in all_chunks
            ]

            # Index in Qdrant
            self.indexer.upsert_chunks(all_chunks, dense_embeddings, sparse_vectors)

            # Store in MongoDB
            for doc, meta in enriched:
                await self.mongo.store_document(doc, meta)
            await self.mongo.store_chunks(all_chunks)

            total_chunks += len(all_chunks)
            logger.info(f"Ingested {len(all_chunks)} chunks from batch. Total so far: {total_chunks}")

        logger.info(f"Ingestion complete. Total chunks: {total_chunks}")
        return total_chunks

    def _infer_source_type(self, path: Path, source: str) -> SourceType:
        if path.suffix == '.xml':
            return SourceType.PUBMED_XML
        elif source in ('icmr', 'nmc_guideline', 'rssdi'):
            return SourceType.PDF_GUIDELINE
        else:
            return SourceType.PDF_PAPER
```

**CLI to run ingestion:**

```python
# app/ingestion/run_ingestion.py
import asyncio
import argparse

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True, help="Path to documents directory")
    parser.add_argument("--source", required=True,
                        choices=["pubmed", "icmr", "cochrane", "nmc_guideline", "rssdi", "who"])
    parser.add_argument("--recreate", action="store_true", help="Recreate Qdrant collection")
    args = parser.parse_args()

    pipeline = IngestionPipeline()
    total = await pipeline.ingest_directory(args.dir, args.source, args.recreate)
    print(f"Done. Ingested {total} chunks.")

if __name__ == "__main__":
    asyncio.run(main())
```

```bash
# Usage:
python -m app.ingestion.run_ingestion --dir /data/pubmed_xml --source pubmed
python -m app.ingestion.run_ingestion --dir /data/icmr_pdfs --source icmr --recreate
```

---

## 4. Phase B — Standard Search (Query Path)

### B1 — Query Understanding Layer

**Location:** `app/search/query_understanding.py`

```python
import re
import spacy
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum


class QueryIntent(str, Enum):
    DIAGNOSTIC = "diagnostic"
    THERAPEUTIC = "therapeutic"
    PROGNOSTIC = "prognostic"
    DRUG_INFO = "drug_info"
    GUIDELINE = "guideline"
    GENERAL = "general"


@dataclass
class QueryAnalysis:
    original_query: str
    intent: QueryIntent
    entities: Dict[str, List[str]]     # {"disease": [...], "drug": [...], "symptom": [...]}
    rewritten_query: Optional[str]     # HyDE or expanded query
    metadata_filters: List[Dict]       # Qdrant filter conditions
    use_hyde: bool
    expanded_terms: List[str]          # additional search terms from synonym expansion


class QueryUnderstanding:
    """
    Pre-search query processing layer.
    Steps:
      1. Intent classification (rule-based + keyword matching)
      2. Entity extraction (scispaCy)
      3. Metadata filter inference from entities + intent
      4. Conditional HyDE rewrite (only for diagnostic/prognostic)
      5. Query expansion with medical synonyms
    """

    DIAGNOSTIC_PATTERNS = [
        "what causes", "cause of", "differential diagnosis", "ddx", "how to diagnose",
        "symptoms of", "signs of", "presentation of", "clinical features of",
        "what is", "what are the causes", "aetiology"
    ]

    THERAPEUTIC_PATTERNS = [
        "treatment", "treat", "therapy", "management", "how to manage",
        "drug of choice", "first line", "second line", "dose", "dosage",
        "drug for", "medication for", "antibiotic for"
    ]

    PROGNOSTIC_PATTERNS = [
        "prognosis", "outcome", "survival", "mortality", "morbidity",
        "risk of", "chance of", "likelihood", "long term"
    ]

    DRUG_INFO_PATTERNS = [
        "side effects", "adverse effects", "interactions", "contraindications",
        "mechanism of", "pharmacology", "half life"
    ]

    GUIDELINE_PATTERNS = [
        "guideline", "recommendation", "protocol", "standard of care",
        "icmr", "nmc", "who recommendation", "evidence based"
    ]

    # Medical synonym table (query expansion)
    MEDICAL_SYNONYMS = {
        "heart attack": ["myocardial infarction", "mi", "acute coronary syndrome"],
        "diabetes": ["diabetes mellitus", "dm", "type 2 diabetes", "t2dm"],
        "high blood pressure": ["hypertension", "htn"],
        "stroke": ["cerebrovascular accident", "cva", "ischemic stroke"],
        "tb": ["tuberculosis", "mycobacterium tuberculosis"],
        "dengue": ["dengue fever", "dengue hemorrhagic fever"],
        "thyroid": ["hypothyroidism", "hyperthyroidism", "thyroid disease"],
    }

    def __init__(self):
        # Load scispaCy model for entity extraction
        # Required: pip install scispacy && python -m spacy download en_core_sci_md
        try:
            self.nlp = spacy.load("en_core_sci_md")
        except OSError:
            # Fallback if scispaCy model not installed
            self.nlp = None

    def analyze(self, query: str) -> QueryAnalysis:
        query_lower = query.lower().strip()

        intent = self._classify_intent(query_lower)
        entities = self._extract_entities(query)
        metadata_filters = self._infer_metadata_filters(query_lower, entities, intent)
        expanded_terms = self._expand_query(query_lower)
        use_hyde = intent in (QueryIntent.DIAGNOSTIC, QueryIntent.PROGNOSTIC)

        # For HyDE: generate a hypothetical answer snippet
        # This is done async in the retrieval step, not here
        # We just set the flag here

        return QueryAnalysis(
            original_query=query,
            intent=intent,
            entities=entities,
            rewritten_query=None,  # filled by async HyDE step
            metadata_filters=metadata_filters,
            use_hyde=use_hyde,
            expanded_terms=expanded_terms
        )

    def _classify_intent(self, query_lower: str) -> QueryIntent:
        for pattern in self.DIAGNOSTIC_PATTERNS:
            if pattern in query_lower:
                return QueryIntent.DIAGNOSTIC
        for pattern in self.THERAPEUTIC_PATTERNS:
            if pattern in query_lower:
                return QueryIntent.THERAPEUTIC
        for pattern in self.PROGNOSTIC_PATTERNS:
            if pattern in query_lower:
                return QueryIntent.PROGNOSTIC
        for pattern in self.DRUG_INFO_PATTERNS:
            if pattern in query_lower:
                return QueryIntent.DRUG_INFO
        for pattern in self.GUIDELINE_PATTERNS:
            if pattern in query_lower:
                return QueryIntent.GUIDELINE
        return QueryIntent.GENERAL

    def _extract_entities(self, query: str) -> Dict[str, List[str]]:
        entities = {"disease": [], "drug": [], "symptom": [], "procedure": [], "lab_value": []}
        if self.nlp is None:
            return entities

        doc = self.nlp(query)
        for ent in doc.ents:
            label = ent.label_.lower()
            text = ent.text.lower()
            if label in ("disease", "disorder"):
                entities["disease"].append(text)
            elif label in ("chemical", "drug", "simple_chemical"):
                entities["drug"].append(text)
            elif label == "sign_symptom":
                entities["symptom"].append(text)
            elif label in ("medical_procedure", "diagnostic_procedure"):
                entities["procedure"].append(text)
            elif label == "lab_value":
                entities["lab_value"].append(text)

        return entities

    def _infer_metadata_filters(self, query_lower: str, entities: dict, intent: QueryIntent) -> List[Dict]:
        """
        Build Qdrant filter conditions from query content.
        These pre-filter the search space before hybrid retrieval.
        """
        from qdrant_client.http import models

        conditions = []

        # Temporal filter
        if any(w in query_lower for w in ["recent", "latest", "current", "new", "2024", "2025"]):
            conditions.append(
                models.FieldCondition(key="year", range=models.Range(gte=2020))
            )

        # Evidence type filter
        if intent == QueryIntent.GUIDELINE or any(w in query_lower for w in ["guideline", "recommendation", "protocol"]):
            conditions.append(
                models.FieldCondition(key="doc_type",
                                      match=models.MatchAny(any=["guideline", "systematic_review", "meta_analysis"]))
            )

        # India filter
        if any(w in query_lower for w in ["india", "indian", "indians", "our population"]):
            conditions.append(
                models.FieldCondition(key="india_relevant", match=models.MatchValue(value=True))
            )

        # Drug dosing filter
        if intent == QueryIntent.THERAPEUTIC and any(w in query_lower for w in ["dose", "dosage", "dosing"]):
            conditions.append(
                models.FieldCondition(key="has_drug_dosing", match=models.MatchValue(value=True))
            )

        return conditions

    def _expand_query(self, query_lower: str) -> List[str]:
        """Add medical synonyms for query expansion."""
        extra_terms = []
        for term, synonyms in self.MEDICAL_SYNONYMS.items():
            if term in query_lower:
                extra_terms.extend(synonyms)
        return extra_terms
```

---

### B2 — Parallel Hybrid Retrieval

**Location:** `app/search/retriever.py`

```python
import asyncio
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class RetrievedChunk:
    chunk_id: str
    doc_id: str
    score: float            # retrieval score (before reranking)
    text: str
    contextual_text: str
    metadata: dict
    retrieval_source: str   # "dense" | "sparse" | "both"


class HybridRetriever:
    def __init__(self, qdrant_url: str = "http://qdrant:6333"):
        self.client = QdrantClient(url=qdrant_url)
        self.collection = COLLECTION_NAME
        self.embedder = DualEmbedder()

    async def retrieve(
        self,
        query: str,
        query_analysis: 'QueryAnalysis',
        top_k: int = 50
    ) -> List[RetrievedChunk]:
        """
        Step 1: Optionally apply HyDE (generate hypothetical answer for embedding)
        Step 2: Embed query (dense + sparse) — parallel
        Step 3: Dense search + sparse search — parallel (asyncio.gather)
        Step 4: Return combined results for RRF fusion
        """

        # HyDE rewrite — only for diagnostic/prognostic queries
        embed_query = query
        if query_analysis.use_hyde:
            hyde_text = await self._generate_hyde(query)
            embed_query = hyde_text if hyde_text else query

        # Add expanded terms to query string for sparse search
        if query_analysis.expanded_terms:
            sparse_query = query + " " + " ".join(query_analysis.expanded_terms)
        else:
            sparse_query = query

        # Compute embeddings (CPU-bound, run in thread to not block event loop)
        loop = asyncio.get_event_loop()
        dense_embedding, sparse_vector = await asyncio.gather(
            loop.run_in_executor(None, self.embedder.embed_query, embed_query),
            loop.run_in_executor(None, self.embedder.compute_sparse_vector, sparse_query),
        )

        # Build Qdrant filter from metadata conditions
        qdrant_filter = self._build_filter(query_analysis.metadata_filters)

        # Run dense and sparse search in parallel
        dense_results, sparse_results = await asyncio.gather(
            self._dense_search(dense_embedding, qdrant_filter, top_k),
            self._sparse_search(sparse_vector, qdrant_filter, top_k),
        )

        return dense_results, sparse_results

    async def _dense_search(self, embedding, qdrant_filter, top_k: int) -> List[RetrievedChunk]:
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None,
            lambda: self.client.search(
                collection_name=self.collection,
                query_vector=models.NamedVector(name="dense", vector=embedding.tolist()),
                query_filter=qdrant_filter,
                limit=top_k,
                with_payload=True,
            )
        )
        return [self._to_chunk(r, "dense") for r in results]

    async def _sparse_search(self, sparse_vector: dict, qdrant_filter, top_k: int) -> List[RetrievedChunk]:
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None,
            lambda: self.client.search(
                collection_name=self.collection,
                query_vector=models.NamedSparseVector(
                    name="sparse",
                    vector=models.SparseVector(
                        indices=sparse_vector["indices"],
                        values=sparse_vector["values"]
                    )
                ),
                query_filter=qdrant_filter,
                limit=top_k,
                with_payload=True,
            )
        )
        return [self._to_chunk(r, "sparse") for r in results]

    def _build_filter(self, conditions: list) -> Optional[models.Filter]:
        if not conditions:
            return None
        if len(conditions) == 1:
            return models.Filter(must=conditions)
        return models.Filter(must=conditions)

    def _to_chunk(self, qdrant_result, source: str) -> RetrievedChunk:
        payload = qdrant_result.payload or {}
        return RetrievedChunk(
            chunk_id=payload.get("chunk_id", str(qdrant_result.id)),
            doc_id=payload.get("doc_id", ""),
            score=qdrant_result.score,
            text=payload.get("raw_text", ""),
            contextual_text=payload.get("raw_text", ""),  # contextual_text not stored; raw_text is enough for display
            metadata=payload,
            retrieval_source=source
        )

    async def _generate_hyde(self, query: str) -> Optional[str]:
        """
        Generate a hypothetical answer paragraph using the LLM.
        This hypothetical answer is then embedded and used for dense retrieval.
        The idea: if we ask "what causes X?", generate a fake answer paragraph
        about X, embed it, and find real chunks that are semantically close to
        what a real answer would look like.
        
        Only use for diagnostic/prognostic intents where semantic similarity
        to a hypothetical answer helps more than keyword matching.
        Skip for drug_info and guideline queries — exact terms matter more.
        """
        import httpx
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.post(
                    "http://nim-llm:8000/v1/completions",  # adjust to your NIM endpoint
                    json={
                        "model": "meta/llama-3.1-70b-instruct",
                        "prompt": f"Write a brief medical paragraph answering: {query}\n\nAnswer:",
                        "max_tokens": 200,
                        "temperature": 0.1
                    }
                )
                data = response.json()
                return data["choices"][0]["text"].strip()
        except Exception:
            return None  # graceful fallback — proceed with original query
```

---

### B3 — RRF Fusion + Metadata Boost

**Location:** `app/search/fusion.py`

```python
from typing import List, Dict, Tuple
from collections import defaultdict


EVIDENCE_BOOST_SCORE = {
    "1a": 1.35,  # systematic review / meta-analysis
    "1b": 1.25,  # RCT
    "2a": 1.15,
    "2b": 1.10,
    "3": 1.05,
    "4": 1.00,
    "5": 1.10,   # guidelines — still clinically valuable
    "unknown": 1.00,
}

RECENCY_BOOST = {
    2025: 1.10,
    2024: 1.08,
    2023: 1.05,
    2022: 1.03,
}


def reciprocal_rank_fusion(
    dense_results: List['RetrievedChunk'],
    sparse_results: List['RetrievedChunk'],
    k: int = 60,
    top_n: int = 20
) -> List['RetrievedChunk']:
    """
    Reciprocal Rank Fusion (RRF).
    
    RRF formula: score(doc) = Σ 1/(k + rank)
    k=60 is the standard recommendation from the original RRF paper.
    
    Why RRF over weighted sum?
    - Dense scores (cosine similarity): typically 0.5–0.95
    - Sparse scores (BM25): typically 0.1–12.0
    These scales are completely different — you can't add them directly.
    RRF converts to rank positions which are always comparable.
    
    After RRF:
    - Apply evidence level boost (multiply RRF score)
    - Apply recency boost for recent years
    - Sort by final boosted score
    - Return top_n unique chunks
    """
    # Build {chunk_id: chunk} lookup from both result sets
    all_chunks: Dict[str, 'RetrievedChunk'] = {}
    for result in dense_results + sparse_results:
        if result.chunk_id not in all_chunks:
            all_chunks[result.chunk_id] = result
        else:
            # Keep the version with higher raw score
            if result.score > all_chunks[result.chunk_id].score:
                all_chunks[result.chunk_id] = result

    # Compute RRF scores
    rrf_scores: Dict[str, float] = defaultdict(float)

    for rank, chunk in enumerate(dense_results):
        rrf_scores[chunk.chunk_id] += 1.0 / (k + rank + 1)

    for rank, chunk in enumerate(sparse_results):
        rrf_scores[chunk.chunk_id] += 1.0 / (k + rank + 1)

    # Apply evidence level boost
    for chunk_id, chunk in all_chunks.items():
        evidence_level = chunk.metadata.get("evidence_level", "unknown")
        rrf_scores[chunk_id] *= EVIDENCE_BOOST_SCORE.get(evidence_level, 1.0)

        # Apply recency boost
        year = chunk.metadata.get("year", 0)
        if year in RECENCY_BOOST:
            rrf_scores[chunk_id] *= RECENCY_BOOST[year]

    # Sort by final score and return top_n
    sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

    result = []
    for chunk_id in sorted_ids[:top_n]:
        chunk = all_chunks[chunk_id]
        chunk.score = rrf_scores[chunk_id]  # update score to RRF score
        result.append(chunk)

    return result
```

---

### B4 — Two-Stage Reranking

**Location:** `app/search/reranker.py`

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List
import numpy as np


class CrossEncoderReranker:
    """
    BAAI/bge-reranker-base cross-encoder reranker.
    
    Takes query + candidate chunks, scores each pair, returns top_k.
    
    CRITICAL: Run on top-20 from RRF (not top-50 from raw retrieval).
    This is the main speed fix. Cross-encoder is O(n) — halving candidates
    from 50 to 20 = ~2.5x speed improvement.
    
    Keep model warm in memory (load once at startup).
    Use torch.inference_mode() for inference.
    """

    MODEL_NAME = "BAAI/bge-reranker-base"

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.MODEL_NAME)
        self.model.eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda()

    def rerank(self, query: str, chunks: List['RetrievedChunk'], top_k: int = 8) -> List['RetrievedChunk']:
        """
        Score each (query, chunk) pair using the cross-encoder.
        Return top_k chunks sorted by reranker score.
        
        The cross-encoder reads both query and chunk jointly — it captures
        relevance that embedding similarity can miss (e.g. negation, specificity).
        """
        if not chunks:
            return []

        # Prepare input pairs
        pairs = [[query, chunk.text[:512]] for chunk in chunks]  # truncate to 512 tokens

        with torch.inference_mode():
            inputs = self.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}

            scores = self.model(**inputs).logits.squeeze(-1)
            scores = scores.cpu().numpy()

        # Update chunk scores and sort
        for chunk, score in zip(chunks, scores):
            chunk.score = float(score)

        ranked = sorted(chunks, key=lambda c: c.score, reverse=True)
        return ranked[:top_k]
```

---

### B5 — MMR Deduplication

**Location:** `app/search/mmr.py`

```python
import numpy as np
from typing import List


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def maximal_marginal_relevance(
    chunks: List['RetrievedChunk'],
    embedder: 'DualEmbedder',
    lambda_param: float = 0.7,
    n_select: int = 6
) -> List['RetrievedChunk']:
    """
    Apply Maximal Marginal Relevance (MMR) to select diverse, relevant chunks.
    
    λ=0.7 means: 70% weight on relevance, 30% on diversity.
    Reduces to 70% standard relevance ranking when λ=1.0.
    Reduces to pure diversity when λ=0.0.
    
    λ=0.7 recommended for clinical QA — we want relevant but not 5 chunks
    from the same paper repeating the same fact.
    
    Algorithm:
      1. Select the highest-scoring chunk unconditionally.
      2. For each remaining chunk, compute:
           MMR_score = λ * reranker_score - (1-λ) * max_sim_to_selected
      3. Select the chunk with highest MMR_score.
      4. Repeat until n_select chunks selected.
    
    The embeddings for similarity computation are generated on the raw text
    of the candidates — NOT on the contextual_text. We want to know if two
    chunks are semantically similar to each other, not to the query.
    """
    if len(chunks) <= n_select:
        return chunks

    # Embed the candidate texts to compare chunk-to-chunk similarity
    chunk_texts = [c.text[:400] for c in chunks]  # truncate for speed
    embeddings = embedder.embed_batch(chunk_texts, batch_size=16)

    selected_indices = []
    selected_embeddings = []
    candidate_indices = list(range(len(chunks)))

    # Step 1: select highest-scoring chunk unconditionally
    best_idx = max(candidate_indices, key=lambda i: chunks[i].score)
    selected_indices.append(best_idx)
    selected_embeddings.append(embeddings[best_idx])
    candidate_indices.remove(best_idx)

    # Step 2: iteratively select next best by MMR
    while len(selected_indices) < n_select and candidate_indices:
        mmr_scores = {}

        for i in candidate_indices:
            # Relevance: reranker score (already normalised by reranker)
            relevance = chunks[i].score

            # Diversity: negative max similarity to already-selected chunks
            max_sim = max(
                cosine_similarity(embeddings[i], sel_emb)
                for sel_emb in selected_embeddings
            )

            mmr_scores[i] = lambda_param * relevance - (1 - lambda_param) * max_sim

        # Select chunk with highest MMR score
        next_idx = max(mmr_scores.keys(), key=lambda i: mmr_scores[i])
        selected_indices.append(next_idx)
        selected_embeddings.append(embeddings[next_idx])
        candidate_indices.remove(next_idx)

    return [chunks[i] for i in selected_indices]
```

---

### B6 — Context Assembly

**Location:** `app/search/context_builder.py`

```python
from typing import List


EVIDENCE_LEVEL_LABELS = {
    "1a": "Systematic Review / Meta-Analysis",
    "1b": "Randomised Controlled Trial",
    "2a": "Systematic Review of Cohort Studies",
    "2b": "Cohort Study",
    "3": "Case-Control Study",
    "4": "Case Series",
    "5": "Expert Opinion / Guideline",
    "unknown": "Not Classified",
}


def assemble_context(chunks: List['RetrievedChunk'], max_tokens: int = 3000) -> str:
    """
    Build the context string to pass to the LLM.
    
    Each chunk is formatted with:
      - Citation number [N]
      - Title + year + journal
      - Evidence level label
      - Document type
      - Whether India-relevant
      - The chunk text
    
    Context is truncated to max_tokens (approximate via character count).
    Chunks are ordered by reranker score (best first).
    
    max_tokens=3000 leaves ~1000 tokens for system prompt + response in
    a 4096-token context. Adjust based on your NIM endpoint's context window.
    For Llama 3.1 70B with 8K context: can increase to 6000.
    """
    parts = []
    total_chars = 0
    char_limit = max_tokens * 4  # rough token-to-char ratio

    for i, chunk in enumerate(chunks, 1):
        m = chunk.metadata
        title = m.get("title", "Untitled")
        year = m.get("year", "")
        journal = m.get("journal", "")
        evidence_level = m.get("evidence_level", "unknown")
        doc_type = m.get("doc_type", "").replace("_", " ").title()
        india = " 🇮🇳 India-relevant" if m.get("india_relevant") else ""

        header = (
            f"[{i}] {title} ({year}"
            f"{', ' + journal if journal else ''})\n"
            f"Evidence: {EVIDENCE_LEVEL_LABELS.get(evidence_level, evidence_level)}"
            f" | {doc_type}{india}"
        )

        chunk_block = f"{header}\n{chunk.text}\n"
        chunk_chars = len(chunk_block)

        if total_chars + chunk_chars > char_limit:
            break

        parts.append(chunk_block)
        total_chars += chunk_chars

    return "\n---\n".join(parts)


def build_citation_list(chunks: List['RetrievedChunk']) -> List[dict]:
    """Build structured citation list for the API response."""
    citations = []
    for i, chunk in enumerate(chunks, 1):
        m = chunk.metadata
        citations.append({
            "number": i,
            "doc_id": chunk.doc_id,
            "chunk_id": chunk.chunk_id,
            "title": m.get("title", ""),
            "authors": m.get("authors", []),
            "year": m.get("year", 0),
            "journal": m.get("journal", ""),
            "doi": m.get("doi", ""),
            "pmid": m.get("pmid", ""),
            "evidence_level": m.get("evidence_level", "unknown"),
            "doc_type": m.get("doc_type", ""),
            "india_relevant": m.get("india_relevant", False),
            "source_url": f"https://pubmed.ncbi.nlm.nih.gov/{m.get('pmid', '')}" if m.get("pmid") else "",
        })
    return citations
```

---

### B7 — Redis Caching

**Location:** `app/search/cache.py`

```python
import redis.asyncio as aioredis
import hashlib
import json
from typing import Optional, List


class SearchCache:
    """
    Redis cache for expensive search operations.
    
    What we cache:
      1. Reranker output — keyed on (query_hash + sorted chunk_ids)
         TTL: 1 hour — reranker is the most expensive step
      2. Full search results — keyed on query_hash + filters_hash
         TTL: 30 minutes — for identical repeated queries
    
    We do NOT cache:
      - Embedding generation (fast, changes with model updates)
      - Qdrant retrieval (Qdrant has its own in-memory cache)
    
    Cache key format: "openinsight:{version}:{operation}:{hash}"
    Version bump when you re-ingest or change models.
    """

    CACHE_VERSION = "v2"

    def __init__(self, redis_url: str = "redis://redis:6379"):
        self.redis = aioredis.from_url(redis_url, encoding="utf-8", decode_responses=True)

    def _make_key(self, operation: str, *components) -> str:
        content = "|".join(str(c) for c in components)
        hash_ = hashlib.sha256(content.encode()).hexdigest()[:16]
        return f"openinsight:{self.CACHE_VERSION}:{operation}:{hash_}"

    async def get_search_result(self, query: str, filters: list) -> Optional[dict]:
        key = self._make_key("search", query.lower().strip(), json.dumps(filters, sort_keys=True))
        cached = await self.redis.get(key)
        if cached:
            return json.loads(cached)
        return None

    async def set_search_result(self, query: str, filters: list, result: dict, ttl: int = 1800):
        key = self._make_key("search", query.lower().strip(), json.dumps(filters, sort_keys=True))
        await self.redis.setex(key, ttl, json.dumps(result))

    async def get_reranked(self, query: str, chunk_ids: List[str]) -> Optional[List[dict]]:
        sorted_ids = sorted(chunk_ids)
        key = self._make_key("rerank", query.lower().strip(), "|".join(sorted_ids))
        cached = await self.redis.get(key)
        if cached:
            return json.loads(cached)
        return None

    async def set_reranked(self, query: str, chunk_ids: List[str], reranked: List[dict], ttl: int = 3600):
        sorted_ids = sorted(chunk_ids)
        key = self._make_key("rerank", query.lower().strip(), "|".join(sorted_ids))
        await self.redis.setex(key, ttl, json.dumps(reranked))

    async def invalidate_all(self):
        """Call this when re-ingesting corpus to clear stale cache."""
        pattern = f"openinsight:{self.CACHE_VERSION}:*"
        keys = await self.redis.keys(pattern)
        if keys:
            await self.redis.delete(*keys)
```

---

### Putting It All Together — Search Endpoint

**Location:** `app/api/search.py`

```python
from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional, List

router = APIRouter()


class SearchRequest(BaseModel):
    query: str
    filters: Optional[dict] = None  # user-level overrides
    top_k_final: int = 6  # final chunks sent to LLM


class SearchResponse(BaseModel):
    answer: str
    citations: List[dict]
    query_intent: str
    chunks_retrieved: int
    cache_hit: bool


@router.post("/search", response_model=SearchResponse)
async def standard_search(request: SearchRequest):
    """
    Standard search endpoint.
    Pipeline:
      1. Query understanding (intent, entities, filters, HyDE flag)
      2. Check Redis cache for identical query
      3. Parallel hybrid retrieval (dense + sparse)
      4. RRF fusion + evidence boost (→ top-20)
      5. Cross-encoder reranking (→ top-8)
      6. MMR deduplication (→ top-6)
      7. Context assembly
      8. LLM generation (streamed)
    """
    from app.search.query_understanding import QueryUnderstanding
    from app.search.retriever import HybridRetriever
    from app.search.fusion import reciprocal_rank_fusion
    from app.search.reranker import CrossEncoderReranker
    from app.search.mmr import maximal_marginal_relevance
    from app.search.context_builder import assemble_context, build_citation_list
    from app.search.cache import SearchCache

    # Lazy singletons (init once at startup, reuse across requests)
    query_understanding = QueryUnderstanding()
    retriever = HybridRetriever()
    reranker = CrossEncoderReranker()
    cache = SearchCache()
    embedder = retriever.embedder

    # Step 1: Query understanding
    analysis = query_understanding.analyze(request.query)

    # Step 2: Cache check (skip rest of pipeline if cached)
    cache_hit = False
    cached = await cache.get_search_result(request.query, analysis.metadata_filters)
    if cached:
        cache_hit = True
        return SearchResponse(**cached, cache_hit=True)

    # Step 3: Retrieve
    dense_results, sparse_results = await retriever.retrieve(
        request.query,
        analysis,
        top_k=50
    )

    # Step 4: RRF fusion → top-20
    fused = reciprocal_rank_fusion(dense_results, sparse_results, k=60, top_n=20)

    # Step 5: Reranking → top-8
    # Check reranker cache first
    chunk_ids = [c.chunk_id for c in fused]
    cached_reranked = await cache.get_reranked(request.query, chunk_ids)
    if cached_reranked:
        reranked = [_chunk_from_dict(c) for c in cached_reranked]
    else:
        reranked = reranker.rerank(request.query, fused, top_k=8)
        await cache.set_reranked(request.query, chunk_ids, [_chunk_to_dict(c) for c in reranked])

    # Step 6: MMR → top-6
    final_chunks = maximal_marginal_relevance(reranked, embedder, lambda_param=0.7, n_select=6)

    # Step 7: Context assembly
    context = assemble_context(final_chunks, max_tokens=3000)
    citations = build_citation_list(final_chunks)

    # Step 8: LLM generation (streaming handled by frontend)
    answer = await _generate_answer(request.query, context, analysis.intent)

    result = {
        "answer": answer,
        "citations": citations,
        "query_intent": analysis.intent.value,
        "chunks_retrieved": len(fused),
        "cache_hit": False,
    }

    # Cache the full result
    await cache.set_search_result(request.query, analysis.metadata_filters, result)

    return SearchResponse(**result, cache_hit=cache_hit)
```

---

## 5. Speed Optimisations

| Component | Change | Expected Gain |
|---|---|---|
| Model loading | Load once at startup via lifespan event, not per request | Eliminates ~2s cold start |
| Dense + sparse retrieval | `asyncio.gather` (parallel) | ~1.5–2× faster retrieval |
| Reranker input | RRF → 20 candidates before reranking (was 50+) | ~2.5× faster reranking |
| Embedding model | `torch.inference_mode()` + `model.eval()` | ~30% faster embedding |
| Batch embedding at ingestion | `batch_size=32` in `encode()` | ~10× faster ingestion |
| Redis: reranker cache | Cache reranked results for 1h per unique query | Near-instant for repeat queries |
| Redis: full result cache | Cache complete search result for 30min | Zero pipeline cost on repeat |
| Qdrant config | `on_disk=False` for vectors (keep in RAM) | Eliminates disk I/O on search |

**Startup lifespan (FastAPI):**

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI

# Singletons
_query_understanding = None
_retriever = None
_reranker = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _query_understanding, _retriever, _reranker
    # Load all heavy models at startup
    _query_understanding = QueryUnderstanding()
    _retriever = HybridRetriever()
    _reranker = CrossEncoderReranker()  # loads BAAI/bge-reranker-base
    yield
    # Cleanup (if needed)

app = FastAPI(lifespan=lifespan)
```

---

## 6. File and Folder Structure

```
app/
├── ingestion/
│   ├── __init__.py
│   ├── parsers.py          # DocumentParser, ParsedDocument, ParsedSection
│   ├── chunker.py          # HierarchicalChunker, Chunk
│   ├── metadata.py         # ChunkMetadata, MetadataEnricher, DocType, EvidenceLevel
│   ├── embedder.py         # DualEmbedder (dense + sparse)
│   ├── qdrant_indexer.py   # QdrantIndexer, collection creation
│   ├── mongo_store.py      # MongoDocStore
│   ├── pipeline.py         # IngestionPipeline (orchestrates all above)
│   └── run_ingestion.py    # CLI entry point
├── search/
│   ├── __init__.py
│   ├── query_understanding.py  # QueryUnderstanding, QueryAnalysis, QueryIntent
│   ├── retriever.py            # HybridRetriever
│   ├── fusion.py               # reciprocal_rank_fusion + evidence boost
│   ├── reranker.py             # CrossEncoderReranker
│   ├── mmr.py                  # maximal_marginal_relevance
│   ├── context_builder.py      # assemble_context, build_citation_list
│   └── cache.py                # SearchCache (Redis)
├── api/
│   ├── search.py               # /search endpoint
│   └── ingest.py               # /ingest endpoint (trigger ingestion via API)
└── main.py                     # FastAPI app with lifespan
```

---

## 7. Environment Variables

Add these to your `.env` / Codespaces secrets:

```env
# Existing
QDRANT_URL=http://qdrant:6333
MONGODB_URL=mongodb://mongodb:27017
REDIS_URL=redis://redis:6379
NVIDIA_NIM_URL=http://nim-llm:8000

# New
QDRANT_COLLECTION=openinsight_v2
DENSE_MODEL_NAME=pritamdeka/S-PubMedBert-MS-MARCO
RERANKER_MODEL_NAME=BAAI/bge-reranker-base
GROBID_URL=http://grobid:8070
CACHE_VERSION=v2
CACHE_TTL_SEARCH=1800
CACHE_TTL_RERANK=3600
INGESTION_BATCH_SIZE=10
EMBED_BATCH_SIZE=32
TOP_K_RETRIEVAL=50
TOP_K_AFTER_FUSION=20
TOP_K_AFTER_RERANK=8
TOP_K_FINAL=6
MMR_LAMBDA=0.7
HYDE_ENABLED=true
```

---

## 8. Docker Compose Changes

Add GROBID service if not already present:

```yaml
services:
  grobid:
    image: lfoppiano/grobid:0.8.0
    ports:
      - "8070:8070"
    environment:
      - JAVA_OPTS=-Xms512m -Xmx2g
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8070/api/isalive"]
      interval: 30s
      timeout: 10s
      retries: 3
```

Add `lxml` and `httpx` to `requirements.txt`:

```
sentence-transformers>=2.7.0
transformers>=4.40.0
torch>=2.2.0
qdrant-client>=1.9.0
scispacy>=0.5.4
en_core_sci_md @ https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_md-0.5.4.tar.gz
lxml>=5.2.0
httpx>=0.27.0
pdf2image>=1.17.0
pytesseract>=0.3.10
pymongo>=4.7.0
motor>=3.4.0
redis>=5.0.0
```

---

## 9. Implementation Order

Follow this exact order. Each step is independently deployable.

| Step | What | Touches | Time Estimate |
|---|---|---|---|
| 1 | Hierarchical chunker + metadata schema | `ingestion/` | 1–2 days |
| 2 | Re-ingest corpus with new chunker | CLI + Qdrant | 2–4 hours (one-time) |
| 3 | Sparse vectors at ingestion + RRF fusion | `ingestion/`, `search/fusion.py` | 1 day |
| 4 | Query understanding layer | `search/query_understanding.py` | 1 day |
| 5 | Reranker on top-20 (not 50+) + Redis cache | `search/reranker.py`, `search/cache.py` | 0.5 days |
| 6 | MMR post-rerank | `search/mmr.py` | 0.5 days |
| 7 | Parallel async retrieval | `search/retriever.py` | 0.5 days |
| 8 | Context assembly refactor | `search/context_builder.py` | 0.5 days |

---

## 10. Testing Checklist

Run these manually after each step before moving to the next.

**After Step 1–2 (ingestion):**
- [ ] Qdrant collection `openinsight_v2` exists
- [ ] Collection has >10,000 vectors (not ~1,400)
- [ ] Sample a random point: `client.scroll(collection, limit=1)` — verify payload has all metadata fields
- [ ] Check chunk types: query `chunk_type=doc_summary` → expect ~1 per document
- [ ] Check contextual prefix: verify `raw_text` field contains the actual chunk text, not the prefix

**After Step 3 (RRF):**
- [ ] Search returns results from both dense and sparse paths
- [ ] RRF result contains chunks from both lists
- [ ] Evidence-boosted chunks (RCTs, systematic reviews) appear near the top

**After Step 4 (query understanding):**
- [ ] Test `QueryUnderstanding().analyze("treatment for type 2 diabetes")` → intent=THERAPEUTIC, entities has "diabetes"
- [ ] Test `QueryUnderstanding().analyze("latest ICMR guideline for diabetes")` → metadata_filters contains `year >= 2020` and `doc_type in ["guideline"]`
- [ ] Test India filter: `"treatment in Indian patients"` → metadata_filters includes `india_relevant=True`

**After Steps 5–8 (full pipeline):**
- [ ] End-to-end query "what is the first-line treatment for type 2 diabetes in India?" returns answer with citations
- [ ] Latency under 5 seconds for the first query
- [ ] Latency under 1 second for the same query (Redis cache hit)
- [ ] Response includes at least 3 citations with PMIDs
- [ ] No duplicate citations from the same paper
- [ ] All 6 final chunks are from different papers (MMR working)
```
