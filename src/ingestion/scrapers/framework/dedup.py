"""Cross-source deduplication for scraped documents.

Same article can appear in multiple sources (PubMed, IndMED, Medknow, PMC).
We want ONE chunk per article, not 3. CrossSourceDedup matches on:

1. DOI (exact, after normalization) — strongest signal
2. PMID (exact) — strong signal for PubMed-indexed content
3. Title + year (Jaccard >= 0.9 + same year) — moderate signal
4. Content hash (SHA-256 of normalized full text) — strong but only when
   both candidates have full text (abstracts are too short for stable hashes)

When duplicate detected, the higher-trust source wins. Trust order:
    nfi > icmr > cdsco > ntep > nvbdcp > csi > rssdi > pubmed > indmed > medknow > pmc > web

The lower-trust source's identifiers are recorded in `also_indexed_in` on the
kept document, preserving provenance.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

# Trust tier — lower number = higher trust (kept on collision)
SOURCE_TRUST: dict[str, int] = {
    "nfi": 1,
    "ipc": 1,
    "icmr": 2,
    "cdsco": 2,
    "ntep": 2,
    "nvbdcp": 2,
    "nmc_curriculum": 2,
    "ncdir": 2,
    "nfhs": 2,
    "csi": 3,
    "rssdi": 3,
    "statpearls": 3,
    "ncbi_bookshelf": 3,
    "pubmed": 4,
    "pmc": 4,
    "indmed": 5,
    "medknow": 5,
    "web": 9,
}

# Jaccard threshold for title similarity
TITLE_SIMILARITY_THRESHOLD = 0.9

# Stopwords for title normalization
_TITLE_STOPWORDS = frozenset({
    "a", "an", "the", "of", "in", "on", "at", "to", "for", "and", "or", "but",
    "with", "from", "by", "as", "is", "are", "was", "were", "be", "been",
    "this", "that", "these", "those", "it", "its", "their", "his", "her",
})


@dataclass
class DedupMatch:
    """Result of a dedup check between a candidate and an existing document."""
    is_duplicate: bool
    match_type: str = ""  # "doi" | "pmid" | "title" | "content_hash"
    existing_doc_id: str | None = None
    confidence: float = 0.0


@dataclass
class DedupIndex:
    """In-memory index of seen documents for fast dedup checks.

    Maintains indexes by DOI, PMID, normalized-title+year, and content-hash.
    Call `add(doc_id, ...)` after each successful ingest.
    """
    by_doi: dict[str, str] = field(default_factory=dict)  # doi -> doc_id
    by_pmid: dict[str, str] = field(default_factory=dict)  # pmid -> doc_id
    by_title_year: dict[str, str] = field(default_factory=dict)  # "title|year" -> doc_id
    by_content_hash: dict[str, str] = field(default_factory=dict)  # hash -> doc_id
    doc_sources: dict[str, str] = field(default_factory=dict)  # doc_id -> source
    doc_alternates: dict[str, list[str]] = field(default_factory=dict)  # doc_id -> [other sources]

    def add(
        self,
        doc_id: str,
        source: str,
        doi: str | None = None,
        pmid: str | None = None,
        title: str | None = None,
        year: str | None = None,
        content_hash: str | None = None,
    ) -> None:
        """Register a document in the index."""
        self.doc_sources[doc_id] = source
        if doi:
            self.by_doi[self._norm_doi(doi)] = doc_id
        if pmid:
            self.by_pmid[str(pmid).strip()] = doc_id
        if title:
            key = self._title_year_key(title, year)
            if key:
                self.by_title_year[key] = doc_id
        if content_hash:
            self.by_content_hash[content_hash] = doc_id

    def check(
        self,
        source: str,
        doi: str | None = None,
        pmid: str | None = None,
        title: str | None = None,
        year: str | None = None,
        content_hash: str | None = None,
    ) -> DedupMatch:
        """Check if a candidate document is a duplicate of an existing one.

        Returns DedupMatch. If is_duplicate=True, the existing doc_id is in
        `existing_doc_id` — caller should record `source` in that doc's
        `also_indexed_in` list rather than re-ingesting.
        """
        # 1. DOI match
        if doi:
            norm = self._norm_doi(doi)
            if norm in self.by_doi:
                existing = self.by_doi[norm]
                return DedupMatch(
                    is_duplicate=True,
                    match_type="doi",
                    existing_doc_id=existing,
                    confidence=1.0,
                )
        # 2. PMID match
        if pmid:
            pmid_str = str(pmid).strip()
            if pmid_str in self.by_pmid:
                existing = self.by_pmid[pmid_str]
                return DedupMatch(
                    is_duplicate=True,
                    match_type="pmid",
                    existing_doc_id=existing,
                    confidence=1.0,
                )
        # 3. Title + year match
        if title:
            key = self._title_year_key(title, year)
            if key and key in self.by_title_year:
                existing = self.by_title_year[key]
                return DedupMatch(
                    is_duplicate=True,
                    match_type="title",
                    existing_doc_id=existing,
                    confidence=0.9,
                )
        # 4. Content hash match
        if content_hash and content_hash in self.by_content_hash:
            existing = self.by_content_hash[content_hash]
            return DedupMatch(
                is_duplicate=True,
                match_type="content_hash",
                existing_doc_id=existing,
                confidence=0.95,
            )
        return DedupMatch(is_duplicate=False)

    def record_alternate_source(self, doc_id: str, source: str) -> None:
        """Record that `source` also indexes this document.

        Called when a duplicate is detected — preserves provenance.
        """
        self.doc_alternates.setdefault(doc_id, [])
        if source not in self.doc_alternates[doc_id]:
            self.doc_alternates[doc_id].append(source)

    @staticmethod
    def _norm_doi(doi: str) -> str:
        """Lowercase + strip URL prefix."""
        doi = doi.strip().lower()
        for prefix in ("https://doi.org/", "http://doi.org/", "https://dx.doi.org/", "doi:"):
            if doi.startswith(prefix):
                doi = doi[len(prefix):]
                break
        return doi

    @staticmethod
    def _title_year_key(title: str | None, year: str | None) -> str | None:
        """Build a normalized title|year key for exact-match dedup.

        Normalization:
        - lowercase
        - strip punctuation
        - remove stopwords
        - collapse whitespace
        - append year (or "unknown")
        """
        if not title:
            return None
        # Strip HTML if present
        title = re.sub(r"<[^>]+>", " ", title)
        # Lowercase + replace non-alphanumeric with space
        title = re.sub(r"[^a-z0-9\s]", " ", title.lower())
        # Tokenize + filter stopwords
        tokens = [t for t in title.split() if t and t not in _TITLE_STOPWORDS]
        if not tokens:
            return None
        normalized = " ".join(tokens)
        year_str = str(year).strip() if year else "unknown"
        return f"{normalized}|{year_str}"


def compute_content_hash(text: str) -> str:
    """SHA-256 hash of normalized text content.

    Normalization: lowercase, collapse whitespace, strip leading/trailing space.
    Used for cross-source dedup when both candidates have full text.
    """
    normalized = re.sub(r"\s+", " ", text.lower().strip())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def jaccard_similarity(a: str, b: str) -> float:
    """Token Jaccard similarity between two strings.

    Used for fuzzy title matching when exact normalized match fails.
    """
    tokens_a = set(a.lower().split())
    tokens_b = set(b.lower().split())
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)


def pick_winner(source_a: str, source_b: str) -> str:
    """Pick the higher-trust source between two candidates.

    Returns the source name that should be kept (lower trust number = higher trust).
    """
    trust_a = SOURCE_TRUST.get(source_a, 9)
    trust_b = SOURCE_TRUST.get(source_b, 9)
    return source_a if trust_a <= trust_b else source_b
