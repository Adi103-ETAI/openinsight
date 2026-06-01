# BUILT: ContentExtractor
"""
Content Extractor — Extracts structured medical sources from fetched content.
Works with both HTTP-fetched pages and CDP browser content.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from loguru import logger


# Trust tier mapping
TRUSTED_DOMAINS = {
    # Tier 1 — Authoritative
    "who.int": 1, "icmr.gov.in": 1, "cdc.gov": 1, "nice.org.uk": 1,
    "ahajournals.org": 1, "acc.org": 1, "nccn.org": 1,
    "pubmed.ncbi.nlm.nih.gov": 1, "pmc.ncbi.nlm.nih.gov": 1,
    # Tier 2 — Peer-reviewed
    "nejm.org": 2, "jamanetwork.com": 2, "thelancet.com": 2,
    "bmj.com": 2, "japi.org": 2, "ijmr.org.in": 2,
    # Tier 3 — Institutional
    "aiims.edu": 3, "pgimer.edu.in": 3, "cdsco.gov.in": 3, "fda.gov": 3,
    # Tier 4 — Aggregators
    "medscape.com": 4, "mayoclinic.org": 4, "uptodate.com": 4,
    # Tier 5 — News
    "reuters.com": 5, "statnews.com": 5, "thewire.in": 5,
}

TIER_LABELS = {
    1: "Authoritative", 2: "Peer-reviewed", 3: "Institutional",
    4: "Aggregator", 5: "News",
}

MAX_TIER = 4  # Filter threshold


@dataclass
class ExtractedSource:
    """A structured medical source extracted from web content."""
    id: str
    title: str
    url: str
    excerpt: str
    date: str
    tier: int
    tier_label: str
    word_count: int = 0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "url": self.url,
            "excerpt": self.excerpt,
            "date": self.date,
            "tier": self.tier,
            "tier_label": self.tier_label,
        }


# Date extraction patterns
_DATE_PATTERNS = [
    re.compile(r"(?:published|updated|posted|date)[:\s]*(\w+ \d{1,2},? \d{4})", re.I),
    re.compile(r"(\w+ \d{1,2},? \d{4})"),  # "January 15, 2024"
    re.compile(r"(\d{4}-\d{2}-\d{2})"),     # "2024-01-15"
    re.compile(r"(\d{2}/\d{2}/\d{4})"),     # "01/15/2024"
    re.compile(r"(©\s*\d{4})"),              # "© 2024"
]


class ContentExtractor:
    """
    Extracts structured medical sources from fetched content.
    Handles both HTTP-fetched pages and CDP browser content.
    """

    def extract_from_http_page(
        self,
        url: str,
        title: str,
        text_content: str,
        meta_description: str = "",
        source_index: int = 1,
    ) -> ExtractedSource | None:
        """
        Extract a structured source from an HTTP-fetched page.
        Returns None if content is too thin to be useful.
        """
        if not text_content or len(text_content) < 50:
            return None

        # Get tier from domain
        tier = self._get_tier_from_url(url)
        if tier > MAX_TIER:
            return None

        # Extract date
        date = self._extract_date(text_content)

        # Extract relevant excerpt (first ~500 chars of meaningful content)
        excerpt = self._extract_excerpt(text_content, max_chars=500)

        if not excerpt or len(excerpt) < 50:
            return None

        return ExtractedSource(
            id=f"web_{source_index}",
            title=title or url,
            url=url,
            excerpt=excerpt,
            date=date,
            tier=tier,
            tier_label=TIER_LABELS.get(tier, "Unknown"),
            word_count=len(text_content.split()),
        )

    def extract_from_cdp_content(
        self,
        url: str,
        title: str,
        page_text: str,
        source_index: int = 1,
    ) -> ExtractedSource | None:
        """
        Extract a structured source from CDP browser content.
        Similar to HTTP but handles richer content.
        """
        if not page_text or len(page_text) < 50:
            return None

        tier = self._get_tier_from_url(url)
        if tier > MAX_TIER:
            return None

        date = self._extract_date(page_text)
        excerpt = self._extract_excerpt(page_text, max_chars=800)

        if not excerpt or len(excerpt) < 50:
            return None

        return ExtractedSource(
            id=f"web_{source_index}",
            title=title or url,
            url=url,
            excerpt=excerpt,
            date=date,
            tier=tier,
            tier_label=TIER_LABELS.get(tier, "Unknown"),
            word_count=len(page_text.split()),
        )

    def extract_from_llm_sources(
        self,
        sources: list[dict],
        start_index: int = 1,
    ) -> list[ExtractedSource]:
        """Convert LLM-provided source dicts to ExtractedSource objects."""
        results = []
        for i, s in enumerate(sources, start_index):
            tier = s.get("tier", 5)
            if tier > MAX_TIER:
                continue
            results.append(ExtractedSource(
                id=f"web_{i}",
                title=s.get("title", ""),
                url=s.get("url", ""),
                excerpt=s.get("excerpt", "")[:500],
                date=s.get("date", ""),
                tier=tier,
                tier_label=TIER_LABELS.get(tier, "Unknown"),
            ))
        return results

    def _get_tier_from_url(self, url: str) -> int:
        """Determine trust tier from URL domain."""
        url_lower = url.lower()
        for domain, tier in TRUSTED_DOMAINS.items():
            if domain in url_lower:
                return tier
        return 5  # Unknown = lowest

    def _extract_date(self, text: str) -> str:
        """Extract publication date from text."""
        # Try each pattern
        for pattern in _DATE_PATTERNS:
            match = pattern.search(text[:2000])  # Check first 2000 chars
            if match:
                date_str = match.group(1) if match.lastindex else match.group(0)
                return self._normalize_date(date_str)
        return ""

    def _normalize_date(self, date_str: str) -> str:
        """Normalize date string to YYYY-MM-DD or YYYY format."""
        date_str = date_str.strip()

        # Already YYYY-MM-DD
        if re.match(r"^\d{4}-\d{2}-\d{2}$", date_str):
            return date_str

        # Extract year
        year_match = re.search(r"(20\d{2})", date_str)
        if year_match:
            return year_match.group(1)

        return date_str[:20]

    def _extract_excerpt(self, text: str, max_chars: int = 500) -> str:
        """Extract a relevant excerpt from text content."""
        # Split into sentences
        sentences = re.split(r"(?<=[.!?])\s+", text)

        # Score sentences by medical relevance
        medical_keywords = [
            "treatment", "diagnosis", "dose", "dosage", "drug", "therapy",
            "patient", "clinical", "evidence", "guideline", "recommendation",
            "study", "trial", "systematic review", "meta-analysis",
            "contraindication", "adverse", "efficacy", "safety",
            "monitoring", "protocol", "management", "prevention",
        ]

        scored_sentences = []
        for sentence in sentences:
            if len(sentence) < 20:
                continue
            score = sum(1 for kw in medical_keywords if kw in sentence.lower())
            scored_sentences.append((score, sentence))

        # Sort by relevance score
        scored_sentences.sort(key=lambda x: x[0], reverse=True)

        # Build excerpt from most relevant sentences
        excerpt = ""
        for _, sentence in scored_sentences:
            if len(excerpt) + len(sentence) + 1 > max_chars:
                break
            excerpt += sentence.strip() + " "

        # If no scored sentences, take first N chars
        if not excerpt:
            excerpt = text[:max_chars]

        return excerpt.strip()
