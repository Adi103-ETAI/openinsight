"""HTML metadata extractor for academic journal articles.

Extracts:
- Highwire Press tags (citation_title, citation_author, citation_doi, etc.)
  These are the de facto standard on PubMed/Medknow/JAPI/PLOS/Elsevier.
- JSON-LD blocks (schema.org ScholarlyArticle, MedicalScholarlyArticle)
- Dublin Core (DC.Title, DC.Creator, DC.Identifier, DC.Date)
- OpenGraph (og:title, og:description, og:published_time)
- Fallback: <title>, <meta name="description">

Falls back gracefully — if no citation_* tags are present, JSON-LD is tried,
then DC, then OG, then <title>. The first non-empty value wins per field.
"""
from __future__ import annotations

import json
import re
from typing import Any

from bs4 import BeautifulSoup

from src.ingestion.scrapers.framework.models import MetadataSelectors


class MetadataExtractor:
    """Extract structured metadata from HTML."""

    def __init__(self, selectors: MetadataSelectors | None = None) -> None:
        self.selectors = selectors or MetadataSelectors()

    def extract(self, html: bytes | str, encoding: str = "utf-8") -> dict[str, Any]:
        """Extract all available metadata from `html`.

        Returns a dict with keys: title, authors, journal, doi, pmid, pubdate,
        abstract, plus a `raw` dict containing every citation_*, DC.*, og:*,
        and JSON-LD entry found (for debugging + custom extraction).
        """
        if isinstance(html, bytes):
            try:
                html = html.decode(encoding or "utf-8", errors="replace")
            except Exception:
                html = html.decode("utf-8", errors="replace")
        soup = BeautifulSoup(html, "lxml")

        # Collect all meta tags by name/property
        meta_by_name: dict[str, str] = {}
        meta_by_property: dict[str, str] = {}
        for tag in soup.find_all("meta"):
            name = (tag.get("name") or "").strip()
            prop = (tag.get("property") or "").strip()
            content = (tag.get("content") or "").strip()
            if name and content:
                meta_by_name[name.lower()] = content
            if prop and content:
                meta_by_property[prop.lower()] = content

        # Combine into a single lookup (name + property namespaces)
        all_meta: dict[str, str] = {**meta_by_name, **meta_by_property}

        # Multi-valued fields (authors)
        author_lists: dict[str, list[str]] = {}
        for tag in soup.find_all("meta"):
            name = (tag.get("name") or tag.get("property") or "").strip().lower()
            content = (tag.get("content") or "").strip()
            if not name or not content:
                continue
            if name in ("citation_author", "dc.creator", "dc.creator.personal_name", "author"):
                author_lists.setdefault(name, []).append(content)

        # JSON-LD blocks
        jsonld: list[dict] = []
        for script in soup.find_all("script", type="application/ld+json"):
            try:
                data = json.loads(script.string or script.get_text() or "")
                if isinstance(data, list):
                    jsonld.extend(data)
                elif isinstance(data, dict):
                    # Could be a single object or have @graph
                    if "@graph" in data and isinstance(data["@graph"], list):
                        jsonld.extend(data["@graph"])
                    else:
                        jsonld.append(data)
            except (json.JSONDecodeError, TypeError):
                continue

        # Build the result with priority chains
        title = self._first_match(self.selectors.title, all_meta)
        if not title:
            # Try JSON-LD
            for obj in jsonld:
                if isinstance(obj, dict) and obj.get("name") and obj.get("@type") in ("ScholarlyArticle", "MedicalScholarlyArticle", "Article", "NewsArticle"):
                    title = obj["name"]
                    break
        if not title and soup.title:
            title = soup.title.get_text(strip=True)[:500]

        journal = self._first_match(self.selectors.journal, all_meta)
        doi = self._first_match(self.selectors.doi, all_meta)
        # Normalize DOI to lowercase + strip URL prefix
        if doi:
            doi = self._normalize_doi(doi)
        pmid = self._first_match(self.selectors.pmid, all_meta)
        pubdate = self._first_match(self.selectors.pubdate, all_meta)
        abstract = self._first_match(self.selectors.abstract, all_meta)
        if abstract:
            # Strip HTML tags from abstract
            abstract = BeautifulSoup(abstract, "lxml").get_text(" ", strip=True)[:5000]

        # Authors — combine all author meta tags, dedup preserving order
        authors: list[str] = []
        for key in ("citation_author", "dc.creator", "author"):
            for a in author_lists.get(key, []):
                if a and a not in authors:
                    authors.append(a)

        # Fallback: try JSON-LD author
        if not authors:
            for obj in jsonld:
                if isinstance(obj, dict) and obj.get("author"):
                    author_field = obj["author"]
                    if isinstance(author_field, list):
                        for a in author_field:
                            if isinstance(a, dict) and a.get("name"):
                                authors.append(a["name"])
                            elif isinstance(a, str):
                                authors.append(a)
                    elif isinstance(author_field, dict) and author_field.get("name"):
                        authors.append(author_field["name"])
                    elif isinstance(author_field, str):
                        authors.append(author_field)
                    if authors:
                        break

        return {
            "title": title,
            "authors": authors,
            "journal": journal,
            "doi": doi,
            "pmid": pmid,
            "pubdate": pubdate,
            "abstract": abstract,
            "raw": {
                "meta": all_meta,
                "jsonld": jsonld,
            },
        }

    def _first_match(self, keys: list[str], meta: dict[str, str]) -> str | None:
        """Return the first non-empty value among `keys` in `meta`."""
        for key in keys:
            value = meta.get(key.lower())
            if value and value.strip():
                return value.strip()
        return None

    @staticmethod
    def _normalize_doi(doi: str) -> str:
        """Normalize DOI: strip URL prefix, lowercase the registrar."""
        doi = doi.strip()
        # Strip common URL prefixes
        for prefix in ("https://doi.org/", "http://doi.org/", "https://dx.doi.org/", "http://dx.doi.org/", "doi:"):
            if doi.lower().startswith(prefix.lower()):
                doi = doi[len(prefix):]
                break
        # DOIs are case-insensitive on the registrar, case-sensitive on the suffix
        if "/" in doi:
            registrar, _, suffix = doi.partition("/")
            return f"{registrar.lower()}/{suffix}"
        return doi.lower()


# PDF link extraction helper
_PDF_LINK_RE = re.compile(r"\.pdf(?:\?|$|#)", re.IGNORECASE)


def find_pdf_links(html: bytes | str, base_url: str = "") -> list[str]:
    """Extract PDF link URLs from an HTML page.

    Looks for <a> tags whose href ends in .pdf or whose text contains
    'PDF' / 'Full Text'. Returns absolute URLs (resolved against `base_url`).
    """
    from urllib.parse import urljoin, urlparse

    if isinstance(html, bytes):
        html = html.decode("utf-8", errors="replace")
    soup = BeautifulSoup(html, "lxml")
    urls: list[str] = []
    seen: set[str] = set()

    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href or href.startswith("#") or href.startswith("javascript:"):
            continue
        text = a.get_text(" ", strip=True).lower()
        is_pdf_link = bool(_PDF_LINK_RE.search(href)) or "pdf" in text or "full text" in text
        if not is_pdf_link:
            continue
        absolute = urljoin(base_url, href)
        # Normalize: strip fragment
        parsed = urlparse(absolute)
        absolute = parsed._replace(fragment="").geturl()
        if absolute not in seen:
            seen.add(absolute)
            urls.append(absolute)
    return urls
