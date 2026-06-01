# BUILT: HTTPFetcher
"""
HTTP Fetcher — Fast static site fetching via httpx.
Tier 1 of the web search pipeline. No browser needed.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urljoin, urlparse

import httpx
from loguru import logger


@dataclass
class FetchedPage:
    """A fetched web page with extracted content."""
    url: str
    title: str = ""
    text_content: str = ""
    meta_description: str = ""
    status_code: int = 0
    content_type: str = ""
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.status_code == 200 and self.text_content and not self.error


# Patterns to extract readable text from HTML
_TAG_RE = re.compile(r"<[^>]+>")
_SCRIPT_RE = re.compile(r"<script[^>]*>[\s\S]*?</script>", re.IGNORECASE)
_STYLE_RE = re.compile(r"<style[^>]*>[\s\S]*?</style>", re.IGNORECASE)
_MULTILINE_WS_RE = re.compile(r"\n{3,}")
_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)
_META_DESC_RE = re.compile(
    r'<meta[^>]*name=["\']description["\'][^>]*content=["\']([^"\']*)["\']',
    re.IGNORECASE,
)
_META_DESC_RE2 = re.compile(
    r'<meta[^>]*content=["\']([^"\']*)["\'][^>]*name=["\']description["\']',
    re.IGNORECASE,
)
_HEADING_RE = re.compile(r"<h[1-6][^>]*>(.*?)</h[1-6]>", re.IGNORECASE | re.DOTALL)
_PARA_RE = re.compile(r"<p[^>]*>(.*?)</p>", re.IGNORECASE | re.DOTALL)
_LIST_RE = re.compile(r"<li[^>]*>(.*?)</li>", re.IGNORECASE | re.DOTALL)
_TABLE_RE = re.compile(r"<table[^>]*>(.*?)</table>", re.IGNORECASE | re.DOTALL)
_ROW_RE = re.compile(r"<tr[^>]*>(.*?)</tr>", re.IGNORECASE | re.DOTALL)
_CELL_RE = re.compile(r"<t[dh][^>]*>(.*?)</t[dh]>", re.IGNORECASE | re.DOTALL)


class HTTPFetcher:
    """
    Fast HTTP fetcher for static medical sites.
    Fetches HTML, extracts title + readable text content.
    No browser needed — works for WHO, ICMR, CDC, PubMed, NICE, etc.
    """

    DEFAULT_TIMEOUT = 15.0
    DEFAULT_HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (compatible; OpenInsightBot/1.0; "
            "+https://github.com/Adi103-ETAI/openinsight)"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }

    def __init__(self, timeout: float = DEFAULT_TIMEOUT):
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self._timeout,
                follow_redirects=True,
                headers=self.DEFAULT_HEADERS,
            )
        return self._client

    async def fetch(self, url: str) -> FetchedPage:
        """
        Fetch a URL and extract readable content.

        Returns FetchedPage with title, text_content, and metadata.
        Handles redirects, timeouts, and encoding issues.
        """
        try:
            client = await self._get_client()
            response = await client.get(url)
            response.raise_for_status()

            content_type = response.headers.get("content-type", "")
            html = response.text

            title = self._extract_title(html)
            meta_desc = self._extract_meta_description(html)
            text_content = self._extract_readable_text(html)

            return FetchedPage(
                url=str(response.url),
                title=title,
                text_content=text_content,
                meta_description=meta_desc,
                status_code=response.status_code,
                content_type=content_type,
            )

        except httpx.TimeoutException:
            logger.debug(f"[HTTPFetcher] Timeout fetching {url}")
            return FetchedPage(url=url, error="timeout")
        except httpx.HTTPStatusError as e:
            logger.debug(f"[HTTPFetcher] HTTP {e.response.status_code} for {url}")
            return FetchedPage(url=url, status_code=e.response.status_code, error=f"HTTP {e.response.status_code}")
        except Exception as e:
            logger.debug(f"[HTTPFetcher] Error fetching {url}: {e}")
            return FetchedPage(url=url, error=str(e)[:200])

    async def fetch_many(self, urls: list[str]) -> list[FetchedPage]:
        """Fetch multiple URLs concurrently."""
        import asyncio
        tasks = [self.fetch(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        pages = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                pages.append(FetchedPage(url=urls[i], error=str(result)[:200]))
            else:
                pages.append(result)
        return pages

    def _extract_title(self, html: str) -> str:
        match = _TITLE_RE.search(html)
        return self._clean_text(match.group(1)) if match else ""

    def _extract_meta_description(self, html: str) -> str:
        match = _META_DESC_RE.search(html) or _META_DESC_RE2.search(html)
        return self._clean_text(match.group(1)) if match else ""

    def _extract_readable_text(self, html: str) -> str:
        """Extract readable text from HTML, preserving structure."""
        # Remove scripts and styles
        text = _SCRIPT_RE.sub("", html)
        text = _STYLE_RE.sub("", text)

        # Extract structured elements
        parts = []

        # Headings
        for match in _HEADING_RE.finditer(text):
            heading = self._clean_text(match.group(1))
            if heading and len(heading) > 3:
                parts.append(f"\n{heading}\n")

        # Paragraphs
        for match in _PARA_RE.finditer(text):
            para = self._clean_text(match.group(1))
            if para and len(para) > 20:
                parts.append(para)

        # List items
        for match in _LIST_RE.finditer(text):
            item = self._clean_text(match.group(1))
            if item and len(item) > 10:
                parts.append(f"- {item}")

        # Tables (simplified)
        for table_match in _TABLE_RE.finditer(text):
            table_text = self._extract_table(table_match.group(0))
            if table_text:
                parts.append(table_text)

        # Fallback: if no structured elements found, strip all tags
        if not parts:
            clean = _TAG_RE.sub(" ", text)
            clean = self._clean_text(clean)
            if clean:
                parts.append(clean)

        result = "\n\n".join(parts)
        # Limit to ~5000 chars (enough for summarization)
        return result[:5000]

    def _extract_table(self, table_html: str) -> str:
        """Extract table content as markdown-ish format."""
        rows = []
        for row_match in _ROW_RE.finditer(table_html):
            cells = []
            for cell_match in _CELL_RE.finditer(row_match.group(1)):
                cell = self._clean_text(cell_match.group(1))
                cells.append(cell)
            if cells:
                rows.append(" | ".join(cells))

        if rows:
            return "\n".join(rows)
        return ""

    def _clean_text(self, text: str) -> str:
        """Clean extracted text — decode entities, normalize whitespace."""
        # Decode common HTML entities
        text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
        text = text.replace("&nbsp;", " ").replace("&quot;", '"').replace("&#39;", "'")
        text = text.replace("&#x27;", "'").replace("&#x2F;", "/")
        # Strip tags if any remain
        text = _TAG_RE.sub(" ", text)
        # Normalize whitespace
        text = re.sub(r"[ \t]+", " ", text)
        text = _MULTILINE_WS_RE.sub("\n\n", text)
        return text.strip()

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None
