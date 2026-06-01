# BUILT: WebSearchAgent
"""
Web Search Agent — Three-tier live web retrieval.

Tier 1: HTTP fetch (fast, no browser, works for 80% of medical sites)
Tier 2: CDP browser (Lightpanda/Chrome for JS-heavy sites)
Tier 3: Gemini Flash (LLM fallback when no sources found)
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from loguru import logger

from src.config.settings import get_settings
from src.services.llm.router import LLMRouter
from src.services.browser.http_fetcher import HTTPFetcher
from src.services.browser.content_extractor import ContentExtractor, ExtractedSource, MAX_TIER
from src.query.deepinsight.agents.skills import get_system_prompt


@dataclass
class WebSearchResult:
    """Result from web search agent."""
    summary: str = ""
    sources: list[dict] = field(default_factory=list)
    conflict_flag: bool = False
    conflict_detail: str | None = None
    retrieved_at: str = ""
    found: bool = False
    tier_used: int = 0  # Which tier succeeded (1, 2, or 3)


class WebSearchAgent:
    """
    Three-tier web search agent for clinical evidence retrieval.

    Tier 1: HTTP fetch — fast, no browser, handles static medical sites
    Tier 2: CDP browser — Lightpanda/Chrome for JS-heavy sites
    Tier 3: Gemini Flash — LLM knowledge fallback

    Auto-discovers Lightpanda/Chrome via CDP at startup.
    """

    def __init__(self, settings: Any = None, llm_router: LLMRouter | None = None):
        self.settings = settings or get_settings()
        self.llm_router = llm_router or LLMRouter()
        self._http = HTTPFetcher()
        self._extractor = ContentExtractor()

        # CDP browser detection
        self._cdp_url: str | None = None
        self._has_browser = False

        # Detect browser on init (non-blocking)
        self._browser_detection_task: asyncio.Task | None = None

    async def _ensure_browser_detected(self) -> None:
        """Detect available browser (Lightpanda/Chrome) via CDP."""
        if self._cdp_url is not None:
            return

        # Check env var first
        self._cdp_url = os.getenv("CDP_BROWSER_URL") or os.getenv("LIGHTPANDA_CDP_URL")

        if self._cdp_url:
            self._has_browser = True
            logger.info(f"[WebSearch] CDP browser from env: {self._cdp_url}")
            return

        # Auto-discover
        from src.services.browser.cdp_browser import discover_cdp_url
        self._cdp_url = await discover_cdp_url()
        if self._cdp_url:
            self._has_browser = True
            logger.info(f"[WebSearch] Auto-discovered CDP browser: {self._cdp_url}")
        else:
            logger.info("[WebSearch] No CDP browser found — using HTTP-only mode")

    async def run(self, query: str, original_query: str) -> WebSearchResult:
        """
        Execute three-tier web search.

        Args:
            query: The query to search for
            original_query: The original user query

        Returns:
            WebSearchResult with sources, summary, and conflict info
        """
        retrieved_at = datetime.now(timezone.utc).isoformat()

        # Ensure browser detection is done
        await self._ensure_browser_detected()

        # Build target URLs
        search_queries = self._build_search_queries(query, original_query)
        target_urls = self._build_target_urls(query, original_query)

        # ── Tier 1: HTTP fetch ───────────────────────────────────────────────
        sources = await self._tier1_http(target_urls, query)

        # ── Tier 2: CDP browser (if Tier  insufficient and browser available) ─
        if len(sources) < 2 and self._has_browser:
            browser_sources = await self._tier2_browser(target_urls, query)
            sources.extend(browser_sources)

        # Deduplicate by URL
        sources = self._deduplicate_sources(sources)

        # ── Tier 3: Gemini Flash (if no sources at all) ──────────────────────
        if not sources:
            return await self._tier3_gemini(query, search_queries, retrieved_at)

        # ── Summarize via LLM ────────────────────────────────────────────────
        summary = await self._summarize_sources(query, sources)

        # Detect conflicts
        conflict_flag, conflict_detail = self._detect_conflicts(summary)

        return WebSearchResult(
            summary=summary,
            sources=[s.to_dict() for s in sources],
            conflict_flag=conflict_flag,
            conflict_detail=conflict_detail,
            retrieved_at=retrieved_at,
            found=True,
            tier_used=1 if not self._has_browser or len(sources) >= 2 else 2,
        )

    # ── Tier 1: HTTP Fetch ───────────────────────────────────────────────────

    async def _tier1_http(self, urls: list[str], query: str) -> list[ExtractedSource]:
        """Fast HTTP fetch for static medical sites."""
        logger.info(f"[WebSearch] Tier 1: Fetching {len(urls)} URLs via HTTP")
        pages = await self._http.fetch_many(urls)

        sources = []
        for i, page in enumerate(pages, 1):
            if not page.ok:
                continue
            source = self._extractor.extract_from_http_page(
                url=page.url,
                title=page.title,
                text_content=page.text_content,
                meta_description=page.meta_description,
                source_index=i,
            )
            if source:
                sources.append(source)

        logger.info(f"[WebSearch] Tier 1: Got {len(sources)} sources")
        return sources

    # ── Tier 2: CDP Browser ──────────────────────────────────────────────────

    async def _tier2_browser(self, urls: list[str], query: str) -> list[ExtractedSource]:
        """CDP browser for JS-heavy sites."""
        logger.info(f"[WebSearch] Tier 2: Browsing {len(urls)} URLs via CDP")

        from src.services.browser.cdp_browser import CDPBrowser

        browser = CDPBrowser(self._cdp_url, timeout=20.0)
        sources = []

        try:
            connected = await browser.connect()
            if not connected:
                logger.warning("[WebSearch] Tier 2: CDP connection failed")
                return []

            for i, url in enumerate(urls, 1):
                try:
                    # Navigate
                    await browser.navigate(url)
                    await browser.wait_for_load(timeout_ms=10000)

                    # Extract
                    title = await browser.get_title()
                    text = await browser.get_page_text(max_chars=5000)

                    if text and len(text) > 50:
                        source = self._extractor.extract_from_cdp_content(
                            url=browser.get_url() if hasattr(browser, 'get_url') else url,
                            title=title,
                            page_text=text,
                            source_index=i,
                        )
                        if source:
                            sources.append(source)

                except Exception as e:
                    logger.debug(f"[WebSearch] Tier 2: Failed to browse {url}: {e}")

        finally:
            await browser.close()

        logger.info(f"[WebSearch] Tier 2: Got {len(sources)} sources")
        return sources

    # ── Tier 3: Gemini Flash Fallback ────────────────────────────────────────

    async def _tier3_gemini(
        self, query: str, search_queries: list[str], retrieved_at: str
    ) -> WebSearchResult:
        """Gemini Flash structured prompt fallback."""
        logger.info("[WebSearch] Tier 3: Using Gemini Flash fallback")

        client = self.llm_router.get_client_for_agent("web")
        system_prompt = get_system_prompt("web_search_agent", fallback=self._build_system_prompt())
        user_prompt = self._build_user_prompt(query, search_queries)

        try:
            response = await client.chat_completions(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                max_tokens=2048,
            )
        except Exception as e:
            logger.error(f"[WebSearch] Tier 3 LLM failed: {e}")
            return WebSearchResult(
                retrieved_at=retrieved_at,
                found=False,
                summary=f"Web search failed: {str(e)[:200]}",
            )

        if not response:
            return WebSearchResult(
                retrieved_at=retrieved_at,
                found=False,
                summary="Web search returned empty response.",
            )

        # Parse LLM response
        result = self._parse_llm_response(response, retrieved_at)
        result.tier_used = 3
        return result

    # ── Shared Helpers ────────────────────────────────────────────────────────

    async def _summarize_sources(self, query: str, sources: list[ExtractedSource]) -> str:
        """Use LLM to summarize collected sources into a clinical summary."""
        client = self.llm_router.get_client_for_agent("web")

        sources_text = ""
        for s in sources:
            sources_text += (
                f"[{s.id}] {s.title} ({s.date})\n"
                f"URL: {s.url} | Tier: {s.tier_label}\n"
                f"{s.excerpt[:400]}\n\n"
            )

        prompt = (
            f"Clinical query: {query}\n\n"
            f"Retrieved sources:\n{sources_text}\n\n"
            "Summarize the key clinical findings from these sources in 2-3 sentences. "
            "Cite source IDs like [web_1]. Note any conflicts between sources."
        )

        try:
            response = await client.chat_completions(
                messages=[
                    {"role": "system", "content": (
                        "You are a medical evidence summarizer. "
                        "Be concise, factual, and cite sources."
                    )},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=512,
            )
            return response or "Failed to generate summary."
        except Exception as e:
            logger.warning(f"[WebSearch] Summarization failed: {e}")
            return f"Sources retrieved but summarization failed: {str(e)[:200]}"

    def _detect_conflicts(self, summary: str) -> tuple[bool, str | None]:
        """Detect if summary mentions guideline updates or conflicts."""
        conflict_keywords = [
            "guideline update", "new recommendation", "changed from",
            "previously", "conflicts with", "differs from", "updated",
            "supersedes", "replaced by", "no longer recommended",
        ]
        summary_lower = summary.lower()
        for kw in conflict_keywords:
            if kw in summary_lower:
                return True, f"Summary mentions: {kw}"
        return False, None

    def _deduplicate_sources(self, sources: list[ExtractedSource]) -> list[ExtractedSource]:
        """Deduplicate sources by URL, keeping highest tier."""
        seen_urls: dict[str, ExtractedSource] = {}
        for source in sources:
            existing = seen_urls.get(source.url)
            if existing is None or source.tier < existing.tier:
                seen_urls[source.url] = source
        return list(seen_urls.values())

    def _build_search_queries(self, query: str, original_query: str) -> list[str]:
        """Build 2-3 targeted search queries."""
        queries = [query]

        temporal_patterns = [
            r"\b(20\d{2})\b", r"\blast(?:est)?\b", r"\bnew\b",
            r"\brecent\b", r"\bcurrent\b", r"\bupdated\b",
        ]
        has_temporal = any(re.search(p, original_query.lower()) for p in temporal_patterns)

        india_patterns = [r"\bindia\b", r"\bicmr\b", r"\bindian\b"]
        has_india = any(re.search(p, original_query.lower()) for p in india_patterns)

        if has_temporal:
            queries.append(f"{query} guideline update 2024 2025")
        if has_india:
            queries.append(f"{query} India ICMR")
        if not has_temporal and not has_india:
            queries.append(f"{query} clinical guideline evidence")

        return queries[:3]

    def _build_target_urls(self, query: str, original_query: str) -> list[str]:
        """Build target URLs based on query content."""
        urls = []
        q = original_query.lower()

        if any(w in q for w in ["india", "icmr", "indian"]):
            urls.append("https://www.icmr.gov.in/")
        if any(w in q for w in ["heart", "cardiac", "aha", "acc", "heart failure"]):
            urls.extend(["https://www.ahajournals.org/", "https://www.acc.org/"])
        if any(w in q for w in ["nice", "uk", "british"]):
            urls.append("https://www.nice.org.uk/")
        if any(w in q for w in ["who", "world health"]):
            urls.append("https://www.who.int/")
        if any(w in q for w in ["fda", "drug", "approval"]):
            urls.append("https://www.fda.gov/")
        if any(w in q for w in ["nih", "national institutes"]):
            urls.append("https://www.nih.gov/")
        if any(w in q for w in ["cdc", "infection", "outbreak", "vaccine"]):
            urls.append("https://www.cdc.gov/")
        if any(w in q for w in ["pubmed", "study", "trial", "research"]):
            urls.append("https://pubmed.ncbi.nlm.nih.gov/")

        # Fallback: PubMed search
        if not urls:
            encoded = query.replace(" ", "+")
            urls.append(f"https://pubmed.ncbi.nlm.nih.gov/?term={encoded}&sort=date")

        return urls[:5]

    def _build_system_prompt(self) -> str:
        """Fallback system prompt when SKILL.md not available."""
        return (
            "You are a medical web search agent for a clinical decision support system. "
            "Find trustworthy online medical sources and summarize them.\n\n"
            "RULES:\n"
            "1. Only cite Tier 1-4 sources (WHO, ICMR, CDC, NEJM, JAMA, Lancet, BMJ)\n"
            "2. Never cite Wikipedia, social media, or pharma marketing\n"
            "3. Provide real URLs\n"
            "4. Flag conflicts between sources\n\n"
            "OUTPUT: JSON with summary, sources array, conflict_flag, conflict_detail"
        )

    def _build_user_prompt(self, query: str, search_queries: list[str]) -> str:
        """Build user prompt for Gemini Flash."""
        queries_text = "\n".join(f"  - {q}" for q in search_queries)
        return (
            f"Clinical query: {query}\n\n"
            f"Search queries:\n{queries_text}\n\n"
            "Find trustworthy medical sources. Return structured JSON."
        )

    def _parse_llm_response(self, response: str, retrieved_at: str) -> WebSearchResult:
        """Parse LLM JSON response into WebSearchResult."""
        json_match = re.search(r"\{[\s\S]*\}", response)
        if not json_match:
            return WebSearchResult(summary=response[:500], retrieved_at=retrieved_at, found=False)

        try:
            data = json.loads(json_match.group())
        except json.JSONDecodeError:
            return WebSearchResult(summary=response[:500], retrieved_at=retrieved_at, found=False)

        # Convert to ExtractedSource objects for filtering
        raw_sources = data.get("sources", [])
        sources = self._extractor.extract_from_llm_sources(raw_sources)

        return WebSearchResult(
            summary=data.get("summary", ""),
            sources=[s.to_dict() for s in sources],
            conflict_flag=data.get("conflict_flag", False),
            conflict_detail=data.get("conflict_detail"),
            retrieved_at=retrieved_at,
            found=len(sources) > 0,
        )

    async def close(self) -> None:
        """Clean up resources."""
        await self._http.close()
