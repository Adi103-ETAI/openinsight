"""robots.txt parser + crawl-delay respect.

Politeness is non-negotiable. Every fetch goes through RobotsChecker first
(unless the source explicitly opts out, which we never do for medical sources).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

from loguru import logger
from src.ingestion.scrapers.framework.cache import ScrapeCache


@dataclass
class RobotsRule:
    """A single Allow/Disallow rule from robots.txt."""
    path: str
    allow: bool  # True = Allow, False = Disallow


@dataclass
class RobotsFile:
    """Parsed robots.txt for one origin."""
    origin: str  # scheme://host
    user_agents: dict[str, list[RobotsRule]] = field(default_factory=dict)
    crawl_delay: float | None = None  # seconds
    sitemaps: list[str] = field(default_factory=list)
    fetched_at: float = 0.0  # unix timestamp

    def is_allowed(self, url: str, user_agent: str = "OpenInsight-Bot") -> bool:
        """Check if `url` can be fetched by `user_agent`.

        Per RFC 9309: only the most specific matching UA group's rules apply.
        If a specific group matches the UA, use ONLY that group's rules.
        Otherwise fall back to the `*` group. If no group matches, allow.
        """
        path = urlparse(url).path or "/"
        if path.startswith("/"):
            rel = path
        else:
            rel = "/" + path
        # Add query string back (some robots.txt rules include query params)
        query = urlparse(url).query
        if query:
            rel = rel + "?" + query

        # Find the most specific matching UA group (longest matching prefix wins)
        ua_lower = user_agent.lower()
        best_group: str | None = None
        best_prefix_len = -1
        for ua_key in self.user_agents.keys():
            if ua_key == "*":
                continue
            if ua_lower.startswith(ua_key.lower()):
                if len(ua_key) > best_prefix_len:
                    best_prefix_len = len(ua_key)
                    best_group = ua_key

        # If a specific UA group matched, use only its rules. Otherwise fall back to `*`.
        if best_group is not None:
            applicable_rules: list[RobotsRule] = self.user_agents[best_group]
        elif "*" in self.user_agents:
            applicable_rules = self.user_agents["*"]
        else:
            return True  # no rules at all = allow

        # Per RFC: longest-match wins. If allow + disallow both match, the
        # longer pattern wins. Empty disallow = allow all.
        best_match: RobotsRule | None = None
        best_len = -1
        for rule in applicable_rules:
            if rule.path == "":
                # Empty pattern: matches everything with length 0
                pattern_len = 0
            elif rel.startswith(rule.path):
                pattern_len = len(rule.path)
            else:
                continue
            if pattern_len > best_len:
                best_len = pattern_len
                best_match = rule

        if best_match is None:
            return True  # no rule = allow
        return best_match.allow


class RobotsChecker:
    """Fetches + caches robots.txt per origin.

    Cache duration: 24h (robots.txt rarely changes more often).
    On fetch failure: assume ALLOWED (permissive) but log warning — we don't
    want a transient network error to block the entire crawl. The HTTP client
    will still apply rate limiting.
    """

    CACHE_TTL_SECONDS = 24 * 60 * 60
    USER_AGENT = "OpenInsight-Bot"

    def __init__(self, cache: ScrapeCache | None = None) -> None:
        self._cache = cache
        self._memory: dict[str, RobotsFile] = {}  # origin → parsed

    async def can_fetch(self, url: str, user_agent: str = USER_AGENT) -> bool:
        """Check robots.txt permission for `url`."""
        try:
            robots = await self._get_robots_for_url(url)
            return robots.is_allowed(url, user_agent)
        except Exception as e:
            logger.warning(f"[robots] failed to check {url}: {e} — allowing")
            return True

    async def get_crawl_delay(self, url: str, user_agent: str = USER_AGENT) -> float | None:
        """Get crawl-delay for `url`'s origin, if specified."""
        try:
            robots = await self._get_robots_for_url(url)
            return robots.crawl_delay
        except Exception:
            return None

    async def _get_robots_for_url(self, url: str) -> RobotsFile:
        origin = self._origin(url)
        # L1: memory
        cached = self._memory.get(origin)
        if cached and (time.time() - cached.fetched_at) < self.CACHE_TTL_SECONDS:
            return cached
        # L2: shared cache (Redis) — store raw text, parse on hit
        cache_key = f"robots:{origin}"
        raw_text: str | None = None
        if self._cache is not None:
            raw_text = await self._cache.get_string(cache_key)
        if raw_text is None:
            raw_text = await self._fetch_robots(origin)
            if self._cache is not None and raw_text:
                await self._cache.set_string(cache_key, raw_text, ttl=self.CACHE_TTL_SECONDS)
        robots = self._parse(origin, raw_text)
        robots.fetched_at = time.time()
        self._memory[origin] = robots
        return robots

    async def _fetch_robots(self, origin: str) -> str:
        """Fetch robots.txt from `origin/robots.txt`."""
        import httpx
        url = f"{origin}/robots.txt"
        try:
            async with httpx.AsyncClient(
                timeout=10.0,
                follow_redirects=True,
                headers={"User-Agent": self.USER_AGENT},
            ) as client:
                resp = await client.get(url)
                if resp.status_code == 200:
                    return resp.text
                if resp.status_code == 404:
                    # No robots.txt = allow all
                    return ""
                logger.warning(f"[robots] {url} returned {resp.status_code}")
                return ""
        except Exception as e:
            logger.warning(f"[robots] fetch failed for {url}: {e}")
            return ""

    def _parse(self, origin: str, text: str) -> RobotsFile:
        """Parse robots.txt per RFC 9309 (subset)."""
        robots = RobotsFile(origin=origin)
        if not text:
            return robots

        current_uas: list[str] = []
        current_rules: list[RobotsRule] = []

        for raw_line in text.splitlines():
            line = raw_line.split("#", 1)[0].strip()  # strip comments
            if not line:
                continue
            if ":" not in line:
                continue
            field_name, _, value = line.partition(":")
            field_name = field_name.strip().lower()
            value = value.strip()

            if field_name == "user-agent":
                # If we were accumulating rules for previous UAs, flush them
                if current_rules and current_uas:
                    for ua in current_uas:
                        robots.user_agents.setdefault(ua, []).extend(current_rules)
                # Start a new UA group
                current_uas = [value]
                current_rules = []
            elif field_name == "allow":
                current_rules.append(RobotsRule(path=value, allow=True))
            elif field_name == "disallow":
                current_rules.append(RobotsRule(path=value, allow=False))
            elif field_name == "crawl-delay":
                try:
                    robots.crawl_delay = float(value)
                except ValueError:
                    pass
            elif field_name == "sitemap":
                robots.sitemaps.append(value)

        # Flush last group
        if current_rules and current_uas:
            for ua in current_uas:
                robots.user_agents.setdefault(ua, []).extend(current_rules)

        return robots

    @staticmethod
    def _origin(url: str) -> str:
        parsed = urlparse(url)
        return f"{parsed.scheme}://{parsed.netloc}"
