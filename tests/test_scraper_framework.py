"""Unit tests for the scraper framework (Phase 0.5).

Tests run without external dependencies (no Redis, no network) — everything
is mocked or uses in-memory implementations.

Coverage:
- robots.txt parsing (allow/deny/crawl-delay)
- rate limiter (token bucket math)
- cache (3-tier lookup, backfill, TTL expiry)
- metadata extractor (Highwire + JSON-LD + DC + OG + fallback)
- dedup (DOI / PMID / title+year / content_hash matching)
- find_pdf_links
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import pytest

# --- Robots.txt parser ----------------------------------------------------

from src.ingestion.scrapers.framework.robots import RobotsChecker, RobotsFile, RobotsRule


class TestRobotsParser:
    """robots.txt parsing logic."""

    def test_parse_simple_disallow(self) -> None:
        text = """
        User-agent: *
        Disallow: /private/
        Disallow: /admin
        """
        checker = RobotsChecker()
        robots = checker._parse("https://example.com", text)
        assert robots.is_allowed("https://example.com/public/page.html")
        assert not robots.is_allowed("https://example.com/private/secret.html")
        assert not robots.is_allowed("https://example.com/admin/login")

    def test_parse_empty_robots_allows_all(self) -> None:
        """Empty robots.txt (or 404) means everything is allowed."""
        checker = RobotsChecker()
        robots = checker._parse("https://example.com", "")
        assert robots.is_allowed("https://example.com/anything")
        assert robots.is_allowed("https://example.com/")

    def test_parse_crawl_delay(self) -> None:
        text = """
        User-agent: *
        Crawl-delay: 5
        Disallow: /private
        """
        checker = RobotsChecker()
        robots = checker._parse("https://example.com", text)
        assert robots.crawl_delay == 5.0

    def test_parse_sitemap(self) -> None:
        text = """
        User-agent: *
        Sitemap: https://example.com/sitemap.xml
        Sitemap: https://example.com/sitemap2.xml
        """
        checker = RobotsChecker()
        robots = checker._parse("https://example.com", text)
        assert len(robots.sitemaps) == 2

    def test_parse_user_agent_specific(self) -> None:
        """Specific UA rules override wildcard for that UA."""
        text = """
        User-agent: *
        Disallow: /

        User-agent: OpenInsight-Bot
        Disallow: /private
        """
        checker = RobotsChecker()
        robots = checker._parse("https://example.com", text)
        # OpenInsight-Bot can access public but not private
        assert robots.is_allowed("https://example.com/public", "OpenInsight-Bot")
        assert not robots.is_allowed("https://example.com/private", "OpenInsight-Bot")
        # Other bots denied everything
        assert not robots.is_allowed("https://example.com/public", "OtherBot")

    def test_parse_comments_and_whitespace(self) -> None:
        """Comments and extra whitespace should not break parsing."""
        text = """
        # This is a comment
        User-agent: *
        Disallow: /private  # inline comment
        """
        checker = RobotsChecker()
        robots = checker._parse("https://example.com", text)
        assert not robots.is_allowed("https://example.com/private/page")

    def test_parse_longest_match_wins(self) -> None:
        """Per RFC 9309, longest pattern match wins."""
        text = """
        User-agent: *
        Disallow: /private/
        Allow: /private/public/
        """
        checker = RobotsChecker()
        robots = checker._parse("https://example.com", text)
        # /private/public/ is more specific than /private/ → allowed
        assert robots.is_allowed("https://example.com/private/public/page")
        # /private/secret/ is only matched by Disallow → denied
        assert not robots.is_allowed("https://example.com/private/secret")


# --- Rate limiter ---------------------------------------------------------

from src.ingestion.scrapers.framework.rate_limiter import RateLimiter, TokenBucket


class TestRateLimiter:
    """Token bucket rate limiting."""

    @pytest.mark.asyncio
    async def test_first_acquire_succeeds_immediately(self) -> None:
        limiter = RateLimiter(default_rate=1.0, default_burst=3.0)
        # First acquire should succeed without waiting
        start = time.monotonic()
        ok = await limiter.acquire("https://example.com/page")
        elapsed = time.monotonic() - start
        assert ok
        assert elapsed < 0.1  # should be near-instant

    @pytest.mark.asyncio
    async def test_burst_capacity(self) -> None:
        """Burst capacity should allow multiple immediate acquires."""
        limiter = RateLimiter(default_rate=1.0, default_burst=3.0)
        # 3 acquires should be near-instant (burst)
        for i in range(3):
            ok = await limiter.acquire(f"https://example.com/{i}")
            assert ok
        # 4th acquire should block (needs to wait for token refill)
        # Use short timeout to verify it actually blocks
        start = time.monotonic()
        ok = await limiter.acquire("https://example.com/4")
        elapsed = time.monotonic() - start
        # Should have waited ~1 second for the next token
        assert ok
        assert elapsed > 0.5

    @pytest.mark.asyncio
    async def test_per_domain_isolation(self) -> None:
        """Different domains have independent buckets."""
        limiter = RateLimiter(default_rate=1.0, default_burst=1.0)
        # Exhaust domain A's bucket
        await limiter.acquire("https://a.example.com/1")
        # Domain B should still have a token
        start = time.monotonic()
        ok = await limiter.acquire("https://b.example.com/1")
        elapsed = time.monotonic() - start
        assert ok
        assert elapsed < 0.1

    @pytest.mark.asyncio
    async def test_domain_override(self) -> None:
        """Per-domain rate override should be respected."""
        limiter = RateLimiter(
            default_rate=10.0,
            default_burst=10.0,
            per_domain_overrides={"slow.example.com": (0.5, 1.0)},
        )
        # First acquire ok (burst)
        await limiter.acquire("https://slow.example.com/1")
        # Second should block ~2 seconds (rate 0.5/sec)
        start = time.monotonic()
        await limiter.acquire("https://slow.example.com/2")
        elapsed = time.monotonic() - start
        assert elapsed > 1.0

    def test_set_domain_rate_runtime(self) -> None:
        """Runtime rate override should replace existing bucket config."""
        limiter = RateLimiter(default_rate=10.0, default_burst=10.0)
        limiter.set_domain_rate("special.example.com", 0.1, 1.0)
        stats = limiter.stats()
        assert "special.example.com" not in stats  # bucket not created yet
        # Now trigger creation by acquiring
        asyncio.run(limiter.acquire("https://special.example.com/1"))
        stats = limiter.stats()
        assert "special.example.com" in stats
        assert stats["special.example.com"]["refill_rate"] == 0.1


# --- Cache ----------------------------------------------------------------

from src.ingestion.scrapers.framework.cache import MemoryTier, ScrapeCache, _cache_key


class TestCache:
    """3-tier cache behavior."""

    def test_cache_key_stable(self) -> None:
        """Same inputs produce same cache key."""
        k1 = _cache_key("GET", "https://example.com/page")
        k2 = _cache_key("GET", "https://example.com/page")
        k3 = _cache_key("GET", "https://example.com/other")
        assert k1 == k2
        assert k1 != k3

    def test_cache_key_method_matters(self) -> None:
        k1 = _cache_key("GET", "https://example.com/page")
        k2 = _cache_key("POST", "https://example.com/page")
        assert k1 != k2

    def test_cache_key_body_matters(self) -> None:
        k1 = _cache_key("POST", "https://example.com/page", b"body1")
        k2 = _cache_key("POST", "https://example.com/page", b"body2")
        assert k1 != k2

    def test_memory_tier_set_get(self) -> None:
        tier = MemoryTier()
        tier.set("key1", {"data": "value"})
        assert tier.get("key1") == {"data": "value"}

    def test_memory_tier_lru_eviction(self) -> None:
        """Memory tier should evict oldest entries when at capacity."""
        tier = MemoryTier()
        tier.MAX_ENTRIES = 3  # override for test
        tier.set("k1", "v1")
        tier.set("k2", "v2")
        tier.set("k3", "v3")
        tier.set("k4", "v4")  # should evict k1
        assert tier.get("k1") is None
        assert tier.get("k2") == "v2"
        assert tier.get("k4") == "v4"

    def test_memory_tier_ttl_expiry(self) -> None:
        tier = MemoryTier()
        tier.set("key", "value", ttl=0)  # 0 means no expiry
        assert tier.get("key") == "value"
        # Set with TTL = -1 (already expired)
        tier.set("key2", "value2", ttl=-1)
        assert tier.get("key2") is None

    def test_scrape_cache_sync_lookup(self) -> None:
        """ScrapeCache.get_cached_response should check memory + filesystem (sync)."""
        # Use a temp dir for filesystem tier
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ScrapeCache(redis_url=None, fs_root=tmpdir)
            # Store directly in memory tier
            cache.memory.set(
                _cache_key("GET", "https://example.com/x"),
                {"status_code": 200, "content": b"hello"},
            )
            cached = cache.get_cached_response("GET", "https://example.com/x")
            assert cached is not None
            assert cached["status_code"] == 200
            assert cached["_cache_layer"] == "memory"

    def test_scrape_cache_filesystem_backfill(self) -> None:
        """Filesystem cache should backfill memory tier on hit."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ScrapeCache(redis_url=None, fs_root=tmpdir)
            # Store in filesystem only (skip memory)
            key = _cache_key("GET", "https://example.com/y")
            cache.fs.set(key, {"status_code": 200, "content": b"from-fs"})
            # Memory should be empty
            assert cache.memory.get(key) is None
            # Lookup should hit filesystem and backfill memory
            cached = cache.get_cached_response("GET", "https://example.com/y")
            assert cached is not None
            assert cached["_cache_layer"] == "filesystem"
            assert cached["content"] == b"from-fs"
            # Memory should now have it
            assert cache.memory.get(key) is not None


# --- Metadata extractor ---------------------------------------------------

from src.ingestion.scrapers.framework.metadata_extractor import MetadataExtractor, find_pdf_links


# Sample HTML with Highwire Press citation_* tags (the standard for academic journals)
SAMPLE_HTML_HIGHWIRE = b"""<!DOCTYPE html>
<html>
<head>
<title>Diabetes management in Indian adults: a cross-sectional study</title>
<meta name="citation_title" content="Diabetes management in Indian adults: a cross-sectional study">
<meta name="citation_author" content="Sharma, Priya">
<meta name="citation_author" content="Kumar, Rajesh">
<meta name="citation_author" content="Menon, Anjali">
<meta name="citation_journal_title" content="Indian Journal of Medical Research">
<meta name="citation_doi" content="10.4103/ijmr.ijmr_1234_24">
<meta name="citation_pubmed_id" content="39123456">
<meta name="citation_publication_date" content="2024-08-15">
<meta name="citation_abstract" content="<p>This study examined diabetes management practices among 1,200 adults in urban India.</p>">
</head>
<body>Article body</body>
</html>"""

# Sample HTML with JSON-LD (schema.org ScholarlyArticle)
SAMPLE_HTML_JSONLD = b"""<!DOCTYPE html>
<html>
<head>
<title>Some Article</title>
<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "ScholarlyArticle",
  "name": "A JSON-LD Title",
  "author": [
    {"@type": "Person", "name": "Alice Smith"},
    {"@type": "Person", "name": "Bob Jones"}
  ],
  "datePublished": "2024-01-01",
  "isPartOf": {"@type": "PublicationIssue", "isPartOf": {"@type": "PublicationVolume", "title": "Test Journal"}}
}
</script>
</head>
<body></body>
</html>"""

# Sample HTML with only OpenGraph (rare for academic, common for news)
SAMPLE_HTML_OG = b"""<!DOCTYPE html>
<html>
<head>
<title>OG Title</title>
<meta property="og:title" content="OG Title From Meta">
<meta property="og:description" content="OG description here">
<meta property="og:published_time" content="2024-06-15T10:30:00Z">
</head>
<body></body>
</html>"""


class TestMetadataExtractor:
    """HTML metadata extraction."""

    def test_highwire_extraction(self) -> None:
        extractor = MetadataExtractor()
        meta = extractor.extract(SAMPLE_HTML_HIGHWIRE)
        assert meta["title"] == "Diabetes management in Indian adults: a cross-sectional study"
        assert len(meta["authors"]) == 3
        assert meta["authors"][0] == "Sharma, Priya"
        assert meta["journal"] == "Indian Journal of Medical Research"
        assert meta["doi"] == "10.4103/ijmr.ijmr_1234_24"
        assert meta["pmid"] == "39123456"
        assert meta["pubdate"] == "2024-08-15"
        # Abstract should have HTML stripped
        assert "<p>" not in (meta["abstract"] or "")

    def test_jsonld_extraction(self) -> None:
        extractor = MetadataExtractor()
        meta = extractor.extract(SAMPLE_HTML_JSONLD)
        # JSON-LD title takes precedence over <title>
        assert meta["title"] == "A JSON-LD Title"
        assert "Alice Smith" in meta["authors"]
        assert "Bob Jones" in meta["authors"]

    def test_opengraph_fallback(self) -> None:
        """OG tags fill in when citation_* is absent."""
        extractor = MetadataExtractor()
        meta = extractor.extract(SAMPLE_HTML_OG)
        # og:title is in the default selectors chain
        assert meta["title"] == "OG Title From Meta"
        assert meta["pubdate"] == "2024-06-15T10:30:00Z"

    def test_doi_normalization(self) -> None:
        """DOI should be normalized (strip URL prefix, lowercase registrar)."""
        extractor = MetadataExtractor()
        html = b'<meta name="citation_doi" content="https://doi.org/10.4103/IJMR.ijmr_1234_24">'
        meta = extractor.extract(html)
        assert meta["doi"] == "10.4103/IJMR.ijmr_1234_24"

    def test_title_fallback_to_html_title(self) -> None:
        """If no meta tag has the title, fall back to <title>."""
        extractor = MetadataExtractor()
        html = b"<html><head><title>Plain Title</title></head><body></body></html>"
        meta = extractor.extract(html)
        assert meta["title"] == "Plain Title"

    def test_empty_html(self) -> None:
        extractor = MetadataExtractor()
        meta = extractor.extract(b"")
        assert meta["title"] is None
        assert meta["authors"] == []


class TestFindPdfLinks:
    """PDF link extraction from HTML."""

    def test_direct_pdf_link(self) -> None:
        html = b'<a href="/articles/paper.pdf">Download</a>'
        urls = find_pdf_links(html, base_url="https://example.com")
        assert urls == ["https://example.com/articles/paper.pdf"]

    def test_absolute_pdf_url(self) -> None:
        html = b'<a href="https://other.com/paper.pdf">PDF</a>'
        urls = find_pdf_links(html, base_url="https://example.com")
        assert urls == ["https://other.com/paper.pdf"]

    def test_full_text_link(self) -> None:
        """Links with 'Full Text' text should be detected even without .pdf."""
        html = b'<a href="/articles/12345/full">Full Text</a>'
        urls = find_pdf_links(html, base_url="https://example.com")
        assert urls == ["https://example.com/articles/12345/full"]

    def test_dedup_urls(self) -> None:
        html = b"""
        <a href="/paper.pdf">PDF</a>
        <a href="/paper.pdf">Download PDF</a>
        """
        urls = find_pdf_links(html, base_url="https://example.com")
        assert len(urls) == 1

    def test_skip_javascript_links(self) -> None:
        html = b'<a href="javascript:downloadPdf()">PDF</a>'
        urls = find_pdf_links(html, base_url="https://example.com")
        assert urls == []

    def test_strip_fragment(self) -> None:
        html = b'<a href="/paper.pdf#page=1">PDF</a>'
        urls = find_pdf_links(html, base_url="https://example.com")
        assert urls == ["https://example.com/paper.pdf"]


# --- Dedup ----------------------------------------------------------------

from src.ingestion.scrapers.framework.dedup import (
    DedupIndex,
    SOURCE_TRUST,
    compute_content_hash,
    jaccard_similarity,
    pick_winner,
)


class TestDedup:
    """Cross-source deduplication."""

    def test_doi_match(self) -> None:
        idx = DedupIndex()
        idx.add("doc1", source="pubmed", doi="10.4103/ijmr.ijmr_1234_24")
        match = idx.check(source="indmed", doi="10.4103/IJMR.ijmr_1234_24")
        assert match.is_duplicate
        assert match.match_type == "doi"
        assert match.existing_doc_id == "doc1"

    def test_pmid_match(self) -> None:
        idx = DedupIndex()
        idx.add("doc1", source="pubmed", pmid="39123456")
        match = idx.check(source="medknow", pmid="39123456")
        assert match.is_duplicate
        assert match.match_type == "pmid"

    def test_title_year_match(self) -> None:
        idx = DedupIndex()
        idx.add(
            "doc1",
            source="pubmed",
            title="Diabetes management in Indian adults",
            year="2024",
        )
        # Same title (different case + punctuation) + same year
        match = idx.check(
            source="indmed",
            title="Diabetes Management in Indian Adults.",
            year="2024",
        )
        assert match.is_duplicate
        assert match.match_type == "title"

    def test_title_year_different_year_not_match(self) -> None:
        """Same title but different year should NOT match (could be a different paper)."""
        idx = DedupIndex()
        idx.add("doc1", source="pubmed", title="Annual review of diabetes", year="2024")
        match = idx.check(source="indmed", title="Annual review of diabetes", year="2023")
        assert not match.is_duplicate

    def test_content_hash_match(self) -> None:
        idx = DedupIndex()
        text = "This is the full text of the article. It is long enough to hash."
        h = compute_content_hash(text)
        idx.add("doc1", source="pubmed", content_hash=h)
        match = idx.check(source="medknow", content_hash=h)
        assert match.is_duplicate
        assert match.match_type == "content_hash"

    def test_no_match_returns_is_duplicate_false(self) -> None:
        idx = DedupIndex()
        idx.add("doc1", source="pubmed", doi="10.1/abc", pmid="123", title="Paper A", year="2024")
        match = idx.check(source="indmed", doi="10.2/xyz", pmid="456", title="Paper B", year="2024")
        assert not match.is_duplicate

    def test_compute_content_hash_normalizes_whitespace(self) -> None:
        """Content hash should be insensitive to whitespace differences."""
        h1 = compute_content_hash("Hello  world\n\nfoo")
        h2 = compute_content_hash("Hello world foo")
        assert h1 == h2

    def test_compute_content_hash_lowercase(self) -> None:
        h1 = compute_content_hash("Hello World")
        h2 = compute_content_hash("hello world")
        assert h1 == h2

    def test_jaccard_similarity(self) -> None:
        sim = jaccard_similarity("the quick brown fox", "the quick brown dog")
        # 3 shared / 5 total = 0.6
        assert 0.5 < sim < 0.7

    def test_pick_winner_higher_trust(self) -> None:
        """ICMR should win over PubMed (lower trust number = higher trust)."""
        winner = pick_winner("icmr", "pubmed")
        assert winner == "icmr"

    def test_pick_winner_unknown_source(self) -> None:
        """Unknown sources default to lowest trust (9)."""
        winner = pick_winner("pubmed", "unknown_source")
        assert winner == "pubmed"

    def test_record_alternate_source(self) -> None:
        """When a duplicate is detected, the source should be recorded as alternate."""
        idx = DedupIndex()
        idx.add("doc1", source="pubmed", doi="10.1/abc")
        match = idx.check(source="indmed", doi="10.1/abc")
        assert match.is_duplicate
        assert match.existing_doc_id == "doc1"
        idx.record_alternate_source("doc1", "indmed")
        assert "indmed" in idx.doc_alternates["doc1"]

    def test_source_trust_ordering(self) -> None:
        """NFI should be highest trust, web should be lowest."""
        assert SOURCE_TRUST["nfi"] < SOURCE_TRUST["icmr"]
        assert SOURCE_TRUST["icmr"] < SOURCE_TRUST["pubmed"]
        assert SOURCE_TRUST["pubmed"] < SOURCE_TRUST["indmed"]
        assert SOURCE_TRUST["web"] == 9


# --- SourceConfig ---------------------------------------------------------

from src.ingestion.scrapers.framework.models import SourceConfig


class TestSourceConfig:
    """SourceConfig declarative configuration."""

    def test_user_agent_includes_contact(self) -> None:
        cfg = SourceConfig(name="test", base_url="https://example.com")
        assert "OpenInsight-Bot" in cfg.user_agent
        assert "hello@openinsight.in" in cfg.user_agent

    def test_user_agent_suffix(self) -> None:
        cfg = SourceConfig(name="test", base_url="https://example.com", user_agent_suffix="IndMED-indexer")
        assert "IndMED-indexer" in cfg.user_agent

    def test_domain_extraction(self) -> None:
        cfg = SourceConfig(name="test", base_url="https://indmedinfo.nic.in/index")
        assert cfg.domain == "indmedinfo.nic.in"

    def test_default_metadata_selectors(self) -> None:
        cfg = SourceConfig(name="test", base_url="https://example.com")
        # Default selectors should include Highwire Press citation_* tags
        assert "citation_title" in cfg.metadata_selectors.title
        assert "citation_author" in cfg.metadata_selectors.authors
        assert "citation_doi" in cfg.metadata_selectors.doi


# --- Models ---------------------------------------------------------------

from src.ingestion.scrapers.framework.models import ScrapeResult, ScrapedDocument, CrawlJob


class TestScrapeResult:
    def test_is_pdf_by_content_type(self) -> None:
        r = ScrapeResult(url="https://x.com/doc", ok=True, content_type="application/pdf")
        assert r.is_pdf
        assert not r.is_html

    def test_is_pdf_by_url_extension(self) -> None:
        r = ScrapeResult(url="https://x.com/doc.pdf", ok=True, content_type=None)
        assert r.is_pdf

    def test_is_html(self) -> None:
        r = ScrapeResult(url="https://x.com/doc", ok=True, content_type="text/html; charset=utf-8")
        assert r.is_html
        assert not r.is_pdf

    def test_is_xml(self) -> None:
        r = ScrapeResult(url="https://x.com/api", ok=True, content_type="application/xml")
        assert r.is_xml
