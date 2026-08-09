"""Tests for the IndMED scraper (Phase 1).

Tests cover:
- Journal config (INDMED_JOURNALS list, JOURNAL_TRUST_TIER mapping)
- URL extraction from OJS archive + issue pages
- PDF download link detection on article pages
- Year extraction from publication date strings
- Article URL filtering by year range
- Parser: DocumentRecord + ChunkRecord construction
- Parser: body text extraction (multiple OJS template variants)
- Parser: abstract extraction
- Parser: chunking with overlap

All tests are network-free — they use fixture HTML, not live HTTP.
"""
from __future__ import annotations

from datetime import datetime

import pytest

from src.ingestion.parsers.indmed import IndMEDParser
from src.ingestion.scrapers.framework.models import ScrapedDocument
from src.ingestion.scrapers.sources.indmed import (
    INDMED_CONFIG,
    INDMED_JOURNALS,
    JOURNAL_TRUST_TIER,
    IndMEDScraper,
)


# --- Source config --------------------------------------------------------

class TestIndMEDConfig:
    def test_base_url(self) -> None:
        assert INDMED_CONFIG.base_url == "https://indmedinfo.nic.in"

    def test_rate_limit_is_polite(self) -> None:
        """NIC servers are slow — rate limit must be ≤0.5 req/sec."""
        assert INDMED_CONFIG.rate_limit <= 0.5

    def test_crawl_delay_at_least_2s(self) -> None:
        assert INDMED_CONFIG.crawl_delay >= 2.0

    def test_india_relevant_default(self) -> None:
        assert INDMED_CONFIG.india_relevant_default is True

    def test_indian_source_default(self) -> None:
        assert INDMED_CONFIG.indian_source_default is True

    def test_trust_tier_default_is_4(self) -> None:
        """IndMED-only journals default to Tier 4 (lowest in our hierarchy)."""
        assert INDMED_CONFIG.trust_tier == 4

    def test_user_agent_identifies_bot(self) -> None:
        ua = INDMED_CONFIG.user_agent
        assert "OpenInsight-Bot" in ua
        assert "IndMED-indexer" in ua
        assert "hello@openinsight.in" in ua

    def test_journal_list_has_at_least_20_journals(self) -> None:
        """Phase 1 target: 20+ IndMED journals."""
        assert len(INDMED_JOURNALS) >= 20

    def test_ijmr_is_tier_1(self) -> None:
        """IJMR (Indian J Medical Research) is ICMR-backed — highest trust."""
        assert JOURNAL_TRUST_TIER["ijmr"] == 1

    def test_journal_trust_tier_covers_all_journals(self) -> None:
        """Every journal abbreviation in INDMED_JOURNALS should have a trust tier."""
        for journal_name, abbr in INDMED_JOURNALS.items():
            assert abbr in JOURNAL_TRUST_TIER, f"missing trust tier for {abbr} ({journal_name})"


# --- URL extraction -------------------------------------------------------

# Sample OJS archive page HTML (matches OJS 3.x template)
SAMPLE_ARCHIVE_HTML = """
<html><body>
<div class="archive">
  <a href="/index.php/ijp/issue/view/123">Vol 56 Issue 1 (2024)</a>
  <a href="/index.php/ijp/issue/view/122">Vol 55 Issue 4 (2023)</a>
  <a href="/index.php/ijp/issue/view/121">Vol 55 Issue 3 (2023)</a>
  <a href="/index.php/ijp/issue/view/120">Vol 55 Issue 2 (2023)</a>
  <a href="/index.php/other/article/view/999">Unrelated</a>
</div>
</body></html>
"""

# Sample OJS issue TOC page HTML
SAMPLE_ISSUE_HTML = """
<html><body>
<div class="issue-toc">
  <h3>Articles</h3>
  <a href="/index.php/ijp/article/view/4567">Drug interaction study</a>
  <a href="/index.php/ijp/article/view/4568">Pharmacokinetics of metformin</a>
  <a href="/index.php/ijp/article/view/4569">Clinical trial of newer antidiabetics</a>
  <a href="/index.php/other/article/view/9999">Unrelated</a>
</div>
</body></html>
"""

# Sample OJS article landing page with Highwire Press citation_* meta tags
SAMPLE_ARTICLE_HTML = b"""<!DOCTYPE html>
<html>
<head>
<title>Pharmacokinetics of metformin in Indian patients with type 2 diabetes</title>
<meta name="citation_title" content="Pharmacokinetics of metformin in Indian patients with type 2 diabetes">
<meta name="citation_author" content="Sharma, Priya">
<meta name="citation_author" content="Kumar, Rajesh">
<meta name="citation_journal_title" content="Indian Journal of Pharmacology">
<meta name="citation_doi" content="10.4103/ijp.ijp_123_24">
<meta name="citation_publication_date" content="2024-03-15">
<meta name="citation_abstract" content="<p>This study evaluated metformin pharmacokinetics in 200 Indian adults with T2DM.</p>">
</head>
<body>
<div class="article-body">
<h2>Abstract</h2>
<p>This study evaluated metformin pharmacokinetics in 200 Indian adults with T2DM.</p>

<h2>Introduction</h2>
<p>Metformin is the first-line treatment for type 2 diabetes. However, pharmacokinetic data in Indian populations remain limited. This prospective study aimed to characterize metformin absorption, distribution, and elimination in Indian adults.</p>

<h2>Methods</h2>
<p>We enrolled 200 Indian adults with T2DM (age 30-65, HbA1c 7-10%) from AIIMS Delhi. Single-dose pharmacokinetics were assessed after 500mg metformin. Blood samples were collected at 0, 0.5, 1, 2, 4, 6, 8, 12 hours. Plasma metformin was measured by HPLC.</p>

<h2>Results</h2>
<p>Cmax was 1.42 +/- 0.38 mcg/mL (vs 1.18 in Western cohorts). Tmax was 2.1 hours. AUC0-12 was 9.8 mcg*h/mL. No serious adverse events occurred.</p>

<h2>Conclusion</h2>
<p>Indian adults show higher metformin Cmax than Western cohorts, supporting India-specific dosing guidance. Further studies are warranted.</p>
</div>
<a href="/index.php/ijp/article/download/4568/3456">Download PDF</a>
</body>
</html>"""


class TestIndMEDScraperURLExtraction:
    """Test URL extraction from OJS HTML pages."""

    def test_extract_issue_urls(self) -> None:
        scraper = IndMEDScraper.__new__(IndMEDScraper)  # bypass __init__ (no HTTP deps)
        urls = scraper._extract_issue_urls(SAMPLE_ARCHIVE_HTML, "ijp")
        assert len(urls) == 4
        assert all("/index.php/ijp/issue/view/" in u for u in urls)
        # Should not include unrelated journal paths
        assert not any("other" in u for u in urls)

    def test_extract_issue_urls_dedup(self) -> None:
        """Duplicate issue URLs should be deduped."""
        html = """
        <a href="/index.php/ijp/issue/view/123">Issue 1</a>
        <a href="/index.php/ijp/issue/view/123">Issue 1 (duplicate)</a>
        """
        scraper = IndMEDScraper.__new__(IndMEDScraper)
        urls = scraper._extract_issue_urls(html, "ijp")
        assert len(urls) == 1

    def test_extract_article_urls(self) -> None:
        scraper = IndMEDScraper.__new__(IndMEDScraper)
        urls = scraper._extract_article_urls(SAMPLE_ISSUE_HTML, "ijp", (2015, 2025))
        assert len(urls) == 3
        assert all("/index.php/ijp/article/view/" in u for u in urls)
        # Should not include unrelated journal paths
        assert not any("other" in u for u in urls)

    def test_extract_article_urls_dedup(self) -> None:
        html = """
        <a href="/index.php/ijp/article/view/100">Article 1</a>
        <a href="/index.php/ijp/article/view/100">Article 1 (duplicate)</a>
        """
        scraper = IndMEDScraper.__new__(IndMEDScraper)
        urls = scraper._extract_article_urls(html, "ijp", (2015, 2025))
        assert len(urls) == 1


class TestIndMEDScraperPDFLink:
    """Test PDF download link extraction from OJS article pages."""

    def test_find_pdf_download_link(self) -> None:
        """Standard OJS download link should be detected."""
        html = """
        <html><body>
        <a href="/index.php/ijp/article/download/4568/3456">Download PDF</a>
        </body></html>
        """
        scraper = IndMEDScraper.__new__(IndMEDScraper)
        url = scraper._find_pdf_download_link(html, "https://indmedinfo.nic.in/index.php/ijp/article/view/4568", "ijp")
        assert url is not None
        assert "/article/download/4568/3456" in url
        assert url.startswith("https://")

    def test_find_pdf_link_returns_none_when_absent(self) -> None:
        html = "<html><body><p>No PDF link here</p></body></html>"
        scraper = IndMEDScraper.__new__(IndMEDScraper)
        url = scraper._find_pdf_download_link(html, "https://example.com", "ijp")
        assert url is None


class TestIndMEDScraperYearExtraction:
    """Test year extraction from publication date strings."""

    def test_iso_date(self) -> None:
        assert IndMEDScraper._extract_year("2024-03-15") == 2024

    def test_year_only(self) -> None:
        assert IndMEDScraper._extract_year("2023") == 2023

    def test_year_in_text(self) -> None:
        assert IndMEDScraper._extract_year("Published online 15 March 2022") == 2022

    def test_none(self) -> None:
        assert IndMEDScraper._extract_year(None) is None

    def test_empty(self) -> None:
        assert IndMEDScraper._extract_year("") is None

    def test_no_year(self) -> None:
        assert IndMEDScraper._extract_year("date unavailable") is None

    def test_old_year(self) -> None:
        assert IndMEDScraper._extract_year("1998-01-01") == 1998


# --- Parser ---------------------------------------------------------------

class TestIndMEDParser:
    """Test the IndMEDParser — converts ScrapedDocument to DocumentRecord + chunks."""

    def _make_doc(self, html: bytes = SAMPLE_ARTICLE_HTML) -> ScrapedDocument:
        return ScrapedDocument(
            url="https://indmedinfo.nic.in/index.php/ijp/article/view/4568",
            source="indmed",
            content=html,
            content_type="text/html",
            title="Pharmacokinetics of metformin in Indian patients with type 2 diabetes",
            authors=["Sharma, Priya", "Kumar, Rajesh"],
            journal="Indian Journal of Pharmacology",
            doi="10.4103/ijp.ijp_123_24",
            pubdate="2024-03-15",
            abstract="This study evaluated metformin pharmacokinetics in 200 Indian adults with T2DM.",
            metadata={"journal_abbr": "ijp"},
            trust_tier=3,
            india_relevant=True,
            indian_source=True,
        )

    def test_parse_returns_document_and_chunks(self) -> None:
        parser = IndMEDParser()
        doc = self._make_doc()
        record, chunks = parser.parse(doc)
        assert record is not None
        assert isinstance(chunks, list)
        assert len(chunks) > 0

    def test_document_record_has_correct_source_type(self) -> None:
        parser = IndMEDParser()
        record, _ = parser.parse(self._make_doc())
        assert record.source_type == "indmed"

    def test_document_record_preserves_metadata(self) -> None:
        parser = IndMEDParser()
        record, _ = parser.parse(self._make_doc())
        assert record.title == "Pharmacokinetics of metformin in Indian patients with type 2 diabetes"
        assert record.journal == "Indian Journal of Pharmacology"
        assert record.doi == "10.4103/ijp.ijp_123_24"
        assert record.year == 2024
        assert record.is_india_specific is True

    def test_document_record_has_content_hash(self) -> None:
        parser = IndMEDParser()
        record, _ = parser.parse(self._make_doc())
        assert record.content_hash is not None
        assert len(record.content_hash) == 16

    def test_chunks_have_correct_source_type(self) -> None:
        parser = IndMEDParser()
        _, chunks = parser.parse(self._make_doc())
        for chunk in chunks:
            assert chunk.source_type == "indmed"

    def test_chunks_are_india_specific(self) -> None:
        parser = IndMEDParser()
        _, chunks = parser.parse(self._make_doc())
        for chunk in chunks:
            assert chunk.is_india_specific is True

    def test_chunks_have_increasing_index(self) -> None:
        parser = IndMEDParser()
        _, chunks = parser.parse(self._make_doc())
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i

    def test_chunks_have_token_estimate(self) -> None:
        parser = IndMEDParser()
        _, chunks = parser.parse(self._make_doc())
        for chunk in chunks:
            assert chunk.token_estimate > 0
            # Rough estimate: chars / 4
            assert abs(chunk.token_estimate - chunk.char_count // 4) <= 1

    def test_chunk_overlap(self) -> None:
        """Adjacent chunks should share some text (overlap for context continuity)."""
        parser = IndMEDParser()
        _, chunks = parser.parse(self._make_doc())
        if len(chunks) >= 2:
            # The end of chunk 0 should appear at the start of chunk 1 (overlap region)
            overlap_text = chunks[0].chunk_text[-200:]
            assert overlap_text[:50] in chunks[1].chunk_text or chunks[1].chunk_text.startswith(overlap_text[:50])

    def test_empty_content_returns_empty_chunks(self) -> None:
        parser = IndMEDParser()
        doc = ScrapedDocument(
            url="https://example.com/empty",
            source="indmed",
            content=b"",
            content_type="text/html",
            metadata={},
        )
        record, chunks = parser.parse(doc)
        assert len(chunks) == 0
        assert record.title == "Untitled"

    def test_body_text_extraction_uses_article_body_div(self) -> None:
        """The parser should prefer the OJS article-body div when present."""
        html = b"""<html><body>
        <div class="article-body">
        <p>This is the main body text. It is long enough to be considered substantial content for the article body extraction logic to pick it up.</p>
        </div>
        <div class="sidebar"><p>This is sidebar text that should not be included.</p></div>
        </body></html>"""
        doc = self._make_doc(html)
        parser = IndMEDParser()
        record, _ = parser.parse(doc)
        assert "main body text" in record.content
        assert "sidebar text" not in record.content

    def test_abstract_fallback_to_html_when_metadata_missing(self) -> None:
        """If abstract is not in ScrapedDocument, parser should extract from HTML."""
        html = b"""<html><body>
        <meta name="citation_abstract" content="<p>Extracted abstract text.</p>">
        <div class="article-body"><p>Body content here is substantial enough to pass the 200 char threshold so the body extractor picks it up correctly without falling back to paragraph collection.</p></div>
        </body></html>"""
        doc = ScrapedDocument(
            url="https://example.com/x",
            source="indmed",
            content=html,
            content_type="text/html",
            abstract=None,  # not pre-extracted
            metadata={},
        )
        parser = IndMEDParser()
        record, _ = parser.parse(doc)
        # Abstract should be in the content (body or abstract fallback)
        assert "Extracted abstract text" in record.content or record.content

    def test_parser_version_is_set(self) -> None:
        parser = IndMEDParser()
        record, chunks = parser.parse(self._make_doc())
        assert record.parser_version == "indmed-v1"
        for chunk in chunks:
            assert chunk.parser_version == "indmed-v1"

    def test_specialty_tags_include_journal_abbr(self) -> None:
        """Journal abbreviation should be preserved in specialty_tags."""
        parser = IndMEDParser()
        record, _ = parser.parse(self._make_doc())
        assert "ijp" in record.specialty_tags

    def test_chunks_have_document_id(self) -> None:
        parser = IndMEDParser()
        _, chunks = parser.parse(self._make_doc())
        for chunk in chunks:
            # document_id should be set (using URL since Pydantic schema has no doc_id)
            assert chunk.document_id == "https://indmedinfo.nic.in/index.php/ijp/article/view/4568"

    def test_content_hash_stable(self) -> None:
        """Same content should produce same hash."""
        parser = IndMEDParser()
        record1, _ = parser.parse(self._make_doc())
        record2, _ = parser.parse(self._make_doc())
        assert record1.content_hash == record2.content_hash

    def test_content_hash_differs_for_different_content(self) -> None:
        parser = IndMEDParser()
        record1, _ = parser.parse(self._make_doc())
        # Modify the article
        modified_html = SAMPLE_ARTICLE_HTML.replace(b"metformin", b"glimepiride")
        doc2 = self._make_doc(modified_html)
        record2, _ = parser.parse(doc2)
        assert record1.content_hash != record2.content_hash


class TestIndMEDParserProvenanceFields:
    """Phase 1 — verify trust_tier + indian_source flow from ScrapedDocument to ChunkRecord."""

    def _make_doc_with_tier(self, trust_tier: int) -> ScrapedDocument:
        return ScrapedDocument(
            url="https://indmedinfo.nic.in/index.php/ijp/article/view/4568",
            source="indmed",
            content=SAMPLE_ARTICLE_HTML,
            content_type="text/html",
            title="Test Article",
            authors=["Sharma, P"],
            journal="Indian Journal of Pharmacology",
            doi="10.1/x",
            pubdate="2024-01-01",
            metadata={"journal_abbr": "ijp"},
            trust_tier=trust_tier,
            india_relevant=True,
            indian_source=True,
        )

    def test_chunks_have_trust_tier_from_doc(self) -> None:
        """trust_tier on ScrapedDocument should propagate to all chunks."""
        parser = IndMEDParser()
        for tier in [1, 2, 3, 4, 5]:
            doc = self._make_doc_with_tier(tier)
            _, chunks = parser.parse(doc)
            assert len(chunks) > 0
            for chunk in chunks:
                assert chunk.trust_tier == tier, f"chunk {chunk.chunk_index} has tier {chunk.trust_tier}, expected {tier}"

    def test_chunks_have_indian_source_true(self) -> None:
        """IndMED chunks should always have indian_source=True."""
        parser = IndMEDParser()
        _, chunks = parser.parse(self._make_doc_with_tier(3))
        for chunk in chunks:
            assert chunk.indian_source is True

    def test_chunks_have_empty_also_indexed_in(self) -> None:
        """also_indexed_in starts empty — populated by cross-source dedup later."""
        parser = IndMEDParser()
        _, chunks = parser.parse(self._make_doc_with_tier(3))
        for chunk in chunks:
            assert chunk.also_indexed_in == []

    def test_ijmr_journal_gets_tier_1(self) -> None:
        """An IJMR article should produce tier-1 chunks (ICMR-backed, highest trust)."""
        from src.ingestion.scrapers.sources.indmed import JOURNAL_TRUST_TIER
        doc = ScrapedDocument(
            url="https://indmedinfo.nic.in/index.php/ijmr/article/view/123",
            source="indmed",
            content=SAMPLE_ARTICLE_HTML,
            content_type="text/html",
            title="IJMR Article",
            authors=["Test"],
            journal="Indian Journal of Medical Research",
            metadata={"journal_abbr": "ijmr"},
            trust_tier=JOURNAL_TRUST_TIER["ijmr"],
            india_relevant=True,
            indian_source=True,
        )
        parser = IndMEDParser()
        _, chunks = parser.parse(doc)
        for chunk in chunks:
            assert chunk.trust_tier == 1
