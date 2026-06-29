"""Tests for the Medknow scraper + parser (Phase 1).

Covers:
- Journal config (MEDKNOW_JOURNALS, ISSN mapping)
- Issue URL extraction with year filtering
- Article URL extraction by ISSN
- PDF URL generation (?type=2 convention)
- Parser: DocumentRecord + ChunkRecord construction
- Parser: body text extraction (multiple Medknow template variants)
- Parser: chunking with overlap
"""
from __future__ import annotations

import pytest

from src.ingestion.parsers.medknow import MedknowParser
from src.ingestion.scrapers.framework.models import ScrapedDocument
from src.ingestion.scrapers.sources.medknow import (
    MEDKNOW_CONFIG,
    MEDKNOW_JOURNALS,
    MedknowScraper,
)


# --- Config ---------------------------------------------------------------

class TestMedknowConfig:
    def test_base_url(self) -> None:
        assert MEDKNOW_CONFIG.base_url == "https://www.medknow.com"

    def test_rate_limit_at_least_1_per_sec(self) -> None:
        """Medknow infra can handle 1 req/sec."""
        assert MEDKNOW_CONFIG.rate_limit >= 1.0

    def test_india_relevant_default(self) -> None:
        assert MEDKNOW_CONFIG.india_relevant_default is True

    def test_indian_source_default(self) -> None:
        assert MEDKNOW_CONFIG.indian_source_default is True

    def test_journal_list_has_at_least_15_journals(self) -> None:
        assert len(MEDKNOW_JOURNALS) >= 15

    def test_every_journal_has_issn(self) -> None:
        """Every Medknow journal must have an ISSN (for cross-source dedup)."""
        for name, info in MEDKNOW_JOURNALS.items():
            assert info.get("issn_print") or info.get("issn_online"), f"{name} missing ISSN"

    def test_every_journal_has_medknow_path(self) -> None:
        """Every journal needs a medknow_path for URL construction."""
        for name, info in MEDKNOW_JOURNALS.items():
            assert info.get("medknow_path"), f"{name} missing medknow_path"

    def test_user_agent_identifies_bot(self) -> None:
        ua = MEDKNOW_CONFIG.user_agent
        assert "OpenInsight-Bot" in ua
        assert "Medknow-enrichment" in ua


# --- URL extraction -------------------------------------------------------

SAMPLE_ARCHIVE_HTML = """
<html><body>
<div class="archive">
  <a href="/journals/ijp/2024/56_1.htm">Vol 56 Issue 1 (2024)</a>
  <a href="/journals/ijp/2023/55_4.htm">Vol 55 Issue 4 (2023)</a>
  <a href="/journals/ijp/2022/55_3.htm">Vol 55 Issue 3 (2022)</a>
  <a href="/journals/ijp/2010/52_1.htm">Vol 52 Issue 1 (2010)</a>
</div>
</body></html>
"""

SAMPLE_ISSUE_HTML = """
<html><body>
<div class="issue-toc">
  <a href="/article.asp?issn=1998-3751;year=2024;volume=56;issue=1;spage=1;epage=8;aulast=Sharma">Article 1</a>
  <a href="/article.asp?issn=1998-3751;year=2024;volume=56;issue=1;spage=9;epage=15;aulast=Kumar">Article 2</a>
  <a href="/article.asp?issn=1998-3751;year=2024;volume=56;issue=1;spage=16;epage=22;aulast=Menon;type=2">[PDF]</a>
  <a href="/article.asp?issn=0000-0000;year=2024;volume=99;issue=1;spage=1;epage=2;aulast=Other">Other journal</a>
</div>
</body></html>
"""


class TestMedknowScraperURLExtraction:
    def test_extract_issue_urls_with_year_filter(self) -> None:
        scraper = MedknowScraper.__new__(MedknowScraper)
        urls = scraper._extract_issue_urls(SAMPLE_ARCHIVE_HTML, "ijp", (2015, 2025))
        # 2024, 2023, 2022 are in range; 2010 should be filtered out
        assert len(urls) == 3
        assert not any("2010" in u for u in urls)

    def test_extract_issue_urls_dedup(self) -> None:
        html = """
        <a href="/journals/ijp/2024/56_1.htm">Issue 1</a>
        <a href="/journals/ijp/2024/56_1.htm">Issue 1 (duplicate)</a>
        """
        scraper = MedknowScraper.__new__(MedknowScraper)
        urls = scraper._extract_issue_urls(html, "ijp", (2015, 2025))
        assert len(urls) == 1

    def test_extract_article_urls_by_issn(self) -> None:
        """Article URLs should be filtered by the journal's ISSN."""
        scraper = MedknowScraper.__new__(MedknowScraper)
        journal_info = {"issn_online": "1998-3751", "issn_print": "0253-7613"}
        urls = scraper._extract_article_urls(SAMPLE_ISSUE_HTML, "https://medknow.com/issue", journal_info)
        # Should include the 2 article.asp URLs for this ISSN, exclude the
        # type=2 PDF link and the other-ISSN article
        assert len(urls) == 2
        for url in urls:
            # ISSN comparison is dash-insensitive
            assert "19983751" in url.replace("-", "") or "02537613" in url.replace("-", "")
            assert "type=2" not in url  # PDF links excluded


class TestMedknowPDFURL:
    """Test PDF URL generation for Medknow articles."""

    def test_pdf_url_appends_type_2(self) -> None:
        """Medknow PDF URLs append ;type=2 to the article URL."""
        article_url = "https://www.medknow.com/article.asp?issn=1998-3751;year=2024;volume=56;issue=1;spage=1;epage=8;aulast=Sharma"
        pdf_url = MedknowScraper._find_pdf_url(article_url, "<html></html>")
        assert pdf_url is not None
        assert "type=2" in pdf_url

    def test_pdf_url_skips_if_type_already_present(self) -> None:
        """If the URL already has type=, don't double-append."""
        article_url = "https://www.medknow.com/article.asp?issn=1998-3751;year=2024;volume=56;issue=1;spage=1;epage=8;aulast=Sharma;type=2"
        # When type= is already in URL, method 1 skips; method 2 looks for HTML links
        # With empty HTML, returns None
        pdf_url = MedknowScraper._find_pdf_url(article_url, "<html></html>")
        assert pdf_url is None

    def test_pdf_url_finds_explicit_pdf_link_in_html(self) -> None:
        """If a PDF link is present in the HTML, use it."""
        html = '<a href="/article.asp?issn=1998-3751;year=2024;volume=56;issue=1;spage=1;epage=8;aulast=Sharma;type=2">PDF</a>'
        article_url = "https://www.medknow.com/some/other/path"
        pdf_url = MedknowScraper._find_pdf_url(article_url, html)
        assert pdf_url is not None
        assert "type=2" in pdf_url


# --- Parser ---------------------------------------------------------------

SAMPLE_ARTICLE_HTML = b"""<!DOCTYPE html>
<html>
<head>
<title>Metformin pharmacokinetics in Indian patients</title>
<meta name="citation_title" content="Metformin pharmacokinetics in Indian patients">
<meta name="citation_author" content="Sharma, Priya">
<meta name="citation_author" content="Kumar, Rajesh">
<meta name="citation_journal_title" content="Indian Journal of Pharmacology">
<meta name="citation_doi" content="10.4103/ijp.ijp_123_24">
<meta name="citation_publication_date" content="2024-03-15">
<meta name="citation_abstract" content="<p>Study of metformin PK in 200 Indian adults.</p>">
</head>
<body>
<div id="article">
<h2>Abstract</h2>
<p>Study of metformin PK in 200 Indian adults.</p>

<h2>Introduction</h2>
<p>Metformin is the first-line treatment for type 2 diabetes in India. However, pharmacokinetic data in Indian populations remain limited, with most dosing recommendations derived from Western studies.</p>

<h2>Methods</h2>
<p>We enrolled 200 Indian adults with T2DM (age 30-65, HbA1c 7-10%) from AIIMS Delhi. Single-dose pharmacokinetics were assessed after 500mg metformin. Blood samples were collected at 0, 0.5, 1, 2, 4, 6, 8, 12 hours. Plasma metformin was measured by HPLC.</p>

<h2>Results</h2>
<p>Cmax was 1.42 +/- 0.38 mcg/mL. Tmax was 2.1 hours. AUC0-12 was 9.8 mcg*h/mL. No serious adverse events occurred during the study period.</p>

<h2>Conclusion</h2>
<p>Indian adults show higher metformin Cmax than Western cohorts, supporting India-specific dosing guidance. Further studies are warranted to confirm these findings.</p>
</div>
</body>
</html>"""


class TestMedknowParser:
    def _make_doc(self, html: bytes = SAMPLE_ARTICLE_HTML) -> ScrapedDocument:
        return ScrapedDocument(
            url="https://www.medknow.com/article.asp?issn=1998-3751;year=2024;volume=56;issue=1;spage=1;epage=8;aulast=Sharma",
            source="medknow",
            content=html,
            content_type="text/html",
            title="Metformin pharmacokinetics in Indian patients",
            authors=["Sharma, Priya", "Kumar, Rajesh"],
            journal="Indian Journal of Pharmacology",
            doi="10.4103/ijp.ijp_123_24",
            pubdate="2024-03-15",
            abstract="Study of metformin PK in 200 Indian adults.",
            metadata={"journal_name": "Indian Journal of Pharmacology", "journal_abbr": "ijp", "issn": "1998-3751"},
            trust_tier=3,
            india_relevant=True,
            indian_source=True,
        )

    def test_parse_returns_document_and_chunks(self) -> None:
        parser = MedknowParser()
        record, chunks = parser.parse(self._make_doc())
        assert record is not None
        assert isinstance(chunks, list)
        assert len(chunks) > 0

    def test_document_record_source_type(self) -> None:
        parser = MedknowParser()
        record, _ = parser.parse(self._make_doc())
        assert record.source_type == "medknow"

    def test_document_record_metadata_preserved(self) -> None:
        parser = MedknowParser()
        record, _ = parser.parse(self._make_doc())
        assert record.title == "Metformin pharmacokinetics in Indian patients"
        assert record.journal == "Indian Journal of Pharmacology"
        assert record.doi == "10.4103/ijp.ijp_123_24"
        assert record.year == 2024
        assert record.is_india_specific is True

    def test_document_record_has_content_hash(self) -> None:
        parser = MedknowParser()
        record, _ = parser.parse(self._make_doc())
        assert record.content_hash is not None
        assert len(record.content_hash) == 16

    def test_chunks_source_type(self) -> None:
        parser = MedknowParser()
        _, chunks = parser.parse(self._make_doc())
        for chunk in chunks:
            assert chunk.source_type == "medknow"

    def test_chunks_are_india_specific(self) -> None:
        parser = MedknowParser()
        _, chunks = parser.parse(self._make_doc())
        for chunk in chunks:
            assert chunk.is_india_specific is True

    def test_chunks_have_increasing_index(self) -> None:
        parser = MedknowParser()
        _, chunks = parser.parse(self._make_doc())
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i

    def test_chunks_have_token_estimate(self) -> None:
        parser = MedknowParser()
        _, chunks = parser.parse(self._make_doc())
        for chunk in chunks:
            assert chunk.token_estimate > 0

    def test_chunk_overlap(self) -> None:
        parser = MedknowParser()
        _, chunks = parser.parse(self._make_doc())
        if len(chunks) >= 2:
            overlap_text = chunks[0].chunk_text[-200:]
            assert overlap_text[:50] in chunks[1].chunk_text or chunks[1].chunk_text.startswith(overlap_text[:50])

    def test_empty_content(self) -> None:
        parser = MedknowParser()
        doc = ScrapedDocument(
            url="https://example.com/empty",
            source="medknow",
            content=b"",
            content_type="text/html",
            metadata={},
        )
        record, chunks = parser.parse(doc)
        assert len(chunks) == 0
        assert record.title == "Untitled"

    def test_body_text_extraction_uses_article_div(self) -> None:
        """Medknow body lives in <div id="article">."""
        html = b"""<html><body>
        <div id="article">
        <p>This is the main body text. It is long enough to be considered substantial content for the article body extraction logic to pick it up correctly.</p>
        </div>
        <div class="sidebar"><p>Sidebar text should not be included in body.</p></div>
        </body></html>"""
        doc = self._make_doc(html)
        parser = MedknowParser()
        record, _ = parser.parse(doc)
        assert "main body text" in record.content
        assert "Sidebar text" not in record.content

    def test_parser_version(self) -> None:
        parser = MedknowParser()
        record, chunks = parser.parse(self._make_doc())
        assert record.parser_version == "medknow-v1"
        for chunk in chunks:
            assert chunk.parser_version == "medknow-v1"

    def test_specialty_tags_include_journal_abbr(self) -> None:
        parser = MedknowParser()
        record, _ = parser.parse(self._make_doc())
        assert "ijp" in record.specialty_tags

    def test_content_hash_stable(self) -> None:
        parser = MedknowParser()
        r1, _ = parser.parse(self._make_doc())
        r2, _ = parser.parse(self._make_doc())
        assert r1.content_hash == r2.content_hash

    def test_content_hash_differs_for_different_content(self) -> None:
        parser = MedknowParser()
        r1, _ = parser.parse(self._make_doc())
        modified = SAMPLE_ARTICLE_HTML.replace(b"metformin", b"glimepiride")
        r2, _ = parser.parse(self._make_doc(modified))
        assert r1.content_hash != r2.content_hash
