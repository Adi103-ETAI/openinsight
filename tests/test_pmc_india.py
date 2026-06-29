"""Tests for the PMC India scraper + parser (Phase 1)."""
from __future__ import annotations

import pytest

from src.ingestion.parsers.pmc_india import PMCIndiaParser
from src.ingestion.scrapers.framework.models import ScrapedDocument
from src.ingestion.scrapers.sources.pmc_india import PMCIndiaScraper, PMC_INDIA_CONFIG


# Sample PMC XML (NLM Journal Publishing DTD format)
SAMPLE_PMC_XML = b"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE article PUBLIC "-//NLM//DTD JOURNAL ARCHIVING DTD V3.0//EN" "https://dtd.nlm.nih.gov/3.0/journalpublishing.dtd">
<article article-type="research-article">
  <front>
    <journal-meta>
      <journal-id journal-id-type="iso-abbrev">Indian J Pharmacol</journal-id>
      <journal-title-group>
        <journal-title>Indian Journal of Pharmacology</journal-title>
      </journal-title-group>
    </journal-meta>
    <article-meta>
      <article-id pub-id-type="doi">10.4103/ijp.ijp_123_24</article-id>
      <article-id pub-id-type="pmid">39123456</article-id>
      <article-id pub-id-type="pmc">PMC11223344</article-id>
      <title-group>
        <article-title>Metformin pharmacokinetics in Indian patients with type 2 diabetes</article-title>
      </title-group>
      <contrib-group>
        <contrib contrib-type="author">
          <name>
            <surname>Sharma</surname>
            <given-names>Priya</given-names>
          </name>
        </contrib>
        <contrib contrib-type="author">
          <name>
            <surname>Kumar</surname>
            <given-names>Rajesh</given-names>
          </name>
        </contrib>
      </contrib-group>
      <pub-date pub-type="epub">
        <day>15</day>
        <month>3</month>
        <year>2024</year>
      </pub-date>
    </article-meta>
    <abstract>
      <p>This study evaluated metformin pharmacokinetics in 200 Indian adults with T2DM. Cmax was higher than Western cohorts.</p>
    </abstract>
  </front>
  <body>
    <sec>
      <title>Introduction</title>
      <p>Metformin is the first-line treatment for type 2 diabetes. Pharmacokinetic data in Indian populations remain limited.</p>
      <p>Most dosing recommendations are derived from Western studies with predominantly Caucasian populations.</p>
    </sec>
    <sec>
      <title>Methods</title>
      <p>We enrolled 200 Indian adults with T2DM from AIIMS Delhi. Single-dose pharmacokinetics were assessed after 500mg metformin.</p>
      <p>Blood samples were collected at 0, 0.5, 1, 2, 4, 6, 8, 12 hours. Plasma metformin was measured by HPLC.</p>
    </sec>
    <sec>
      <title>Results</title>
      <p>Cmax was 1.42 +/- 0.38 mcg/mL. Tmax was 2.1 hours. AUC0-12 was 9.8 mcg*h/mL.</p>
      <p>No serious adverse events occurred during the study period.</p>
    </sec>
    <sec>
      <title>Conclusion</title>
      <p>Indian adults show higher metformin Cmax than Western cohorts, supporting India-specific dosing guidance.</p>
    </sec>
  </body>
</article>"""


class TestPMCIndiaConfig:
    def test_base_url(self) -> None:
        assert PMC_INDIA_CONFIG.base_url == "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    def test_rate_limit_at_least_3(self) -> None:
        assert PMC_INDIA_CONFIG.rate_limit >= 3.0

    def test_india_relevant_default_true(self) -> None:
        assert PMC_INDIA_CONFIG.india_relevant_default is True

    def test_indian_source_default_false(self) -> None:
        """PMC is international — articles have Indian authors, not Indian publisher."""
        assert PMC_INDIA_CONFIG.indian_source_default is False

    def test_trust_tier_3(self) -> None:
        assert PMC_INDIA_CONFIG.trust_tier == 3

    def test_user_agent_includes_pmc(self) -> None:
        ua = PMC_INDIA_CONFIG.user_agent
        assert "PMC-India-indexer" in ua


class TestPMCIndiaParser:
    def _make_doc(self, xml: bytes = SAMPLE_PMC_XML) -> ScrapedDocument:
        return ScrapedDocument(
            url="https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&id=11223344",
            source="pmc_india",
            content=xml,
            content_type="application/xml",
            metadata={"pmc_id": "11223344", "pmc_id_full": "PMC11223344"},
            trust_tier=3,
            india_relevant=True,
            indian_source=False,
        )

    def test_parse_returns_document_and_chunks(self) -> None:
        parser = PMCIndiaParser()
        record, chunks = parser.parse(self._make_doc())
        assert record is not None
        assert isinstance(chunks, list)
        assert len(chunks) > 0

    def test_document_source_type(self) -> None:
        parser = PMCIndiaParser()
        record, _ = parser.parse(self._make_doc())
        assert record.source_type == "pmc_india"

    def test_extract_title(self) -> None:
        parser = PMCIndiaParser()
        record, _ = parser.parse(self._make_doc())
        assert record.title == "Metformin pharmacokinetics in Indian patients with type 2 diabetes"

    def test_extract_journal(self) -> None:
        parser = PMCIndiaParser()
        record, _ = parser.parse(self._make_doc())
        assert record.journal == "Indian Journal of Pharmacology"

    def test_extract_doi(self) -> None:
        parser = PMCIndiaParser()
        record, _ = parser.parse(self._make_doc())
        assert record.doi == "10.4103/ijp.ijp_123_24"

    def test_extract_year(self) -> None:
        parser = PMCIndiaParser()
        record, _ = parser.parse(self._make_doc())
        assert record.year == 2024

    def test_extract_pubdate_iso(self) -> None:
        parser = PMCIndiaParser()
        record, _ = parser.parse(self._make_doc())
        assert record.published_date == "2024-03-15"

    def test_is_india_specific(self) -> None:
        parser = PMCIndiaParser()
        record, _ = parser.parse(self._make_doc())
        assert record.is_india_specific is True

    def test_content_hash_present(self) -> None:
        parser = PMCIndiaParser()
        record, _ = parser.parse(self._make_doc())
        assert record.content_hash is not None
        assert len(record.content_hash) == 16

    def test_chunks_have_sections(self) -> None:
        """Each chunk should preserve its section title (the key PMC advantage)."""
        parser = PMCIndiaParser()
        _, chunks = parser.parse(self._make_doc())
        sections = [c.section for c in chunks]
        # Should have abstract + Introduction + Methods + Results + Conclusion
        assert "abstract" in sections
        assert "Introduction" in sections
        assert "Methods" in sections
        assert "Results" in sections
        assert "Conclusion" in sections

    def test_chunks_are_india_specific(self) -> None:
        parser = PMCIndiaParser()
        _, chunks = parser.parse(self._make_doc())
        for chunk in chunks:
            assert chunk.is_india_specific is True

    def test_chunks_have_increasing_index(self) -> None:
        parser = PMCIndiaParser()
        _, chunks = parser.parse(self._make_doc())
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i

    def test_parser_version(self) -> None:
        parser = PMCIndiaParser()
        record, chunks = parser.parse(self._make_doc())
        assert record.parser_version == "pmc_india-v1"
        for chunk in chunks:
            assert chunk.parser_version == "pmc_india-v1"

    def test_empty_content(self) -> None:
        parser = PMCIndiaParser()
        doc = ScrapedDocument(
            url="https://example.com/empty",
            source="pmc_india",
            content=b"",
            content_type="application/xml",
            metadata={},
        )
        record, chunks = parser.parse(doc)
        assert len(chunks) == 0
        assert record.title == "Untitled"

    def test_content_hash_stable(self) -> None:
        parser = PMCIndiaParser()
        r1, _ = parser.parse(self._make_doc())
        r2, _ = parser.parse(self._make_doc())
        assert r1.content_hash == r2.content_hash

    def test_content_hash_differs_for_different_content(self) -> None:
        parser = PMCIndiaParser()
        r1, _ = parser.parse(self._make_doc())
        modified = SAMPLE_PMC_XML.replace(b"metformin", b"glimepiride")
        r2, _ = parser.parse(self._make_doc(modified))
        assert r1.content_hash != r2.content_hash

    def test_long_section_split_into_subchunks(self) -> None:
        """A section longer than 2000 chars should be split into sub-chunks."""
        long_paragraph = "This is a long paragraph. " * 200  # ~4600 chars
        xml = b"""<?xml version="1.0"?>
<article>
  <front>
    <article-meta>
      <title-group><article-title>Long Section Test</article-title></title-group>
      <pub-date><year>2024</year></pub-date>
    </article-meta>
  </front>
  <body>
    <sec>
      <title>Long Methods</title>
      <p>""" + long_paragraph.encode() + b"""</p>
    </sec>
  </body>
</article>"""
        parser = PMCIndiaParser()
        _, chunks = parser.parse(self._make_doc(xml))
        # Should produce multiple chunks for the long section
        assert len(chunks) >= 2
        # Each sub-chunk should mention the section name
        for chunk in chunks:
            assert "Long Methods" in chunk.section
