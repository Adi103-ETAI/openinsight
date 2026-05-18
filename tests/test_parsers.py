"""
Tests for the parser modules.

Covers:
- BaseParser abstract interface
- GROBIDParser TEI XML parsing
- ICMRParser PDF parsing
- OCRParser scanned PDF detection
- Parser fallback behavior

Markers:
- unit: Unit tests (no external service calls)
- requires_grobid: Tests that need GROBID service
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# GROBIDParser imports from ml.ner which imports spacy
# ICMRParser may also have heavy dependencies
pytest.importorskip("bs4", reason="beautifulsoup4 required for GROBID parser")

from src.ingestion.parsers.base import BaseParser
from src.ingestion.parsers.grobid import GROBIDParser
from src.ingestion.document_db import DocumentRecord


@pytest.mark.unit
class TestBaseParser:
    """Tests for the BaseParser abstract class."""

    def test_cannot_instantiate_base_parser(self):
        """BaseParser should not be directly instantiable."""
        with pytest.raises(TypeError):
            BaseParser()

    def test_concrete_parser_must_implement_parse(self):
        """Concrete parser must implement parse method."""
        class IncompleteParser(BaseParser):
            @property
            def source_type(self) -> str:
                return "test"

        with pytest.raises(TypeError):
            IncompleteParser()

    def test_concrete_parser_must_implement_source_type(self):
        """Concrete parser must implement source_type property."""
        class IncompleteParser(BaseParser):
            def parse(self) -> list[DocumentRecord]:
                return []

        with pytest.raises(TypeError):
            IncompleteParser()

    def test_valid_concrete_parser(self):
        """Concrete parser with all methods should instantiate."""
        class ValidParser(BaseParser):
            @property
            def source_type(self) -> str:
                return "test"

            def parse(self) -> list[DocumentRecord]:
                return []

        parser = ValidParser()
        assert parser.source_type == "test"
        assert parser.parse() == []


@pytest.mark.unit
class TestGROBIDParser:
    """Tests for the GROBID parser."""

    @pytest.fixture
    def grobid_parser(self, tmp_path: Path) -> GROBIDParser:
        """Create GROBIDParser with a dummy file."""
        dummy_file = tmp_path / "test.pdf"
        dummy_file.write_bytes(b"%PDF-1.4 dummy content")
        return GROBIDParser(dummy_file, source_type="pubmed")

    def test_source_type(self, grobid_parser: GROBIDParser):
        """Source type should be set correctly."""
        assert grobid_parser.source_type == "pubmed"

    def test_file_not_found_returns_empty(self, tmp_path: Path):
        """Non-existent file should return empty list."""
        non_existent = tmp_path / "nonexistent.pdf"
        parser = GROBIDParser(non_existent, source_type="pubmed")
        result = parser.parse()
        assert result == []

    def test_grobid_url_from_settings(self, grobid_parser: GROBIDParser):
        """GROBID URL should come from settings."""
        url = grobid_parser.grobid_url
        assert isinstance(url, str)
        assert url.startswith("http")

    def test_configurable_timeout(self, grobid_parser: GROBIDParser):
        """Timeout should be configurable from settings."""
        timeout = grobid_parser.timeout
        assert isinstance(timeout, int)
        assert timeout > 0

    def test_configurable_max_retries(self, grobid_parser: GROBIDParser):
        """Max retries should be configurable from settings."""
        retries = grobid_parser.max_retries
        assert isinstance(retries, int)
        assert retries > 0

    @patch.object(GROBIDParser, "_call_grobid")
    def test_parse_with_grobid_success(
        self, mock_call_grobid: MagicMock, grobid_parser: GROBIDParser,
    ):
        """Successful GROBID call should return parsed document."""
        mock_call_grobid.return_value = """<?xml version="1.0"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0">
  <teiHeader>
    <fileDesc>
      <titleStmt><title>Test Article Title</title></titleStmt>
      <publicationStmt><date type="published" when="2024-01-01"/></publicationStmt>
    </fileDesc>
  </teiHeader>
  <text>
    <body>
      <div><head>Introduction</head><p>This is the introduction text.</p></div>
    </body>
  </text>
</TEI>"""
        result = grobid_parser.parse()

        assert len(result) == 1
        assert isinstance(result[0], DocumentRecord)
        assert "Test Article Title" in result[0].title

    @patch.object(GROBIDParser, "_call_grobid")
    def test_parse_grobid_fallback_to_icmr(
        self, mock_call_grobid: MagicMock, grobid_parser: GROBIDParser,
    ):
        """GROBID failure should fallback to ICMR parser."""
        mock_call_grobid.return_value = None

        with patch("src.ingestion.parsers.icmr.ICMRParser") as mock_icmr:
            mock_icmr.return_value.parse.return_value = [
                DocumentRecord(
                    source_type="pubmed",
                    title="Fallback Title",
                    content="Fallback content",
                )
            ]
            result = grobid_parser.parse()

            assert len(result) == 1
            assert result[0].title == "Fallback Title"

    @patch.object(GROBIDParser, "_call_grobid")
    def test_parse_empty_grobid_response(
        self, mock_call_grobid: MagicMock, grobid_parser: GROBIDParser,
    ):
        """Empty GROBID response should fallback to ICMR."""
        mock_call_grobid.return_value = ""

        with patch("src.ingestion.parsers.icmr.ICMRParser") as mock_icmr:
            mock_icmr.return_value.parse.return_value = []
            result = grobid_parser.parse()
            assert result == []

    def test_check_health(self):
        """Health check should return boolean."""
        # Without GROBID running, this should return False
        result = GROBIDParser.check_health("http://localhost:9999", timeout=1)
        assert result is False


@pytest.mark.unit
class TestGROBIDTEIParsing:
    """Tests for TEI XML parsing within GROBID parser."""

    @pytest.fixture
    def grobid_parser(self, tmp_path: Path) -> GROBIDParser:
        """Create GROBIDParser with a dummy file."""
        dummy_file = tmp_path / "test.pdf"
        dummy_file.write_bytes(b"%PDF-1.4 dummy content")
        return GROBIDParser(dummy_file, source_type="pubmed")

    def test_parse_tei_basic(self, grobid_parser: GROBIDParser):
        """Basic TEI XML should be parsed correctly."""
        tei_xml = """<?xml version="1.0"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0">
  <teiHeader>
    <fileDesc>
      <titleStmt><title>Research on Tuberculosis</title></titleStmt>
      <publicationStmt><date type="published" when="2024"/></publicationStmt>
    </fileDesc>
  </teiHeader>
  <text>
    <body>
      <div><head>Methods</head><p>Study methodology here.</p></div>
      <div><head>Results</head><p>Results found.</p></div>
    </body>
  </text>
</TEI>"""
        result = grobid_parser._parse_tei(tei_xml)

        assert "Tuberculosis" in result["title"]
        assert result["year"] == 2024
        assert len(result["sections"]) == 2

    def test_parse_tei_empty(self, grobid_parser: GROBIDParser):
        """Empty TEI XML should return empty fields."""
        tei_xml = '<?xml version="1.0"?><TEI></TEI>'
        result = grobid_parser._parse_tei(tei_xml)

        assert result["title"] == ""
        assert result["abstract"] == ""
        assert result["sections"] == []
        assert result["year"] is None

    def test_parse_tei_with_abstract(self, grobid_parser: GROBIDParser):
        """TEI with abstract should extract it."""
        tei_xml = """<?xml version="1.0"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0">
  <teiHeader>
    <fileDesc>
      <titleStmt><title>Study Title</title></titleStmt>
    </fileDesc>
  </teiHeader>
  <text>
    <front><abstract><p>This is the abstract text.</p></abstract></front>
    <body></body>
  </text>
</TEI>"""
        result = grobid_parser._parse_tei(tei_xml)
        assert "abstract" in result["abstract"].lower()
