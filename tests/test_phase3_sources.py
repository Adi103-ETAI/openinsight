"""Tests for Phase 3 scrapers + parsers (Layer 4 — Drug & Regulatory).

Covers:
- CDSCO scraper (approved drugs database, search by drug name)
- CDSCO parser (structured drug record → one chunk per drug)
- CTRI scraper (clinical trials registry, trial ID extraction)
- CTRI parser (structured trial record → one chunk per trial)
- PvPI scraper (drug safety alerts, month/year extraction from link text)
- NFI parser (drug monograph, HTML + plain text extraction)
- Registry integration (all Phase 3 sources registered)
"""
from __future__ import annotations

import pytest

from src.ingestion.scrapers.framework.models import ScrapedDocument


# --- CDSCO ----------------------------------------------------------------

from src.ingestion.scrapers.sources.cdsco import CDSCO_CONFIG, CDSCOScraper
from src.ingestion.parsers.cdsco import CDSCOParser


class TestCDSCOConfig:
    def test_base_url(self) -> None:
        assert "cdscoonline.gov.in" in CDSCO_CONFIG.base_url

    def test_trust_tier_1(self) -> None:
        """CDSCO = India's FDA equivalent — Tier 1."""
        assert CDSCO_CONFIG.trust_tier == 1

    def test_india_relevant_default_true(self) -> None:
        assert CDSCO_CONFIG.india_relevant_default is True

    def test_indian_source_default_true(self) -> None:
        assert CDSCO_CONFIG.indian_source_default is True

    def test_rate_limit_is_polite(self) -> None:
        """Indian gov servers — ≤0.5 req/sec."""
        assert CDSCO_CONFIG.rate_limit <= 0.5


class TestCDSCOScraperLinkExtraction:
    def test_extract_drug_record_urls(self) -> None:
        html = """
        <table>
          <tr><td><a href="/drugDetail?id=123">Metformin 500mg</a></td></tr>
          <tr><td><a href="/drugDetail?id=456">Glimepiride 2mg</a></td></tr>
          <tr><td><a href="/other/page">Not a drug link</a></td></tr>
        </table>
        """
        scraper = CDSCOScraper.__new__(CDSCOScraper)
        records = scraper._extract_drug_record_urls(html)
        assert len(records) == 2
        assert records[0][1] == "Metformin 500mg"
        assert records[1][1] == "Glimepiride 2mg"

    def test_extract_drug_record_urls_dedup(self) -> None:
        html = """
        <a href="/drugDetail?id=123">Drug A</a>
        <a href="/drugDetail?id=123">Drug A (duplicate)</a>
        """
        scraper = CDSCOScraper.__new__(CDSCOScraper)
        records = scraper._extract_drug_record_urls(html)
        assert len(records) == 1


class TestCDSCOScraperFieldExtraction:
    def test_extract_drug_fields_from_table(self) -> None:
        """Extract structured fields from a CDSCO-style HTML table."""
        html = """
        <table>
          <tr><th>Drug Name</th><td>Metformin Hydrochloride</td></tr>
          <tr><th>Manufacturer</th><td>Sun Pharmaceutical Industries</td></tr>
          <tr><th>Approval Date</th><td>15-Jan-2020</td></tr>
          <tr><th>Indication</th><td>Type 2 Diabetes Mellitus</td></tr>
          <tr><th>Strength</th><td>500mg tablet</td></tr>
          <tr><th>Schedule</th><td>Schedule H</td></tr>
        </table>
        """
        fields = CDSCOScraper._extract_drug_fields(html)
        assert fields.get("drug_name") == "Metformin Hydrochloride"
        assert fields.get("manufacturer") == "Sun Pharmaceutical Industries"
        assert fields.get("approval_date") == "15-Jan-2020"
        assert "Type 2 Diabetes" in fields.get("indication", "")
        assert "500mg" in fields.get("strength", "")
        assert "Schedule H" in fields.get("schedule", "")


class TestCDSCOParser:
    def _make_doc(self, structured: dict | None = None) -> ScrapedDocument:
        return ScrapedDocument(
            url="https://cdscoonline.gov.in/drugDetail?id=123",
            source="cdsco",
            content=b"<html><body>Drug detail page</body></html>",
            content_type="text/html",
            title="Metformin 500mg",
            metadata={
                "drug_name": "Metformin",
                "search_type": "approved",
                "structured": structured or {
                    "drug_name": "Metformin Hydrochloride",
                    "manufacturer": "Sun Pharma",
                    "approval_date": "15-Jan-2020",
                    "indication": "Type 2 Diabetes Mellitus",
                    "strength": "500mg tablet",
                    "schedule": "Schedule H",
                },
            },
            trust_tier=1,
            india_relevant=True,
            indian_source=True,
        )

    def test_parse_returns_one_document_one_chunk(self) -> None:
        """Each CDSCO drug record → exactly one document with one chunk."""
        parser = CDSCOParser()
        record, chunks = parser.parse(self._make_doc())
        assert len(chunks) == 1
        assert record.source_type == "cdsco"

    def test_chunk_has_drug_record_section(self) -> None:
        """Chunk section should be 'drug_record' for the lookup fast path."""
        parser = CDSCOParser()
        _, chunks = parser.parse(self._make_doc())
        assert chunks[0].section == "drug_record"

    def test_chunk_has_high_content_weight(self) -> None:
        """Drug records should be boosted in retrieval (content_weight > 1)."""
        parser = CDSCOParser()
        _, chunks = parser.parse(self._make_doc())
        assert chunks[0].content_weight > 1.0

    def test_chunk_has_trust_tier_1(self) -> None:
        parser = CDSCOParser()
        _, chunks = parser.parse(self._make_doc())
        assert chunks[0].trust_tier == 1

    def test_chunk_has_evidence_level_1(self) -> None:
        """Regulatory records = highest evidence level."""
        parser = CDSCOParser()
        _, chunks = parser.parse(self._make_doc())
        assert chunks[0].evidence_level == 1

    def test_chunk_text_contains_all_fields(self) -> None:
        parser = CDSCOParser()
        _, chunks = parser.parse(self._make_doc())
        text = chunks[0].chunk_text
        assert "Metformin" in text
        assert "Sun Pharma" in text
        assert "Type 2 Diabetes" in text
        assert "500mg" in text
        assert "Schedule H" in text
        assert "CDSCO" in text

    def test_chunk_drugs_field_populated(self) -> None:
        """The drugs list should contain the drug name."""
        parser = CDSCOParser()
        _, chunks = parser.parse(self._make_doc())
        assert "Metformin Hydrochloride" in chunks[0].drugs

    def test_content_hash_stable(self) -> None:
        parser = CDSCOParser()
        r1, _ = parser.parse(self._make_doc())
        r2, _ = parser.parse(self._make_doc())
        assert r1.content_hash == r2.content_hash


# --- CTRI -----------------------------------------------------------------

from src.ingestion.scrapers.sources.ctri import CTRI_CONFIG, CTRIScraper
from src.ingestion.parsers.ctri import CTRIParser


class TestCTRIConfig:
    def test_base_url(self) -> None:
        assert "ctri.nic.in" in CTRI_CONFIG.base_url

    def test_trust_tier_2(self) -> None:
        """CTRI = registry but trials may not have published results — Tier 2."""
        assert CTRI_CONFIG.trust_tier == 2

    def test_india_relevant_default_true(self) -> None:
        assert CTRI_CONFIG.india_relevant_default is True

    def test_indian_source_default_true(self) -> None:
        assert CTRI_CONFIG.indian_source_default is True


class TestCTRIScraper:
    def test_extract_trial_urls(self) -> None:
        html = """
        <table>
          <tr><td><a href="/trialview.aspx?trialid=12345">CTRI/2024/01/012345 - Diabetes Study</a></td></tr>
          <tr><td><a href="/trialview.aspx?trialid=67890">CTRI/2023/05/067890 - Cancer Trial</a></td></tr>
          <tr><td><a href="/other/page">Not a trial link</a></td></tr>
        </table>
        """
        scraper = CTRIScraper.__new__(CTRIScraper)
        results = scraper._extract_trial_urls(html)
        assert len(results) == 2
        assert results[0][1] == "CTRI/2024/01/012345"  # trial_id
        assert "Diabetes Study" in results[0][2]  # title

    def test_extract_trial_id_from_text(self) -> None:
        """CTRI IDs have format CTRI/YYYY/MM/NNNNNN."""
        assert CTRIScraper._extract_trial_id("CTRI/2024/01/012345 - Diabetes Study") == "CTRI/2024/01/012345"
        assert CTRIScraper._extract_trial_id("some text CTRI/2023/05/067890 here") == "CTRI/2023/05/067890"
        assert CTRIScraper._extract_trial_id("no id here") == "unknown"


class TestCTRIParser:
    def _make_doc(self) -> ScrapedDocument:
        return ScrapedDocument(
            url="http://ctri.nic.in/trialview.aspx?trialid=12345",
            source="ctri",
            content=b"<html><body>Trial detail page</body></html>",
            content_type="text/html",
            title="CTRI/2024/01/012345 - Diabetes Study",
            metadata={
                "trial_id": "CTRI/2024/01/012345",
                "title": "Phase 3 Trial of Metformin in Indian T2DM Patients",
                "structured": {
                    "trial_id": "CTRI/2024/01/012345",
                    "title": "Phase 3 Trial of Metformin in Indian T2DM Patients",
                    "sponsor": "AIIMS Delhi",
                    "phase": "Phase 3",
                    "status": "Recruiting",
                    "condition": "Type 2 Diabetes Mellitus",
                    "intervention": "Metformin 500mg",
                    "enrollment": "200",
                    "registration_date": "2024-01-15",
                },
            },
            trust_tier=2,
            india_relevant=True,
            indian_source=True,
        )

    def test_parse_returns_one_chunk(self) -> None:
        parser = CTRIParser()
        record, chunks = parser.parse(self._make_doc())
        assert len(chunks) == 1
        assert record.source_type == "ctri"

    def test_chunk_has_trial_record_section(self) -> None:
        parser = CTRIParser()
        _, chunks = parser.parse(self._make_doc())
        assert chunks[0].section == "trial_record"

    def test_chunk_has_trust_tier_2(self) -> None:
        parser = CTRIParser()
        _, chunks = parser.parse(self._make_doc())
        assert chunks[0].trust_tier == 2

    def test_chunk_text_contains_trial_fields(self) -> None:
        parser = CTRIParser()
        _, chunks = parser.parse(self._make_doc())
        text = chunks[0].chunk_text
        assert "CTRI/2024/01/012345" in text
        assert "AIIMS Delhi" in text
        assert "Phase 3" in text
        assert "Type 2 Diabetes" in text

    def test_chunk_diseases_populated_from_condition(self) -> None:
        parser = CTRIParser()
        _, chunks = parser.parse(self._make_doc())
        assert "Type 2 Diabetes Mellitus" in chunks[0].diseases

    def test_chunk_drugs_populated_from_intervention(self) -> None:
        parser = CTRIParser()
        _, chunks = parser.parse(self._make_doc())
        assert "Metformin 500mg" in chunks[0].drugs


# --- PvPI -----------------------------------------------------------------

from src.ingestion.scrapers.sources.pvpi import PVPI_CONFIG, PVPIScraper


class TestPVPIConfig:
    def test_base_url(self) -> None:
        assert "ipc.gov.in" in PVPI_CONFIG.base_url

    def test_trust_tier_1(self) -> None:
        """PvPI = official pharmacovigilance authority — Tier 1."""
        assert PVPI_CONFIG.trust_tier == 1

    def test_india_relevant_default_true(self) -> None:
        assert PVPI_CONFIG.india_relevant_default is True

    def test_indian_source_default_true(self) -> None:
        assert PVPI_CONFIG.indian_source_default is True

    def test_pdf_link_patterns_include_alert(self) -> None:
        patterns = PVPI_CONFIG.extra["pdf_link_patterns"]
        assert "alert" in patterns
        assert "signal" in patterns


class TestPVPIScraperLinkExtraction:
    def test_extract_pdf_links_filters_by_pattern(self) -> None:
        html = """
        <a href="/PvPI/alerts/jan2024.pdf">January 2024 Drug Safety Alert</a>
        <a href="/PvPI/alerts/feb2024.pdf">February 2024 Signal Alert</a>
        <a href="/PvPI/reports/annual.pdf">Annual Report 2024</a>  <!-- should be skipped -->
        """
        scraper = PVPIScraper.__new__(PVPIScraper)
        links = scraper._extract_pdf_links(html, "https://www.ipc.gov.in/PvPI/alerts.html")
        # Only the alert/signal PDFs should be included
        assert len(links) == 2
        titles = [t for _, t, _ in links]
        assert any("January 2024" in t for t in titles)
        assert any("February 2024" in t for t in titles)

    def test_extract_month_year_from_link_text(self) -> None:
        """Month/year should be extracted from link text."""
        assert PVPIScraper._extract_month_year("January 2024 Drug Safety Alert") == "January 2024"
        assert PVPIScraper._extract_month_year("Q1 2024 Newsletter") == "Q1 2024"
        assert PVPIScraper._extract_month_year("Alert 2024") == "2024"
        assert PVPIScraper._extract_month_year("No date here") == ""


# --- NFI parser (stub — tested with fixture data) -------------------------

from src.ingestion.parsers.nfi import NFIParser


SAMPLE_NFI_HTML = b"""<html>
<head><title>Metformin - National Formulary of India</title></head>
<body>
<h1>Metformin</h1>
<table>
  <tr><th>Indications</th><td>Type 2 diabetes mellitus, especially in overweight patients</td></tr>
  <tr><th>Dosage</th><td>Adult: 500mg twice daily, titrated to 2g daily. Pediatric (10+): 500mg twice daily.</td></tr>
  <tr><th>Contraindications</th><td>Severe renal impairment (eGFR < 30), diabetic ketoacidosis, hypersensitivity</td></tr>
  <tr><th>Adverse Effects</th><td>GI upset, lactic acidosis (rare), vitamin B12 deficiency</td></tr>
  <tr><th>Pregnancy Category</th><td>Category B</td></tr>
  <tr><th>Schedule</th><td>Schedule H</td></tr>
  <tr><th>Brand Names</th><td>Glycomet, Glucophage, Obimet</td></tr>
</table>
</body></html>"""


SAMPLE_NFI_TEXT = b"""Metformin

Indications: Type 2 diabetes mellitus, especially in overweight patients.

Dosage: Adult: 500mg twice daily, titrated to 2g daily. Pediatric (10+): 500mg twice daily.

Contraindications: Severe renal impairment (eGFR < 30), diabetic ketoacidosis, hypersensitivity.

Adverse Effects: GI upset, lactic acidosis (rare), vitamin B12 deficiency.

Pregnancy Category: Category B

Schedule: Schedule H

Brand Names: Glycomet, Glucophage, Obimet
"""


class TestNFIParser:
    def _make_html_doc(self) -> ScrapedDocument:
        return ScrapedDocument(
            url="https://nhp.gov.in/drugs/metformin",
            source="nfi",
            content=SAMPLE_NFI_HTML,
            content_type="text/html",
            title="Metformin",
            metadata={},
            trust_tier=1,
            india_relevant=True,
            indian_source=True,
        )

    def _make_text_doc(self) -> ScrapedDocument:
        return ScrapedDocument(
            url="https://example.com/nfi/metformin",
            source="nfi",
            content=SAMPLE_NFI_TEXT,
            content_type="text/plain",
            title="Metformin",
            metadata={},
            trust_tier=1,
            india_relevant=True,
            indian_source=True,
        )

    def test_parse_html_returns_one_chunk(self) -> None:
        parser = NFIParser()
        record, chunks = parser.parse(self._make_html_doc())
        assert len(chunks) == 1
        assert record.source_type == "nfi"

    def test_parse_text_returns_one_chunk(self) -> None:
        parser = NFIParser()
        record, chunks = parser.parse(self._make_text_doc())
        assert len(chunks) == 1

    def test_chunk_has_drug_monograph_section(self) -> None:
        """Chunk section should be 'drug_monograph' for the lookup fast path."""
        parser = NFIParser()
        _, chunks = parser.parse(self._make_html_doc())
        assert chunks[0].section == "drug_monograph"

    def test_chunk_has_trust_tier_1(self) -> None:
        parser = NFIParser()
        _, chunks = parser.parse(self._make_html_doc())
        assert chunks[0].trust_tier == 1

    def test_chunk_has_evidence_level_1(self) -> None:
        """Regulatory formulary = highest evidence level."""
        parser = NFIParser()
        _, chunks = parser.parse(self._make_html_doc())
        assert chunks[0].evidence_level == 1

    def test_chunk_has_high_content_weight(self) -> None:
        """Drug monographs should be boosted in retrieval."""
        parser = NFIParser()
        _, chunks = parser.parse(self._make_html_doc())
        assert chunks[0].content_weight > 1.0

    def test_html_extraction_captures_dosage(self) -> None:
        parser = NFIParser()
        record, _ = parser.parse(self._make_html_doc())
        assert "500mg" in record.content
        assert "2g daily" in record.content

    def test_html_extraction_captures_contraindications(self) -> None:
        parser = NFIParser()
        record, _ = parser.parse(self._make_html_doc())
        assert "renal impairment" in record.content.lower()

    def test_html_extraction_captures_pregnancy_category(self) -> None:
        parser = NFIParser()
        record, _ = parser.parse(self._make_html_doc())
        assert "Category B" in record.content

    def test_html_extraction_captures_brand_names(self) -> None:
        parser = NFIParser()
        record, _ = parser.parse(self._make_html_doc())
        assert "Glycomet" in record.content

    def test_text_extraction_captures_sections(self) -> None:
        """Plain text extraction should split on section headers."""
        parser = NFIParser()
        record, _ = parser.parse(self._make_text_doc())
        assert "500mg" in record.content
        assert "Category B" in record.content
        assert "Schedule H" in record.content

    def test_chunk_drugs_field_populated(self) -> None:
        parser = NFIParser()
        _, chunks = parser.parse(self._make_html_doc())
        assert "Metformin" in chunks[0].drugs

    def test_chunk_has_safety_flag_when_contraindications_present(self) -> None:
        """If contraindications are present, has_safety_flag should be True."""
        parser = NFIParser()
        _, chunks = parser.parse(self._make_html_doc())
        assert chunks[0].has_safety_flag is True

    def test_content_hash_stable(self) -> None:
        parser = NFIParser()
        r1, _ = parser.parse(self._make_html_doc())
        r2, _ = parser.parse(self._make_html_doc())
        assert r1.content_hash == r2.content_hash


# --- Registry integration -------------------------------------------------

from src.ingestion.scrapers.framework.registry import get_scraper, list_sources


class TestRegistryIncludesPhase3Sources:
    def test_list_sources_includes_phase3_sources(self) -> None:
        sources = list_sources()
        assert "cdsco" in sources
        assert "ctri" in sources
        assert "pvpi" in sources

    def test_get_scraper_cdsco(self) -> None:
        scraper = get_scraper("cdsco")
        assert scraper.config.name == "cdsco"
        assert scraper.config.trust_tier == 1

    def test_get_scraper_ctri(self) -> None:
        scraper = get_scraper("ctri")
        assert scraper.config.name == "ctri"
        assert scraper.config.trust_tier == 2

    def test_get_scraper_pvpi(self) -> None:
        scraper = get_scraper("pvpi")
        assert scraper.config.name == "pvpi"
        assert scraper.config.trust_tier == 1
