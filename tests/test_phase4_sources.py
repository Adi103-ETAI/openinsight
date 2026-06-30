"""Tests for Phase 4 specialty society scrapers (Layer 3 — Clinical Guidelines).

Covers:
- 7 specialty societies registered (RSSDI, CSI, ISCCM, IAP, FOGSI, AIOS, ISN)
- All have Tier 1 trust, India-specific
- PDF link extraction filters for guideline/consensus/protocol patterns
- Registry integration
"""
from __future__ import annotations

import pytest

from src.ingestion.scrapers.framework.registry import get_scraper, list_sources
from src.ingestion.scrapers.sources.specialty_societies import (
    SPECIALTY_SOCIETIES,
    SpecialtySocietyScraper,
    make_society_config,
)


SOCIETIES = ["rssdi", "csi", "isccm", "iap", "fogsi", "aios", "isn"]


class TestSpecialtySocietyConfig:
    def test_all_societies_configured(self) -> None:
        assert set(SPECIALTY_SOCIETIES.keys()) == set(SOCIETIES)

    def test_all_societies_have_guidelines_paths(self) -> None:
        for key, society in SPECIALTY_SOCIETIES.items():
            assert society.guidelines_paths, f"{key} missing guidelines_paths"
            assert len(society.guidelines_paths) >= 1

    def test_all_societies_have_full_name(self) -> None:
        for key, society in SPECIALTY_SOCIETIES.items():
            assert society.full_name, f"{key} missing full_name"

    def test_all_societies_are_tier_1(self) -> None:
        """Specialty society guidelines = Tier 1 (highest trust)."""
        for key, society in SPECIALTY_SOCIETIES.items():
            assert society.trust_tier == 1, f"{key} should be Tier 1"

    def test_all_societies_are_india_specific(self) -> None:
        for key, society in SPECIALTY_SOCIETIES.items():
            assert society.india_relevant is True
            assert society.indian_source is True

    def test_make_society_config_rate_limit(self) -> None:
        """Society sites are on commercial hosting — 1 req/sec is OK."""
        for key, society in SPECIALTY_SOCIETIES.items():
            config = make_society_config(society)
            assert config.rate_limit >= 1.0, f"{key} rate limit too low"

    def test_rssdi_config(self) -> None:
        society = SPECIALTY_SOCIETIES["rssdi"]
        assert "rssdi.in" in society.base_url
        assert "Diabetes" in society.full_name

    def test_csi_config(self) -> None:
        society = SPECIALTY_SOCIETIES["csi"]
        assert "csi-india.org" in society.base_url
        assert "Cardiological" in society.full_name

    def test_iap_config(self) -> None:
        society = SPECIALTY_SOCIETIES["iap"]
        assert "iapindia.org" in society.base_url
        assert "Pediatrics" in society.full_name


class TestSpecialtySocietyRegistration:
    """Verify all 7 societies are registered in the source registry."""

    def test_list_sources_includes_all_societies(self) -> None:
        sources = list_sources()
        for society in SOCIETIES:
            assert society in sources, f"{society} not registered"

    def test_get_scraper_returns_correct_config(self) -> None:
        for society_key in SOCIETIES:
            scraper = get_scraper(society_key)
            assert scraper.config.name == society_key
            assert scraper.config.trust_tier == 1

    def test_get_scraper_rssdi(self) -> None:
        scraper = get_scraper("rssdi")
        assert "rssdi.in" in scraper.config.base_url
        assert "Diabetes" in scraper.config.extra["full_name"]

    def test_get_scraper_csi(self) -> None:
        scraper = get_scraper("csi")
        assert "csi-india.org" in scraper.config.base_url

    def test_get_scraper_isccm(self) -> None:
        scraper = get_scraper("isccm")
        assert "isccm.org" in scraper.config.base_url

    def test_get_scraper_fogsi(self) -> None:
        scraper = get_scraper("fogsi")
        assert "fogsi.org" in scraper.config.base_url

    def test_all_society_scrapers_are_india_relevant(self) -> None:
        for society in SOCIETIES:
            scraper = get_scraper(society)
            assert scraper.config.india_relevant_default is True

    def test_all_society_scrapers_are_indian_source(self) -> None:
        for society in SOCIETIES:
            scraper = get_scraper(society)
            assert scraper.config.indian_source_default is True


class TestSpecialtySocietyPDFLinkExtraction:
    """Test PDF link extraction with guideline pattern filtering."""

    def test_extract_pdf_links_filters_by_guideline_pattern(self) -> None:
        """Only PDFs matching guideline/consensus/protocol patterns should be extracted."""
        html = """
        <a href="/docs/rssdi_clinical_guidelines_2024.pdf">RSSDI Clinical Practice Guidelines 2024</a>
        <a href="/docs/diabetes_consensus.pdf">Consensus Statement on Diabetes</a>
        <a href="/docs/annual_report.pdf">Annual Report 2024</a>  <!-- should be skipped -->
        <a href="/docs/membership_form.pdf">Membership Form</a>  <!-- should be skipped -->
        """
        scraper = SpecialtySocietyScraper.__new__(SpecialtySocietyScraper)
        # Set a minimal config for the test
        from src.ingestion.scrapers.framework.models import SourceConfig
        scraper.config = SourceConfig(name="test", base_url="https://example.com")
        links = scraper._extract_pdf_links(html, "https://example.com/guidelines")
        # Only the guideline + consensus PDFs should be included
        assert len(links) == 2
        titles = [t for _, t in links]
        assert any("Guidelines" in t for t in titles)
        assert any("Consensus" in t for t in titles)

    def test_extract_pdf_links_dedup(self) -> None:
        html = """
        <a href="/docs/guideline.pdf">Guideline</a>
        <a href="/docs/guideline.pdf">Guideline (duplicate)</a>
        """
        from src.ingestion.scrapers.framework.models import SourceConfig
        scraper = SpecialtySocietyScraper.__new__(SpecialtySocietyScraper)
        scraper.config = SourceConfig(name="test", base_url="https://example.com")
        links = scraper._extract_pdf_links(html, "https://example.com")
        assert len(links) == 1

    def test_extract_pdf_links_matches_protocol_pattern(self) -> None:
        html = '<a href="/docs/protocol.pdf">Treatment Protocol</a>'
        from src.ingestion.scrapers.framework.models import SourceConfig
        scraper = SpecialtySocietyScraper.__new__(SpecialtySocietyScraper)
        scraper.config = SourceConfig(name="test", base_url="https://example.com")
        links = scraper._extract_pdf_links(html, "https://example.com")
        assert len(links) == 1

    def test_extract_pdf_links_matches_recommendation_pattern(self) -> None:
        html = '<a href="/docs/recommendations.pdf">Clinical Recommendations</a>'
        from src.ingestion.scrapers.framework.models import SourceConfig
        scraper = SpecialtySocietyScraper.__new__(SpecialtySocietyScraper)
        scraper.config = SourceConfig(name="test", base_url="https://example.com")
        links = scraper._extract_pdf_links(html, "https://example.com")
        assert len(links) == 1

    def test_extract_pdf_links_skips_non_guideline_pdfs(self) -> None:
        html = """
        <a href="/docs/newsletter.pdf">Newsletter</a>
        <a href="/docs/brochure.pdf">Conference Brochure</a>
        <a href="/docs/form.pdf">Registration Form</a>
        """
        from src.ingestion.scrapers.framework.models import SourceConfig
        scraper = SpecialtySocietyScraper.__new__(SpecialtySocietyScraper)
        scraper.config = SourceConfig(name="test", base_url="https://example.com")
        links = scraper._extract_pdf_links(html, "https://example.com")
        assert len(links) == 0  # None match guideline patterns
