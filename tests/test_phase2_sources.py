"""Tests for Phase 2 scrapers + parsers (Layer 1 — Foundational).

Covers:
- StatPearls scraper config + parser (section-aware chunking)
- NCBI Bookshelf scraper config + parser (GeneReviews + Medical Genetics)
- NMC curriculum scraper config + PDF link extraction
- Government manuals scraper config + registration of NTEP/NVBDCP/NHM/NPCDS

All tests are network-free — they use fixture HTML/XML.
"""
from __future__ import annotations

import pytest

# --- StatPearls config + scraper -----------------------------------------

from src.ingestion.scrapers.sources.statpearls import STATPEARLS_CONFIG, StatPearlsScraper


class TestStatPearlsConfig:
    def test_base_url(self) -> None:
        assert "eutils.ncbi.nlm.nih.gov" in STATPEARLS_CONFIG.base_url

    def test_trust_tier_2(self) -> None:
        """StatPearls = peer-reviewed, NCBI-hosted — Tier 2."""
        assert STATPEARLS_CONFIG.trust_tier == 2

    def test_india_relevant_default_false(self) -> None:
        """StatPearls is international reference, not India-specific."""
        assert STATPEARLS_CONFIG.india_relevant_default is False

    def test_indian_source_default_false(self) -> None:
        """NCBI = US government, not Indian source."""
        assert STATPEARLS_CONFIG.indian_source_default is False

    def test_user_agent_identifies_bot(self) -> None:
        ua = STATPEARLS_CONFIG.user_agent
        assert "StatPearls-indexer" in ua
        assert "hello@openinsight.in" in ua


# --- StatPearls parser ----------------------------------------------------

from src.ingestion.parsers.statpearls_v2 import StatPearlsParser
from src.ingestion.scrapers.framework.models import ScrapedDocument

SAMPLE_STATPEARLS_HTML = b"""<!DOCTYPE html>
<html>
<head>
<title>Type 2 Diabetes Mellitus - StatPearls</title>
<meta name="citation_title" content="Type 2 Diabetes Mellitus">
<meta name="citation_author" content="Sharma, Anita">
<meta name="citation_author" content="Patel, Kumar">
<meta name="citation_journal_title" content="StatPearls">
<meta name="citation_publication_date" content="2024-05-10">
<meta name="citation_abstract" content="<p>Type 2 diabetes mellitus is a chronic metabolic disorder characterized by hyperglycemia.</p>">
</head>
<body>
<h1>Type 2 Diabetes Mellitus</h1>
<div class="abstract">
<p>Type 2 diabetes mellitus is a chronic metabolic disorder characterized by hyperglycemia resulting from insulin resistance and relative insulin deficiency.</p>
</div>

<h2>Introduction</h2>
<p>Type 2 diabetes mellitus (T2DM) is the most common form of diabetes, accounting for over 90 percent of all diabetes cases worldwide. The disease has reached epidemic proportions globally, with particularly high prevalence in South Asian populations including India.</p>

<h2>Etiology</h2>
<p>T2DM results from a combination of insulin resistance and beta-cell dysfunction. Genetic factors, obesity, sedentary lifestyle, and aging are major risk factors. South Asians have higher insulin resistance at lower BMI thresholds compared to Western populations.</p>

<h2>Epidemiology</h2>
<p>Global prevalence of diabetes is estimated at 537 million adults. India has the second-highest number of adults with diabetes after China, with approximately 101 million cases. The prevalence is higher in urban areas (12-15%) compared to rural areas (5-7%).</p>

<h2>Treatment</h2>
<p>First-line treatment is lifestyle modification (diet, exercise, weight loss) plus metformin. Metformin is started at 500mg twice daily and titrated to a maximum of 2g daily. Second-line agents include SGLT2 inhibitors, GLP-1 receptor agonists, DPP-4 inhibitors, and sulfonylureas. Insulin therapy is initiated when HbA1c exceeds 9 percent despite dual oral therapy.</p>

<h2>Differential Diagnosis</h2>
<p>T2DM must be distinguished from type 1 diabetes, gestational diabetes, maturity-onset diabetes of the young (MODY), and secondary diabetes from pancreatic disease or drug-induced hyperglycemia.</p>

<h2>Complications</h2>
<p>Chronic complications include microvascular (retinopathy, nephropathy, neuropathy) and macrovascular (cardiovascular disease, stroke, peripheral arterial disease). Acute complications include hyperosmolar hyperglycemic state (HHS).</p>
</body>
</html>"""


class TestStatPearlsParser:
    def _make_doc(self, html: bytes = SAMPLE_STATPEARLS_HTML) -> ScrapedDocument:
        return ScrapedDocument(
            url="https://www.ncbi.nlm.nih.gov/books/NBK513215/",
            source="statpearls",
            content=html,
            content_type="text/html",
            title="Type 2 Diabetes Mellitus",
            authors=["Sharma, Anita", "Patel, Kumar"],
            journal="StatPearls",
            doi=None,
            pubdate="2024-05-10",
            # Use the full abstract from the HTML (≥80 chars to pass validation)
            abstract="Type 2 diabetes mellitus is a chronic metabolic disorder characterized by hyperglycemia resulting from insulin resistance and relative insulin deficiency.",
            metadata={"book_id": "NBK513215"},
            trust_tier=2,
            india_relevant=False,
            indian_source=False,
        )

    def test_parse_returns_document_and_chunks(self) -> None:
        parser = StatPearlsParser()
        record, chunks = parser.parse(self._make_doc())
        assert record is not None
        assert isinstance(chunks, list)
        assert len(chunks) > 0

    def test_document_source_type(self) -> None:
        parser = StatPearlsParser()
        record, _ = parser.parse(self._make_doc())
        assert record.source_type == "statpearls"

    def test_document_title_preserved(self) -> None:
        parser = StatPearlsParser()
        record, _ = parser.parse(self._make_doc())
        assert record.title == "Type 2 Diabetes Mellitus"

    def test_document_journal_is_statpearls(self) -> None:
        parser = StatPearlsParser()
        record, _ = parser.parse(self._make_doc())
        assert record.journal == "StatPearls"

    def test_document_is_india_specific_false(self) -> None:
        parser = StatPearlsParser()
        record, _ = parser.parse(self._make_doc())
        assert record.is_india_specific is False

    def test_chunks_have_sections(self) -> None:
        """Each chunk should preserve its section title (StatPearls structure)."""
        parser = StatPearlsParser()
        _, chunks = parser.parse(self._make_doc())
        sections = [c.section for c in chunks]
        assert "abstract" in sections
        assert "Introduction" in sections
        assert "Etiology" in sections
        assert "Epidemiology" in sections
        assert "Treatment" in sections
        assert "Differential Diagnosis" in sections
        assert "Complications" in sections

    def test_chunks_have_trust_tier_2(self) -> None:
        parser = StatPearlsParser()
        _, chunks = parser.parse(self._make_doc())
        for chunk in chunks:
            assert chunk.trust_tier == 2

    def test_chunks_have_indian_source_false(self) -> None:
        parser = StatPearlsParser()
        _, chunks = parser.parse(self._make_doc())
        for chunk in chunks:
            assert chunk.indian_source is False

    def test_content_hash_present(self) -> None:
        parser = StatPearlsParser()
        record, _ = parser.parse(self._make_doc())
        assert record.content_hash is not None
        assert len(record.content_hash) == 16

    def test_content_hash_stable(self) -> None:
        parser = StatPearlsParser()
        r1, _ = parser.parse(self._make_doc())
        r2, _ = parser.parse(self._make_doc())
        assert r1.content_hash == r2.content_hash


# --- NCBI Bookshelf config + scraper --------------------------------------

from src.ingestion.scrapers.sources.ncbi_bookshelf import (
    BOOKSHELF_COLLECTIONS,
    NCBI_BOOKSHELF_CONFIG,
    NCBIBookshelfScraper,
)


class TestNCBIBookshelfConfig:
    def test_base_url(self) -> None:
        assert "eutils.ncbi.nlm.nih.gov" in NCBI_BOOKSHELF_CONFIG.base_url

    def test_default_trust_tier_2(self) -> None:
        assert NCBI_BOOKSHELF_CONFIG.trust_tier == 2

    def test_genereviews_collection_exists(self) -> None:
        assert "genereviews" in BOOKSHELF_COLLECTIONS
        assert BOOKSHELF_COLLECTIONS["genereviews"]["label"] == "GeneReviews"

    def test_medical_genetics_collection_exists(self) -> None:
        assert "medical_genetics" in BOOKSHELF_COLLECTIONS

    def test_all_collections_have_required_fields(self) -> None:
        for key, collection in BOOKSHELF_COLLECTIONS.items():
            assert "term" in collection, f"{key} missing term"
            assert "label" in collection, f"{key} missing label"
            assert "trust_tier" in collection, f"{key} missing trust_tier"


# --- NCBI Bookshelf parser ------------------------------------------------

from src.ingestion.parsers.ncbi_bookshelf import NCBIBookshelfParser

SAMPLE_GENEREVIEWS_HTML = b"""<!DOCTYPE html>
<html>
<head>
<title>BRCA1- and BRCA2-Associated Hereditary Breast and Ovarian Cancer - GeneReviews</title>
<meta name="citation_title" content="BRCA1- and BRCA2-Associated Hereditary Breast and Ovarian Cancer">
<meta name="citation_author" content="Peterson, Barbara">
<meta name="citation_journal_title" content="GeneReviews">
<meta name="citation_publication_date" content="2024-03-15">
</head>
<body>
<h1>BRCA1- and BRCA2-Associated Hereditary Breast and Ovarian Cancer</h1>

<h2>Summary</h2>
<p>Hereditary breast and ovarian cancer syndrome (HBOC) is an autosomal dominant condition caused by pathogenic variants in BRCA1 or BRCA2. Affected individuals have significantly increased risks for breast, ovarian, pancreatic, and prostate cancers.</p>

<h2>Diagnosis</h2>
<p>Diagnosis is established in a proband with a pathogenic variant in BRCA1 or BRCA2 identified by molecular genetic testing. Multi-gene panel testing is preferred when HBOC is suspected.</p>

<h2>Clinical Characteristics</h2>
<p>Females with BRCA1 pathogenic variants have approximately a 72 percent lifetime risk of breast cancer and 44 percent risk of ovarian cancer. BRCA2 carriers have 69 percent breast cancer risk and 17 percent ovarian cancer risk.</p>

<h2>Management</h2>
<p>Management includes enhanced breast cancer screening starting at age 25 (annual MRI), risk-reducing mastectomy, and risk-reducing bilateral salpingo-oophorectomy (RRSO) between ages 35-40 for BRCA1 carriers and 40-45 for BRCA2 carriers.</p>
</body>
</html>"""


class TestNCBIBookshelfParser:
    def _make_doc(self, html: bytes = SAMPLE_GENEREVIEWS_HTML) -> ScrapedDocument:
        return ScrapedDocument(
            url="https://www.ncbi.nlm.nih.gov/books/NBK1247/",
            source="ncbi_bookshelf",
            content=html,
            content_type="text/html",
            title="BRCA1- and BRCA2-Associated Hereditary Breast and Ovarian Cancer",
            authors=["Peterson, Barbara"],
            journal="GeneReviews",
            pubdate="2024-03-15",
            metadata={
                "book_id": "NBK1247",
                "collection": "genereviews",
                "collection_label": "GeneReviews",
                "trust_tier": 2,
            },
            trust_tier=2,
            india_relevant=False,
            indian_source=False,
        )

    def test_parse_returns_document_and_chunks(self) -> None:
        parser = NCBIBookshelfParser()
        record, chunks = parser.parse(self._make_doc())
        assert record is not None
        assert isinstance(chunks, list)
        assert len(chunks) > 0

    def test_document_source_type(self) -> None:
        parser = NCBIBookshelfParser()
        record, _ = parser.parse(self._make_doc())
        assert record.source_type == "ncbi_bookshelf"

    def test_document_journal_is_collection_label(self) -> None:
        """Journal field should be set to the collection label (GeneReviews)."""
        parser = NCBIBookshelfParser()
        record, _ = parser.parse(self._make_doc())
        assert record.journal == "GeneReviews"

    def test_chunks_have_sections(self) -> None:
        parser = NCBIBookshelfParser()
        _, chunks = parser.parse(self._make_doc())
        sections = [c.section for c in chunks]
        assert "Summary" in sections
        assert "Diagnosis" in sections
        assert "Management" in sections

    def test_chunks_have_trust_tier_from_doc(self) -> None:
        parser = NCBIBookshelfParser()
        _, chunks = parser.parse(self._make_doc())
        for chunk in chunks:
            assert chunk.trust_tier == 2

    def test_specialty_tags_include_collection(self) -> None:
        parser = NCBIBookshelfParser()
        record, _ = parser.parse(self._make_doc())
        assert "genereviews" in record.specialty_tags


# --- NMC curriculum scraper ------------------------------------------------

from src.ingestion.scrapers.sources.nmc_curriculum import NMC_CONFIG, NMCCurriculumScraper


class TestNMCConfig:
    def test_base_url(self) -> None:
        assert NMC_CONFIG.base_url == "https://www.nmc.org.in"

    def test_trust_tier_1(self) -> None:
        """NMC = apex medical education body — Tier 1."""
        assert NMC_CONFIG.trust_tier == 1

    def test_india_relevant_default_true(self) -> None:
        assert NMC_CONFIG.india_relevant_default is True

    def test_indian_source_default_true(self) -> None:
        assert NMC_CONFIG.indian_source_default is True

    def test_rate_limit_is_polite(self) -> None:
        """Indian gov servers are slow — ≤0.5 req/sec."""
        assert NMC_CONFIG.rate_limit <= 0.5

    def test_pdf_link_patterns_include_curriculum(self) -> None:
        patterns = NMC_CONFIG.extra["pdf_link_patterns"]
        assert "curriculum" in patterns
        assert "competency" in patterns


class TestNMCCurriculumScraperLinkExtraction:
    def test_extract_pdf_links_filters_by_pattern(self) -> None:
        """Only PDFs matching curriculum/competency patterns should be extracted."""
        html = """
        <a href="/docs/curriculum_pg.pdf">PG Curriculum 2024</a>
        <a href="/docs/competency_ug.pdf">UG Competency Framework</a>
        <a href="/docs/annual_report.pdf">Annual Report 2024</a>  <!-- should be skipped -->
        <a href="/docs/meeting_minutes.docx">Minutes</a>  <!-- not a PDF, skipped -->
        """
        scraper = NMCCurriculumScraper.__new__(NMCCurriculumScraper)
        links = scraper._extract_pdf_links(html, "https://www.nmc.org.in/resources")
        # Should only include the 2 curriculum/competency PDFs
        assert len(links) == 2
        titles = [title for _, title in links]
        assert any("Curriculum" in t for t in titles)
        assert any("Competency" in t for t in titles)

    def test_extract_pdf_links_dedup(self) -> None:
        html = """
        <a href="/docs/curriculum.pdf">Curriculum</a>
        <a href="/docs/curriculum.pdf">Curriculum (duplicate)</a>
        """
        scraper = NMCCurriculumScraper.__new__(NMCCurriculumScraper)
        links = scraper._extract_pdf_links(html, "https://www.nmc.org.in")
        assert len(links) == 1

    def test_extract_pdf_links_absolute_urls(self) -> None:
        html = '<a href="/docs/curriculum.pdf">Curriculum</a>'
        scraper = NMCCurriculumScraper.__new__(NMCCurriculumScraper)
        links = scraper._extract_pdf_links(html, "https://www.nmc.org.in/resources")
        assert len(links) == 1
        url, _ = links[0]
        assert url.startswith("https://www.nmc.org.in/")


# --- Government manuals scraper registration ------------------------------

from src.ingestion.scrapers.framework.registry import get_scraper, list_sources


class TestGovtManualsRegistration:
    """Verify all 4 government programmes are registered."""

    def test_list_sources_includes_all_govt_programmes(self) -> None:
        sources = list_sources()
        assert "ntep" in sources
        assert "nvbdcp" in sources
        assert "nhm" in sources
        assert "npcds" in sources

    def test_get_scraper_ntep(self) -> None:
        scraper = get_scraper("ntep")
        assert scraper.config.name == "ntep"
        assert "tbcindia.gov.in" in scraper.config.base_url

    def test_get_scraper_nvbdcp(self) -> None:
        scraper = get_scraper("nvbdcp")
        assert scraper.config.name == "nvbdcp"
        assert "nvbdcp.gov.in" in scraper.config.base_url

    def test_govt_scrapers_have_tier_1(self) -> None:
        """Govt programmes = Tier 1 (highest trust)."""
        for prog in ["ntep", "nvbdcp", "nhm", "npcds"]:
            scraper = get_scraper(prog)
            assert scraper.config.trust_tier == 1, f"{prog} should be Tier 1"

    def test_govt_scrapers_are_india_relevant(self) -> None:
        for prog in ["ntep", "nvbdcp", "nhm", "npcds"]:
            scraper = get_scraper(prog)
            assert scraper.config.india_relevant_default is True

    def test_govt_scrapers_are_indian_source(self) -> None:
        for prog in ["ntep", "nvbdcp", "nhm", "npcds"]:
            scraper = get_scraper(prog)
            assert scraper.config.indian_source_default is True

    def test_govt_scrapers_polite_rate_limit(self) -> None:
        """Indian gov servers are slow — ≤0.5 req/sec."""
        for prog in ["ntep", "nvbdcp", "nhm", "npcds"]:
            scraper = get_scraper(prog)
            assert scraper.config.rate_limit <= 0.5


# --- Registry integration -------------------------------------------------

class TestRegistryIncludesPhase2Sources:
    """Verify the registry auto-imports Phase 2 sources."""

    def test_list_sources_includes_phase2_sources(self) -> None:
        sources = list_sources()
        assert "statpearls" in sources
        assert "ncbi_bookshelf" in sources
        assert "nmc_curriculum" in sources

    def test_get_scraper_statpearls(self) -> None:
        scraper = get_scraper("statpearls")
        assert scraper.config.name == "statpearls"

    def test_get_scraper_ncbi_bookshelf(self) -> None:
        scraper = get_scraper("ncbi_bookshelf")
        assert scraper.config.name == "ncbi_bookshelf"

    def test_get_scraper_nmc_curriculum(self) -> None:
        scraper = get_scraper("nmc_curriculum")
        assert scraper.config.name == "nmc_curriculum"
