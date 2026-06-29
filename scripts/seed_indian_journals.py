"""Seed Indian medical journals — orchestration script for Phase 1 Layer 2.

Runs discovery + ingestion across all 3 Phase 1 sources:
1. PubMed — Indian journals via journal[TA] filter (~30K articles target)
2. IndMED — 30 Indian journals hosted on indmedinfo.nic.in (~5-10K articles)
3. Medknow — 20 Indian journals on medknow.com (full-text enrichment)
4. PMC India — PubMed Central full-text with Indian affiliations (~20K articles)

Usage:
    # Discover only (no ingest) — see what would be indexed
    python scripts/seed_indian_journals.py --discover-only --source indmed

    # Discover + ingest from all sources
    python scripts/seed_indian_journals.py --source all

    # Ingest from one source with a custom journal list
    python scripts/seed_indian_journals.py --source pubmed \\
        --journals 'Indian J Med Res,Natl Med J India'

    # Ingest from PMC India for a specific specialty
    python scripts/seed_indian_journals.py --source pmc_india \\
        --specialty endocrinology --max-results 500

This script is the Phase 1 entrypoint for populating the vector store with
Indian journal content. Run on Kaggle for GPU-accelerated embedding; runs
on local CPU for small test sets.

Prerequisites:
- MongoDB running (config MONGODB_URL)
- Milvus/Zilliz reachable (config VECTOR_URI / VECTOR_TOKEN)
- NCBI_API_KEY env var set (for PubMed + PMC, raises rate to 10 req/sec)
- GROBID running on localhost:8070 (for any PDF parsing)
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import sys
from typing import Any

from loguru import logger

# Configure logging
logger.remove()
logger.add(sys.stderr, level="INFO", format="<level>{level: <8}</level> | {message}")


# All Indian journal PubMed abbreviations for Phase 1
INDIAN_JOURNALS_PUBMED = [
    "Indian J Med Res",                # IJMR
    "Natl Med J India",                # NMJI
    "J Assoc Physicians India",        # JAPI
    "J Indian Med Assoc",              # JIMA
    "Indian Heart J",                  # IHJ
    "Indian J Anaesth",                # IJA
    "Indian Pediatr",                  # IP
    "Indian J Community Med",          # IJCM
    "Indian J Dermatol",               # IJD
    "Indian J Psychiatry",             # IJP
    "Indian J Pharmacol",              # IJPh
    "Indian J Public Health",          # IJPH
    "Indian J Ophthalmol",             # IJO
    "Indian J Orthop",                 # IJOr
    "Indian J Surg",                   # IJS
    "Indian J Radiol Imaging",         # IJRI
    "Indian J Nephrol",                # IJN
    "Indian J Endocr Metab",           # IJEM
    "Indian J Gastroenterol",          # IJG
    "Indian J Med Microbiol",          # IJMM
    "Indian J Med Paediatr Oncol",     # IJMPO
    "Indian J Palliat Care",           # IJPC
    "Indian J Pathol Microbiol",       # IJPM
    "Indian J Rheumatol",              # IJR
    "Indian J Urol",                   # IJU
    "Indian J Med Sci",                # IJMS
    "J Lab Physicians",                # JLP
    "J Postgrad Med",                  # JPM
    "Med J Armed Forces India",        # MJAFI
    "Ann Indian Acad Neurol",          # AIAN
]


async def discover_pubmed(
    journals: list[str] | None,
    max_per_journal: int,
    discover_only: bool,
    limit: int | None = None,
) -> int:
    """Discover + (optionally) ingest PubMed articles from Indian journals."""
    from src.ingestion.scrapers import get_scraper
    from src.ingestion.parsers.pubmed import PubMedParser

    scraper = get_scraper("pubmed")
    target_journals = journals or INDIAN_JOURNALS_PUBMED
    total_ingested = 0
    pipeline = None

    if not discover_only:
        from src.ingestion.pipeline import IngestionPipeline
        pipeline = IngestionPipeline()

    try:
        for journal in target_journals:
            if limit and total_ingested >= limit:
                break
            logger.info(f"[pubmed] discovering articles for: {journal}")
            try:
                jobs = await scraper.discover_by_journal(
                    journal_abbrev=journal,
                    max_results=max_per_journal,
                    date_range="2015:2025[DP]",
                )
                if limit:
                    remaining = limit - total_ingested
                    jobs = jobs[:remaining]
                logger.info(f"[pubmed] {journal}: {len(jobs)} articles discovered")

                if not discover_only and jobs:
                    # Fetch + parse each article, then batch-ingest
                    parsed_docs = []
                    for job in jobs:
                        scraped = await scraper.fetch_one(job)
                        if scraped:
                            # PubMedParser currently expects a file path or XML;
                            # we pass the scraped XML content directly
                            parser = PubMedParser.__new__(PubMedParser)
                            # Use the parser's _parse_xml method if available,
                            # otherwise wrap content in a temp structure
                            try:
                                records = parser.parse() if hasattr(parser, 'parse') and False else []
                            except Exception:
                                records = []
                            # For now, create a minimal DocumentRecord from the scraped XML
                            if not records and scraped.content:
                                from src.ingestion.document_db import DocumentRecord
                                record = DocumentRecord(
                                    source_type="pubmed",
                                    title=scraped.title or "Untitled",
                                    content=scraped.content.decode('utf-8', errors='replace')[:50000],
                                    url=scraped.url,
                                    doi=scraped.doi,
                                    published_date=scraped.pubdate,
                                    journal=scraped.journal,
                                    is_india_specific=True,
                                    parser_version="pubmed-scraped-v1",
                                    content_hash=hashlib.sha256(scraped.content).hexdigest()[:16] if scraped.content else "",
                                )
                                parsed_docs.append((record, []))
                    if parsed_docs:
                        result = await pipeline.ingest_scraped_documents(
                            documents=parsed_docs,
                            source="pubmed",
                        )
                        total_ingested += result.get("documents_stored", 0)
                        logger.info(f"[pubmed] {journal}: ingested {result}")
                else:
                    total_ingested += len(jobs)
            except Exception as e:
                logger.error(f"[pubmed] {journal}: failed: {e}")
    finally:
        await scraper.close()

    return total_ingested


async def discover_indmed(
    journals: list[str] | None,
    max_per_journal: int,
    discover_only: bool,
    limit: int | None = None,
) -> int:
    """Discover + (optionally) ingest IndMED articles."""
    from src.ingestion.scrapers import get_scraper
    from src.ingestion.parsers.indmed import IndMEDParser

    scraper = get_scraper("indmed")
    parser = IndMEDParser()
    pipeline = None

    if not discover_only:
        from src.ingestion.pipeline import IngestionPipeline
        pipeline = IngestionPipeline()

    try:
        jobs = await scraper.discover(
            journals=journals,
            max_articles_per_journal=max_per_journal,
            year_range=(2015, 2025),
        )
        if limit:
            jobs = jobs[:limit]
        logger.info(f"[indmed] {len(jobs)} articles discovered across {len(journals) if journals else 'all'} journals")

        if not discover_only and jobs:
            parsed_docs = []
            for job in jobs:
                scraped = await scraper.fetch_one(job)
                if scraped:
                    record, chunks = parser.parse(scraped)
                    if chunks:
                        parsed_docs.append((record, chunks))
            if parsed_docs:
                result = await pipeline.ingest_scraped_documents(
                    documents=parsed_docs,
                    source="indmed",
                )
                logger.info(f"[indmed] ingestion result: {result}")
                return result.get("documents_stored", 0)
            return 0
        return len(jobs)
    finally:
        await scraper.close()


async def discover_medknow(
    journals: list[str] | None,
    max_per_journal: int,
    discover_only: bool,
    limit: int | None = None,
) -> int:
    """Discover + (optionally) ingest Medknow articles."""
    from src.ingestion.scrapers import get_scraper
    from src.ingestion.parsers.medknow import MedknowParser

    scraper = get_scraper("medknow")
    parser = MedknowParser()
    pipeline = None

    if not discover_only:
        from src.ingestion.pipeline import IngestionPipeline
        pipeline = IngestionPipeline()

    try:
        # If journals specified, look up full journal names from abbreviations
        from src.ingestion.scrapers.sources.medknow import MEDKNOW_JOURNALS
        if journals:
            # Reverse-lookup: abbreviation → full journal name
            abbr_to_name = {info["abbr"]: name for name, info in MEDKNOW_JOURNALS.items()}
            target_journals = [abbr_to_name.get(j, j) for j in journals]
        else:
            target_journals = None  # all

        jobs = await scraper.discover(
            journals=target_journals,
            max_articles_per_journal=max_per_journal,
            year_range=(2015, 2025),
        )
        if limit:
            jobs = jobs[:limit]
        logger.info(f"[medknow] {len(jobs)} articles discovered")

        if not discover_only and jobs:
            parsed_docs = []
            for job in jobs:
                scraped = await scraper.fetch_one(job)
                if scraped:
                    record, chunks = parser.parse(scraped)
                    if chunks:
                        parsed_docs.append((record, chunks))
            if parsed_docs:
                result = await pipeline.ingest_scraped_documents(
                    documents=parsed_docs,
                    source="medknow",
                )
                logger.info(f"[medknow] ingestion result: {result}")
                return result.get("documents_stored", 0)
            return 0
        return len(jobs)
    finally:
        await scraper.close()


async def discover_pmc_india(
    specialty: str | None,
    max_results: int,
    discover_only: bool,
    limit: int | None = None,
) -> int:
    """Discover + (optionally) ingest PMC India articles."""
    from src.ingestion.scrapers import get_scraper
    from src.ingestion.parsers.pmc_india import PMCIndiaParser

    scraper = get_scraper("pmc_india")
    parser = PMCIndiaParser()
    pipeline = None

    if not discover_only:
        from src.ingestion.pipeline import IngestionPipeline
        pipeline = IngestionPipeline()

    try:
        if specialty:
            jobs = await scraper.discover_by_specialty(
                specialty=specialty,
                max_results=max_results,
            )
            logger.info(f"[pmc_india] {len(jobs)} articles discovered for specialty: {specialty}")
        else:
            jobs = await scraper.discover(max_results=max_results)
            logger.info(f"[pmc_india] {len(jobs)} articles discovered (general India query)")
        if limit:
            jobs = jobs[:limit]

        if not discover_only and jobs:
            parsed_docs = []
            for job in jobs:
                scraped = await scraper.fetch_one(job)
                if scraped:
                    record, chunks = parser.parse(scraped)
                    if chunks:
                        parsed_docs.append((record, chunks))
            if parsed_docs:
                result = await pipeline.ingest_scraped_documents(
                    documents=parsed_docs,
                    source="pmc_india",
                )
                logger.info(f"[pmc_india] ingestion result: {result}")
                return result.get("documents_stored", 0)
            return 0
        return len(jobs)
    finally:
        await scraper.close()


async def main() -> int:
    parser = argparse.ArgumentParser(
        description="Seed Indian medical journals (Phase 1 — Layer 2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--source",
        choices=["pubmed", "indmed", "medknow", "pmc_india", "all"],
        default="all",
        help="Which source to ingest from (default: all)",
    )
    parser.add_argument(
        "--journals",
        type=str,
        default=None,
        help="Comma-separated list of journal abbreviations (default: all configured journals)",
    )
    parser.add_argument(
        "--max-per-journal",
        type=int,
        default=500,
        help="Max articles per journal (default: 500)",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=500,
        help="Max results for PMC India (default: 500)",
    )
    parser.add_argument(
        "--specialty",
        type=str,
        default=None,
        help="For PMC India: cardiology, endocrinology, pediatrics, oncology, neurology, infectious_disease, public_health",
    )
    parser.add_argument(
        "--discover-only",
        action="store_true",
        help="Only discover URLs, do not ingest (useful for testing)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap total articles across all sources (useful for test runs). "
             "Example: --limit 20 ingests at most 20 articles total.",
    )
    args = parser.parse_args()

    # Parse journal list
    journals = None
    if args.journals:
        journals = [j.strip() for j in args.journals.split(",") if j.strip()]

    logger.info("=" * 70)
    logger.info("OpenInsight — Phase 1 — Indian Journals Seeder")
    logger.info("=" * 70)
    logger.info(f"Source: {args.source}")
    logger.info(f"Discover only: {args.discover_only}")
    if args.limit:
        logger.info(f"Total cap: {args.limit} articles across all sources")
    if journals:
        logger.info(f"Journals: {journals}")
    logger.info("-" * 70)

    total = 0

    if args.source in ("pubmed", "all"):
        logger.info("[1/4] PubMed — Indian journals")
        remaining = args.limit - total if args.limit else None
        count = await discover_pubmed(journals, args.max_per_journal, args.discover_only, limit=remaining)
        total += count

    if args.source in ("indmed", "all"):
        if not args.limit or total < args.limit:
            logger.info("[2/4] IndMED — Indian journals on indmedinfo.nic.in")
            remaining = args.limit - total if args.limit else None
            count = await discover_indmed(journals, args.max_per_journal, args.discover_only, limit=remaining)
            total += count

    if args.source in ("medknow", "all"):
        if not args.limit or total < args.limit:
            logger.info("[3/4] Medknow — Full-text enrichment from medknow.com")
            # For Medknow, journals arg uses abbreviations (e.g., "ijp") not full names
            remaining = args.limit - total if args.limit else None
            count = await discover_medknow(journals, args.max_per_journal, args.discover_only, limit=remaining)
            total += count

    if args.source in ("pmc_india", "all"):
        if not args.limit or total < args.limit:
            logger.info("[4/4] PMC India — Full-text with Indian affiliations")
            remaining = args.limit - total if args.limit else None
            count = await discover_pmc_india(args.specialty, args.max_results, args.discover_only, limit=remaining)
            total += count

    logger.info("-" * 70)
    logger.info(f"Total articles discovered: {total}")
    if args.discover_only:
        logger.info("Run without --discover-only to ingest (requires MongoDB + Milvus + NCBI_API_KEY)")
    else:
        logger.info("Ingestion wiring is pending — see TODOs in this script")
    logger.info("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
