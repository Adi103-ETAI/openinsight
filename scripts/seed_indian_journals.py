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
) -> int:
    """Discover + (optionally) ingest PubMed articles from Indian journals."""
    from src.ingestion.scrapers import get_scraper

    scraper = get_scraper("pubmed")
    target_journals = journals or INDIAN_JOURNALS_PUBMED
    total_jobs = 0

    for journal in target_journals:
        logger.info(f"[pubmed] discovering articles for: {journal}")
        try:
            jobs = await scraper.discover_by_journal(
                journal_abbrev=journal,
                max_results=max_per_journal,
                date_range="2015:2025[DP]",
            )
            logger.info(f"[pubmed] {journal}: {len(jobs)} articles discovered")
            total_jobs += len(jobs)

            if not discover_only and jobs:
                # TODO: wire to ingestion pipeline (Phase 1 final step)
                # For now, just log the first 3 URLs as a sanity check
                for job in jobs[:3]:
                    logger.debug(f"  → {job.url}")
                logger.info(f"[pubmed] {journal}: ingestion not yet wired (would ingest {len(jobs)} articles)")
        except Exception as e:
            logger.error(f"[pubmed] {journal}: failed: {e}")
        finally:
            await scraper.close()

    return total_jobs


async def discover_indmed(
    journals: list[str] | None,
    max_per_journal: int,
    discover_only: bool,
) -> int:
    """Discover + (optionally) ingest IndMED articles."""
    from src.ingestion.scrapers import get_scraper

    scraper = get_scraper("indmed")
    try:
        jobs = await scraper.discover(
            journals=journals,
            max_articles_per_journal=max_per_journal,
            year_range=(2015, 2025),
        )
        logger.info(f"[indmed] {len(jobs)} articles discovered across {len(journals) if journals else 'all'} journals")
        if not discover_only and jobs:
            # TODO: wire to ingestion pipeline
            logger.info(f"[indmed] ingestion not yet wired (would ingest {len(jobs)} articles)")
        return len(jobs)
    finally:
        await scraper.close()


async def discover_medknow(
    journals: list[str] | None,
    max_per_journal: int,
    discover_only: bool,
) -> int:
    """Discover + (optionally) ingest Medknow articles."""
    from src.ingestion.scrapers import get_scraper

    scraper = get_scraper("medknow")
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
        logger.info(f"[medknow] {len(jobs)} articles discovered")
        if not discover_only and jobs:
            # TODO: wire to ingestion pipeline
            logger.info(f"[medknow] ingestion not yet wired (would ingest {len(jobs)} articles)")
        return len(jobs)
    finally:
        await scraper.close()


async def discover_pmc_india(
    specialty: str | None,
    max_results: int,
    discover_only: bool,
) -> int:
    """Discover + (optionally) ingest PMC India articles."""
    from src.ingestion.scrapers import get_scraper

    scraper = get_scraper("pmc_india")
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
        if not discover_only and jobs:
            # TODO: wire to ingestion pipeline
            logger.info(f"[pmc_india] ingestion not yet wired (would ingest {len(jobs)} articles)")
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
    if journals:
        logger.info(f"Journals: {journals}")
    logger.info("-" * 70)

    total = 0

    if args.source in ("pubmed", "all"):
        logger.info("[1/4] PubMed — Indian journals")
        total += await discover_pubmed(journals, args.max_per_journal, args.discover_only)

    if args.source in ("indmed", "all"):
        logger.info("[2/4] IndMED — Indian journals on indmedinfo.nic.in")
        total += await discover_indmed(journals, args.max_per_journal, args.discover_only)

    if args.source in ("medknow", "all"):
        logger.info("[3/4] Medknow — Full-text enrichment from medknow.com")
        # For Medknow, journals arg uses abbreviations (e.g., "ijp") not full names
        total += await discover_medknow(journals, args.max_per_journal, args.discover_only)

    if args.source in ("pmc_india", "all"):
        logger.info("[4/4] PMC India — Full-text with Indian affiliations")
        total += await discover_pmc_india(args.specialty, args.max_results, args.discover_only)

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
