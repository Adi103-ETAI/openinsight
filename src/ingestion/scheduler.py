"""
Ingestion Scheduler
Runs scheduled ingestion jobs for all configured medical data sources.

Uses APScheduler (AsyncIOScheduler) with cron triggers defined in Settings.
Each source runs its own cron job so schedules can be tuned independently.

Note: Parser-based scheduled ingestion is currently disabled.
Use directory-based ingestion with the CLI instead.
"""

from __future__ import annotations

from typing import Optional

from loguru import logger

try:
    from apscheduler.schedulers.asyncio import AsyncIOScheduler

    _APScheduler_available = True
except ImportError:
    _APScheduler_available = False
    logger.warning(
        "APScheduler not installed. Scheduled ingestion is disabled. "
        "Install with: pip install apscheduler"
    )

from src.config.settings import get_settings

# Parser-based scheduled ingestion is disabled - use CLI-based ingestion instead
_PARSER_BASED_INGESTION_AVAILABLE = False

settings = get_settings()

_PUBMED_QUERIES = [
    "hypertension treatment guidelines",
    "diabetes mellitus management",
    "pneumonia antibiotics clinical trial",
    "COVID-19 treatment outcomes",
    "tuberculosis drug resistance",
    "malaria prevention treatment",
    "sepsis management intensive care",
    "cancer immunotherapy systematic review",
    "heart failure evidence based treatment",
    "stroke acute management guidelines",
]

_COCHRANE_QUERIES = [
    "hypertension treatment",
    "diabetes management",
    "antibiotic stewardship",
    "cancer screening",
    "mental health interventions",
]

_WHO_QUERIES = [
    "essential medicines list",
    "infectious disease guidelines",
    "maternal health recommendations",
    "chronic disease prevention",
    "antimicrobial resistance",
]

_CDC_QUERIES = [
    "disease prevention guidelines",
    "vaccination recommendations",
    "infectious disease outbreak",
    "chronic disease management",
    "antimicrobial stewardship",
]

_STATPEARLS_QUERIES = [
    "pharmacology drug interactions",
    "clinical diagnosis criteria",
    "emergency medicine protocols",
    "internal medicine review",
    "pediatric disease management",
]


async def _run_pubmed_sync() -> None:
    if not _PARSER_BASED_INGESTION_AVAILABLE:
        logger.warning("[scheduler] PubMed sync skipped - parser-based ingestion unavailable")
        return
    logger.info("[scheduler] Starting PubMed weekly sync")
    # TODO: Re-implement when parser-based ingestion is available


async def _run_cochrane_sync() -> None:
    if not _PARSER_BASED_INGESTION_AVAILABLE:
        logger.warning("[scheduler] Cochrane sync skipped - parser-based ingestion unavailable")
        return
    logger.info("[scheduler] Starting Cochrane monthly sync")
    # TODO: Re-implement when parser-based ingestion is available


async def _run_who_sync() -> None:
    if not _PARSER_BASED_INGESTION_AVAILABLE:
        logger.warning("[scheduler] WHO sync skipped - parser-based ingestion unavailable")
        return
    logger.info("[scheduler] Starting WHO monthly sync")
    # TODO: Re-implement when parser-based ingestion is available


async def _run_cdc_sync() -> None:
    if not _PARSER_BASED_INGESTION_AVAILABLE:
        logger.warning("[scheduler] CDC sync skipped - parser-based ingestion unavailable")
        return
    logger.info("[scheduler] Starting CDC monthly sync")
    # TODO: Re-implement when parser-based ingestion is available


async def _run_statpearls_sync() -> None:
    if not _PARSER_BASED_INGESTION_AVAILABLE:
        logger.warning("[scheduler] StatPearls sync skipped - parser-based ingestion unavailable")
        return
    logger.info("[scheduler] Starting StatPearls monthly sync")
    # TODO: Re-implement when parser-based ingestion is available


def _parse_cron(cron_expr: str) -> dict:
    parts = cron_expr.strip().split()
    if len(parts) != 5:
        raise ValueError(f"Invalid cron expression: {cron_expr!r} (expected 5 fields)")
    keys = ["minute", "hour", "day", "month", "day_of_week"]
    return dict(zip(keys, parts))


def start_scheduler() -> Optional["AsyncIOScheduler"]:
    if not _APScheduler_available:
        logger.warning("[scheduler] APScheduler unavailable - scheduled sync disabled")
        return None

    scheduler = AsyncIOScheduler()

    # Log that parser-based scheduled ingestion is disabled
    logger.info(
        "[scheduler] Parser-based scheduled ingestion is disabled. "
        "Use directory-based ingestion with: python -m src.ingestion.run_ingestion --dir <path> --source <source>"
    )

    # Note: All scheduled jobs are currently disabled pending re-implementation
    # of parser-based ingestion. The CLI-based ingestion can be used instead.

    scheduler.start()
    logger.info("[scheduler] Ingestion scheduler started (parser-based jobs disabled)")
    return scheduler


def stop_scheduler(scheduler: Optional["AsyncIOScheduler"]) -> None:
    if scheduler is not None:
        scheduler.shutdown(wait=False)
        logger.info("[scheduler] Ingestion scheduler stopped")