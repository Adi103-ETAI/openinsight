"""
Ingestion Scheduler
Runs scheduled ingestion jobs for all configured medical data sources.

Uses APScheduler (AsyncIOScheduler) with cron triggers defined in Settings.
Each source runs its own cron job so schedules can be tuned independently.
"""

from __future__ import annotations

from typing import Optional

from loguru import logger

try:
    from apscheduler.schedulers.asyncio import AsyncIOScheduler
    from apscheduler.triggers.cron import CronTrigger

    _APScheduler_available = True
except ImportError:
    _APScheduler_available = False
    logger.warning(
        "APScheduler not installed. Scheduled ingestion is disabled. "
        "Install with: pip install apscheduler"
    )

from src.core.config import get_settings
from src.ingestion.parsers.cdc import CDCParser
from src.ingestion.parsers.cochrane import CochraneParser
from src.ingestion.parsers.pubmed import PubMedParser
from src.ingestion.parsers.statpearls import StatPearlsParser
from src.ingestion.parsers.who import WHOParser
from src.ingestion.pipeline_v3 import run_pipeline_v3_from_parser

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
    logger.info("[scheduler] Starting PubMed weekly sync")
    for query in _PUBMED_QUERIES:
        try:
            await run_pipeline_v3_from_parser(
                PubMedParser, query, max_results=100, concurrency=3
            )
        except (RuntimeError, ValueError, TypeError, OSError) as exc:
            logger.error(f"[scheduler] PubMed job failed for '{query}': {exc}")


async def _run_cochrane_sync() -> None:
    logger.info("[scheduler] Starting Cochrane monthly sync")
    for query in _COCHRANE_QUERIES:
        try:
            await run_pipeline_v3_from_parser(
                CochraneParser, query, max_results=50, concurrency=2
            )
        except (RuntimeError, ValueError, TypeError, OSError) as exc:
            logger.error(f"[scheduler] Cochrane job failed for '{query}': {exc}")


async def _run_who_sync() -> None:
    logger.info("[scheduler] Starting WHO monthly sync")
    for query in _WHO_QUERIES:
        try:
            await run_pipeline_v3_from_parser(
                WHOParser, query, max_results=30, concurrency=2
            )
        except (RuntimeError, ValueError, TypeError, OSError) as exc:
            logger.error(f"[scheduler] WHO job failed for '{query}': {exc}")


async def _run_cdc_sync() -> None:
    logger.info("[scheduler] Starting CDC monthly sync")
    for query in _CDC_QUERIES:
        try:
            await run_pipeline_v3_from_parser(
                CDCParser, query, max_results=30, concurrency=2
            )
        except (RuntimeError, ValueError, TypeError, OSError) as exc:
            logger.error(f"[scheduler] CDC job failed for '{query}': {exc}")


async def _run_statpearls_sync() -> None:
    logger.info("[scheduler] Starting StatPearls monthly sync")
    for query in _STATPEARLS_QUERIES:
        try:
            await run_pipeline_v3_from_parser(
                StatPearlsParser, query, max_results=20, concurrency=2
            )
        except (RuntimeError, ValueError, TypeError, OSError) as exc:
            logger.error(f"[scheduler] StatPearls job failed for '{query}': {exc}")


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

    try:
        scheduler.add_job(
            _run_pubmed_sync,
            CronTrigger(**_parse_cron(settings.scheduler_pubmed_cron)),
            id="pubmed_weekly_sync",
            name="PubMed Weekly Sync",
            replace_existing=True,
            misfire_grace_time=3600,
        )
    except (RuntimeError, ValueError, TypeError, OSError) as exc:
        logger.error(f"[scheduler] Failed to schedule PubMed job: {exc}")

    try:
        scheduler.add_job(
            _run_cochrane_sync,
            CronTrigger(**_parse_cron(settings.scheduler_cochrane_cron)),
            id="cochrane_monthly_sync",
            name="Cochrane Monthly Sync",
            replace_existing=True,
            misfire_grace_time=7200,
        )
    except (RuntimeError, ValueError, TypeError, OSError) as exc:
        logger.error(f"[scheduler] Failed to schedule Cochrane job: {exc}")

    try:
        scheduler.add_job(
            _run_who_sync,
            CronTrigger(**_parse_cron(settings.scheduler_who_cron)),
            id="who_monthly_sync",
            name="WHO Monthly Sync",
            replace_existing=True,
            misfire_grace_time=7200,
        )
    except (RuntimeError, ValueError, TypeError, OSError) as exc:
        logger.error(f"[scheduler] Failed to schedule WHO job: {exc}")

    try:
        scheduler.add_job(
            _run_cdc_sync,
            CronTrigger(**_parse_cron(settings.scheduler_cdc_cron)),
            id="cdc_monthly_sync",
            name="CDC Monthly Sync",
            replace_existing=True,
            misfire_grace_time=7200,
        )
    except (RuntimeError, ValueError, TypeError, OSError) as exc:
        logger.error(f"[scheduler] Failed to schedule CDC job: {exc}")

    try:
        scheduler.add_job(
            _run_statpearls_sync,
            CronTrigger(**_parse_cron(settings.scheduler_cdc_cron)),
            id="statpearls_monthly_sync",
            name="StatPearls Monthly Sync",
            replace_existing=True,
            misfire_grace_time=7200,
        )
    except (RuntimeError, ValueError, TypeError, OSError) as exc:
        logger.error(f"[scheduler] Failed to schedule StatPearls job: {exc}")

    scheduler.start()
    logger.info("[scheduler] Ingestion scheduler started")
    return scheduler


def stop_scheduler(scheduler: Optional["AsyncIOScheduler"]) -> None:
    if scheduler is not None:
        scheduler.shutdown(wait=False)
        logger.info("[scheduler] Ingestion scheduler stopped")
