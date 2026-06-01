"""
Ingestion Scheduler (DEPRECATED — removed)

Parser-based scheduled ingestion has been removed. Use CLI-based
directory ingestion instead:

    python -m src.ingestion.run_ingestion --dir <path> --source <source>

This module is retained as a stub to avoid ImportError for any
residual references. All functions are no-ops.
"""

from __future__ import annotations

from typing import Optional

from loguru import logger


def start_scheduler() -> Optional[object]:
    """No-op. Scheduled ingestion has been removed."""
    logger.info("[scheduler] Scheduled ingestion is disabled. Use CLI ingestion instead.")
    return None


def stop_scheduler(scheduler: Optional[object]) -> None:
    """No-op. Scheduled ingestion has been removed."""
    pass
