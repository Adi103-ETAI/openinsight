"""
Centralized loguru configuration for OpenInsight.

Provides a single entry point for configuring loguru across the application.
Handles log level, format, file rotation, and structured logging.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from loguru import logger

from src.config.settings import get_settings


def configure_loguru(
    log_level: Optional[str] = None,
    log_dir: Optional[str] = None,
    rotation: str = "500 MB",
    retention: str = "10 days",
    compression: str = "zip",
    serialize: bool = False,
) -> None:
    """
    Configure loguru with application-wide settings.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR). Defaults to settings.
        log_dir: Directory for log files. Defaults to 'logs/' in project root.
        rotation: When to rotate log file (size or time-based).
        retention: How long to keep old log files.
        compression: Compression format for rotated logs.
        serialize: Whether to output logs as JSON.
    """
    settings = get_settings()
    level = (log_level or settings.log_level).upper()

    # Remove all default handlers before adding custom ones
    logger.remove()

    # --- Console handler (stderr) ---
    if serialize:
        # JSON format for production / log aggregation
        logger.add(
            sys.stderr,
            level=level,
            serialize=True,
            backtrace=True,
            diagnose=False,
        )
    else:
        # Human-readable format with request_id support
        log_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level:<8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
            "{extra[request_id_str]}"
        )

        def _request_id_formatter(record: dict) -> None:
            """Append request_id to extra if present."""
            request_id = record["extra"].get("request_id")
            record["extra"]["request_id_str"] = (
                f" | request_id={request_id}" if request_id else ""
            )

        logger.add(
            sys.stderr,
            format=log_format,
            level=level,
            backtrace=True,
            diagnose=True,
            enqueue=True,  # Thread-safe async logging
        )
        logger.configure(patch=_request_id_formatter)

    # --- File handler with rotation ---
    if log_dir is None:
        log_dir = str(Path(__file__).resolve().parents[2] / "logs")

    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # All logs (rotating)
    logger.add(
        str(log_path / "openinsight_{time:YYYY-MM-DD}.log"),
        level=level,
        rotation=rotation,
        retention=retention,
        compression=compression,
        format=(
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | "
            "{name}:{function}:{line} - {message}"
            "{extra[request_id_str]}"
        ),
        enqueue=True,
        backtrace=True,
        diagnose=False,
    )

    # Error-only log for quick triage
    logger.add(
        str(log_path / "error_{time:YYYY-MM-DD}.log"),
        level="ERROR",
        rotation=rotation,
        retention=retention,
        compression=compression,
        format=(
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | "
            "{name}:{function}:{line} - {message}"
            "{extra[request_id_str]}"
        ),
        enqueue=True,
        backtrace=True,
        diagnose=False,
    )

    logger.info(
        "Loguru configured | level={} | log_dir={} | rotation={} | retention={}",
        level,
        log_path,
        rotation,
        retention,
    )


def get_logger() -> logger:
    """Return the configured loguru logger instance."""
    return logger
