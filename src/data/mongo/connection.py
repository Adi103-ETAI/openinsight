"""
MongoDB Connection Pool Manager.

Provides a singleton connection pool for MongoDB that can be shared across
the application to avoid connection overhead per batch.
"""

from __future__ import annotations

import importlib
from typing import Any, Optional

from loguru import logger

from src.config.settings import get_settings

# Singleton instance
_mongo_client: Optional[Any] = None


def get_mongo_client(
    mongo_url: Optional[str] = None,
    max_pool_size: Optional[int] = None,
    min_pool_size: Optional[int] = None,
    max_idle_time_ms: Optional[int] = None,
    connect_timeout_ms: Optional[int] = None,
    server_selection_timeout_ms: Optional[int] = None,
) -> Any:
    """
    Get or create a pooled MongoDB client.

    Uses singleton pattern to ensure connection reuse across the application.
    Connection pool settings are applied on first creation and reused thereafter.

    Args:
        mongo_url: MongoDB connection URL (defaults to settings)
        max_pool_size: Maximum connections in pool
        min_pool_size: Minimum connections to maintain
        max_idle_time_ms: Max idle time before connection is closed (ms)
        connect_timeout_ms: Connection timeout (ms)
        server_selection_timeout_ms: Server selection timeout (ms)

    Returns:
        AsyncIOMotorClient with connection pooling enabled
    """
    global _mongo_client

    if _mongo_client is not None:
        return _mongo_client

    settings = get_settings()

    # Use provided values or fall back to settings
    url = mongo_url or settings.mongodb_url

    # Import motor asynchronously
    motor_asyncio = importlib.import_module("motor.motor_asyncio")
    AsyncIOMotorClient = getattr(motor_asyncio, "AsyncIOMotorClient")

    # Build connection pool options
    pool_options = {
        "maxPoolSize": max_pool_size or settings.mongodb_max_pool_size,
        "minPoolSize": min_pool_size or settings.mongodb_min_pool_size,
        "maxIdleTimeMS": max_idle_time_ms or settings.mongodb_max_idle_time_ms,
        "connectTimeoutMS": connect_timeout_ms or settings.mongodb_connect_timeout_ms,
        "serverSelectionTimeoutMS": server_selection_timeout_ms
        or settings.mongodb_server_selection_timeout_ms,
    }

    logger.info(
        "[MongoPool] Creating connection pool: maxPoolSize=%d, minPoolSize=%d",
        pool_options["maxPoolSize"],
        pool_options["minPoolSize"],
    )

    _mongo_client = AsyncIOMotorClient(url, **pool_options)

    # Avoid sync connection tests here; Motor is async-only and this can
    # conflict with an already running event loop.
    logger.info("[MongoPool] Connection pool initialized")

    return _mongo_client


def close_mongo_client() -> None:
    """Close the MongoDB client and release all pool connections."""
    global _mongo_client

    if _mongo_client is not None:
        logger.info("[MongoPool] Closing connection pool")
        _mongo_client.close()
        _mongo_client = None


def get_mongo_db(db_name: Optional[str] = None) -> Any:
    """
    Get a database instance from the pooled connection.

    Args:
        db_name: Database name (defaults to settings.mongodb_db)

    Returns:
        Database instance
    """
    settings = get_settings()
    client = get_mongo_client()
    return client[db_name or settings.mongodb_db]
