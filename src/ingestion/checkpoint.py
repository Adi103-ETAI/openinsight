"""
Checkpoint Manager for Ingestion Pipeline

Provides checkpoint/resume functionality for long-running ingestion jobs.
Persists progress to MongoDB to allow resuming from last successful batch on failure.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from loguru import logger

from src.config.settings import get_settings
from src.data.mongo.connection import get_mongo_db


class IngestionCheckpoint:
    """
    Checkpoint state for an ingestion run.
    
    Tracks progress so ingestion can resume from last successful batch
    instead of restarting from the beginning.
    """

    def __init__(
        self,
        source: str,
        directory: str,
        batch_size: int = 10,
        last_batch_index: int = -1,
        total_batches: int = 0,
        files_processed: list[str] | None = None,
        total_files: int = 0,
        status: str = "in_progress",
        error: str | None = None,
        created_at: datetime | None = None,
        updated_at: datetime | None = None,
    ):
        self.source = source
        self.directory = directory
        self.batch_size = batch_size
        self.last_batch_index = last_batch_index  # -1 means no batches completed yet
        self.total_batches = total_batches
        self.files_processed = files_processed or []
        self.total_files = total_files
        self.status = status  # in_progress, completed, failed
        self.error = error
        self.created_at = created_at or datetime.utcnow()
        self.updated_at = updated_at or datetime.utcnow()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for MongoDB storage."""
        return {
            "source": self.source,
            "directory": self.directory,
            "batch_size": self.batch_size,
            "last_batch_index": self.last_batch_index,
            "total_batches": self.total_batches,
            "files_processed": self.files_processed,
            "total_files": self.total_files,
            "status": self.status,
            "error": self.error,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IngestionCheckpoint:
        """Create from MongoDB document."""
        return cls(
            source=data.get("source", ""),
            directory=data.get("directory", ""),
            batch_size=data.get("batch_size", 10),
            last_batch_index=data.get("last_batch_index", -1),
            total_batches=data.get("total_batches", 0),
            files_processed=data.get("files_processed", []),
            total_files=data.get("total_files", 0),
            status=data.get("status", "in_progress"),
            error=data.get("error"),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
        )


class CheckpointManager:
    """
    Manages checkpoint state for ingestion pipeline.
    
    Provides:
    - Save checkpoint after each successful batch
    - Load checkpoint to resume from last successful position
    - List/clear checkpoints for different sources
    
    Storage: MongoDB collection "ingestion_checkpoints"
    """

    def __init__(self, mongo_url: str | None = None, db_name: str | None = None):
        """Initialize checkpoint manager with shared MongoDB connection pool."""
        settings = get_settings()
        
        # Use shared connection pool
        self.db = get_mongo_db(db_name or settings.mongodb_db)
        self.checkpoints = self.db["ingestion_checkpoints"]

    async def create_checkpoint(
        self,
        source: str,
        directory: str,
        batch_size: int = 10,
        total_files: int = 0,
    ) -> IngestionCheckpoint:
        """Create a new checkpoint for an ingestion run."""
        total_batches = (total_files + batch_size - 1) // batch_size  # ceil division
        
        checkpoint = IngestionCheckpoint(
            source=source,
            directory=directory,
            batch_size=batch_size,
            total_files=total_files,
            total_batches=total_batches,
            status="in_progress",
        )

        await self.checkpoints.update_one(
            {"source": source, "directory": directory},
            {"$set": checkpoint.to_dict()},
            upsert=True,
        )

        logger.info(
            "[checkpoint] Created checkpoint: source=%s dir=%s batches=%d",
            source,
            directory,
            total_batches,
        )
        return checkpoint

    async def save_batch_complete(
        self,
        source: str,
        directory: str,
        batch_index: int,
        processed_files: list[str],
    ) -> None:
        """Save checkpoint after successful batch completion."""
        update = {
            "last_batch_index": batch_index,
            "files_processed": processed_files,
            "updated_at": datetime.utcnow().isoformat(),
            "status": "in_progress",
        }

        await self.checkpoints.update_one(
            {"source": source, "directory": directory},
            {"$set": update},
        )

        logger.info(
            "[checkpoint] Saved batch %d complete for %s", batch_index, source
        )

    async def mark_complete(
        self,
        source: str,
        directory: str,
    ) -> None:
        """Mark ingestion as completed."""
        await self.checkpoints.update_one(
            {"source": source, "directory": directory},
            {
                "$set": {
                    "status": "completed",
                    "last_batch_index": -1,  # All done
                    "updated_at": datetime.utcnow().isoformat(),
                }
            },
        )
        logger.info("[checkpoint] Marked complete: source=%s", source)

    async def mark_failed(
        self,
        source: str,
        directory: str,
        error: str,
    ) -> None:
        """Mark ingestion as failed with error message."""
        await self.checkpoints.update_one(
            {"source": source, "directory": directory},
            {
                "$set": {
                    "status": "failed",
                    "error": error,
                    "updated_at": datetime.utcnow().isoformat(),
                }
            },
        )
        logger.warning("[checkpoint] Marked failed: source=%s error=%s", source, error)

    async def get_checkpoint(
        self,
        source: str,
        directory: str,
    ) -> IngestionCheckpoint | None:
        """Get checkpoint for resuming, or None if no checkpoint exists."""
        doc = await self.checkpoints.find_one(
            {"source": source, "directory": directory}
        )
        if doc:
            return IngestionCheckpoint.from_dict(doc)
        return None

    async def get_resume_batch_index(
        self,
        source: str,
        directory: str,
    ) -> int:
        """
        Get the batch index to resume from.
        
        Returns:
            - -1 if no checkpoint exists (start from beginning)
            - -2 if checkpoint status is "completed"
            - last_batch_index + 1 if in_progress (resume from next batch)
        """
        checkpoint = await self.get_checkpoint(source, directory)
        if not checkpoint:
            return -1  # No checkpoint, start from beginning
        
        if checkpoint.status == "completed":
            return -2  # Already completed
        
        if checkpoint.status == "failed":
            return checkpoint.last_batch_index  # Resume from where it failed
        
        # in_progress - resume from next batch after last successful
        return checkpoint.last_batch_index + 1

    async def clear_checkpoint(
        self,
        source: str,
        directory: str,
    ) -> bool:
        """Clear checkpoint to start fresh. Returns True if checkpoint existed."""
        result = await self.checkpoints.delete_one(
            {"source": source, "directory": directory}
        )
        if result.deleted_count > 0:
            logger.info("[checkpoint] Cleared checkpoint: source=%s", source)
            return True
        return False

    async def list_checkpoints(self) -> list[IngestionCheckpoint]:
        """List all checkpoints."""
        cursor = self.checkpoints.find().sort("updated_at", -1)
        checkpoints = []
        async for doc in cursor:
            checkpoints.append(IngestionCheckpoint.from_dict(doc))
        return checkpoints

    async def get_active_checkpoint(self) -> IngestionCheckpoint | None:
        """Get the most recent active checkpoint (in_progress or failed)."""
        doc = await self.checkpoints.find_one(
            {"status": {"$in": ["in_progress", "failed"]}},
            sort=[("updated_at", -1)],
        )
        if doc:
            return IngestionCheckpoint.from_dict(doc)
        return None