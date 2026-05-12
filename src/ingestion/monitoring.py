"""
Ingestion Monitoring & Analytics
Tracks per-run and cumulative ingestion metrics.
Persists metrics to MongoDB (collection: ingestion_metrics).
"""
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional

from loguru import logger
from motor.motor_asyncio import AsyncIOMotorDatabase


@dataclass
class RunMetrics:
    """Metrics collected during a single ingestion run."""
    run_id: str
    started_at: datetime = field(default_factory=datetime.utcnow)
    finished_at: Optional[datetime] = None
    source_type: str = "unknown"
    # Document counters
    documents_fetched: int = 0
    documents_stored: int = 0
    documents_skipped_duplicate: int = 0
    documents_failed_validation: int = 0
    # Chunk counters
    chunks_created: int = 0
    chunks_embedded: int = 0
    chunks_skipped_noise: int = 0
    chunks_skipped_quality: int = 0
    chunks_failed_validation: int = 0
    # Embedding
    embedding_errors: int = 0
    # Timing
    duration_seconds: float = 0.0
    # Status
    status: str = "running"   # "running" | "completed" | "failed"
    error_message: Optional[str] = None

    def finish(self, status: str = "completed", error: Optional[str] = None) -> None:
        self.finished_at = datetime.utcnow()
        self.duration_seconds = (self.finished_at - self.started_at).total_seconds()
        self.status = status
        self.error_message = error

    def summary(self) -> dict:
        return {
            "run_id": self.run_id,
            "source_type": self.source_type,
            "status": self.status,
            "documents_stored": self.documents_stored,
            "documents_skipped_duplicate": self.documents_skipped_duplicate,
            "chunks_embedded": self.chunks_embedded,
            "chunks_skipped_noise": self.chunks_skipped_noise,
            "duration_seconds": round(self.duration_seconds, 1),
        }


class IngestionMonitor:
    """
    Manages ingestion run metrics.
    Persists each run to MongoDB (collection: ingestion_metrics).
    """

    def __init__(self, db: AsyncIOMotorDatabase) -> None:
        self._db = db
        self._collection = db["ingestion_metrics"]

    async def save_run(self, metrics: RunMetrics) -> None:
        """Upsert a RunMetrics record into MongoDB."""
        data = asdict(metrics)
        # Convert datetime to ISO string for MongoDB compatibility
        for key in ("started_at", "finished_at"):
            if isinstance(data.get(key), datetime):
                data[key] = data[key].isoformat()
        try:
            await self._collection.update_one(
                {"run_id": metrics.run_id},
                {"$set": data},
                upsert=True,
            )
            logger.debug(f"[monitor] Saved run metrics: {metrics.run_id}")
        except Exception as exc:
            logger.warning(f"[monitor] Failed to save run metrics: {exc}")

    async def get_recent_runs(self, limit: int = 20) -> list[dict]:
        """Return the most recent ingestion runs."""
        cursor = self._collection.find({}, {"_id": 0}).sort("started_at", -1).limit(limit)
        return await cursor.to_list(length=limit)

    async def get_storage_stats(self) -> dict:
        """Return document and chunk counts per source_type."""
        docs_col = self._db["documents_v2"]
        chunks_col = self._db["chunks_v2"]

        pipeline = [{"$group": {"_id": "$source_type", "count": {"$sum": 1}}}]
        doc_counts: dict[str, int] = {}
        async for row in docs_col.aggregate(pipeline):
            doc_counts[row["_id"]] = row["count"]

        chunk_counts: dict[str, int] = {}
        async for row in chunks_col.aggregate(pipeline):
            chunk_counts[row["_id"]] = row["count"]

        embedded_pipeline = [
            {"$match": {"embedded": True}},
            {"$group": {"_id": "$source_type", "count": {"$sum": 1}}},
        ]
        embedded_counts: dict[str, int] = {}
        async for row in chunks_col.aggregate(embedded_pipeline):
            embedded_counts[row["_id"]] = row["count"]

        return {
            "documents_by_source": doc_counts,
            "chunks_by_source": chunk_counts,
            "embedded_chunks_by_source": embedded_counts,
            "total_documents": sum(doc_counts.values()),
            "total_chunks": sum(chunk_counts.values()),
            "total_embedded_chunks": sum(embedded_counts.values()),
        }

    async def alert_on_failure(self, metrics: RunMetrics) -> None:
        """
        Log a structured alert when a run fails or has high error rates.
        Extend this to send emails / Slack / PagerDuty as needed.
        """
        if metrics.status == "failed":
            logger.error(
                f"[ALERT] Ingestion run FAILED | source={metrics.source_type} "
                f"run_id={metrics.run_id} error={metrics.error_message}"
            )
        elif metrics.embedding_errors > 0:
            logger.warning(
                f"[ALERT] Embedding errors in run | source={metrics.source_type} "
                f"errors={metrics.embedding_errors} run_id={metrics.run_id}"
            )
