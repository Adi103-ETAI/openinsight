from __future__ import annotations

import hashlib
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class DocumentDeduplicator:
    """
    Prevents duplicate documents from being re-ingested.
    
    Checks:
    1. doc_id exists in MongoDB
    2. Content hash to detect changes
    3. Returns skip/update decision
    """

    def __init__(self, mongo_store: Any):
        self.mongo = mongo_store

    async def check_document(
        self,
        doc: dict[str, Any],
        force_reindex: bool = False,
    ) -> tuple[bool, Optional[str]]:
        """
        Check if document should be skipped or re-processed.
        
        Returns:
            (should_skip, reason)
            - (True, "unchanged") - skip, doc unchanged
            - (True, "exists") - skip, force_reindex=False
            - (False, None) - process document
        """
        doc_id = doc.get("doc_id", "")
        if not doc_id:
            return False, None

        existing = await self.mongo.get_document(doc_id)
        if not existing:
            return False, None

        if force_reindex:
            return False, "force_reindex"

        content_hash = self._compute_content_hash(doc)
        existing_hash = existing.get("content_hash", "")

        if content_hash == existing_hash:
            logger.info(f"[dedupe] Skipping unchanged doc: {doc_id}")
            return True, "unchanged"

        logger.info(f"[dedupe] Doc changed, will re-process: {doc_id}")
        return False, "changed"

    async def check_chunk_exists(self, chunk_id: str) -> bool:
        """Check if chunk already exists."""
        chunk = await self.mongo.get_chunk(chunk_id)
        return chunk is not None

    def _compute_content_hash(self, doc: dict[str, Any]) -> str:
        """Compute hash of document content for change detection."""
        content_parts = [
            doc.get("title", ""),
            doc.get("abstract", ""),
            doc.get("content", ""),
        ]
        joined = "|".join(p.strip() for p in content_parts if p.strip())
        return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]

    async def get_existing_doc_ids(self, source_type: str) -> set[str]:
        """Get all existing doc_ids for a source type."""
        from motor.motor_asyncio import AsyncIOMotorClient
        from src.core.config import get_settings
        
        settings = get_settings()
        client = AsyncIOMotorClient(settings.mongodb_url)
        db = client[settings.mongodb_db]
        
        cursor = db["documents_v2"].find(
            {"source_type": source_type},
            {"doc_id": 1}
        )
        
        doc_ids = set()
        async for doc in cursor:
            if doc.get("doc_id"):
                doc_ids.add(doc["doc_id"])
        
        return doc_ids


class ChunkDeduplicator:
    """
    Prevents duplicate chunks within a document.
    Uses chunk text hash to detect duplicates.
    """

    def __init__(self):
        self.seen_hashes: set[str] = set()

    def is_duplicate(self, text: str) -> bool:
        """Check if chunk text is duplicate within current batch."""
        text_hash = hashlib.md5(text.encode("utf-8")).hexdigest()[:12]
        if text_hash in self.seen_hashes:
            return True
        self.seen_hashes.add(text_hash)
        return False

    def reset(self):
        """Reset seen hashes for new document."""
        self.seen_hashes.clear()