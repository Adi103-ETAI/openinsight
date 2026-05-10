from __future__ import annotations

import importlib
import logging
from datetime import datetime
from typing import Any, Optional

from src.config.settings import get_settings
from src.data.mongo.connection import get_mongo_db

logger = logging.getLogger(__name__)


class MongoDocStoreV2:
    """
    Mongo store for v2 ingestion artifacts.

    - documents_v2: full parsed documents plus enriched metadata
    - chunks_v2: raw chunk text and chunk-level metadata linkage
    
    Uses shared connection pool from mongo_connection module to avoid
    connection overhead per batch.
    """

    def __init__(
        self, mongo_url: str | None = None, db_name: str | None = None
    ):
        settings = get_settings()
        
        # Use shared connection pool instead of creating new client
        self.db = get_mongo_db(db_name or settings.mongodb_db)
        self.documents = self.db["documents_v2"]
        self.chunks = self.db["chunks_v2"]

    async def store_document(self, doc: Any, enriched_metadata: dict[str, Any]) -> None:
        doc_id = self._get_field(doc, "doc_id", "")
        sections = self._get_field(doc, "sections", [])

        section_payload = []
        if isinstance(sections, list):
            for i, sec in enumerate(sections):
                if isinstance(sec, dict):
                    title = sec.get("title", "")
                    text = sec.get("text", "")
                    index = sec.get("section_index", i)
                else:
                    title = self._get_field(sec, "title", "")
                    text = self._get_field(sec, "text", "")
                    index = self._get_field(sec, "section_index", i)
                section_payload.append(
                    {"title": str(title), "text": str(text), "index": int(index)}
                )

        payload = {
            "doc_id": doc_id,
            "title": self._get_field(doc, "title", ""),
            "abstract": self._get_field(doc, "abstract", ""),
            "authors": self._coerce_str_list(self._get_field(doc, "authors", [])),
            "year": self._safe_int(self._get_field(doc, "year", 0)),
            "journal": self._get_field(doc, "journal", ""),
            "doi": self._get_field(doc, "doi", None),
            "pmid": self._get_field(doc, "pmid", None),
            "sections": section_payload,
            "mesh_terms": self._coerce_str_list(self._get_field(doc, "mesh_terms", [])),
            "keywords": self._coerce_str_list(self._get_field(doc, "keywords", [])),
            **enriched_metadata,
            "ingested_at": datetime.utcnow().isoformat(),
        }

        await self.documents.update_one(
            {"doc_id": doc_id}, {"$set": payload}, upsert=True
        )

    async def store_chunks(self, chunks: list[Any]) -> None:
        pymongo = importlib.import_module("pymongo")
        UpdateOne = getattr(pymongo, "UpdateOne")

        ops: list[Any] = []
        for chunk in chunks:
            ops.append(
                UpdateOne(
                    {"chunk_id": chunk.chunk_id},
                    {
                        "$set": {
                            "chunk_id": chunk.chunk_id,
                            "doc_id": chunk.doc_id,
                            "chunk_type": chunk.chunk_type,
                            "section_title": chunk.section_title,
                            "text": chunk.text,
                            "char_count": chunk.char_count,
                            "token_estimate": chunk.token_estimate,
                            "chunk_index": chunk.chunk_index,
                            "total_chunks": chunk.total_chunks,
                            "metadata": dict(chunk.metadata),
                        }
                    },
                    upsert=True,
                )
            )

        if ops:
            await self.chunks.bulk_write(ops)

    async def get_document(self, doc_id: str) -> Optional[dict[str, Any]]:
        return await self.documents.find_one({"doc_id": doc_id}, {"_id": 0})

    async def get_chunk(self, chunk_id: str) -> Optional[dict[str, Any]]:
        return await self.chunks.find_one({"chunk_id": chunk_id}, {"_id": 0})

    def _get_field(self, obj: Any, key: str, default: Any) -> Any:
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    def _coerce_str_list(self, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(v).strip() for v in value if str(v).strip()]
        if isinstance(value, str):
            val = value.strip()
            return [val] if val else []
        return []

    def _safe_int(self, value: Any) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0
