"""
Conversation store — MongoDB persistence for chat threads (BIS Stage 5 port).

Collections: `conversations`, `messages`. Owner-scoped by user_id; all
reads filter on (user_id via conversation) so one user never sees another's.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from loguru import logger
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorCollection


def _oid_str(doc: dict[str, Any]) -> dict[str, Any]:
    if doc and "_id" in doc:
        doc["_id"] = str(doc["_id"])
    return doc


class ConversationStore:
    def __init__(self, mongo_url: str, db_name: str) -> None:
        self._client = AsyncIOMotorClient(mongo_url)
        self._db = self._client[db_name]
        self._convs: AsyncIOMotorCollection = self._db["conversations"]
        self._msgs: AsyncIOMotorCollection = self._db["messages"]
        self._indexes_created = False

    async def _ensure_indexes(self) -> None:
        if self._indexes_created:
            return
        try:
            await self._convs.create_index([("user_id", 1), ("updated_at", -1)])
            await self._msgs.create_index([("conversation_id", 1), ("created_at", 1)])
            await self._msgs.create_index("client_id", sparse=True)
            self._indexes_created = True
        except Exception as e:
            logger.warning(f"Failed to create conversation indexes: {e}")

    @staticmethod
    def generate_title(query: str) -> str:
        cleaned = " ".join(query.strip().split())
        return (cleaned[:57] + "…") if len(cleaned) > 60 else cleaned or "New conversation"

    async def create_conversation(self, user_id: str, title: str | None = None) -> dict[str, Any]:
        await self._ensure_indexes()
        now = datetime.utcnow()
        doc = {
            "user_id": user_id,
            "title": (title or "").strip() or "New conversation",
            "status": "active",
            "message_count": 0,
            "last_message_at": None,
            "created_at": now,
            "updated_at": now,
        }
        res = await self._convs.insert_one(doc)
        doc["_id"] = str(res.inserted_id)
        return doc

    async def owned_conversation(self, user_id: str, conversation_id: str) -> dict[str, Any] | None:
        from bson import ObjectId

        try:
            doc = await self._convs.find_one({"_id": ObjectId(conversation_id), "user_id": user_id})
        except Exception:
            return None
        return _oid_str(doc) if doc else None

    async def list_conversations(
        self, user_id: str, status: str = "active", page: int = 1, limit: int = 20
    ) -> tuple[list[dict[str, Any]], int]:
        await self._ensure_indexes()
        q: dict[str, Any] = {"user_id": user_id}
        if status in ("active", "archived"):
            q["status"] = status
        total = await self._convs.count_documents(q)
        cursor = self._convs.find(q).sort("updated_at", -1).skip((page - 1) * limit).limit(limit)
        out = []
        async for doc in cursor:
            out.append(_oid_str(doc))
        return out, total

    async def add_messages(
        self,
        conversation_id: str,
        user_query: str,
        answer: str,
        citations: list[dict[str, Any]],
        chunks_retrieved: int,
        client_id: str | None = None,
    ) -> dict[str, Any]:
        """Persist user + assistant rows, touch conversation. Returns assistant doc."""
        from bson import ObjectId

        await self._ensure_indexes()
        now = datetime.utcnow()
        await self._msgs.insert_one({
            "conversation_id": conversation_id,
            "role": "user",
            "content": user_query,
            "citations": [],
            "status": "completed",
            "client_id": client_id,
            "created_at": now,
            "completed_at": now,
        })
        assistant = {
            "conversation_id": conversation_id,
            "role": "assistant",
            "content": answer,
            "citations": citations,
            "status": "completed" if citations or answer else "failed",
            "chunks_retrieved": chunks_retrieved,
            "created_at": now,
            "completed_at": now,
        }
        res = await self._msgs.insert_one(assistant)
        assistant["_id"] = str(res.inserted_id)
        conv = await self._convs.find_one({"_id": ObjectId(conversation_id)})
        update: dict[str, Any] = {
            "message_count": (conv.get("message_count", 0) if conv else 0) + 2,
            "last_message_at": now,
            "updated_at": now,
        }
        if conv and conv.get("title") == "New conversation":
            update["title"] = self.generate_title(user_query)
        await self._convs.update_one({"_id": ObjectId(conversation_id)}, {"$set": update})
        return assistant

    async def list_messages(
        self, conversation_id: str, page: int = 1, limit: int = 50
    ) -> tuple[list[dict[str, Any]], int]:
        total = await self._msgs.count_documents({"conversation_id": conversation_id})
        cursor = self._msgs.find({"conversation_id": conversation_id}).sort("created_at", 1).skip((page - 1) * limit).limit(limit)
        out = []
        async for doc in cursor:
            out.append(_oid_str(doc))
        return out, total

    async def import_entries(self, user_id: str, entries: list[dict[str, Any]]) -> tuple[int, int]:
        """Idempotent localStorage import — one conversation per entry, max 50."""
        imported, skipped = 0, 0
        for entry in entries[:50]:
            client_id = str(entry.get("client_id", "") or entry.get("id", ""))
            query = str(entry.get("query", "")).strip()
            if not client_id or not query:
                skipped += 1
                continue
            if await self._msgs.find_one({"client_id": client_id}) is not None:
                skipped += 1
                continue
            conv = await self.create_conversation(user_id, str(entry.get("title", ""))[:200] or self.generate_title(query))
            resp = entry.get("response") or {}
            await self.add_messages(
                str(conv["_id"]),
                query,
                str(resp.get("answer", "")) if isinstance(resp, dict) else "",
                resp.get("citations", []) if isinstance(resp, dict) and isinstance(resp.get("citations"), list) else [],
                int(resp.get("chunks_retrieved", 0)) if isinstance(resp, dict) else 0,
                client_id=client_id,
            )
            imported += 1
        return imported, skipped
