"""
Research Vault — MongoDB store for saved citations, notes, and collections.
Doctors can save search results and organize them into collections for later review.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from loguru import logger
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorCollection


class VaultStore:
    """Async MongoDB store for Research Vault items and collections."""

    def __init__(self, mongo_url: str, db_name: str) -> None:
        self._client = AsyncIOMotorClient(mongo_url)
        self._db = self._client[db_name]
        self._items: AsyncIOMotorCollection = self._db["vault_items"]
        self._collections: AsyncIOMotorCollection = self._db["vault_collections"]
        self._indexes_created = False

    async def _ensure_indexes(self) -> None:
        """Create indexes on first access."""
        if self._indexes_created:
            return
        try:
            await self._items.create_index([("user_id", 1), ("created_at", -1)])
            await self._items.create_index([("user_id", 1), ("item_type", 1)])
            await self._items.create_index([("user_id", 1), ("tags", 1)])
            await self._items.create_index("source_url", sparse=True)
            await self._collections.create_index([("user_id", 1)])
            self._indexes_created = True
        except Exception as e:
            logger.warning(f"Failed to create vault indexes: {e}")

    # ── Items ────────────────────────────────────────────────────────────────

    async def create_item(
        self,
        user_id: str,
        item_type: str,
        title: str,
        content: str = "",
        source_type: str | None = None,
        source_url: str | None = None,
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        collection_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """Create a new vault item."""
        await self._ensure_indexes()
        now = datetime.utcnow()
        doc = {
            "user_id": user_id,
            "item_type": item_type,
            "title": title,
            "content": content,
            "source_type": source_type,
            "source_url": source_url,
            "metadata": metadata or {},
            "tags": tags or [],
            "collection_ids": collection_ids or [],
            "created_at": now,
            "updated_at": now,
        }
        result = await self._items.insert_one(doc)
        doc["_id"] = str(result.inserted_id)
        return doc

    async def get_item(self, item_id: str) -> dict[str, Any] | None:
        """Get a single vault item by ID."""
        from bson import ObjectId

        doc = await self._items.find_one({"_id": ObjectId(item_id)})
        if doc:
            doc["_id"] = str(doc["_id"])
        return doc

    async def list_items(
        self,
        user_id: str,
        item_type: str | None = None,
        tag: str | None = None,
        collection_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """List vault items for a user with optional filters."""
        await self._ensure_indexes()
        query: dict[str, Any] = {"user_id": user_id}
        if item_type:
            query["item_type"] = item_type
        if tag:
            query["tags"] = tag
        if collection_id:
            query["collection_ids"] = collection_id

        cursor = (
            self._items.find(query)
            .sort("created_at", -1)
            .skip(offset)
            .limit(limit)
        )
        items = []
        async for doc in cursor:
            doc["_id"] = str(doc["_id"])
            items.append(doc)
        return items

    async def update_item(
        self,
        item_id: str,
        title: str | None = None,
        content: str | None = None,
        tags: list[str] | None = None,
        collection_ids: list[str] | None = None,
    ) -> dict[str, Any] | None:
        """Update a vault item. Returns updated item or None if not found."""
        from bson import ObjectId

        update_fields: dict[str, Any] = {"updated_at": datetime.utcnow()}
        if title is not None:
            update_fields["title"] = title
        if content is not None:
            update_fields["content"] = content
        if tags is not None:
            update_fields["tags"] = tags
        if collection_ids is not None:
            update_fields["collection_ids"] = collection_ids

        result = await self._items.update_one(
            {"_id": ObjectId(item_id)},
            {"$set": update_fields},
        )
        if result.matched_count == 0:
            return None
        return await self.get_item(item_id)

    async def delete_item(self, item_id: str) -> bool:
        """Delete a vault item. Returns True if deleted."""
        from bson import ObjectId

        result = await self._items.delete_one({"_id": ObjectId(item_id)})
        return result.deleted_count > 0

    async def add_item_to_collection(self, item_id: str, collection_id: str) -> bool:
        """Add a collection reference to an item."""
        from bson import ObjectId

        result = await self._items.update_one(
            {"_id": ObjectId(item_id)},
            {
                "$addToSet": {"collection_ids": collection_id},
                "$set": {"updated_at": datetime.utcnow()},
            },
        )
        return result.matched_count > 0

    async def remove_item_from_collection(self, item_id: str, collection_id: str) -> bool:
        """Remove a collection reference from an item."""
        from bson import ObjectId

        result = await self._items.update_one(
            {"_id": ObjectId(item_id)},
            {
                "$pull": {"collection_ids": collection_id},
                "$set": {"updated_at": datetime.utcnow()},
            },
        )
        return result.matched_count > 0

    # ── Collections ──────────────────────────────────────────────────────────

    async def create_collection(
        self,
        user_id: str,
        name: str,
        description: str = "",
        color: str = "#3B82F6",
    ) -> dict[str, Any]:
        """Create a new vault collection."""
        await self._ensure_indexes()
        now = datetime.utcnow()
        doc = {
            "user_id": user_id,
            "name": name,
            "description": description,
            "color": color,
            "item_count": 0,
            "created_at": now,
            "updated_at": now,
        }
        result = await self._collections.insert_one(doc)
        doc["_id"] = str(result.inserted_id)
        return doc

    async def get_collection(self, collection_id: str) -> dict[str, Any] | None:
        """Get a single collection by ID."""
        from bson import ObjectId

        doc = await self._collections.find_one({"_id": ObjectId(collection_id)})
        if doc:
            doc["_id"] = str(doc["_id"])
        return doc

    async def list_collections(self, user_id: str) -> list[dict[str, Any]]:
        """List all collections for a user."""
        await self._ensure_indexes()
        cursor = self._collections.find({"user_id": user_id}).sort("created_at", -1)
        collections = []
        async for doc in cursor:
            doc["_id"] = str(doc["_id"])
            collections.append(doc)
        return collections

    async def update_collection(
        self,
        collection_id: str,
        name: str | None = None,
        description: str | None = None,
        color: str | None = None,
    ) -> dict[str, Any] | None:
        """Update a collection. Returns updated collection or None."""
        from bson import ObjectId

        update_fields: dict[str, Any] = {"updated_at": datetime.utcnow()}
        if name is not None:
            update_fields["name"] = name
        if description is not None:
            update_fields["description"] = description
        if color is not None:
            update_fields["color"] = color

        result = await self._collections.update_one(
            {"_id": ObjectId(collection_id)},
            {"$set": update_fields},
        )
        if result.matched_count == 0:
            return None
        return await self.get_collection(collection_id)

    async def delete_collection(self, collection_id: str) -> bool:
        """
        Delete a collection. Items are unlinked but not deleted.
        Returns True if deleted.
        """
        from bson import ObjectId

        # Remove collection reference from all items
        await self._items.update_many(
            {"collection_ids": collection_id},
            {"$pull": {"collection_ids": collection_id}},
        )

        result = await self._collections.delete_one({"_id": ObjectId(collection_id)})
        return result.deleted_count > 0

    async def recalculate_item_count(self, collection_id: str) -> int:
        """Recalculate and update the item_count for a collection."""
        from bson import ObjectId

        count = await self._items.count_documents(
            {"collection_ids": collection_id}
        )
        await self._collections.update_one(
            {"_id": ObjectId(collection_id)},
            {"$set": {"item_count": count, "updated_at": datetime.utcnow()}},
        )
        return count
