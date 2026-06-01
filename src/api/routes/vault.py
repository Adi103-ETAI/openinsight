"""
Research Vault API — CRUD for saved citations, notes, and collections.
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from loguru import logger

from src.api.models.vault import (
    VaultCollectionCreate,
    VaultCollectionResponse,
    VaultCollectionUpdate,
    VaultItemCreate,
    VaultItemResponse,
    VaultItemUpdate,
)
from src.config.settings import get_settings
from src.data.mongo.vault_store import VaultStore

router = APIRouter()
settings = get_settings()


def _get_vault_store(request: Request) -> VaultStore:
    """Get or create VaultStore singleton from app state."""
    store = getattr(request.app.state, "vault_store", None)
    if store is None:
        store = VaultStore(
            mongo_url=settings.mongodb_url,
            db_name=settings.mongodb_db,
        )
        request.app.state.vault_store = store
    return store


def _get_user_id(request: Request) -> str:
    """
    Extract user ID from request.
    In production, this would come from auth middleware.
    For now, uses a header or defaults to 'default_user'.
    """
    return request.headers.get("X-User-ID", "default_user")


# ── Items ────────────────────────────────────────────────────────────────────


@router.post("/items", response_model=VaultItemResponse, status_code=201)
async def create_item(payload: VaultItemCreate, request: Request) -> VaultItemResponse:
    """Save a new item to the vault."""
    user_id = _get_user_id(request)
    store = _get_vault_store(request)

    try:
        item = await store.create_item(
            user_id=user_id,
            item_type=payload.item_type,
            title=payload.title,
            content=payload.content,
            source_type=payload.source_type,
            source_url=payload.source_url,
            metadata=payload.metadata,
            tags=payload.tags,
            collection_ids=payload.collection_ids,
        )
        return VaultItemResponse(**item)
    except Exception as e:
        logger.error(f"Failed to create vault item: {e}")
        raise HTTPException(status_code=500, detail="Failed to create vault item")


@router.get("/items", response_model=list[VaultItemResponse])
async def list_items(
    request: Request,
    item_type: str | None = Query(None, description="Filter by item type"),
    tag: str | None = Query(None, description="Filter by tag"),
    collection_id: str | None = Query(None, description="Filter by collection ID"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> list[VaultItemResponse]:
    """List vault items with optional filters."""
    user_id = _get_user_id(request)
    store = _get_vault_store(request)

    try:
        items = await store.list_items(
            user_id=user_id,
            item_type=item_type,
            tag=tag,
            collection_id=collection_id,
            limit=limit,
            offset=offset,
        )
        return [VaultItemResponse(**item) for item in items]
    except Exception as e:
        logger.error(f"Failed to list vault items: {e}")
        raise HTTPException(status_code=500, detail="Failed to list vault items")


@router.get("/items/{item_id}", response_model=VaultItemResponse)
async def get_item(item_id: str, request: Request) -> VaultItemResponse:
    """Get a single vault item by ID."""
    store = _get_vault_store(request)

    item = await store.get_item(item_id)
    if not item:
        raise HTTPException(status_code=404, detail="Vault item not found")
    return VaultItemResponse(**item)


@router.put("/items/{item_id}", response_model=VaultItemResponse)
async def update_item(
    item_id: str,
    payload: VaultItemUpdate,
    request: Request,
) -> VaultItemResponse:
    """Update a vault item."""
    store = _get_vault_store(request)

    item = await store.update_item(
        item_id=item_id,
        title=payload.title,
        content=payload.content,
        tags=payload.tags,
        collection_ids=payload.collection_ids,
    )
    if not item:
        raise HTTPException(status_code=404, detail="Vault item not found")
    return VaultItemResponse(**item)


@router.delete("/items/{item_id}", status_code=204)
async def delete_item(item_id: str, request: Request) -> None:
    """Delete a vault item."""
    store = _get_vault_store(request)

    deleted = await store.delete_item(item_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Vault item not found")


# ── Item-Collection Links ────────────────────────────────────────────────────


@router.post("/items/{item_id}/collections/{collection_id}", status_code=204)
async def add_item_to_collection(
    item_id: str,
    collection_id: str,
    request: Request,
) -> None:
    """Add an item to a collection."""
    store = _get_vault_store(request)

    success = await store.add_item_to_collection(item_id, collection_id)
    if not success:
        raise HTTPException(status_code=404, detail="Item not found")

    await store.recalculate_item_count(collection_id)


@router.delete("/items/{item_id}/collections/{collection_id}", status_code=204)
async def remove_item_from_collection(
    item_id: str,
    collection_id: str,
    request: Request,
) -> None:
    """Remove an item from a collection."""
    store = _get_vault_store(request)

    success = await store.remove_item_from_collection(item_id, collection_id)
    if not success:
        raise HTTPException(status_code=404, detail="Item not found")

    await store.recalculate_item_count(collection_id)


# ── Collections ──────────────────────────────────────────────────────────────


@router.post("/collections", response_model=VaultCollectionResponse, status_code=201)
async def create_collection(
    payload: VaultCollectionCreate,
    request: Request,
) -> VaultCollectionResponse:
    """Create a new collection."""
    user_id = _get_user_id(request)
    store = _get_vault_store(request)

    try:
        collection = await store.create_collection(
            user_id=user_id,
            name=payload.name,
            description=payload.description,
            color=payload.color,
        )
        return VaultCollectionResponse(**collection)
    except Exception as e:
        logger.error(f"Failed to create collection: {e}")
        raise HTTPException(status_code=500, detail="Failed to create collection")


@router.get("/collections", response_model=list[VaultCollectionResponse])
async def list_collections(request: Request) -> list[VaultCollectionResponse]:
    """List all collections for the user."""
    user_id = _get_user_id(request)
    store = _get_vault_store(request)

    try:
        collections = await store.list_collections(user_id=user_id)
        return [VaultCollectionResponse(**c) for c in collections]
    except Exception as e:
        logger.error(f"Failed to list collections: {e}")
        raise HTTPException(status_code=500, detail="Failed to list collections")


@router.get("/collections/{collection_id}", response_model=VaultCollectionResponse)
async def get_collection(
    collection_id: str,
    request: Request,
) -> VaultCollectionResponse:
    """Get a single collection by ID."""
    store = _get_vault_store(request)

    collection = await store.get_collection(collection_id)
    if not collection:
        raise HTTPException(status_code=404, detail="Collection not found")
    return VaultCollectionResponse(**collection)


@router.put("/collections/{collection_id}", response_model=VaultCollectionResponse)
async def update_collection(
    collection_id: str,
    payload: VaultCollectionUpdate,
    request: Request,
) -> VaultCollectionResponse:
    """Update a collection."""
    store = _get_vault_store(request)

    collection = await store.update_collection(
        collection_id=collection_id,
        name=payload.name,
        description=payload.description,
        color=payload.color,
    )
    if not collection:
        raise HTTPException(status_code=404, detail="Collection not found")
    return VaultCollectionResponse(**collection)


@router.delete("/collections/{collection_id}", status_code=204)
async def delete_collection(collection_id: str, request: Request) -> None:
    """Delete a collection (items are unlinked, not deleted)."""
    store = _get_vault_store(request)

    deleted = await store.delete_collection(collection_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Collection not found")
