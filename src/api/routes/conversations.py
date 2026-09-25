"""
Conversations API — persistence-first chat threads (BIS Stage 5 port).

Owner-scoped by X-User-ID header until session-cookie auth lands
(same placeholder as vault). All lookups are owner-or-404.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request
from loguru import logger
from pydantic import BaseModel, Field

from src.config.settings import get_settings
from src.data.mongo.conversation_store import ConversationStore

router = APIRouter()
settings = get_settings()


def _store(request: Request) -> ConversationStore:
    store = getattr(request.app.state, "conversation_store", None)
    if store is None:
        store = ConversationStore(mongo_url=settings.mongodb_url, db_name=settings.mongodb_db)
        request.app.state.conversation_store = store
    return store


def _user_id(request: Request) -> str:
    return request.headers.get("X-User-ID", "default_user")


class CreateBody(BaseModel):
    title: str | None = None


class ImportBody(BaseModel):
    entries: list[dict] = Field(default_factory=list)


@router.post("")
async def create_conversation(body: CreateBody, request: Request):
    conv = await _store(request).create_conversation(_user_id(request), body.title)
    return conv


@router.get("")
async def list_conversations(
    request: Request, status: str = "active", page: int = Query(1, ge=1), limit: int = Query(20, ge=1, le=100)
):
    items, total = await _store(request).list_conversations(_user_id(request), status, page, limit)
    return {"items": items, "page": page, "limit": limit, "total": total}


@router.get("/{conversation_id}")
async def get_conversation(conversation_id: str, request: Request):
    conv = await _store(request).owned_conversation(_user_id(request), conversation_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conv


@router.get("/{conversation_id}/messages")
async def list_messages(
    conversation_id: str, request: Request, page: int = Query(1, ge=1), limit: int = Query(50, ge=1, le=100)
):
    store = _store(request)
    if await store.owned_conversation(_user_id(request), conversation_id) is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    items, total = await store.list_messages(conversation_id, page, limit)
    return {"items": items, "page": page, "limit": limit, "total": total}


@router.post("/{conversation_id}/messages")
async def ask(conversation_id: str, body: dict, request: Request):
    """Persist a completed search turn to the thread (called after /search)."""
    store = _store(request)
    if await store.owned_conversation(_user_id(request), conversation_id) is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    try:
        assistant = await store.add_messages(
            conversation_id,
            str(body.get("query", "")),
            str(body.get("answer", "")),
            body.get("citations", []) if isinstance(body.get("citations"), list) else [],
            int(body.get("chunks_retrieved", 0)),
            client_id=str(body.get("client_id", "") or "") or None,
        )
    except Exception as e:
        logger.error(f"Failed to persist conversation message: {e}")
        raise HTTPException(status_code=500, detail="Failed to persist message")
    return assistant


@router.post("/import")
async def import_history(body: ImportBody, request: Request):
    if len(body.entries) > 50:
        raise HTTPException(status_code=422, detail="at most 50 entries per call")
    imported, skipped = await _store(request).import_entries(_user_id(request), body.entries)
    return {"imported": imported, "skipped": skipped}


@router.post("/{conversation_id}/messages:stream")
async def stream_reserved():
    raise HTTPException(status_code=501, detail="Streaming ships post-MVP")
