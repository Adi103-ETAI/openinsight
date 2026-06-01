"""Pydantic models for Research Vault API."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


# ── Item Models ──────────────────────────────────────────────────────────────

class VaultItemCreate(BaseModel):
    """Request to create a new vault item."""
    item_type: str = Field(
        ...,
        pattern="^(citation|search_result|note|evidence)$",
        description="Type of vault item",
    )
    title: str = Field(..., max_length=500, description="Item title")
    content: str = Field(default="", max_length=10000, description="Item content/text")
    source_type: Optional[str] = Field(None, description="Source type (pubmed, cochrane, etc.)")
    source_url: Optional[str] = Field(None, description="DOI, PMID, or URL")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Flexible metadata")
    tags: list[str] = Field(default_factory=list, description="User-defined tags")
    collection_ids: list[str] = Field(default_factory=list, description="Collection IDs to add to")


class VaultItemUpdate(BaseModel):
    """Request to update an existing vault item."""
    title: Optional[str] = Field(None, max_length=500)
    content: Optional[str] = Field(None, max_length=10000)
    tags: Optional[list[str]] = None
    collection_ids: Optional[list[str]] = None


class VaultItemResponse(BaseModel):
    """Response for a vault item."""
    id: str
    user_id: str
    item_type: str
    title: str
    content: str
    source_type: Optional[str] = None
    source_url: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)
    collection_ids: list[str] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime


# ── Collection Models ────────────────────────────────────────────────────────

class VaultCollectionCreate(BaseModel):
    """Request to create a new collection."""
    name: str = Field(..., max_length=200, description="Collection name")
    description: str = Field(default="", max_length=1000, description="Description")
    color: str = Field(default="#3B82F6", description="UI color hint")


class VaultCollectionUpdate(BaseModel):
    """Request to update a collection."""
    name: Optional[str] = Field(None, max_length=200)
    description: Optional[str] = Field(None, max_length=1000)
    color: Optional[str] = None


class VaultCollectionResponse(BaseModel):
    """Response for a collection."""
    id: str
    user_id: str
    name: str
    description: str
    color: str
    item_count: int
    created_at: datetime
    updated_at: datetime


# ── Search Integration ───────────────────────────────────────────────────────

class SaveToVaultRequest(BaseModel):
    """Request to save search results to vault from the search endpoint."""
    item_type: str = Field(default="search_result", pattern="^(citation|search_result|note|evidence)$")
    title: str = Field(..., max_length=500)
    content: str = Field(default="", max_length=10000)
    source_type: Optional[str] = None
    source_url: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)
    collection_id: Optional[str] = Field(None, description="Collection to add item to")
