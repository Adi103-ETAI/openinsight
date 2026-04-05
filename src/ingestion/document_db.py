"""
Document DB — MongoDB
Stores raw + parsed documents before they go into the vector index.
Collections:
  - documents   : raw ingested docs (PDF text, PubMed XML, etc.)
  - chunks      : split passages ready for embedding
  - sources     : metadata about each knowledge source
"""
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
from src.core.config import get_settings

settings = get_settings()

# ── Client ──────────────────────────────────────────────────────────────────
_client: Optional[AsyncIOMotorClient] = None


def get_db():
    global _client
    if _client is None:
        _client = AsyncIOMotorClient(settings.mongodb_url)
    return _client[settings.mongodb_db]


# ── Document model ───────────────────────────────────────────────────────────
class DocumentRecord(BaseModel):
    """A single ingested source document."""
    source_type: str          # "icmr" | "pubmed" | "cochrane" | "who" | "cdc" | "statpearls"
    title: str
    content: str              # full raw text
    url: Optional[str] = None
    doi: Optional[str] = None
    published_date: Optional[str] = None
    condition_tags: list[str] = Field(default_factory=list)
    specialty_tags: list[str] = Field(default_factory=list)
    ingested_at: datetime = Field(default_factory=datetime.utcnow)
    year: Optional[int] = None
    journal: Optional[str] = None
    study_type: Optional[str] = None
    population: Optional[str] = None
    evidence_level: int = 5
    is_india_specific: bool = False
    parser_version: str = "v1"
    total_chunks: int = 0
    # Deduplication fields
    content_hash: Optional[str] = None   # SHA-256 of normalised content
    is_duplicate: bool = False
    duplicate_of: Optional[str] = None   # document_id of canonical copy


# ── Chunk model ──────────────────────────────────────────────────────────────
class ChunkRecord(BaseModel):
    """A passage-level chunk, ready to be embedded and stored in Qdrant."""
    document_id: str          # ref to DocumentRecord._id
    source_type: str
    title: str
    chunk_text: str
    chunk_index: int          # position within parent document
    condition_tags: list[str] = Field(default_factory=list)
    specialty_tags: list[str] = Field(default_factory=list)
    char_count: int = 0
    embedded: bool = False    # flipped to True once stored in Qdrant
    embedded_at: Optional[datetime] = None
    section: Optional[str] = None
    diseases: list[str] = Field(default_factory=list)
    drugs: list[str] = Field(default_factory=list)
    symptoms: list[str] = Field(default_factory=list)
    dosages: list[str] = Field(default_factory=list)
    contraindications: list[str] = Field(default_factory=list)
    patient_populations: list[str] = Field(default_factory=list)
    outcomes: list[str] = Field(default_factory=list)
    has_safety_flag: bool = False
    content_type: str = "unknown"
    content_weight: float = 1.0
    quality_score: float = 1.0   # 0–1 overall chunk quality
    is_india_specific: bool = False
    evidence_level: int = 5
    page_number: Optional[int] = None
    token_count: int = 0
    parser_version: str = "v1"
