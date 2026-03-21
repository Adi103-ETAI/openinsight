"""
Embedding Generator
Loads pritamdeka/S-PubMedBert-MS-MARCO once and reuses it.
Drop-in swap: change EMBEDDING_MODEL in .env to switch models.
"""
from sentence_transformers import SentenceTransformer
from src.core.config import get_settings
from loguru import logger
from functools import lru_cache

settings = get_settings()


@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    logger.info(f"Loading embedding model: {settings.embedding_model}")
    model = SentenceTransformer(settings.embedding_model)
    logger.info("Embedding model loaded.")
    return model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts. Returns list of float vectors."""
    model = get_embedder()
    vectors = model.encode(texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
    return vectors.tolist()


def embed_query(text: str) -> list[float]:
    """Embed a single query string."""
    model = get_embedder()
    vector = model.encode([text], normalize_embeddings=True)
    return vector[0].tolist()
