"""Embedding providers: local (SentenceTransformers), HuggingFace Inference API, Cohere."""

from src.ml.embedding.embedder import (
    BaseEmbedder,
    LocalEmbedder,
    DualEmbedderV2,
    HuggingFaceEmbedder,
    CohereEmbedder,
    create_embedder,
    get_embedder,
    reset_embedder,
    embed_texts,
    embed_query,
)

__all__ = [
    "BaseEmbedder",
    "LocalEmbedder",
    "DualEmbedderV2",
    "HuggingFaceEmbedder",
    "CohereEmbedder",
    "create_embedder",
    "get_embedder",
    "reset_embedder",
    "embed_texts",
    "embed_query",
]
