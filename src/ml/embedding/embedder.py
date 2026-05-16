from __future__ import annotations

import hashlib
import re
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from functools import lru_cache
from typing import Any

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from src.config.settings import get_settings
from loguru import logger


# ---------------------------------------------------------------------------
# Abstract base for dense embedders
# ---------------------------------------------------------------------------

class BaseEmbedder(ABC):
    """Abstract base class for dense embedding providers."""

    @abstractmethod
    def embed_batch(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Embed a batch of texts. Returns (N, dim) numpy array."""

    @abstractmethod
    def embed_query(self, query_text: str) -> np.ndarray:
        """Embed a single query string. Returns (dim,) numpy array."""

    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension."""

    def compute_sparse_vector(self, text: str) -> dict[str, list[float] | list[int]]:
        """Compute TF-IDF sparse vector (CPU-only, same for all providers)."""
        # Delegate to shared sparse implementation
        return _compute_sparse_vector(text, self._sparse_vocab_size)

    _sparse_vocab_size: int = 50000  # overridden in __init__


def _compute_sparse_vector(
    text: str, vocab_size: int
) -> dict[str, list[float] | list[int]]:
    """Standalone sparse vector computation (CPU-only, no GPU needed)."""
    tokens = _medical_tokenize(text)
    if not tokens:
        return {"indices": [], "values": []}

    tf = Counter(tokens)
    total_tokens = len(tokens)
    weight_by_index: dict[int, float] = defaultdict(float)

    for term, count in tf.items():
        term_idx = _term_to_index(term, vocab_size)
        tf_norm = count / max(1, total_tokens)
        idf_weight = _get_idf_weight(term)
        weight = tf_norm * idf_weight
        if weight > 0.001:
            weight_by_index[term_idx] += float(weight)

    if not weight_by_index:
        return {"indices": [], "values": []}

    sorted_indices = sorted(weight_by_index.keys())
    sorted_values = [weight_by_index[idx] for idx in sorted_indices]

    return {"indices": sorted_indices, "values": sorted_values}


# ---------------------------------------------------------------------------
# Shared sparse-tokenization helpers (used by all embedders)
# ---------------------------------------------------------------------------

MEDICAL_COMPOUNDS = [
    "type 2 diabetes",
    "type 1 diabetes",
    "heart failure",
    "blood pressure",
    "myocardial infarction",
    "atrial fibrillation",
    "blood glucose",
    "hemoglobin a1c",
    "hba1c",
    "randomized controlled trial",
    "systematic review",
    "meta analysis",
    "meta-analysis",
    "insulin resistance",
    "glycemic control",
    "renal failure",
    "coronary artery disease",
    "coronary heart disease",
]

STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "this", "that", "these",
    "those", "it", "its", "we", "our", "they", "their", "as", "if", "than",
    "more", "most", "such",
}


def _medical_tokenize(text: str) -> list[str]:
    text_lower = text.lower()
    compound_tokens: list[str] = []

    for compound in MEDICAL_COMPOUNDS:
        token_version = compound.replace(" ", "_")
        if compound in text_lower:
            text_lower = text_lower.replace(compound, f" {token_version} ")
            compound_tokens.append(token_version)

    words = re.findall(r"\b[a-z][a-z0-9\-]{2,}\b", text_lower)
    words = [w for w in words if w not in STOPWORDS]
    return compound_tokens + words


def _term_to_index(term: str, vocab_size: int) -> int:
    digest = hashlib.sha256(term.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % vocab_size


def _get_idf_weight(term: str) -> float:
    if "_" in term:
        return 3.5
    if len(term) > 10:
        return 3.0
    if len(term) > 6:
        return 2.0
    return 1.0


# ---------------------------------------------------------------------------
# Local embedder (SentenceTransformers, requires GPU for good perf)
# ---------------------------------------------------------------------------

class LocalEmbedder(BaseEmbedder):
    """
    Dense + sparse embedder using SentenceTransformers locally.

    Dense embeddings are generated from contextual_text so chunk semantics
    include source, type, title, and section context.
    """

    def __init__(self, dense_model_name: str = "pritamdeka/S-PubMedBert-MS-MARCO"):
        from src.config.settings import get_settings
        settings = get_settings()
        self._sparse_vocab_size = settings.sparse_vocab_size
        self._dim = settings.embedding_dim

        self.dense_model = SentenceTransformer(dense_model_name)
        self.dense_model.eval()
        if torch.cuda.is_available():
            self.dense_model = self.dense_model.cuda()

    def dimension(self) -> int:
        return self._dim

    def embed_batch(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        with torch.inference_mode():
            embeddings = self.dense_model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=True,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
        return embeddings

    def embed_query(self, query_text: str) -> np.ndarray:
        with torch.inference_mode():
            embedding = self.dense_model.encode(
                query_text,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
        return embedding


# ---------------------------------------------------------------------------
# HuggingFace Inference API embedder
# ---------------------------------------------------------------------------

class HuggingFaceEmbedder(BaseEmbedder):
    """
    Dense embedder using HuggingFace Inference API (free tier).

    Uses the /feature-extraction endpoint to get embeddings from the same
    model used during ingestion (e.g. S-PubMedBert), ensuring embedding
    consistency between ingestion and query time.

    Free tier limits:
    - Rate limit: 300 requests/hour (with HF token)
    - Models up to 10 GB supported on free tier
    - No GPU needed on client side
    """

    API_URL_TEMPLATE = "https://api-inference.huggingface.co/pipeline/feature-extraction/{}"

    def __init__(
        self,
        model_name: str = "pritamdeka/S-PubMedBert-MS-MARCO",
        api_token: str = "",
    ):
        from src.config.settings import get_settings
        settings = get_settings()
        self._sparse_vocab_size = settings.sparse_vocab_size
        self._dim = settings.embedding_dim
        self._model_name = model_name
        self._api_token = api_token
        self._api_url = self.API_URL_TEMPLATE.format(model_name)
        self._headers = {"Authorization": f"Bearer {api_token}"}
        logger.info(f"[HFEmbedder] Initialized with model={model_name}")

    def dimension(self) -> int:
        return self._dim

    def embed_query(self, query_text: str) -> np.ndarray:
        import httpx

        payload = {"inputs": query_text}
        with httpx.Client(timeout=60.0) as client:
            response = client.post(
                self._api_url, headers=self._headers, json=payload
            )
            response.raise_for_status()
            data = response.json()

        embedding = self._parse_embedding_response(data)
        # Normalize the embedding
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding

    def embed_batch(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Embed texts one by one via HF API (no batch endpoint on free tier).
        
        Raises RuntimeError if ALL embeddings fail, so the pipeline can retry
        instead of silently indexing zero vectors.
        """
        embeddings = []
        failed_count = 0
        for i, text in enumerate(texts):
            try:
                emb = self.embed_query(text)
                embeddings.append(emb)
            except Exception as e:
                logger.warning(f"[HFEmbedder] Failed to embed text {i}: {e}")
                embeddings.append(np.zeros(self._dim, dtype=np.float32))
                failed_count += 1

            # Respect rate limits: small delay between requests
            if (i + 1) % 10 == 0:
                import time
                time.sleep(1.0)

        # If ALL embeddings failed, raise an error so the pipeline can retry
        # instead of silently indexing useless zero vectors
        if failed_count == len(texts) and len(texts) > 0:
            raise RuntimeError(
                f"[HFEmbedder] All {len(texts)} embeddings failed. "
                f"Check that the model '{self._model_name}' has a deployed "
                f"Inference API endpoint on HuggingFace. "
                f"For GPU ingestion (Kaggle/Colab), use EMBED_PROVIDER=local instead."
            )

        if failed_count > 0:
            logger.warning(
                f"[HFEmbedder] {failed_count}/{len(texts)} embeddings failed, "
                f"using zero vectors as fallback"
            )

        return np.array(embeddings, dtype=np.float32)

    def _parse_embedding_response(self, data: Any) -> np.ndarray:
        """Parse HF feature-extraction response into a 1D numpy array."""
        arr = np.array(data, dtype=np.float32)

        # HF returns different shapes depending on the model:
        # - Some return (1, seq_len, dim) — take mean over seq_len
        # - Some return (seq_len, dim) — take mean over seq_len
        # - Some return (dim,) — use directly
        if arr.ndim == 3:
            # (1, seq_len, dim) → mean pool over seq_len → (dim,)
            arr = arr.squeeze(0).mean(axis=0)
        elif arr.ndim == 2:
            # (seq_len, dim) → mean pool over seq_len → (dim,)
            arr = arr.mean(axis=0)
        elif arr.ndim == 1:
            pass  # already (dim,)
        else:
            logger.warning(f"[HFEmbedder] Unexpected embedding shape: {arr.shape}")
            arr = arr.flatten()

        return arr


# ---------------------------------------------------------------------------
# Cohere embedder
# ---------------------------------------------------------------------------

class CohereEmbedder(BaseEmbedder):
    """
    Dense embedder using Cohere Embed API.

    Free tier: 1,000 calls/month (trial key).
    Note: Cohere embeddings have different vector space than S-PubMedBert,
    so DO NOT mix with S-PubMedBert-indexed vectors in the same collection.
    Use this ONLY if you also used Cohere during ingestion.
    """

    API_URL = "https://api.cohere.ai/v1/embed"

    def __init__(
        self,
        api_key: str = "",
        model: str = "embed-english-v3.0",
    ):
        from src.config.settings import get_settings
        settings = get_settings()
        self._sparse_vocab_size = settings.sparse_vocab_size
        self._dim = 1024  # Cohere embed-english-v3.0 outputs 1024d
        self._api_key = api_key
        self._model = model
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        logger.info(f"[CohereEmbedder] Initialized with model={model}")

    def dimension(self) -> int:
        return self._dim

    def embed_query(self, query_text: str) -> np.ndarray:
        import httpx

        payload = {
            "texts": [query_text],
            "model": self._model,
            "input_type": "search_query",
            "embedding_types": ["float"],
        }
        with httpx.Client(timeout=60.0) as client:
            response = client.post(self.API_URL, headers=self._headers, json=payload)
            response.raise_for_status()
            data = response.json()

        embeddings = data.get("embeddings", {}).get("float", [])
        if embeddings and len(embeddings) > 0:
            return np.array(embeddings[0], dtype=np.float32)
        return np.zeros(self._dim, dtype=np.float32)

    def embed_batch(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Embed batch using Cohere API (supports batch natively)."""
        import httpx

        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            payload = {
                "texts": batch,
                "model": self._model,
                "input_type": "search_document",
                "embedding_types": ["float"],
            }
            with httpx.Client(timeout=60.0) as client:
                response = client.post(self.API_URL, headers=self._headers, json=payload)
                response.raise_for_status()
                data = response.json()

            embeddings = data.get("embeddings", {}).get("float", [])
            all_embeddings.extend(embeddings)

        return np.array(all_embeddings, dtype=np.float32)


# ---------------------------------------------------------------------------
# Backward-compatible alias
# ---------------------------------------------------------------------------

DualEmbedderV2 = LocalEmbedder
"""Alias for backward compatibility. Use LocalEmbedder in new code."""


# ---------------------------------------------------------------------------
# Factory: create the right embedder based on config
# ---------------------------------------------------------------------------

def create_embedder(provider: str | None = None) -> BaseEmbedder:
    """
    Factory function to create an embedder based on the configured provider.

    Args:
        provider: Override the EMBED_PROVIDER setting. One of:
            "local", "huggingface", "cohere". If None, reads from settings.

    Returns:
        BaseEmbedder instance appropriate for the provider.
    """
    settings = get_settings()
    provider = provider or settings.embed_provider
    provider = provider.lower().strip()

    if provider == "local":
        model_name = settings.dense_model_name or settings.embedding_model
        logger.info(f"[EmbedderFactory] Creating LocalEmbedder with model={model_name}")
        return LocalEmbedder(dense_model_name=model_name)

    elif provider == "huggingface":
        model_name = settings.hf_embed_model or settings.dense_model_name
        api_token = settings.hf_api_token
        if not api_token:
            raise ValueError(
                "HF_API_TOKEN is required for HuggingFace embedding provider. "
                "Set it in .env or environment variables."
            )
        logger.info(f"[EmbedderFactory] Creating HuggingFaceEmbedder with model={model_name}")
        return HuggingFaceEmbedder(model_name=model_name, api_token=api_token)

    elif provider == "cohere":
        api_key = settings.cohere_api_key
        if not api_key:
            raise ValueError(
                "COHERE_API_KEY is required for Cohere embedding provider. "
                "Set it in .env or environment variables."
            )
        model = settings.cohere_embed_model
        logger.info(f"[EmbedderFactory] Creating CohereEmbedder with model={model}")
        return CohereEmbedder(api_key=api_key, model=model)

    else:
        raise ValueError(
            f"Unknown embed_provider: '{provider}'. "
            f"Supported providers: local, huggingface, cohere"
        )


# ---------------------------------------------------------------------------
# Singleton & convenience functions (backward-compatible)
# ---------------------------------------------------------------------------

_embedder_instance: BaseEmbedder | None = None


def get_embedder() -> BaseEmbedder:
    """Get singleton embedder instance based on EMBED_PROVIDER setting."""
    global _embedder_instance
    if _embedder_instance is None:
        settings = get_settings()
        logger.info(f"Loading embedding provider: {settings.embed_provider}")
        _embedder_instance = create_embedder()
    return _embedder_instance


def reset_embedder() -> None:
    """Reset the singleton embedder (useful after config changes)."""
    global _embedder_instance
    _embedder_instance = None


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts. Returns list of float vectors."""
    model = get_embedder()
    vectors = model.embed_batch(texts, batch_size=32)
    return vectors.tolist()


def embed_query(text: str) -> list[float]:
    """Embed a single query string."""
    model = get_embedder()
    vector = model.embed_query(text)
    return vector.tolist()
