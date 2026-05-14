# (A) Implementation: Code Patches for RemoteEmbedder & RemoteReranker

This guide provides exact code patches to add pluggable embeddings and reranking with remote API support (Cohere, Hugging Face Inference).

---

## Table of Contents
1. Phase 0: Config updates (`settings.py`)
2. Phase 2: Embedder abstraction & `RemoteEmbedder`
3. Phase 3: `RemoteReranker`
4. Phase 4: Wire into retriever and search routes
5. Testing helpers

---

## Phase 0: Update `src/config/settings.py`

Add these config keys to the `Settings` class:

```python
# ===================== Embedding Provider =====================
embed_provider: str = "local"  # options: 'local', 'cohere', 'hf'
embed_api_key: str = ""
embed_api_url: str = "https://api.cohere.ai/v1/embed"
embed_batch_size: int = 32
embed_timeout: int = 30
embed_fallback_to_local: bool = True

# ===================== Reranking Provider =====================
rerank_provider: str = "local"  # options: 'local', 'cohere', 'hf'
rerank_api_key: str = ""
rerank_api_url: str = "https://api.cohere.ai/v1/rerank"
rerank_timeout: int = 30
rerank_fallback_to_local: bool = True
```

Example addition in `src/config/settings.py` (after line ~87, in the Reranking section):

```python
    # ===================== Reranking =====================
    reranker_top_n: int = 8
    reranker_batch_size: int = 16
    reranker_max_chars: int = 1200

    # ===================== Embedding Provider =====================
    embed_provider: str = "local"  # options: 'local', 'cohere', 'hf'
    embed_api_key: str = ""
    embed_api_url: str = "https://api.cohere.ai/v1/embed"
    embed_batch_size: int = 32
    embed_timeout: int = 30
    embed_fallback_to_local: bool = True

    # ===================== Reranking Provider =====================
    rerank_provider: str = "local"  # options: 'local', 'cohere', 'hf'
    rerank_api_key: str = ""
    rerank_api_url: str = "https://api.cohere.ai/v1/rerank"
    rerank_timeout: int = 30
    rerank_fallback_to_local: bool = True
```

---

## Phase 2: Embedder Abstraction

### Step 1: Create `src/ml/embedding/base.py` (interface)

```python
from abc import ABC, abstractmethod
import numpy as np


class Embedder(ABC):
    """Abstract base class for embedding providers."""

    @abstractmethod
    def embed_batch(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """
        Embed a batch of texts.
        
        Args:
            texts: List of text strings to embed.
            batch_size: Batch size for processing (may be ignored by remote APIs).
        
        Returns:
            np.ndarray of shape (len(texts), embedding_dim), normalized if needed.
        """
        pass

    @abstractmethod
    def embed_query(self, query_text: str) -> np.ndarray:
        """
        Embed a single query string.
        
        Args:
            query_text: Query string.
        
        Returns:
            np.ndarray of shape (embedding_dim,), normalized if needed.
        """
        pass

    @abstractmethod
    def compute_sparse_vector(self, text: str) -> dict[str, list[float] | list[int]]:
        """
        Compute sparse vector representation.
        
        Args:
            text: Input text.
        
        Returns:
            Dict with 'indices' and 'values' keys.
        """
        pass
```

### Step 2: Keep `src/ml/embedding/embedder.py` as LocalEmbedder

Refactor the existing `DualEmbedderV2` class to inherit from `Embedder`:

```python
from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict
from functools import lru_cache

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from loguru import logger

from src.config.settings import get_settings
from src.ml.embedding.base import Embedder


class DualEmbedderV2(Embedder):
    """
    Dense + sparse embedder for OpenInsight v2 ingestion/search.
    
    This is the local PyTorch-based embedder (inherits from Embedder interface).
    """

    def __init__(self, dense_model_name: str = "pritamdeka/S-PubMedBert-MS-MARCO"):
        from src.config.settings import get_settings
        settings = get_settings()
        self.VOCAB_SIZE = settings.sparse_vocab_size

        self.dense_model = SentenceTransformer(dense_model_name)
        self.dense_model.eval()
        if torch.cuda.is_available():
            self.dense_model = self.dense_model.cuda()

    # ... rest of existing DualEmbedderV2 implementation remains unchanged ...
```

### Step 3: Create `src/ml/embedding/remote.py` (Cohere / HF)

```python
from __future__ import annotations

import httpx
import numpy as np
from loguru import logger

from src.config.settings import get_settings
from src.ml.embedding.base import Embedder


class RemoteEmbedder(Embedder):
    """
    Remote embedder that calls cloud APIs (Cohere, HF Inference, etc.).
    """

    def __init__(self, provider: str = "cohere", api_key: str = "", api_url: str = ""):
        self.provider = provider.lower()
        self.api_key = api_key or self._get_api_key()
        self.api_url = api_url or self._get_api_url()
        self.timeout = get_settings().embed_timeout or 30
        self.client = httpx.Client(timeout=float(self.timeout))

        if not self.api_key:
            raise ValueError(f"API key not provided for {provider} embedder")

    def _get_api_key(self) -> str:
        settings = get_settings()
        return settings.embed_api_key or ""

    def _get_api_url(self) -> str:
        settings = get_settings()
        if self.provider == "cohere":
            return settings.embed_api_url or "https://api.cohere.ai/v1/embed"
        elif self.provider == "hf":
            return settings.embed_api_url or ""  # Set via config
        return ""

    def embed_batch(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Embed batch of texts via remote API."""
        if not texts:
            return np.array([])

        if self.provider == "cohere":
            return self._embed_batch_cohere(texts, batch_size)
        elif self.provider == "hf":
            return self._embed_batch_hf(texts, batch_size)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def embed_query(self, query_text: str) -> np.ndarray:
        """Embed single query via remote API."""
        embeddings = self.embed_batch([query_text], batch_size=1)
        return embeddings[0] if len(embeddings) > 0 else np.array([])

    def compute_sparse_vector(self, text: str) -> dict[str, list[float] | list[int]]:
        """
        Remote embedder does not compute sparse vectors.
        Return empty to indicate no sparse vector support.
        """
        return {"indices": [], "values": []}

    def _embed_batch_cohere(self, texts: list[str], batch_size: int) -> np.ndarray:
        """Call Cohere embedding API."""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            
            payload = {
                "texts": texts,
                "model": "embed-english-v3.0",  # or config this
                "input_type": "search_document",
            }

            response = self.client.post(self.api_url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()

            embeddings = np.array(data.get("embeddings", []), dtype=np.float32)
            
            # Normalize to unit vectors (cosine similarity)
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / (norms + 1e-9)

            return embeddings
        except httpx.HTTPError as e:
            logger.error(f"Cohere embedding API error: {e}")
            raise

    def _embed_batch_hf(self, texts: list[str], batch_size: int) -> np.ndarray:
        """Call HuggingFace Inference API."""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            payload = {"inputs": texts}

            response = self.client.post(self.api_url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()

            # HF returns list of embeddings or nested structure
            if isinstance(data, list):
                embeddings = np.array(data, dtype=np.float32)
            elif "embeddings" in data:
                embeddings = np.array(data["embeddings"], dtype=np.float32)
            else:
                embeddings = np.array(data, dtype=np.float32)

            # Normalize
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / (norms + 1e-9)

            return embeddings
        except httpx.HTTPError as e:
            logger.error(f"HF Inference API error: {e}")
            raise
```

### Step 4: Create `src/ml/embedding/factory.py` (choose embedder)

```python
from __future__ import annotations

from src.config.settings import get_settings
from src.ml.embedding.base import Embedder
from src.ml.embedding.embedder import DualEmbedderV2
from src.ml.embedding.remote import RemoteEmbedder
from loguru import logger


def get_embedder() -> Embedder:
    """
    Factory function to get the appropriate embedder based on config.
    
    Returns:
        Embedder: Local or remote embedder instance.
    """
    settings = get_settings()
    provider = (settings.embed_provider or "local").lower()

    if provider == "local":
        logger.info(f"Loading local embedder: {settings.dense_model_name}")
        return DualEmbedderV2(settings.dense_model_name or "pritamdeka/S-PubMedBert-MS-MARCO")
    
    elif provider == "cohere":
        logger.info("Loading Cohere remote embedder")
        return RemoteEmbedder(
            provider="cohere",
            api_key=settings.embed_api_key,
            api_url=settings.embed_api_url,
        )
    
    elif provider == "hf":
        logger.info("Loading HuggingFace Inference remote embedder")
        return RemoteEmbedder(
            provider="hf",
            api_key=settings.embed_api_key,
            api_url=settings.embed_api_url,
        )
    
    else:
        raise ValueError(f"Unsupported embed_provider: {provider}")
```

---

## Phase 3: Remote Reranker

### Create `src/query/search/remote_reranker.py`

```python
from __future__ import annotations

import httpx
import numpy as np
from loguru import logger

from src.config.settings import get_settings
from src.query.search.retriever import RetrievedChunk


class RemoteReranker:
    """
    Remote reranker that calls cloud APIs (Cohere rerank, HF Inference cross-encoder, etc.).
    """

    def __init__(self, provider: str = "cohere", api_key: str = "", api_url: str = ""):
        self.provider = provider.lower()
        self.api_key = api_key or self._get_api_key()
        self.api_url = api_url or self._get_api_url()
        self.timeout = get_settings().rerank_timeout or 30
        self.client = httpx.Client(timeout=float(self.timeout))

        if not self.api_key:
            raise ValueError(f"API key not provided for {provider} reranker")

    def _get_api_key(self) -> str:
        settings = get_settings()
        return settings.rerank_api_key or ""

    def _get_api_url(self) -> str:
        settings = get_settings()
        if self.provider == "cohere":
            return settings.rerank_api_url or "https://api.cohere.ai/v1/rerank"
        elif self.provider == "hf":
            return settings.rerank_api_url or ""  # Set via config
        return ""

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int = 8,
    ) -> list[RetrievedChunk]:
        """
        Rerank chunks using remote API.
        
        Args:
            query: Query string.
            chunks: List of retrieved chunks to rerank.
            top_k: Number of top results to return.
        
        Returns:
            Reranked chunks, top_k only.
        """
        if not chunks:
            return []

        if self.provider == "cohere":
            return self._rerank_cohere(query, chunks, top_k)
        elif self.provider == "hf":
            return self._rerank_hf(query, chunks, top_k)
        else:
            raise ValueError(f"Unsupported rerank provider: {self.provider}")

    def _rerank_cohere(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:
        """Cohere rerank API."""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            documents = [{"text": chunk.text[:512]} for chunk in chunks]

            payload = {
                "model": "rerank-english-v2.0",
                "query": query,
                "documents": documents,
                "top_n": top_k,
            }

            response = self.client.post(self.api_url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()

            # Map results back to chunks
            results = data.get("results", [])
            ranked_chunks = []
            for result in results:
                idx = result.get("index", 0)
                score = result.get("relevance_score", 0.0)
                if 0 <= idx < len(chunks):
                    chunk = chunks[idx]
                    chunk.score = float(score)
                    ranked_chunks.append(chunk)

            return ranked_chunks[:top_k]
        except httpx.HTTPError as e:
            logger.error(f"Cohere rerank API error: {e}")
            raise

    def _rerank_hf(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:
        """HuggingFace Inference API (assumes cross-encoder endpoint)."""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            # Prepare pairs: [query, document_text]
            pairs = [[query, chunk.text[:512]] for chunk in chunks]

            payload = {"inputs": pairs}

            response = self.client.post(self.api_url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()

            # Parse scores and attach to chunks
            if isinstance(data, list):
                scores = data
            else:
                scores = data

            for chunk, score in zip(chunks, scores):
                chunk.score = float(score) if isinstance(score, (int, float)) else 0.0

            ranked = sorted(chunks, key=lambda c: c.score, reverse=True)
            return ranked[:top_k]
        except httpx.HTTPError as e:
            logger.error(f"HF Inference API error: {e}")
            raise
```

---

## Phase 4: Wire into Retriever & Search Routes

### Update `src/query/search/retriever.py`

Replace this line:
```python
self.embedder = DualEmbedderV2(settings.dense_model_name)
```

With:
```python
from src.ml.embedding.factory import get_embedder
...
self.embedder = get_embedder()
```

Full diff context:

```python
from src.ml.embedding.factory import get_embedder
from src.config.settings import get_settings

class HybridRetriever:
    def __init__(self):
        settings = get_settings()
        self.settings = settings
        self.collection = settings.vector_collection_v2
        self.vector_store = get_vector_store()
        self.embedder = get_embedder()  # ← Changed from DualEmbedderV2
```

### Update `src/api/routes/search.py`

Add import:
```python
from src.query.search.remote_reranker import RemoteReranker
```

Update `_get_or_create_component` function:

```python
async def _get_or_create_component(request: Request, name: str) -> Any:
    """Thread-safe component creation with async lock to prevent race conditions."""
    components = getattr(request.app.state, "search_components", None)
    if components is None:
        request.app.state.search_components = {}
        components = request.app.state.search_components

    component = components.get(name)
    if component is not None:
        return component

    if name not in _component_locks:
        _component_locks[name] = asyncio.Lock()

    async with _component_locks[name]:
        component = components.get(name)
        if component is not None:
            return component

        if name == "query_understanding":
            component = QueryUnderstanding()
        elif name == "retriever":
            component = HybridRetriever()
        elif name == "reranker":
            # Choose reranker based on config
            settings = get_settings()
            rerank_provider = (settings.rerank_provider or "local").lower()
            
            if rerank_provider == "local":
                component = CrossEncoderReranker()
            elif rerank_provider in ("cohere", "hf"):
                component = RemoteReranker(
                    provider=rerank_provider,
                    api_key=settings.rerank_api_key,
                    api_url=settings.rerank_api_url,
                )
            else:
                logger.warning(f"Unknown rerank_provider: {rerank_provider}, falling back to local")
                component = CrossEncoderReranker()
        elif name == "cache":
            component = SearchCache()
        else:
            raise RuntimeError(f"Unknown component requested: {name}")

        components[name] = component
        return component
```

---

## Phase 5: Testing Helpers

### Unit test example: `tests/test_remote_embedder.py`

```python
import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from src.ml.embedding.remote import RemoteEmbedder


@patch("src.ml.embedding.remote.httpx.Client.post")
def test_remote_embedder_cohere(mock_post):
    """Test Cohere embedder with mocked API."""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "embeddings": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
    }
    mock_post.return_value = mock_response

    embedder = RemoteEmbedder(provider="cohere", api_key="test_key")
    texts = ["hello world", "goodbye"]
    embeddings = embedder.embed_batch(texts)

    assert embeddings.shape == (2, 3)
    assert np.allclose(np.linalg.norm(embeddings, axis=1), 1.0)  # normalized


@patch("src.ml.embedding.remote.httpx.Client.post")
def test_remote_reranker_cohere(mock_post):
    """Test Cohere reranker with mocked API."""
    from src.query.search.remote_reranker import RemoteReranker
    from src.query.search.retriever import RetrievedChunk

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "results": [
            {"index": 1, "relevance_score": 0.95},
            {"index": 0, "relevance_score": 0.85},
        ],
    }
    mock_post.return_value = mock_response

    reranker = RemoteReranker(provider="cohere", api_key="test_key")
    
    chunks = [
        RetrievedChunk("1", "doc1", 0.5, "text1", "ctx1", {}, "dense"),
        RetrievedChunk("2", "doc2", 0.6, "text2", "ctx2", {}, "dense"),
    ]
    
    reranked = reranker.rerank("query", chunks, top_k=2)
    
    assert len(reranked) == 2
    assert reranked[0].score == 0.95
    assert reranked[1].score == 0.85
```

---

## Environment variables for deployment

```bash
# Local (default)
export EMBED_PROVIDER="local"

# Cohere
export EMBED_PROVIDER="cohere"
export EMBED_API_KEY="<your_cohere_key>"
export EMBED_API_URL="https://api.cohere.ai/v1/embed"
export RERANK_PROVIDER="cohere"
export RERANK_API_KEY="<your_cohere_key>"
export RERANK_API_URL="https://api.cohere.ai/v1/rerank"

# HF Inference
export EMBED_PROVIDER="hf"
export EMBED_API_KEY="<your_hf_token>"
export EMBED_API_URL="https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"
```

---

## Summary of new files created:
1. `src/ml/embedding/base.py` — Embedder interface
2. `src/ml/embedding/remote.py` — RemoteEmbedder (Cohere, HF)
3. `src/ml/embedding/factory.py` — Factory to select embedder
4. `src/query/search/remote_reranker.py` — RemoteReranker (Cohere, HF)
5. `tests/test_remote_embedder.py` — Unit tests (optional)

## Files to modify:
1. `src/config/settings.py` — Add config keys
2. `src/ml/embedding/embedder.py` — Inherit from Embedder (minimal change)
3. `src/query/search/retriever.py` — Use factory
4. `src/api/routes/search.py` — Wire RemoteReranker

All changes are backward-compatible; defaults use local embedder & reranker.
