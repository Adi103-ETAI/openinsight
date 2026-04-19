from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import httpx
from qdrant_client import QdrantClient
from qdrant_client.http import models

from src.core.config import get_settings
from src.ingestion.embedder_v2 import DualEmbedderV2


@dataclass
class RetrievedChunk:
    chunk_id: str
    doc_id: str
    score: float
    text: str
    contextual_text: str
    metadata: dict[str, Any]
    retrieval_source: str


class HybridRetriever:
    def __init__(self, qdrant_url: str | None = None):
        settings = get_settings()
        self.settings = settings
        if settings.qdrant_api_key:
            self.client = QdrantClient(
                url=qdrant_url or settings.qdrant_url,
                api_key=settings.qdrant_api_key,
            )
        else:
            self.client = QdrantClient(url=qdrant_url or settings.qdrant_url)
        self.collection = settings.qdrant_collection_v2
        self.embedder = DualEmbedderV2(settings.dense_model_name)

    async def retrieve(
        self,
        query: str,
        query_analysis: Any,
        top_k: int = 50,
    ) -> tuple[list[RetrievedChunk], list[RetrievedChunk]]:
        embed_query_text = query
        if query_analysis.use_hyde:
            hyde_text = await self._generate_hyde(query)
            if hyde_text:
                embed_query_text = hyde_text

        sparse_query_text = query
        if query_analysis.expanded_terms:
            sparse_query_text = f"{query} {' '.join(query_analysis.expanded_terms)}"

        loop = asyncio.get_running_loop()
        dense_embedding, sparse_vector = await asyncio.gather(
            loop.run_in_executor(None, self.embedder.embed_query, embed_query_text),
            loop.run_in_executor(
                None, self.embedder.compute_sparse_vector, sparse_query_text
            ),
        )

        qdrant_filter = self._build_filter(query_analysis.metadata_filters)

        dense_results, sparse_results = await asyncio.gather(
            self._dense_search(dense_embedding, qdrant_filter, top_k),
            self._sparse_search(sparse_vector, qdrant_filter, top_k),
        )
        return dense_results, sparse_results

    async def _dense_search(
        self,
        embedding: Any,
        qdrant_filter: models.Filter | None,
        top_k: int,
    ) -> list[RetrievedChunk]:
        loop = asyncio.get_running_loop()
        results = await loop.run_in_executor(
            None,
            lambda: self.client.search(
                collection_name=self.collection,
                query_vector=models.NamedVector(
                    name="dense",
                    vector=(
                        embedding.tolist()
                        if hasattr(embedding, "tolist")
                        else list(embedding)
                    ),
                ),
                query_filter=qdrant_filter,
                limit=top_k,
                with_payload=True,
            ),
        )
        return [self._to_chunk(result, "dense") for result in results]

    async def _sparse_search(
        self,
        sparse_vector: dict[str, list[int] | list[float]],
        qdrant_filter: models.Filter | None,
        top_k: int,
    ) -> list[RetrievedChunk]:
        loop = asyncio.get_running_loop()
        results = await loop.run_in_executor(
            None,
            lambda: self.client.search(
                collection_name=self.collection,
                query_vector=models.NamedSparseVector(
                    name="sparse",
                    vector=models.SparseVector(
                        indices=[int(i) for i in sparse_vector.get("indices", [])],
                        values=[float(v) for v in sparse_vector.get("values", [])],
                    ),
                ),
                query_filter=qdrant_filter,
                limit=top_k,
                with_payload=True,
            ),
        )
        return [self._to_chunk(result, "sparse") for result in results]

    def _build_filter(self, conditions: list[Any]) -> models.Filter | None:
        if not conditions:
            return None
        return models.Filter(must=conditions)

    def _to_chunk(self, qdrant_result: Any, source: str) -> RetrievedChunk:
        payload = qdrant_result.payload or {}
        return RetrievedChunk(
            chunk_id=payload.get("chunk_id", str(qdrant_result.id)),
            doc_id=payload.get("doc_id", ""),
            score=float(getattr(qdrant_result, "score", 0.0) or 0.0),
            text=payload.get("raw_text", "") or payload.get("chunk_text", ""),
            contextual_text=payload.get("contextual_text", ""),
            metadata=payload,
            retrieval_source=source,
        )

    async def _generate_hyde(self, query: str) -> str | None:
        if not self.settings.hyde_enabled:
            return None

        url = f"{self.settings.nvidia_nim_base_url.rstrip('/')}/completions"
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.settings.nvidia_nim_api_key:
            headers["Authorization"] = f"Bearer {self.settings.nvidia_nim_api_key}"

        body = {
            "model": self.settings.nim_model,
            "prompt": f"Write a brief clinical paragraph that answers this query: {query}\n\nAnswer:",
            "max_tokens": 200,
            "temperature": 0.1,
        }

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.post(url, headers=headers, json=body)
                response.raise_for_status()
                data = response.json()
                choices = data.get("choices", [])
                if not choices:
                    return None
                return (choices[0].get("text") or "").strip() or None
        except (httpx.HTTPError, RuntimeError, ValueError, TypeError):
            return None
