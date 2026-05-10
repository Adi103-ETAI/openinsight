from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import httpx

from src.config.settings import get_settings
from src.ml.embedding.embedder import DualEmbedderV2
from src.vectorstore.registry import get_vector_store
from src.vectorstore.types import ScoredPoint, SparseVector


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
    def __init__(self):
        settings = get_settings()
        self.settings = settings
        self.collection = settings.vector_collection_v2
        self.vector_store = get_vector_store()
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

        sparse_query = SparseVector.from_index_values(
            indices=[int(i) for i in sparse_vector.get("indices", [])],
            values=[float(v) for v in sparse_vector.get("values", [])],
        )

        dense_results, sparse_results = await asyncio.gather(
            loop.run_in_executor(
                None,
                lambda: self.vector_store.search_dense(
                    dense_vector=(
                        dense_embedding.tolist()
                        if hasattr(dense_embedding, "tolist")
                        else list(dense_embedding)
                    ),
                    top_k=top_k,
                    filters=query_analysis.metadata_filters,
                    collection_name=self.collection,
                ),
            ),
            loop.run_in_executor(
                None,
                lambda: self.vector_store.search_sparse(
                    sparse_vector=sparse_query,
                    top_k=top_k,
                    filters=query_analysis.metadata_filters,
                    collection_name=self.collection,
                ),
            ),
        )
        return (
            [self._to_chunk(point, "dense") for point in dense_results],
            [self._to_chunk(point, "sparse") for point in sparse_results],
        )

    def _to_chunk(self, point: ScoredPoint, source: str) -> RetrievedChunk:
        payload = point.payload or {}
        return RetrievedChunk(
            chunk_id=payload.get("chunk_id", point.point_id),
            doc_id=payload.get("doc_id", ""),
            score=float(point.score),
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

