from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from src.config.settings import get_settings
from src.ml.embedding.embedder import BaseEmbedder, get_embedder
from src.services.llm_client import get_nim_client
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


@dataclass
class RetrievedParentChunk:
    """Retrieved parent chunk with its child context for enhanced RAG."""
    chunk_id: str
    doc_id: str
    score: float
    text: str
    contextual_text: str
    metadata: dict[str, Any]
    retrieval_source: str
    # Related child chunks that were used to retrieve this parent
    child_chunks: list[RetrievedChunk] = field(default_factory=list)
    # Reference to the parent chunk ID if this is a child chunk
    parent_chunk_id: str | None = None


class HybridRetriever:
    def __init__(self, embedder: BaseEmbedder | None = None):
        settings = get_settings()
        self.settings = settings
        self.collection = settings.vector_collection_v2
        self.vector_store = get_vector_store()
        # Use config-driven embedder (local, huggingface, or cohere)
        self.embedder = embedder or get_embedder()
        
        # Parent-child retrieval configuration
        self._parent_collection = f"{settings.vector_collection_v2}_parent"
        self._child_collection = f"{settings.vector_collection_v2}_child"

    async def retrieve(
        self,
        query: str,
        query_analysis: Any,
        top_k: int = 50,
    ) -> tuple[list[RetrievedChunk], list[RetrievedChunk]]:
        # Use rewritten query for embedding if available, fall back to HYDE or original
        embed_query_text = query
        if query_analysis.rewritten_query:
            embed_query_text = query_analysis.rewritten_query
        elif query_analysis.use_hyde:
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

        prompt = f"Write a brief clinical paragraph that answers this query: {query}\n\nAnswer:"

        try:
            client = get_nim_client()
            text = await client.completions(
                prompt=prompt,
                temperature=0.1,
                max_tokens=200,
            )
            return text.strip() or None
        except (RuntimeError, ValueError) as exc:
            logger.warning(f"HYDE generation failed ({exc}), falling back to original query")
            return None

    async def retrieve_with_parent(
        self,
        query: str,
        query_analysis: Any,
        top_k: int = 50,
    ) -> tuple[list[RetrievedChunk], list[RetrievedParentChunk]]:
        """
        Hybrid retrieval with parent chunk fetching for enhanced context.
        
        This method performs a two-stage retrieval:
        1. First searches child chunks (in the default collection) for precision
        2. Then fetches parent chunks (from _parent collection) for full context
        
        The returned parent chunks include references to their child chunks,
        allowing the LLM to have both precise matches (children) and
        full section context (parents).
        
        Args:
            query: Search query string
            query_analysis: Query analysis object with filters and expansion
            top_k: Number of chunks to retrieve
            
        Returns:
            Tuple of (child_chunks, parent_chunks_with_children)
        """
        # First perform standard hybrid retrieval on child chunks
        dense_chunks, sparse_chunks = await self.retrieve(
            query=query,
            query_analysis=query_analysis,
            top_k=top_k,
        )
        
        # Combine all child chunks and track unique ones by chunk_id
        all_children: dict[str, RetrievedChunk] = {}
        for chunk in dense_chunks + sparse_chunks:
            if chunk.chunk_id not in all_children or chunk.score > all_children[chunk.chunk_id].score:
                all_children[chunk.chunk_id] = chunk
        
        child_chunks = list(all_children.values())
        
        # Extract parent chunk IDs from child chunks
        # Look for parent_chunk_id in metadata (set during indexing)
        parent_chunk_ids = set()
        for child in child_chunks:
            parent_id = child.metadata.get("parent_chunk_id")
            if parent_id:
                parent_chunk_ids.add(parent_id)
        
        # If no parent IDs found, fall back to simple parent inference from chunk_id
        # (e.g., doc_id_child_... -> doc_id_parent_...)
        if not parent_chunk_ids:
            parent_chunk_ids = self._infer_parent_ids([c.chunk_id for c in child_chunks])
        
        # Fetch parent chunks
        parent_chunks = await self._fetch_parent_chunks(
            list(parent_chunk_ids),
            query_analysis.metadata_filters,
        )
        
        # Attach child chunks to their parents based on parent_chunk_id mapping
        child_by_parent: dict[str, list[RetrievedChunk]] = {}
        for child in child_chunks:
            parent_id = child.metadata.get("parent_chunk_id") or self._get_parent_id_from_child(child.chunk_id)
            if parent_id:
                if parent_id not in child_by_parent:
                    child_by_parent[parent_id] = []
                child_by_parent[parent_id].append(child)
        
        # Update parent chunks with their child references
        for parent in parent_chunks:
            parent.child_chunks = child_by_parent.get(parent.chunk_id, [])
        
        return child_chunks, parent_chunks

    def _infer_parent_ids(self, child_chunk_ids: list[str]) -> set[str]:
        """
        Infer parent chunk IDs from child chunk IDs.
        
        Expected pattern: doc_id_child_parent_0_1 -> doc_id_parent_0
        """
        parent_ids = set()
        for child_id in child_chunk_ids:
            # Split by underscore and reconstruct parent ID
            parts = child_id.split("_child_")
            if len(parts) == 2:
                doc_id = parts[0]
                # The part after "child_" contains parent_index_child_index
                # e.g., "parent_0_1" -> parent is "parent_0"
                child_part = parts[1]
                parent_part = "_".join(child_part.split("_")[:2])  # "parent_0"
                parent_ids.add(f"{doc_id}_{parent_part}")
        return parent_ids

    def _get_parent_id_from_child(self, child_chunk_id: str) -> str | None:
        """Extract parent chunk ID from child chunk ID."""
        parts = child_chunk_id.split("_child_")
        if len(parts) == 2:
            doc_id = parts[0]
            child_part = parts[1]
            parent_part = "_".join(child_part.split("_")[:2])
            return f"{doc_id}_{parent_part}"
        return None

    async def _fetch_parent_chunks(
        self,
        parent_chunk_ids: list[str],
        filters: Any,
    ) -> list[RetrievedParentChunk]:
        """Fetch parent chunks by their IDs."""
        if not parent_chunk_ids:
            return []
        
        # Check if parent collection exists and has data
        # For backward compatibility, if parent collection doesn't exist,
        # return empty list (fallback to child-only retrieval)
        try:
            if not self.vector_store.client.has_collection(self._parent_collection):
                logger.debug("Parent collection does not exist, falling back to child-only retrieval")
                return []
        except Exception:
            # If we can't check, assume parent collection doesn't exist
            return []
        
        # Load the parent collection
        try:
            self.vector_store.client.load_collection(self._parent_collection)
        except Exception as e:
            logger.warning(f"Failed to load parent collection: {e}")
            return []
        
        parent_chunks = []
        settings = get_settings()
        dummy_vector = [0.0] * settings.vector_dim
        
        from src.vectorstore.filters import FilterExpression, FilterCondition, FilterOperator
        
        for parent_id in parent_chunk_ids:
            try:
                parent_filter = FilterExpression(
                    must=[
                        FilterCondition(
                            field="chunk_id",
                            operator=FilterOperator.EQ,
                            value=parent_id,
                        )
                    ]
                )
                
                results = self.vector_store.search_dense(
                    dense_vector=dummy_vector,
                    top_k=1,
                    filters=parent_filter,
                    collection_name=self._parent_collection,
                )
                
                if results:
                    parent_chunks.append(self._to_parent_chunk(results[0]))
            except Exception as e:
                logger.warning(f"Failed to fetch parent chunk {parent_id}: {e}")
                continue
        
        return parent_chunks

    def _to_parent_chunk(self, point: ScoredPoint) -> RetrievedParentChunk:
        """Convert a ScoredPoint to a RetrievedParentChunk."""
        payload = point.payload or {}
        return RetrievedParentChunk(
            chunk_id=payload.get("chunk_id", point.point_id),
            doc_id=payload.get("doc_id", ""),
            score=float(point.score),
            text=payload.get("raw_text", "") or payload.get("chunk_text", ""),
            contextual_text=payload.get("contextual_text", ""),
            metadata=payload,
            retrieval_source="parent",
            child_chunks=[],
            parent_chunk_id=None,
        )

