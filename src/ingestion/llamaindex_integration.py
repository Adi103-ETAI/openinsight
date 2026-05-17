"""
LlamaIndex Integration for Parent Document Retrieval in OpenInsight.

This module implements hierarchical node parsing using LlamaIndex to support:
- Parent chunks (~1000 tokens) containing full section context
- Child chunks (~350 tokens) for precise retrieval
- Relationship tracking between parent and child chunks
- Hybrid search: first search child chunks, then fetch parent chunks

Backward compatible with existing MongoDB/Milvus setup.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

from src.config.settings import get_settings

logger = logging.getLogger(__name__)

# Import settings for default values
_settings = get_settings()


# =============================================================================
# Data Classes for Parent-Child Chunking
# =============================================================================


@dataclass
class ParentChunk:
    """
    Represents a parent chunk containing full section context.
    Typically ~1000 tokens for comprehensive context.
    """
    chunk_id: str
    doc_id: str
    chunk_type: str
    section_title: str
    text: str
    contextual_text: str
    char_count: int
    token_estimate: int
    chunk_index: int
    total_chunks: int
    child_chunk_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ChildChunk:
    """
    Represents a child chunk for precise retrieval.
    Typically ~350 tokens with overlap for context continuity.
    """
    chunk_id: str
    doc_id: str
    chunk_type: str
    section_title: str
    text: str
    contextual_text: str
    char_count: int
    token_estimate: int
    chunk_index: int
    total_chunks: int
    parent_chunk_id: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class HierarchicalChunkResult:
    """Result containing both parent and child chunks."""
    parent_chunks: list[ParentChunk]
    child_chunks: list[ChildChunk]
    parent_to_children: dict[str, list[str]] = field(default_factory=dict)
    child_to_parent: dict[str, str] = field(default_factory=dict)


@dataclass
class RetrievedParentChunk:
    """Retrieved parent chunk with its child context."""
    chunk_id: str
    doc_id: str
    score: float
    text: str
    contextual_text: str
    metadata: dict[str, Any]
    child_chunks: list[Any] = field(default_factory=list)
    retrieval_source: str


# =============================================================================
# Hierarchical Chunk Parser using LlamaIndex Patterns
# =============================================================================


class HierarchicalChunkParser:
    """
    Implements hierarchical node parsing for parent-child chunk relationships.
    
    Uses a two-stage chunking approach:
    1. First create larger parent chunks (~1000 tokens) for full section context
    2. Then create smaller child chunks (~350 tokens) for precise retrieval
    
    This approach maintains the relationship between chunks while providing
    both precision (child) and context (parent) for LLM generation.
    """

    def __init__(
        self,
        parent_tokens: int = 1000,
        child_tokens: int = 350,
        overlap_tokens: int = 50,
    ):
        """
        Initialize the hierarchical chunk parser.
        
        Args:
            parent_tokens: Target token count for parent chunks (default: 1000)
            child_tokens: Target token count for child chunks (default: 350)
            overlap_tokens: Token overlap between child chunks (default: 50)
        """
        self.parent_tokens = parent_tokens
        self.child_tokens = child_tokens
        self.overlap_tokens = overlap_tokens

    def parse_document(
        self, doc: Any, doc_metadata: dict[str, Any]
    ) -> HierarchicalChunkResult:
        """
        Parse a document into parent and child chunks.
        
        Args:
            doc: Document object with title, abstract, sections, etc.
            doc_metadata: Metadata dictionary for the document
            
        Returns:
            HierarchicalChunkResult with parent and child chunks
        """
        doc_id = self._resolve_doc_id(doc)
        title = self._get_field(doc, "title", "")
        abstract = self._get_field(doc, "abstract", "")

        parent_chunks: list[ParentChunk] = []
        child_chunks: list[ChildChunk] = []
        parent_to_children: dict[str, list[str]] = {}
        child_to_parent: dict[str, str] = {}

        # Generate document summary as a parent chunk
        summary_text = self._create_summary_text(doc)
        if summary_text:
            parent_idx = len(parent_chunks)
            parent_id = f"{doc_id}_parent_{parent_idx}"
            parent_chunk = self._create_parent_chunk(
                doc_id=doc_id,
                chunk_id=parent_id,
                chunk_type="doc_summary",
                section_title="Document Summary",
                text=summary_text,
                chunk_index=parent_idx,
                doc_title=title,
                doc_metadata=doc_metadata,
            )
            parent_chunks.append(parent_chunk)

        # Process sections into parent chunks
        sections = self._extract_sections(doc)
        for section in sections:
            section_title = section.title or f"Section {section.section_index + 1}"
            section_text = section.text

            if not section_text.strip():
                continue

            # Create parent chunk for this section
            parent_idx = len(parent_chunks)
            parent_id = f"{doc_id}_parent_{parent_idx}"
            
            parent_chunk = self._create_parent_chunk(
                doc_id=doc_id,
                chunk_id=parent_id,
                chunk_type="section",
                section_title=section_title,
                text=section_text,
                chunk_index=parent_idx,
                doc_title=title,
                doc_metadata=doc_metadata,
            )
            parent_chunks.append(parent_chunk)

            # Create child chunks from the section text
            child_ids_for_parent = []
            section_child_chunks = self._create_child_chunks_from_text(
                doc_id=doc_id,
                parent_id=parent_id,
                section_title=section_title,
                text=section_text,
                doc_title=title,
                doc_metadata=doc_metadata,
            )
            
            for child_chunk in section_child_chunks:
                child_chunks.append(child_chunk)
                child_ids_for_parent.append(child_chunk.chunk_id)
                child_to_parent[child_chunk.chunk_id] = parent_id

            parent_to_children[parent_id] = child_ids_for_parent

        # Update total chunks for all parent chunks
        total_parents = len(parent_chunks)
        for idx, parent in enumerate(parent_chunks):
            parent.total_chunks = total_parents

        # Update total chunks for all child chunks
        total_children = len(child_chunks)
        for idx, child in enumerate(child_chunks):
            child.chunk_index = idx
            child.total_chunks = total_children

        return HierarchicalChunkResult(
            parent_chunks=parent_chunks,
            child_chunks=child_chunks,
            parent_to_children=parent_to_children,
            child_to_parent=child_to_parent,
        )

    def _create_summary_text(self, doc: Any) -> str:
        """Create a summary text from document title, abstract, and metadata."""
        title = self._get_field(doc, "title", "")
        abstract = self._get_field(doc, "abstract", "")
        mesh_terms = self._get_field(doc, "mesh_terms", [])
        keywords = self._get_field(doc, "keywords", [])

        parts = []
        if title.strip():
            parts.append(title.strip())
        if abstract.strip():
            parts.append(abstract.strip())
        if mesh_terms and isinstance(mesh_terms, list):
            parts.append("MeSH Terms: " + ", ".join(str(t) for t in mesh_terms if t))
        if keywords and isinstance(keywords, list):
            parts.append("Keywords: " + ", ".join(str(k) for k in keywords if k))

        return "\n\n".join(parts)

    def _create_parent_chunk(
        self,
        *,
        doc_id: str,
        chunk_id: str,
        chunk_type: str,
        section_title: str,
        text: str,
        chunk_index: int,
        doc_title: str,
        doc_metadata: dict[str, Any],
    ) -> ParentChunk:
        """Create a parent chunk with full section context."""
        token_estimate = self._estimate_tokens(text)
        contextual_text = self._build_contextual_text(
            source=str(doc_metadata.get("source", "unknown")),
            doc_type=str(doc_metadata.get("doc_type", "unknown")),
            doc_title=doc_title,
            section_title=section_title,
            chunk_text=text,
        )

        return ParentChunk(
            chunk_id=chunk_id,
            doc_id=doc_id,
            chunk_type=chunk_type,
            section_title=section_title,
            text=text,
            contextual_text=contextual_text,
            char_count=len(text),
            token_estimate=token_estimate,
            chunk_index=chunk_index,
            total_chunks=0,  # Will be updated after all chunks created
            child_chunk_ids=[],
            metadata=dict(doc_metadata),
        )

    def _create_child_chunks_from_text(
        self,
        doc_id: str,
        parent_id: str,
        section_title: str,
        text: str,
        doc_title: str,
        doc_metadata: dict[str, Any],
    ) -> list[ChildChunk]:
        """
        Create child chunks from text using sentence-based splitting.
        Uses the existing chunker logic adapted for child chunk creation.
        """
        import re

        child_chunks: list[ChildChunk] = []
        
        # Split text into sentences
        sentences = self._split_into_sentences(text)
        if not sentences:
            return []

        current_chunk_text = ""
        current_tokens = 0
        chunk_index = 0

        for sentence in sentences:
            sentence_tokens = self._estimate_tokens(sentence)

            # If adding this sentence exceeds child token limit, create a new chunk
            if current_tokens + sentence_tokens > self.child_tokens and current_chunk_text:
                child_chunk = self._create_child_chunk(
                    doc_id=doc_id,
                    parent_id=parent_id,
                    section_title=section_title,
                    text=current_chunk_text.strip(),
                    chunk_index=chunk_index,
                    doc_title=doc_title,
                    doc_metadata=doc_metadata,
                )
                child_chunks.append(child_chunk)
                chunk_index += 1

                # Start new chunk with overlap
                current_chunk_text = self._get_overlap_text(current_chunk_text) + " " + sentence
                current_tokens = self._estimate_tokens(current_chunk_text)
            else:
                if current_chunk_text:
                    current_chunk_text += " " + sentence
                else:
                    current_chunk_text = sentence
                current_tokens += sentence_tokens

        # Don't forget the last chunk
        if current_chunk_text.strip():
            child_chunk = self._create_child_chunk(
                doc_id=doc_id,
                parent_id=parent_id,
                section_title=section_title,
                text=current_chunk_text.strip(),
                chunk_index=chunk_index,
                doc_title=doc_title,
                doc_metadata=doc_metadata,
            )
            child_chunks.append(child_chunk)

        return child_chunks

    def _create_child_chunk(
        self,
        doc_id: str,
        parent_id: str,
        section_title: str,
        text: str,
        chunk_index: int,
        doc_title: str,
        doc_metadata: dict[str, Any],
    ) -> ChildChunk:
        """Create a child chunk for precise retrieval."""
        token_estimate = self._estimate_tokens(text)
        chunk_id = f"{doc_id}_child_{parent_id}_{chunk_index}"
        
        contextual_text = self._build_contextual_text(
            source=str(doc_metadata.get("source", "unknown")),
            doc_type=str(doc_metadata.get("doc_type", "unknown")),
            doc_title=doc_title,
            section_title=section_title,
            chunk_text=text,
        )

        return ChildChunk(
            chunk_id=chunk_id,
            doc_id=doc_id,
            chunk_type="child",
            section_title=section_title,
            text=text,
            contextual_text=contextual_text,
            char_count=len(text),
            token_estimate=token_estimate,
            chunk_index=chunk_index,
            total_chunks=0,
            parent_chunk_id=parent_id,
            metadata=dict(doc_metadata),
        )

    def _split_into_sentences(self, text: str) -> list[str]:
        """Split text into sentences while handling common abbreviations."""
        import re

        # Protect common abbreviations
        protected = text
        abbreviations = {
            "et al.": "et al<PERIOD>",
            "fig.": "fig<PERIOD>",
            "e.g.": "e<PERIOD>g<PERIOD>",
            "i.e.": "i<PERIOD>e<PERIOD>",
            "vs.": "vs<PERIOD>",
            "mg/dl": "mg<SLASH>dl",
            "p.o.": "p<PERIOD>o<PERIOD>",
            "i.v.": "i<PERIOD>v<PERIOD>",
        }

        for original, replacement in abbreviations.items():
            protected = re.sub(
                re.escape(original), replacement, protected, flags=re.IGNORECASE
            )

        # Split on sentence boundaries
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", protected)
        
        # Restore abbreviations
        restored_sentences = []
        for sentence in sentences:
            restored = sentence
            for original, replacement in abbreviations.items():
                restored = restored.replace(replacement, original)
            restored_sentences.append(restored.strip())

        return [s for s in restored_sentences if s.strip()]

    def _get_overlap_text(self, text: str) -> str:
        """Get the overlap text from the end of the current chunk."""
        words = text.split()
        overlap_words = words[-self.overlap_tokens:]
        return " ".join(overlap_words)

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count using word-based estimation."""
        word_count = len(text.split())
        return max(1, int(word_count * 1.3))

    def _build_contextual_text(
        self,
        *,
        source: str,
        doc_type: str,
        doc_title: str,
        section_title: str,
        chunk_text: str,
    ) -> str:
        """Build contextual text with source information."""
        return (
            f"Source: {source}\n"
            f"Document type: {doc_type}\n"
            f"Title: {doc_title}\n"
            f"Section: {section_title}\n\n"
            f"{chunk_text}"
        )

    def _extract_sections(self, doc: Any) -> list[Any]:
        """Extract sections from document."""
        sections_value = self._get_field(doc, "sections", None)

        if isinstance(sections_value, list) and sections_value:
            parsed_sections = []
            for i, sec in enumerate(sections_value):
                if isinstance(sec, dict):
                    title = sec.get("title", "")
                    text = sec.get("text", "")
                    section_index = sec.get("section_index", i)
                else:
                    title = ""
                    text = str(sec)
                    section_index = i
                    
                parsed_sections.append(
                    type('Section', (), {
                        'title': str(title) if title else "",
                        'text': str(text) if text else "",
                        'section_index': int(section_index)
                    })()
                )
            return parsed_sections

        # Fallback: extract from content field
        content = self._get_field(doc, "content", "")
        if not content.strip():
            return []

        return self._split_content_into_sections(content)

    def _split_content_into_sections(self, content: str) -> list[Any]:
        """Split content into sections using header detection."""
        import re

        header_re = re.compile(
            r"^(?:\d+\.\s*)?("
            r"Abstract|Introduction|Background|Methods?|Materials and Methods|Results|Discussion|"
            r"Conclusion|Conclusions|Results and Discussion|Study Population|Patient Population|"
            r"Demographics|Study Design|Methods and Results|Background and Objectives|Objectives|"
            r"Aims|Supplementary|References|Acknowledgments"
            r")\s*$",
            re.IGNORECASE,
        )

        lines = content.splitlines()
        sections = []
        current_title = "Main"
        current_lines = []

        for line in lines:
            stripped = line.strip()
            if not stripped:
                if current_lines and current_lines[-1] != "":
                    current_lines.append("")
                continue

            if header_re.match(stripped):
                if current_lines:
                    section_text = "\n".join(current_lines).strip()
                    if section_text:
                        sections.append(
                            type('Section', (), {
                                'title': current_title,
                                'text': section_text,
                                'section_index': len(sections)
                            })()
                        )
                current_title = stripped
                current_lines = []
            else:
                current_lines.append(stripped)

        if current_lines:
            section_text = "\n".join(current_lines).strip()
            if section_text:
                sections.append(
                    type('Section', (), {
                        'title': current_title,
                        'text': section_text,
                        'section_index': len(sections)
                    })()
                )

        return sections

    def _resolve_doc_id(self, doc: Any) -> str:
        """Resolve document ID from doc object."""
        import re

        doc_id = self._get_field(doc, "doc_id", None)
        if isinstance(doc_id, str) and doc_id.strip():
            return doc_id

        maybe_pmid = self._get_field(doc, "pmid", None)
        if isinstance(maybe_pmid, str) and maybe_pmid.strip():
            return f"pmid_{maybe_pmid.strip()}"

        title = self._get_field(doc, "title", "untitled")
        slug = re.sub(r"[^a-zA-Z0-9]+", "_", str(title)).strip("_").lower()[:40]
        return f"doc_{slug or 'unknown'}"

    def _get_field(self, obj: Any, key: str, default: Any) -> Any:
        """Get field from object (dict or attribute access)."""
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)


# =============================================================================
# Parent-Child Indexer for Milvus
# =============================================================================


class ParentChildIndexer:
    """
    Indexer for parent-child chunk relationships in Milvus.
    
    Maintains two collections:
    1. Child chunks: For precise semantic search
    2. Parent chunks: For full context retrieval
    
    Uses parent_chunk_id field to link child chunks to their parent.
    """

    def __init__(self, vector_store: Any = None):
        """
        Initialize the parent-child indexer.
        
        Args:
            vector_store: Optional vector store instance. If not provided,
                         uses the registry's default Milvus store.
        """
        from src.vectorstore.registry import get_vector_store
        
        self.vector_store = vector_store or get_vector_store()
        settings = get_settings()
        
        # Configuration for collections
        self.child_collection = f"{settings.vector_collection_v2}_child"
        self.parent_collection = f"{settings.vector_collection_v2}_parent"

    def create_collections(self, recreate: bool = False) -> None:
        """Create both child and parent collections in Milvus."""
        from pymilvus import DataType, MilvusClient
        
        # Get the underlying Milvus client
        client = self.vector_store.client
        
        # Create child collection schema
        child_schema = MilvusClient.create_schema(
            auto_id=False,
            enable_dynamic_field=True,
        )
        self._add_common_fields(child_schema)
        child_schema.add_field(
            field_name="parent_chunk_id",
            datatype=DataType.VARCHAR,
            max_length=128,
        )
        
        # Create child collection index
        child_index_params = client.prepare_index_params()
        self._add_index_fields(child_index_params, client, self.child_collection, child_schema, recreate)

        # Create parent collection schema
        parent_schema = MilvusClient.create_schema(
            auto_id=False,
            enable_dynamic_field=True,
        )
        self._add_common_fields(parent_schema)
        parent_schema.add_field(
            field_name="child_chunk_ids",
            datatype=DataType.VARCHAR,
            max_length=4096,  # JSON array of child IDs
        )
        
        # Create parent collection index
        parent_index_params = client.prepare_index_params()
        self._add_index_fields(parent_index_params, client, self.parent_collection, parent_schema, recreate)

    def _add_common_fields(self, schema: Any) -> None:
        """Add common fields to schema."""
        settings = get_settings()
        
        schema.add_field(
            field_name=self.vector_store.id_field,
            datatype=DataType.VARCHAR,
            is_primary=True,
            max_length=128,
        )
        schema.add_field(
            field_name=self.vector_store.dense_field,
            datatype=DataType.FLOAT_VECTOR,
            dim=settings.vector_dim,
        )
        schema.add_field(
            field_name=self.vector_store.sparse_field,
            datatype=DataType.SPARSE_FLOAT_VECTOR,
        )
        schema.add_field(field_name="year", datatype=DataType.INT64)
        schema.add_field(field_name="doc_type", datatype=DataType.VARCHAR, max_length=64)
        schema.add_field(field_name="source", datatype=DataType.VARCHAR, max_length=64)
        schema.add_field(field_name="source_type", datatype=DataType.VARCHAR, max_length=64)
        schema.add_field(field_name="evidence_level", datatype=DataType.VARCHAR, max_length=16)
        schema.add_field(field_name="india_relevant", datatype=DataType.BOOL)
        schema.add_field(field_name="has_drug_dosing", datatype=DataType.BOOL)
        schema.add_field(field_name="chunk_type", datatype=DataType.VARCHAR, max_length=64)
        schema.add_field(field_name="pmid", datatype=DataType.VARCHAR, max_length=64)

    def _add_index_fields(
        self, 
        index_params: Any, 
        client: Any, 
        collection_name: str, 
        schema: Any, 
        recreate: bool
    ) -> None:
        """Add index fields to collection."""
        if recreate and client.has_collection(collection_name):
            client.drop_collection(collection_name)
        
        settings = get_settings()
        
        index_params.add_index(
            field_name=self.vector_store.dense_field,
            metric_type=settings.vector_dense_metric,
        )
        index_params.add_index(
            field_name=self.vector_store.sparse_field,
            index_type="SPARSE_INVERTED_INDEX",
            metric_type=settings.vector_sparse_metric,
        )
        
        if not client.has_collection(collection_name):
            client.create_collection(
                collection_name=collection_name,
                schema=schema,
                index_params=index_params,
            )
        client.load_collection(collection_name)

    def upsert_parent_chunks(
        self,
        chunks: list[ParentChunk],
        dense_embeddings: Any,
        sparse_vectors: list[dict[str, list[int] | list[float]]],
    ) -> int:
        """Upsert parent chunks to Milvus."""
        from src.vectorstore.types import SparseVector, VectorPoint
        
        points: list[VectorPoint] = []
        for chunk, dense_emb, sparse_vec in zip(chunks, dense_embeddings, sparse_vectors):
            point_id = chunk.chunk_id
            
            payload = dict(chunk.metadata)
            payload.update({
                "raw_text": chunk.text,
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "chunk_type": chunk.chunk_type,
                "section_title": chunk.section_title,
                "chunk_index": chunk.chunk_index,
                "total_chunks": chunk.total_chunks,
                "child_chunk_ids": ",".join(chunk.child_chunk_ids) if chunk.child_chunk_ids else "",
            })
            
            points.append(
                VectorPoint(
                    point_id=point_id,
                    dense_vector=(
                        dense_emb.tolist()
                        if hasattr(dense_emb, "tolist")
                        else list(dense_emb)
                    ),
                    sparse_vector=SparseVector.from_index_values(
                        indices=[int(i) for i in sparse_vec.get("indices", [])],
                        values=[float(v) for v in sparse_vec.get("values", [])],
                    ),
                    payload=payload,
                )
            )
        
        return self.vector_store.upsert_points(
            points,
            collection_name=self.parent_collection,
            batch_size=100,
        )

    def upsert_child_chunks(
        self,
        chunks: list[ChildChunk],
        dense_embeddings: Any,
        sparse_vectors: list[dict[str, list[int] | list[float]]],
    ) -> int:
        """Upsert child chunks to Milvus."""
        from src.vectorstore.types import SparseVector, VectorPoint
        
        points: list[VectorPoint] = []
        for chunk, dense_emb, sparse_vec in zip(chunks, dense_embeddings, sparse_vectors):
            point_id = chunk.chunk_id
            
            payload = dict(chunk.metadata)
            payload.update({
                "raw_text": chunk.text,
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "chunk_type": chunk.chunk_type,
                "section_title": chunk.section_title,
                "chunk_index": chunk.chunk_index,
                "total_chunks": chunk.total_chunks,
                "parent_chunk_id": chunk.parent_chunk_id,
            })
            
            points.append(
                VectorPoint(
                    point_id=point_id,
                    dense_vector=(
                        dense_emb.tolist()
                        if hasattr(dense_emb, "tolist")
                        else list(dense_emb)
                    ),
                    sparse_vector=SparseVector.from_index_values(
                        indices=[int(i) for i in sparse_vec.get("indices", [])],
                        values=[float(v) for v in sparse_vec.get("values", [])],
                    ),
                    payload=payload,
                )
            )
        
        return self.vector_store.upsert_points(
            points,
            collection_name=self.child_collection,
            batch_size=100,
        )


# =============================================================================
# Parent-Child Hybrid Retriever
# =============================================================================


class ParentChildRetriever:
    """
    Hybrid retriever that:
    1. First searches child chunks for precise retrieval
    2. Fetches parent chunks for full context
    3. Returns both for better LLM generation
    """

    def __init__(
        self,
        embedder: Any = None,
        child_collection: str | None = None,
        parent_collection: str | None = None,
    ):
        """
        Initialize the parent-child retriever.
        
        Args:
            embedder: Embedder instance for generating query embeddings
            child_collection: Name of child chunk collection
            parent_collection: Name of parent chunk collection
        """
        from src.vectorstore.registry import get_vector_store
        from src.ml.embedding.embedder import get_embedder
        
        self.vector_store = get_vector_store()
        self.embedder = embedder or get_embedder()
        
        settings = get_settings()
        self.child_collection = child_collection or f"{settings.vector_collection_v2}_child"
        self.parent_collection = parent_collection or f"{settings.vector_collection_v2}_parent"

    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filters: Any = None,
    ) -> tuple[list[RetrievedParentChunk], list[RetrievedParentChunk]]:
        """
        Perform hybrid retrieval: child search + parent fetch.
        
        Args:
            query: Search query string
            top_k: Number of child chunks to retrieve
            filters: Optional metadata filters
            
        Returns:
            Tuple of (child_chunks, parent_chunks_with_children)
        """
        import asyncio
        
        # Generate embeddings for the query
        loop = asyncio.get_running_loop()
        
        # Get dense embedding
        dense_embedding = await loop.run_in_executor(
            None, self.embedder.embed_query, query
        )
        
        # Get sparse vector
        sparse_vector = await loop.run_in_executor(
            None, self.embedder.compute_sparse_vector, query
        )
        
        from src.vectorstore.types import SparseVector
        sparse_query = SparseVector.from_index_values(
            indices=[int(i) for i in sparse_vector.get("indices", [])],
            values=[float(v) for v in sparse_vector.get("values", [])],
        )
        
        # Search child chunks (dense)
        dense_results = await loop.run_in_executor(
            None,
            lambda: self.vector_store.search_dense(
                dense_vector=(
                    dense_embedding.tolist()
                    if hasattr(dense_embedding, "tolist")
                    else list(dense_embedding)
                ),
                top_k=top_k,
                filters=filters,
                collection_name=self.child_collection,
            ),
        )
        
        # Search child chunks (sparse)
        sparse_results = await loop.run_in_executor(
            None,
            lambda: self.vector_store.search_sparse(
                sparse_vector=sparse_query,
                top_k=top_k,
                filters=filters,
                collection_name=self.child_collection,
            ),
        )
        
        # Combine results (simple max-score approach)
        child_chunks = self._combine_results(dense_results, sparse_results, "child")
        
        # Extract parent chunk IDs from child results
        parent_chunk_ids = set()
        for child in child_chunks:
            parent_id = child.metadata.get("parent_chunk_id")
            if parent_id:
                parent_chunk_ids.add(parent_id)
        
        # Fetch parent chunks
        parent_chunks = await self._fetch_parent_chunks(list(parent_chunk_ids))
        
        # Attach child chunks to their parents
        child_map = {child.chunk_id: child for child in child_chunks}
        for parent in parent_chunks:
            child_ids_str = parent.metadata.get("child_chunk_ids", "")
            if child_ids_str:
                child_ids = child_ids_str.split(",")
                parent.child_chunks = [
                    child_map[cid] 
                    for cid in child_ids 
                    if cid in child_map
                ]
        
        return child_chunks, parent_chunks

    def _combine_results(
        self, 
        dense_results: list[Any], 
        sparse_results: list[Any],
        chunk_type: str
    ) -> list[RetrievedParentChunk]:
        """Combine dense and sparse search results."""
        # Use a dict to track best scores per chunk_id
        combined: dict[str, RetrievedParentChunk] = {}
        
        for point in dense_results:
            chunk = self._point_to_chunk(point, "dense")
            existing = combined.get(chunk.chunk_id)
            if existing is None or chunk.score > existing.score:
                combined[chunk.chunk_id] = chunk
        
        for point in sparse_results:
            chunk = self._point_to_chunk(point, "sparse")
            existing = combined.get(chunk.chunk_id)
            if existing is None or chunk.score > existing.score:
                combined[chunk.chunk_id] = chunk
        
        # Sort by score and return
        return sorted(combined.values(), key=lambda x: x.score, reverse=True)

    def _point_to_chunk(self, point: Any, source: str) -> RetrievedParentChunk:
        """Convert a ScoredPoint to a RetrievedParentChunk."""
        payload = point.payload or {}
        return RetrievedParentChunk(
            chunk_id=payload.get("chunk_id", point.point_id),
            doc_id=payload.get("doc_id", ""),
            score=float(point.score),
            text=payload.get("raw_text", "") or payload.get("chunk_text", ""),
            contextual_text=payload.get("contextual_text", ""),
            metadata=payload,
            retrieval_source=source,
        )

    async def _fetch_parent_chunks(
        self, 
        parent_chunk_ids: list[str]
    ) -> list[RetrievedParentChunk]:
        """Fetch parent chunks by their IDs."""
        import asyncio
        
        if not parent_chunk_ids:
            return []
        
        loop = asyncio.get_running_loop()
        
        # For each parent ID, search for it (Milvus doesn't support get by ID directly)
        # We'll search by chunk_id filter
        from src.vectorstore.filters import FilterExpression, FilterCondition, FilterOperator
        
        parent_chunks = []
        
        # Batch fetch - search for each parent
        # Note: In production, consider using a more efficient batch fetch method
        for parent_id in parent_chunk_ids:
            filters = FilterExpression(
                must=[
                    FilterCondition(
                        field="chunk_id",
                        operator=FilterOperator.EQ,
                        value=parent_id,
                    )
                ]
            )
            
            results = await loop.run_in_executor(
                None,
                lambda: self.vector_store.search_dense(
                    dense_vector=[0.0] * get_settings().vector_dim,  # Dummy vector
                    top_k=1,
                    filters=filters,
                    collection_name=self.parent_collection,
                ),
            )
            
            if results:
                parent_chunks.append(self._point_to_chunk(results[0], "parent"))
        
        return parent_chunks


# =============================================================================
# Backward Compatibility Wrapper
# =============================================================================


class HybridRetrieverWithParent:
    """
    Extended HybridRetriever that supports parent chunk retrieval.
    
    This class extends the existing HybridRetriever to add parent chunk
    functionality while maintaining backward compatibility with the 
    existing MongoDB/Milvus setup.
    """

    def __init__(self, embedder: Any = None):
        """
        Initialize the hybrid retriever with parent support.
        
        Args:
            embedder: Optional embedder instance. If not provided,
                      uses the default from settings.
        """
        from src.query.search.retriever import HybridRetriever
        
        # Use existing HybridRetriever for child chunk search
        self.base_retriever = HybridRetriever(embedder=embedder)
        
        # Add parent-child retriever for enhanced retrieval
        self.parent_retriever = ParentChildRetriever(embedder=embedder)

    async def retrieve_with_parent(
        self,
        query: str,
        query_analysis: Any,
        top_k: int = 50,
        include_parent_context: bool = True,
    ) -> tuple[list[Any], list[RetrievedParentChunk]]:
        """
        Retrieve chunks with optional parent context.
        
        Args:
            query: Search query
            query_analysis: Query analysis object with filters and expansion
            top_k: Number of child chunks to retrieve
            include_parent_context: Whether to fetch parent chunks
            
        Returns:
            Tuple of (child_chunks, parent_chunks)
        """
        # Use base retriever for standard hybrid search
        dense_chunks, sparse_chunks = await self.base_retriever.retrieve(
            query=query,
            query_analysis=query_analysis,
            top_k=top_k,
        )
        
        # Optionally fetch parent context
        parent_chunks = []
        if include_parent_context:
            _, parent_chunks = await self.parent_retriever.retrieve(
                query=query,
                top_k=top_k // 2,
                filters=query_analysis.metadata_filters,
            )
        
        # Combine all child chunks
        all_child_chunks = dense_chunks + sparse_chunks
        
        return all_child_chunks, parent_chunks

    async def retrieve(
        self,
        query: str,
        query_analysis: Any,
        top_k: int = 50,
    ) -> tuple[list[Any], list[Any]]:
        """
        Standard retrieval (backward compatible).
        
        Returns tuples of (dense_chunks, sparse_chunks) without parent context.
        """
        return await self.base_retriever.retrieve(
            query=query,
            query_analysis=query_analysis,
            top_k=top_k,
        )


# =============================================================================
# Integration Utilities
# =============================================================================


def create_hierarchical_chunks(
    doc: Any,
    doc_metadata: dict[str, Any],
    parent_tokens: int = 1000,
    child_tokens: int = 350,
    overlap_tokens: int = 50,
) -> HierarchicalChunkResult:
    """
    Utility function to create hierarchical chunks from a document.
    
    Args:
        doc: Document object
        doc_metadata: Document metadata
        parent_tokens: Target tokens for parent chunks
        child_tokens: Target tokens for child chunks
        overlap_tokens: Token overlap between child chunks
        
    Returns:
        HierarchicalChunkResult with parent and child chunks
    """
    parser = HierarchicalChunkParser(
        parent_tokens=parent_tokens,
        child_tokens=child_tokens,
        overlap_tokens=overlap_tokens,
    )
    return parser.parse_document(doc, doc_metadata)


def index_hierarchical_chunks(
    chunks_result: HierarchicalChunkResult,
    embedder: Any,
) -> dict[str, int]:
    """
    Index hierarchical chunks to Milvus.
    
    Args:
        chunks_result: HierarchicalChunkResult from create_hierarchical_chunks
        embedder: Embedder for generating embeddings
        
    Returns:
        Dictionary with 'parent_count' and 'child_count'
    """
    import asyncio
    
    indexer = ParentChildIndexer()
    loop = asyncio.get_event_loop()
    
    # Embed parent chunks
    parent_texts = [p.text for p in chunks_result.parent_chunks]
    parent_embeddings = loop.run_in_executor(
        None, lambda: embedder.embed_documents(parent_texts)
    )
    parent_sparse = [
        embedder.compute_sparse_vector(text) 
        for text in parent_texts
    ]
    
    # Embed child chunks
    child_texts = [c.text for c in chunks_result.child_chunks]
    child_embeddings = loop.run_in_executor(
        None, lambda: embedder.embed_documents(child_texts)
    )
    child_sparse = [
        embedder.compute_sparse_vector(text) 
        for text in child_texts
    ]
    
    # Upsert to Milvus
    parent_count = indexer.upsert_parent_chunks(
        chunks_result.parent_chunks,
        parent_embeddings,
        parent_sparse,
    )
    
    child_count = indexer.upsert_child_chunks(
        chunks_result.child_chunks,
        child_embeddings,
        child_sparse,
    )
    
    return {
        "parent_count": parent_count,
        "child_count": child_count,
    }