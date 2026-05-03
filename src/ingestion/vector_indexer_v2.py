from __future__ import annotations

import uuid
from typing import Any

from src.vectorstore.registry import get_vector_store
from src.vectorstore.types import SparseVector, VectorPoint


COLLECTION_NAME = "openinsight_v2"


class VectorIndexerV2:
    def __init__(self):
        self.store = get_vector_store()

    def create_collection(
        self, recreate: bool = False, collection_name: str = COLLECTION_NAME
    ) -> None:
        self.store.ensure_collection(recreate=recreate, collection_name=collection_name)

    def upsert_chunks(
        self,
        chunks: list[Any],
        dense_embeddings: Any,
        sparse_vectors: list[dict[str, list[int] | list[float]]],
        collection_name: str = COLLECTION_NAME,
    ) -> int:
        if len(chunks) != len(sparse_vectors) or len(chunks) != len(dense_embeddings):
            raise ValueError(
                "chunks, dense_embeddings, and sparse_vectors must have matching lengths"
            )

        points: list[VectorPoint] = []
        for chunk, dense_emb, sparse_vec in zip(
            chunks, dense_embeddings, sparse_vectors
        ):
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.chunk_id))

            payload = dict(chunk.metadata)
            payload.update(
                {
                    "raw_text": chunk.text,
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "chunk_type": chunk.chunk_type,
                    "section_title": chunk.section_title,
                    "chunk_index": chunk.chunk_index,
                    "total_chunks": chunk.total_chunks,
                }
            )

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

        return self.store.upsert_points(
            points,
            collection_name=collection_name,
            batch_size=100,
        )

