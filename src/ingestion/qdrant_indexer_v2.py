from __future__ import annotations

import importlib
import uuid
from typing import Any


COLLECTION_NAME = "openinsight_v2"
DENSE_DIM = 768


class QdrantIndexerV2:
    def __init__(
        self, qdrant_url: str = "http://localhost:6333", qdrant_api_key: str = ""
    ):
        qdrant_client_mod = importlib.import_module("qdrant_client")
        QdrantClient = getattr(qdrant_client_mod, "QdrantClient")

        if qdrant_api_key:
            self.client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        else:
            self.client = QdrantClient(url=qdrant_url)

    def create_collection(
        self, recreate: bool = False, collection_name: str = COLLECTION_NAME
    ) -> None:
        models = importlib.import_module("qdrant_client.http.models")
        exceptions = importlib.import_module("qdrant_client.http.exceptions")
        UnexpectedResponse = getattr(exceptions, "UnexpectedResponse")

        VectorParams = getattr(models, "VectorParams")
        Distance = getattr(models, "Distance")
        SparseVectorParams = getattr(models, "SparseVectorParams")
        SparseIndexParams = getattr(models, "SparseIndexParams")
        HnswConfigDiff = getattr(models, "HnswConfigDiff")
        PayloadSchemaType = getattr(models, "PayloadSchemaType")

        if recreate:
            try:
                self.client.delete_collection(collection_name)
            except (UnexpectedResponse, ValueError, RuntimeError):
                pass

        existing = [c.name for c in self.client.get_collections().collections]
        if collection_name in existing and not recreate:
            return

        self.client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "dense": VectorParams(
                    size=DENSE_DIM, distance=Distance.COSINE, on_disk=False
                ),
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(index=SparseIndexParams(on_disk=False)),
            },
            hnsw_config=HnswConfigDiff(m=16, ef_construct=128),
        )

        payload_indexes = [
            ("year", PayloadSchemaType.INTEGER),
            ("doc_type", PayloadSchemaType.KEYWORD),
            ("source", PayloadSchemaType.KEYWORD),
            ("evidence_level", PayloadSchemaType.KEYWORD),
            ("specialty", PayloadSchemaType.KEYWORD),
            ("india_relevant", PayloadSchemaType.BOOL),
            ("has_drug_dosing", PayloadSchemaType.BOOL),
            ("chunk_type", PayloadSchemaType.KEYWORD),
            ("pmid", PayloadSchemaType.KEYWORD),
        ]

        for field_name, schema_type in payload_indexes:
            self.client.create_payload_index(
                collection_name=collection_name,
                field_name=field_name,
                field_schema=schema_type,
            )

    def upsert_chunks(
        self,
        chunks: list[Any],
        dense_embeddings: Any,
        sparse_vectors: list[dict[str, list[int] | list[float]]],
        collection_name: str = COLLECTION_NAME,
    ) -> int:
        models = importlib.import_module("qdrant_client.http.models")
        PointStruct = getattr(models, "PointStruct")
        SparseVector = getattr(models, "SparseVector")

        if len(chunks) != len(sparse_vectors) or len(chunks) != len(dense_embeddings):
            raise ValueError(
                "chunks, dense_embeddings, and sparse_vectors must have matching lengths"
            )

        points: list[Any] = []
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
                PointStruct(
                    id=point_id,
                    vector={
                        "dense": (
                            dense_emb.tolist()
                            if hasattr(dense_emb, "tolist")
                            else list(dense_emb)
                        ),
                        "sparse": SparseVector(
                            indices=[int(i) for i in sparse_vec.get("indices", [])],
                            values=[float(v) for v in sparse_vec.get("values", [])],
                        ),
                    },
                    payload=payload,
                )
            )

        batch_size = 100
        for i in range(0, len(points), batch_size):
            self.client.upsert(
                collection_name=collection_name, points=points[i : i + batch_size]
            )

        return len(points)
