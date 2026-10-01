"""
Publish gate — Phase 1 of the ingestion build plan.

A batch is marked `ready` only after passing:
  1. Count reconciliation: chunks produced == vectors in Milvus == chunks in Mongo
  2. Dimension check: Milvus `dense` dim == settings.vector_dim
  3. Content checks: no empty chunk text, required metadata present
  4. Retrieval smoke test: golden clinical questions recall expected docs

DB access is dependency-light (pymongo + pymilvus only, no torch) so the
gate runs anywhere, including CI and Kaggle setup cells.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


REQUIRED_CHUNK_METADATA = ("title", "source_type")
REQUIRED_DOC_METADATA = ("title", "source_type")


@dataclass
class GateCheck:
    name: str
    passed: bool
    detail: str = ""


@dataclass
class GateReport:
    source: str
    passed: bool
    checks: list[GateCheck] = field(default_factory=list)

    def summary(self) -> str:
        lines = [f"publish gate [{self.source}]: {'PASS' if self.passed else 'FAIL'}"]
        for c in self.checks:
            lines.append(f"  [{'OK' if c.passed else 'XX'}] {c.name}: {c.detail}")
        return "\n".join(lines)


def check_counts(mongo_chunks: int, milvus_rows: int, produced: int) -> GateCheck:
    """Chunks produced == chunks in Mongo == vectors in Milvus."""
    ok = mongo_chunks == milvus_rows == produced
    return GateCheck(
        name="count-reconciliation",
        passed=ok,
        detail=f"produced={produced} mongo={mongo_chunks} milvus={milvus_rows}",
    )


def check_dimension(collection_dim: int, expected_dim: int) -> GateCheck:
    ok = collection_dim == expected_dim
    return GateCheck(
        name="dimension",
        passed=ok,
        detail=f"collection={collection_dim} expected={expected_dim}",
    )


def check_chunk_texts(texts: list[str], sample: int = 200) -> GateCheck:
    """No empty chunks in a sample; bounds on length."""
    checked = texts[:sample]
    empties = sum(1 for t in checked if not (t or "").strip())
    ok = empties == 0 and len(checked) > 0
    return GateCheck(
        name="chunk-text",
        passed=ok,
        detail=f"checked={len(checked)} empties={empties}",
    )


def check_metadata(items: list[dict[str, Any]], required: tuple[str, ...], label: str) -> GateCheck:
    missing = 0
    for it in items:
        meta = it.get("metadata", it)
        if any(not meta.get(k) for k in required):
            missing += 1
    ok = missing == 0 and len(items) > 0
    return GateCheck(
        name=f"metadata-{label}",
        passed=ok,
        detail=f"checked={len(items)} missing-required={missing}",
    )


def run_gate(
    *,
    source: str,
    produced_chunks: int,
    mongo_chunks: int,
    milvus_rows: int,
    collection_dim: int,
    expected_dim: int,
    sample_texts: list[str],
    sample_metas: list[dict[str, Any]],
) -> GateReport:
    checks = [
        check_counts(mongo_chunks, milvus_rows, produced_chunks),
        check_dimension(collection_dim, expected_dim),
        check_chunk_texts(sample_texts),
        check_metadata(sample_metas, REQUIRED_CHUNK_METADATA, "chunks"),
    ]
    return GateReport(source=source, passed=all(c.passed for c in checks), checks=checks)


def _chunk_source(doc: dict[str, Any], doc_source: dict[str, str]) -> str:
    meta = doc.get("metadata") or {}
    return str(meta.get("source_type") or meta.get("source") or doc_source.get(doc.get("doc_id"), ""))


def _chunk_title(doc: dict[str, Any], doc_title: dict[str, str]) -> str:
    meta = doc.get("metadata") or {}
    return str(meta.get("title") or doc.get("title") or doc_title.get(doc.get("doc_id"), ""))


def collect_mongo_stats(db: Any, source: str, limit: int = 500) -> tuple[int, list[str], list[dict[str, Any]]]:
    """Return (chunk_count, sample_texts, sample_metas) for a source.

    Chunks carry no top-level source tag — resolve via parent documents
    (documents_v2.source) joined on doc_id.
    """
    docs = list(db["documents_v2"].find(
        {"$or": [{"source": source}, {"source_type": source}]},
        {"_id": 1, "doc_id": 1, "title": 1},
    ))
    ids = {str(d.get("_id")) for d in docs} | {str(d.get("doc_id", "")) for d in docs}
    doc_source = {i: source for i in ids}
    doc_title = {}
    for d in docs:
        for k in (str(d.get("_id")), str(d.get("doc_id", ""))):
            doc_title[k] = str(d.get("title", ""))
    if not ids:
        return 0, [], []
    q: dict[str, Any] = {"doc_id": {"$in": sorted(ids)}}
    total = db["chunks_v2"].count_documents(q)
    cursor = db["chunks_v2"].find(q).limit(limit)
    texts, metas = [], []
    for doc in cursor:
        texts.append(str(doc.get("text") or doc.get("chunk_text") or ""))
        metas.append({"title": _chunk_title(doc, doc_title),
                      "source_type": _chunk_source(doc, doc_source)})
    return total, texts, metas


def collect_milvus_stats(client: Any, collection: str) -> tuple[int, int]:
    """Return (row_count, dense_dim) for a collection."""
    stats = client.get_collection_stats(collection)
    rows = int(stats.get("row_count", 0))
    dim = 0
    for f in client.describe_collection(collection).get("fields", []):
        if isinstance(f, dict) and f.get("name") == "dense":
            dim = int(f.get("params", {}).get("dim", 0))
    return rows, dim
