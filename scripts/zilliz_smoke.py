"""
Zilliz Cloud / Milvus Smoke Test
=================================
Verifies the full round-trip via env-driven settings:
  connect  →  create collection  →  insert  →  dense search  →  sparse search  →  cleanup

Usage (from project root):
    python -m scripts.zilliz_smoke
    # or
    python scripts/zilliz_smoke.py

Requires VECTOR_URI and VECTOR_TOKEN (if using Zilliz Cloud) in .env.
For local Milvus Standalone leave VECTOR_TOKEN blank.
"""

from __future__ import annotations

import random
import sys
import time
from pathlib import Path
from uuid import uuid4

# Ensure project root is on sys.path when run directly
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config import get_settings
from src.vectorstore.filters import FilterCondition, FilterExpression, FilterOperator
from src.vectorstore.registry import get_vector_store
from src.vectorstore.types import SparseVector, VectorPoint


# ── helpers ──────────────────────────────────────────────────────────────────


def _random_dense(dim: int) -> list[float]:
    return [random.random() for _ in range(dim)]


def _random_sparse(width: int = 16) -> SparseVector:
    indices = sorted(random.sample(range(1, 50_000), k=width))
    values = [round(random.random(), 4) for _ in range(width)]
    return SparseVector.from_index_values(indices=indices, values=values)


def _ok(msg: str) -> None:
    print(f"  [OK]  {msg}")


def _fail(step: str, exc: Exception) -> None:
    print(f"\n  [FAIL]  FAILED at step [{step}]: {exc}", file=sys.stderr)
    sys.exit(1)


# ── main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    settings = get_settings()
    print()
    print("=" * 60)
    print("Zilliz / Milvus Smoke Test")
    print("=" * 60)
    print(f"  backend : {settings.vector_backend}")
    print(f"  uri     : {settings.vector_uri}")
    print(f"  token   : {'<set>' if settings.vector_token.strip() else '<empty — local mode>'}")
    print(f"  dim     : {settings.vector_dim}")
    print()

    # ── Step 1: connect ──────────────────────────────────────────────────────
    print("[1/5] Connecting …")
    try:
        store = get_vector_store()
        ok = store.health_check()
        if not ok:
            raise RuntimeError("health_check() returned False")
    except Exception as exc:
        _fail("connect", exc)
    _ok(f"Connected to {settings.vector_uri}")

    # ── Step 2: create smoke collection ──────────────────────────────────────
    smoke_col = f"{settings.vector_collection}_smoke_{uuid4().hex[:6]}"
    print(f"[2/5] Creating smoke collection '{smoke_col}' …")
    try:
        store.ensure_collection(recreate=True, collection_name=smoke_col)
    except Exception as exc:
        _fail("ensure_collection", exc)
    _ok("Collection created (with dense + sparse indexes)")

    # ── Step 3: insert ───────────────────────────────────────────────────────
    n = 200
    print(f"[3/5] Inserting {n} hybrid-vector points …")
    points: list[VectorPoint] = []
    for _ in range(n):
        pid = str(uuid4())
        points.append(
            VectorPoint(
                point_id=pid,
                dense_vector=_random_dense(settings.vector_dim),
                sparse_vector=_random_sparse(),
                payload={
                    "chunk_id": pid,
                    "doc_id": f"doc_{pid[:8]}",
                    "title": "Zilliz smoke-test document",
                    "source_type": "smoke",
                    "source": "smoke",
                    "doc_type": "guideline",
                    "chunk_type": "paragraph",
                    "chunk_text": "Zilliz Cloud smoke test chunk.",
                    "year": 2025,
                    "india_relevant": True,
                    "has_drug_dosing": False,
                    "evidence_level": "unknown",
                    "pmid": "",
                },
            )
        )

    try:
        t0 = time.perf_counter()
        inserted = store.upsert_points(points, collection_name=smoke_col, batch_size=100)
        elapsed = time.perf_counter() - t0
    except Exception as exc:
        _fail("upsert_points", exc)
    _ok(f"Inserted {inserted} points in {elapsed:.3f}s")

    # Allow index to settle (Zilliz Cloud flushes asynchronously)
    time.sleep(2)

    # ── Step 4: search ───────────────────────────────────────────────────────
    filter_expr = FilterExpression.from_conditions([
        FilterCondition("source_type", FilterOperator.EQ, "smoke"),
        FilterCondition("year", FilterOperator.GTE, 2020),
    ])

    print("[4/5] Searching …")

    # Dense
    try:
        t1 = time.perf_counter()
        dense_hits = store.search_dense(
            dense_vector=_random_dense(settings.vector_dim),
            top_k=5,
            filters=filter_expr,
            collection_name=smoke_col,
        )
        dense_ms = (time.perf_counter() - t1) * 1000
    except Exception as exc:
        _fail("search_dense", exc)
    _ok(f"Dense search — {len(dense_hits)} hits in {dense_ms:.1f}ms")

    # Sparse
    try:
        t2 = time.perf_counter()
        sparse_hits = store.search_sparse(
            sparse_vector=_random_sparse(),
            top_k=5,
            filters=filter_expr,
            collection_name=smoke_col,
        )
        sparse_ms = (time.perf_counter() - t2) * 1000
    except Exception as exc:
        _fail("search_sparse", exc)
    _ok(f"Sparse search — {len(sparse_hits)} hits in {sparse_ms:.1f}ms")

    # ── Step 5: cleanup ──────────────────────────────────────────────────────
    print("[5/5] Dropping smoke collection …")
    try:
        store.drop_collection(collection_name=smoke_col)
    except Exception as exc:
        _fail("drop_collection", exc)
    _ok("Smoke collection dropped")

    print()
    print("=" * 60)
    print("  ALL STEPS PASSED — Zilliz / Milvus backend is healthy [OK]")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
