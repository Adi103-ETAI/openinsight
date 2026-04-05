"""
Ingestion package — lazy imports to avoid heavy transitive dependencies
at collection time (qdrant-client, sentence-transformers, etc.).
"""

__all__ = [
    "run_pipeline_v3",
    "run_pipeline_v3_from_parser",
    "is_duplicate",
    "enrich_document_hashes",
    "score_chunk",
    "score_chunks",
    "validate_document",
    "validate_chunk",
    "filter_valid_chunks",
    "IngestionMonitor",
    "RunMetrics",
]


def __getattr__(name):  # noqa: N807
    if name in ("run_pipeline_v3", "run_pipeline_v3_from_parser"):
        from src.ingestion.pipeline_v3 import run_pipeline_v3, run_pipeline_v3_from_parser
        return locals()[name]
    if name in ("is_duplicate", "enrich_document_hashes"):
        from src.ingestion.deduplication import is_duplicate, enrich_document_hashes
        return locals()[name]
    if name in ("score_chunk", "score_chunks"):
        from src.ingestion.quality import score_chunk, score_chunks
        return locals()[name]
    if name in ("validate_document", "validate_chunk", "filter_valid_chunks"):
        from src.ingestion.validation import validate_document, validate_chunk, filter_valid_chunks
        return locals()[name]
    if name in ("IngestionMonitor", "RunMetrics"):
        from src.ingestion.monitoring import IngestionMonitor, RunMetrics
        return locals()[name]
    raise AttributeError(f"module 'src.ingestion' has no attribute {name!r}")
