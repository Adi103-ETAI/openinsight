"""
Ingestion package — lazy imports to avoid heavy transitive dependencies
at collection time (vector backend SDKs, sentence-transformers, etc.).
"""

__all__ = [
    "IngestionPipeline",
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
    if name == "IngestionPipeline":
        from src.ingestion.pipeline import IngestionPipeline
        return IngestionPipeline
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
    raise AttributeError(f"module 'src.ingestion' has no attribute '{name}'")