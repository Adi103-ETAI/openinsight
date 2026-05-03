from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # LLM
    nvidia_nim_api_key: str
    nvidia_nim_base_url: str = "https://integrate.api.nvidia.com/v1"
    nim_model: str = "meta/llama-3.1-70b-instruct"

    # MongoDB
    mongodb_url: str = "mongodb://localhost:27017"
    mongodb_db: str = "openinsight"

    # Vector DB (backend-agnostic app config; Milvus/Zilliz is current provider)
    vector_backend: str = "milvus"
    vector_uri: str = "http://localhost:19530"
    vector_token: str = ""
    vector_collection: str = "openinsight_chunks"
    vector_collection_v2: str = "openinsight_v2"
    vector_dim: int = 768
    vector_id_field: str = "id"
    vector_dense_field: str = "dense"
    vector_sparse_field: str = "sparse"
    vector_dense_metric: str = "COSINE"
    vector_sparse_metric: str = "IP"
    milvus_db_name: str = "default"

    # Redis
    redis_url: str = "redis://localhost:6379"

    # PubMed / NCBI
    ncbi_api_key: str = ""
    ncbi_email: str = "sentarc.ai@gmail.com"

    # Embeddings
    embedding_model: str = "pritamdeka/S-PubMedBert-MS-MARCO"
    embedding_dim: int = 768
    dense_model_name: str = "pritamdeka/S-PubMedBert-MS-MARCO"
    reranker_model_name: str = "BAAI/bge-reranker-base"
    grobid_url: str = "http://localhost:8070"
    nim_temperature: float = 0.1
    nim_max_tokens: int = 1024
    retrieval_top_k: int = 8
    retrieval_multiplier: int = 3
    retrieval_min_k: int = 20
    retrieval_max_k: int = 30
    reranker_top_n: int = 8
    reranker_batch_size: int = 16
    reranker_max_chars: int = 1200

    # v2 retrieval pipeline knobs (additive, backward-compatible)
    cache_version: str = "v2"
    cache_ttl_search: int = 1800
    cache_ttl_rerank: int = 3600
    top_k_retrieval: int = 50
    top_k_after_fusion: int = 20
    top_k_after_rerank: int = 8
    top_k_final: int = 6
    mmr_lambda: float = 0.7
    hyde_enabled: bool = True

    # Phase 6 cutover and deprecation controls
    query_default_pipeline: str = "v2"  # allowed: v2, legacy
    enable_legacy_query: bool = True
    query_auto_fallback_to_legacy: bool = True
    enable_query_deprecation_headers: bool = True

    # Ingestion pipeline
    ingestion_batch_size: int = 50  # documents per batch
    ingestion_max_retries: int = 3  # retry attempts for failed documents
    ingestion_retry_delay: float = 2.0  # seconds between retries
    quality_score_threshold: float = 0.3  # drop chunks below this score
    dedup_title_similarity: float = 0.9  # threshold for fuzzy title dedup

    # Scheduler — cron-style (used by APScheduler)
    scheduler_pubmed_cron: str = "0 2 * * 0"  # Sundays 02:00 UTC
    scheduler_who_cron: str = "0 3 1 * *"  # 1st of month 03:00 UTC
    scheduler_cdc_cron: str = "0 4 1 * *"  # 1st of month 04:00 UTC
    scheduler_cochrane_cron: str = "0 5 1 * *"  # 1st of month 05:00 UTC

    # App
    app_env: str = "development"
    log_level: str = "DEBUG"

    class Config:
        env_file = ".env"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
