from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # ===================== LLM =====================
    nvidia_nim_api_key: str
    nvidia_nim_base_url: str = "https://integrate.api.nvidia.com/v1"
    nim_model: str = "meta/llama-3.1-70b-instruct"
    nim_temperature: float = 0.1
    nim_max_tokens: int = 1024

    # ===================== MongoDB =====================
    mongodb_url: str = "mongodb://localhost:27017"
    mongodb_db: str = "openinsight"

    # ===================== Vector DB =====================
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

    # ===================== Redis =====================
    redis_url: str = "redis://localhost:6379"

    # ===================== PubMed / NCBI =====================
    ncbi_api_key: str = ""
    ncbi_email: str = "sentarc.ai@gmail.com"

    # ===================== Embeddings & Reranker =====================
    embedding_model: str = "pritamdeka/S-PubMedBert-MS-MARCO"
    embedding_dim: int = 768
    dense_model_name: str = "pritamdeka/S-PubMedBert-MS-MARCO"
    reranker_model_name: str = "BAAI/bge-reranker-base"
    grobid_url: str = "http://localhost:8070"
    
    # NLP model for entity extraction (scispacy)
    spacy_model: str = "en_core_sci_md"

    # ===================== Embedding Processing =====================
    embedding_batch_size: int = 32
    embedding_retry_batch_size: int = 16
    embedding_timeout: int = 60

    # ===================== Retrieval =====================
    retrieval_top_k: int = 8
    retrieval_multiplier: int = 3
    retrieval_min_k: int = 20
    retrieval_max_k: int = 30

    # ===================== Reranking =====================
    reranker_top_n: int = 8
    reranker_batch_size: int = 16
    reranker_max_chars: int = 1200

    # ===================== Caching =====================
    cache_version: str = "v2"
    cache_ttl_search: int = 1800  # 30 minutes
    cache_ttl_rerank: int = 3600  # 1 hour
    cache_ttl_embedding: int = 1800  # 30 minutes

    # ===================== Search Pipeline =====================
    top_k_retrieval: int = 50
    top_k_after_fusion: int = 20
    top_k_after_rerank: int = 8
    top_k_final: int = 6
    mmr_lambda: float = 0.7
    hyde_enabled: bool = True
    rrf_k: int = 60  # Reciprocal Rank Fusion parameter

    # ===================== Query Rewriting =====================
    llm_query_rewrite: bool = True
    query_rewrite_fallback: bool = True
    query_rewrite_max_tokens: int = 64
    query_rewrite_temperature: float = 0.0
    hyde_timeout: float = 15.0

    # ===================== DeepInsights / Agents =====================
    deep_insights_enabled: bool = True
    deep_insights_max_sub_queries: int = 6
    deep_insights_sub_query_top_k: int = 8
    deep_insights_timeout: int = 60
    deep_insights_context_chars: int = 300

    # ===================== Contradiction Detection =====================
    contradiction_detection: bool = True
    contradiction_min_chunks: int = 3

    # ===================== Distributed Ingestion =====================
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_concurrency: int = 4
    ingestion_workers: int = 4
    ingestion_batch_size: int = 10  # files per batch
    ingestion_max_retries: int = 3
    ingestion_retry_delay: float = 2.0
    ingestion_thread_workers: int = 4
    ingestion_max_chunks_per_doc: int = 100
    parsing_thread_workers: int = 4

    # ===================== Deduplication =====================
    dedup_enabled: bool = True
    dedup_title_similarity: float = 0.9
    dedup_content_hash_length: int = 16
    cache_key_prefix_length: int = 16

    # ===================== Deduplication =====================
    dedup_enabled: bool = True
    dedup_title_similarity: float = 0.9
    dedup_content_hash_length: int = 16

    # ===================== Quality Scoring =====================
    quality_score_threshold: float = 0.3
    quality_high_value_patterns_count: int = 10
    quality_low_value_penalty: float = 0.5

    # ===================== Chunking =====================
    chunk_target_tokens: int = 350
    chunk_max_tokens: int = 500
    chunk_overlap_tokens: int = 50
    chunk_min_tokens: int = 80

    # ===================== Scheduler =====================
    scheduler_pubmed_cron: str = "0 2 * * 0"
    scheduler_who_cron: str = "0 3 1 * *"
    scheduler_cdc_cron: str = "0 4 1 * *"
    scheduler_cochrane_cron: str = "0 5 1 * *"
    scheduler_enabled: bool = True

    # ===================== App =====================
    app_env: str = "development"
    log_level: str = "DEBUG"

    class Config:
        env_file = ".env"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
