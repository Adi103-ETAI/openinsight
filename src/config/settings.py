import os
from pydantic_settings import BaseSettings
from functools import lru_cache


def _calculate_default_workers() -> int:
    """Calculate optimal worker count based on CPU cores.
    
    Uses 75% of available cores, minimum 2, maximum 16.
    This balances CPU-intensive parsing with system responsiveness.
    """
    cpu_count = os.cpu_count() or 4
    return max(2, min(16, int(cpu_count * 0.75)))


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
    mongodb_max_pool_size: int = 50  # Max connections in pool
    mongodb_min_pool_size: int = 5   # Min connections to maintain
    mongodb_max_idle_time_ms: int = 30000  # Connection idle timeout (ms)
    mongodb_connect_timeout_ms: int = 5000  # Connection timeout (ms)
    mongodb_server_selection_timeout_ms: int = 5000  # Server selection timeout (ms)

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
    milvus_cloud: bool = False  # If using Milvus Cloud (managed), set to True to skip load_collection

    # ===================== Redis =====================
    redis_url: str = "redis://localhost:6379"

    # ===================== PubMed / NCBI =====================
    ncbi_api_key: str = ""
    ncbi_email: str = "sentarc.ai@gmail.com"
    pubmed_rate_limit_seconds: float = 0.34  # API requests without key
    pubmed_rate_limit_with_key: float = 0.1   # API requests with API key

    # ===================== Embeddings & Reranker =====================
    embedding_model: str = "pritamdeka/S-PubMedBert-MS-MARCO"
    embedding_dim: int = 768
    sparse_vocab_size: int = 50000  # Vocabulary size for TF-IDF sparse embeddings
    dense_model_name: str = "pritamdeka/S-PubMedBert-MS-MARCO"
    reranker_model_name: str = "BAAI/bge-reranker-v2-m3"  # Upgraded from bge-reranker-base
    # ===================== GROBID =====================
    grobid_url: str = "http://localhost:8070"
    grobid_timeout: int = 120  # Timeout for GROBID API calls in seconds
    grobid_max_retries: int = 3  # Max number of retries for failed GROBID calls
    grobid_retry_delay: float = 2.0  # Initial delay between retries (seconds)
    grobid_health_check_timeout: int = 10  # Timeout for health check requests

    # NLP model for entity extraction (scispacy)
    spacy_model: str = "en_core_sci_md"

    # ===================== Embedding Provider =====================
    # Options: "local", "huggingface", "cohere"
    # - local:       SentenceTransformers on GPU/CPU (requires GPU for good perf)
    # - huggingface: HF Inference API (free tier, same model as local)
    # - cohere:      Cohere Embed API (free trial: 1k calls/mo)
    embed_provider: str = "local"
    hf_api_token: str = ""  # HuggingFace API token for Inference API
    cohere_api_key: str = ""  # Cohere API key for embeddings/reranking
    cohere_embed_model: str = "embed-english-v3.0"  # Cohere embed model name
    hf_embed_model: str = "pritamdeka/S-PubMedBert-MS-MARCO"  # HF model for embeddings API

    # ===================== Reranker Provider =====================
    # Options: "local", "huggingface", "cohere"
    # - local:       CrossEncoder on GPU/CPU (requires GPU for good perf)
    # - huggingface: HF Inference API text-classification endpoint (free tier)
    # - cohere:      Cohere Rerank API (free trial: 1k calls/mo, proper /rerank endpoint)
    rerank_provider: str = "local"
    hf_rerank_model: str = "BAAI/bge-reranker-v2-m3"  # HF model for reranking API
    cohere_rerank_model: str = "rerank-english-v3.0"  # Cohere rerank model name
    reranker_max_length: int = 1024  # Max sequence length for reranker (1024 for bge-m3)

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
    reranker_max_length: int = 1024  # Max token length for reranker input (1024 for bge-m3)

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
    nli_model_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"  # Medical NLI model

    # ===================== Distributed Ingestion =====================
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_concurrency: int = 4
    ingestion_workers: int = 4
    ingestion_batch_size: int = 10  # files per batch
    ingestion_max_retries: int = 3
    ingestion_retry_delay: float = 2.0
    ingestion_thread_workers: int = _calculate_default_workers()  # Default: CPU-based (75% of cores)
    ingestion_max_chunks_per_doc: int = 100
    parsing_thread_workers: int = _calculate_default_workers()  # Default: CPU-based (75% of cores)
    max_concurrent_docs: int = 6  # Max documents to process concurrently
    retry_backoff_multiplier: float = 2.0  # Exponential backoff multiplier for retries
    retry_max_delay: float = 60.0  # Max delay between retries in seconds

    # ===================== Deduplication =====================
    dedup_enabled: bool = True
    dedup_title_similarity: float = 0.9
    dedup_content_hash_length: int = 16
    cache_key_prefix_length: int = 16

    # ===================== Quality Scoring =====================
    quality_score_threshold: float = 0.3
    quality_high_value_patterns_count: int = 10
    quality_low_value_penalty: float = 0.5

    # ===================== Dead Letter Queue =====================
    dead_letter_enabled: bool = True
    dead_letter_collection: str = "failed_documents"

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

    # ===================== Hallucination Detection =====================
    hallucination_enabled: bool = True
    hallucination_threshold: float = 0.75  # Recommended: 0.70-0.85 for medical RAG

    # ===================== App =====================
    app_env: str = "development"
    log_level: str = "DEBUG"

    class Config:
        env_file = ".env"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
