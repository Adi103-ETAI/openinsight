"""
OpenInsight Configuration — Hybrid JSON + .env approach.

Configuration loading priority (highest → lowest):
  1. OS environment variables       (production, CI/CD, Docker)
  2. .env file                      (local development secrets)
  3. config.{APP_ENV}.json          (environment-specific non-secret defaults)
  4. config.base.json               (shared defaults for all environments)
  5. Pydantic field defaults        (fallback values in code)

Usage:
  - Secrets (API keys, passwords) → .env file or OS env vars ONLY
  - Tunable parameters → config.base.json (versioned in Git)
  - Environment overrides → config.production.json (versioned in Git)
  - Local overrides → .env (git-ignored)
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import Field
from pydantic_settings import (
    BaseSettings,
    JsonConfigSettingsSource,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)


# ── Project root detection ──────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _calculate_default_workers() -> int:
    """Calculate optimal worker count based on CPU cores.

    Uses 75% of available cores, minimum 2, maximum 16.
    This balances CPU-intensive parsing with system responsiveness.
    """
    cpu_count = os.cpu_count() or 4
    return max(2, min(16, int(cpu_count * 0.75)))


def _get_env() -> str:
    """Determine current environment from OS env or default to 'development'."""
    return os.getenv("APP_ENV", "development")


def _resolve_config_path(filename: str) -> Path:
    """Resolve config file path relative to project root."""
    return _PROJECT_ROOT / filename


# ── Main Settings class ─────────────────────────────────────────────────────

class Settings(BaseSettings):
    """
    OpenInsight application settings.

    All fields retain backward-compatible flat names for existing code.
    Internally, values can come from JSON config files or .env / OS env vars.

    Loading priority:
      1. Constructor kwargs (highest)
      2. OS environment variables
      3. .env file
      4. config.base.json + config.{env}.json
      5. Field defaults (lowest)
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        # JSON config files (loaded BEFORE .env, so .env overrides them)
        json_file=[
            _resolve_config_path("config.base.json"),
            _resolve_config_path(f"config.{_get_env()}.json"),
        ],
        json_file_encoding="utf-8",
    )

    # ===================== LLM =====================
    nvidia_nim_api_key: str = ""
    nvidia_nim_base_url: str = "https://integrate.api.nvidia.com/v1"
    nim_model: str = "meta/llama-3.1-70b-instruct"
    nim_temperature: float = 0.1
    nim_max_tokens: int = 1024

    # ===================== MongoDB =====================
    mongodb_url: str = "mongodb://localhost:27017"
    mongodb_db: str = "openinsight"
    mongodb_max_pool_size: int = 50
    mongodb_min_pool_size: int = 5
    mongodb_max_idle_time_ms: int = 30000
    mongodb_connect_timeout_ms: int = 5000
    mongodb_server_selection_timeout_ms: int = 5000

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
    milvus_cloud: bool = False

    # ===================== Redis =====================
    redis_url: str = "redis://localhost:6379"

    # ===================== PubMed / NCBI =====================
    ncbi_api_key: str = ""
    ncbi_email: str = "sentarc.ai@gmail.com"
    pubmed_rate_limit_seconds: float = 0.34
    pubmed_rate_limit_with_key: float = 0.1

    # ===================== Embeddings & Reranker =====================
    embedding_model: str = "pritamdeka/S-PubMedBert-MS-MARCO"
    embedding_dim: int = 768
    sparse_vocab_size: int = 50000
    dense_model_name: str = "pritamdeka/S-PubMedBert-MS-MARCO"
    reranker_model_name: str = "BAAI/bge-reranker-v2-m3"

    # ===================== GROBID =====================
    grobid_url: str = "http://localhost:8070"
    grobid_timeout: int = 120
    grobid_max_retries: int = 3
    grobid_retry_delay: float = 2.0
    grobid_health_check_timeout: int = 10

    # NLP model for entity extraction (scispacy)
    spacy_model: str = "en_core_sci_md"

    # ===================== Embedding Provider =====================
    # Options: "local", "huggingface", "cohere"
    # - local:       SentenceTransformers on GPU/CPU (requires GPU for good perf)
    # - huggingface: HF Inference API (free tier, same model as local)
    # - cohere:      Cohere Embed API (free trial: 1k calls/mo)
    embed_provider: str = "local"
    hf_api_token: str = ""
    cohere_api_key: str = ""
    cohere_embed_model: str = "embed-english-v3.0"
    hf_embed_model: str = "pritamdeka/S-PubMedBert-MS-MARCO"

    # ===================== Reranker Provider =====================
    # Options: "local", "huggingface", "cohere"
    # - local:       CrossEncoder on GPU/CPU (requires GPU for good perf)
    # - huggingface: HF Inference API text-classification endpoint (free tier)
    # - cohere:      Cohere Rerank API (free trial: 1k calls/mo, proper /rerank endpoint)
    rerank_provider: str = "local"
    hf_rerank_model: str = "BAAI/bge-reranker-v2-m3"
    cohere_rerank_model: str = "rerank-english-v3.0"
    reranker_max_length: int = 1024

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
    reranker_max_length: int = 1024

    # ===================== Caching =====================
    cache_version: str = "v2"
    cache_ttl_search: int = 1800
    cache_ttl_rerank: int = 3600
    cache_ttl_embedding: int = 1800

    # ===================== Search Pipeline =====================
    top_k_retrieval: int = 50
    top_k_after_fusion: int = 20
    top_k_after_rerank: int = 8
    top_k_final: int = 6
    mmr_lambda: float = 0.7
    hyde_enabled: bool = True
    rrf_k: int = 60

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
    nli_model_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"

    # ===================== Distributed Ingestion =====================
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_concurrency: int = 4
    ingestion_workers: int = 4
    ingestion_batch_size: int = 10
    ingestion_max_retries: int = 3
    ingestion_retry_delay: float = 2.0
    ingestion_thread_workers: int = Field(default_factory=_calculate_default_workers)
    ingestion_max_chunks_per_doc: int = 100
    parsing_thread_workers: int = Field(default_factory=_calculate_default_workers)
    max_concurrent_docs: int = 6
    retry_backoff_multiplier: float = 2.0
    retry_max_delay: float = 60.0

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
    hallucination_threshold: float = 0.75

    # ===================== App =====================
    app_env: str = "development"
    log_level: str = "DEBUG"

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """
        Define loading priority:
        1. init_settings     (constructor kwargs — highest)
        2. env_settings      (OS environment variables)
        3. dotenv_settings   (.env file)
        4. json_settings     (config.base.json + config.{env}.json)
        5. field defaults    (lowest)
        """
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            JsonConfigSettingsSource(settings_cls),
        )


@lru_cache()
def get_settings() -> Settings:
    """Cached singleton — call this everywhere instead of Settings()."""
    return Settings()


def reload_settings() -> Settings:
    """Clear cache and reload settings (useful for testing or hot-reload)."""
    get_settings.cache_clear()
    return get_settings()
