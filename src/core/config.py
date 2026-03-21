from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # LLM
    nvidia_nim_api_key: str
    nvidia_nim_base_url: str = "https://integrate.api.nvidia.com/v1"
    nim_model: str = "meta/llama-3.1-70b-instruct"

    # MongoDB
    mongodb_url: str = "mongodb://localhost:27017"
    mongodb_db: str = "openmed"

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "openmed_chunks"

    # Redis
    redis_url: str = "redis://localhost:6379"

    # PubMed
    ncbi_api_key: str = ""
    ncbi_email: str = "sentarc.ai@gmail.com"

    # Embeddings
    embedding_model: str = "pritamdeka/S-PubMedBert-MS-MARCO"
    embedding_dim: int = 768

    # App
    app_env: str = "development"
    log_level: str = "DEBUG"

    class Config:
        env_file = ".env"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
