from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # LLM
    openai_api_key: str = ""
    llm_model: str = "gpt-4o-mini"
    llm_max_tokens: int = 512
    llm_temperature: float = 0.0
    
    # Azure OpenAI
    # Azure OpenAI
    azure_openai_key: str = ""
    azure_openai_endpoint: str = ""  
    azure_openai_deployment: str = "gpt-4o-mini-2"
    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "brain_health_kb"
    qdrant_api_key: str = ""

    # Redis
    redis_url: str = "redis://localhost:6379/0"
    cache_ttl_seconds: int = 86400

    # Embeddings
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384

    # RAG
    rag_top_k: int = 5
    rag_score_threshold: float = 0.35
    hybrid_alpha: float = 0.7

    # Batch
    batch_size: int = 500
    max_workers: int = 16
    celery_concurrency: int = 8
    max_retries: int = 3
    retry_delay_seconds: int = 5

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 4

    # Logging
    log_level: str = "INFO"
    log_format: str = "json"

    class Config:
        env_file = "config/.env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
