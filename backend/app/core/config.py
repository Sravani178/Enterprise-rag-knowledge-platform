from functools import lru_cache
from uuid import UUID

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables or .env."""

    app_name: str = "Enterprise AI Knowledge Platform"
    environment: str = "development"
    api_v1_prefix: str = "/api/v1"
    database_url: str = "postgresql+psycopg://rag:rag@localhost:5432/rag"
    db_pool_size: int = 10
    db_max_overflow: int = 20
    db_pool_timeout_seconds: float = 30.0
    db_pool_recycle_seconds: int = 1800
    cors_origins: str = "http://localhost:5173"
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/1"
    celery_soft_time_limit_seconds: int = 600
    celery_time_limit_seconds: int = 660
    celery_max_retries: int = 3
    storage_endpoint_url: str = "http://localhost:9000"
    storage_access_key: str = "minioadmin"
    storage_secret_key: str = "minioadmin"
    storage_bucket: str = "documents"
    storage_region: str = "us-east-1"
    storage_connect_timeout_seconds: float = 5.0
    storage_read_timeout_seconds: float = 30.0
    storage_orphan_grace_seconds: int = 900
    max_upload_size_bytes: int = 25 * 1024 * 1024
    default_organization_id: UUID = UUID("00000000-0000-0000-0000-000000000001")
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "document_chunks_openai_1536"
    embedding_provider: str = "openai"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimension: int = 1536
    chunk_size: int = 800
    chunk_overlap: int = 120
    semantic_chunking_enabled: bool = True
    semantic_chunk_similarity_threshold: float = 0.78
    semantic_chunk_min_size: int = 200
    vector_candidate_k: int = 20
    bm25_candidate_k: int = 20
    rrf_k: int = 60
    reranker_provider: str = "cross_encoder"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_model_revision: str | None = None
    reranker_device: str = "auto"
    reranker_max_length: int = 512
    reranker_max_text_chars: int = 4000
    reranker_batch_size: int = 16
    reranker_max_concurrency: int = 2
    reranker_timeout_seconds: float = 10.0
    reranker_fallback_provider: str = "lexical"
    reranker_candidate_k: int = 20
    min_retrieval_score: float = 0.05
    processing_stale_after_seconds: int = 900
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o-mini"
    openai_api_key: str | None = None
    openai_base_url: str = "https://api.openai.com/v1"
    llm_timeout_seconds: float = 30.0
    openai_max_retries: int = 2
    openai_retry_backoff_seconds: float = 0.5
    query_understanding_enabled: bool = True
    query_understanding_max_tokens: int = 200
    answer_evaluation_enabled: bool = True
    answer_evaluation_max_attempts: int = 2
    answer_evaluation_min_score: float = 0.75
    question_cache_enabled: bool = True
    question_cache_redis_url: str = "redis://localhost:6379/2"
    question_cache_ttl_seconds: int = 86400
    question_cache_similarity_threshold: float = 0.94
    question_cache_max_candidates: int = 50
    question_cache_lock_seconds: int = 45
    redis_connect_timeout_seconds: float = 0.5
    redis_socket_timeout_seconds: float = 1.0
    redis_max_connections: int = 100
    redis_pool_timeout_seconds: float = 2.0
    celery_broker_pool_limit: int = 20
    query_concurrency_limit: int = 100
    query_concurrency_wait_seconds: float = 10.0
    query_concurrency_distributed_enabled: bool = True
    query_concurrency_redis_url: str = "redis://localhost:6379/3"
    query_concurrency_lease_seconds: int = 120
    auth_required: bool = False
    jwt_secret: str = "development-only-change-this-secret"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60
    rate_limit_enabled: bool = False
    rate_limit_per_minute: int = 60

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @model_validator(mode="after")
    def validate_runtime_limits(self) -> "Settings":
        production_environment = self.environment.strip().lower() in {
            "production",
            "prod",
        }
        if production_environment and not self.auth_required:
            raise ValueError("AUTH_REQUIRED must be enabled in production")
        if production_environment and not self.rate_limit_enabled:
            raise ValueError("RATE_LIMIT_ENABLED must be enabled in production")
        if self.db_pool_size <= 0:
            raise ValueError("DB_POOL_SIZE must be positive")
        if self.db_max_overflow < 0:
            raise ValueError("DB_MAX_OVERFLOW must be non-negative")
        if self.db_pool_timeout_seconds <= 0:
            raise ValueError("DB_POOL_TIMEOUT_SECONDS must be positive")
        if self.db_pool_recycle_seconds <= 0:
            raise ValueError("DB_POOL_RECYCLE_SECONDS must be positive")
        if self.celery_soft_time_limit_seconds <= 0:
            raise ValueError("CELERY_SOFT_TIME_LIMIT_SECONDS must be positive")
        if self.celery_time_limit_seconds <= self.celery_soft_time_limit_seconds:
            raise ValueError(
                "CELERY_TIME_LIMIT_SECONDS must be greater than CELERY_SOFT_TIME_LIMIT_SECONDS"
            )
        if self.celery_max_retries < 0:
            raise ValueError("CELERY_MAX_RETRIES must be non-negative")
        if self.chunk_size <= 0:
            raise ValueError("CHUNK_SIZE must be positive")
        if self.chunk_overlap < 0 or self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be non-negative and smaller than CHUNK_SIZE")
        if not 0.0 <= self.semantic_chunk_similarity_threshold <= 1.0:
            raise ValueError("SEMANTIC_CHUNK_SIMILARITY_THRESHOLD must be between 0 and 1")
        if self.semantic_chunk_min_size < 0 or self.semantic_chunk_min_size >= self.chunk_size:
            raise ValueError("SEMANTIC_CHUNK_MIN_SIZE must be smaller than CHUNK_SIZE")
        if self.vector_candidate_k <= 0 or self.bm25_candidate_k <= 0:
            raise ValueError("Vector and BM25 candidate limits must be positive")
        if self.reranker_candidate_k <= 0:
            raise ValueError("RERANKER_CANDIDATE_K must be positive")
        if self.reranker_candidate_k > 50:
            raise ValueError("RERANKER_CANDIDATE_K must not exceed 50")
        if self.reranker_max_length <= 0:
            raise ValueError("RERANKER_MAX_LENGTH must be positive")
        if self.reranker_max_text_chars <= 0:
            raise ValueError("RERANKER_MAX_TEXT_CHARS must be positive")
        if self.reranker_batch_size <= 0:
            raise ValueError("RERANKER_BATCH_SIZE must be positive")
        if self.reranker_max_concurrency <= 0:
            raise ValueError("RERANKER_MAX_CONCURRENCY must be positive")
        if self.reranker_timeout_seconds <= 0:
            raise ValueError("RERANKER_TIMEOUT_SECONDS must be positive")
        if self.reranker_fallback_provider not in {"none", "lexical"}:
            raise ValueError("RERANKER_FALLBACK_PROVIDER must be 'none' or 'lexical'")
        if self.processing_stale_after_seconds <= 0:
            raise ValueError("PROCESSING_STALE_AFTER_SECONDS must be positive")
        if self.storage_orphan_grace_seconds <= 0:
            raise ValueError("STORAGE_ORPHAN_GRACE_SECONDS must be positive")
        if self.storage_connect_timeout_seconds <= 0:
            raise ValueError("STORAGE_CONNECT_TIMEOUT_SECONDS must be positive")
        if self.storage_read_timeout_seconds <= 0:
            raise ValueError("STORAGE_READ_TIMEOUT_SECONDS must be positive")
        if self.auth_required:
            if len(self.jwt_secret) < 32:
                raise ValueError(
                    "JWT_SECRET must contain at least 32 characters when auth is enabled"
                )
            if self.jwt_secret == "development-only-change-this-secret":
                raise ValueError("JWT_SECRET must be changed when auth is enabled")
            if self.jwt_algorithm not in {"HS256", "HS384", "HS512"}:
                raise ValueError("JWT_ALGORITHM must be a supported HMAC algorithm")
        if self.access_token_expire_minutes <= 0:
            raise ValueError("ACCESS_TOKEN_EXPIRE_MINUTES must be positive")
        if self.rate_limit_per_minute <= 0:
            raise ValueError("RATE_LIMIT_PER_MINUTE must be positive")
        if self.answer_evaluation_max_attempts <= 0:
            raise ValueError("ANSWER_EVALUATION_MAX_ATTEMPTS must be positive")
        if not 0.0 <= self.answer_evaluation_min_score <= 1.0:
            raise ValueError("ANSWER_EVALUATION_MIN_SCORE must be between 0 and 1")
        if self.question_cache_ttl_seconds <= 0:
            raise ValueError("QUESTION_CACHE_TTL_SECONDS must be positive")
        if not 0.0 <= self.question_cache_similarity_threshold <= 1.0:
            raise ValueError("QUESTION_CACHE_SIMILARITY_THRESHOLD must be between 0 and 1")
        if self.question_cache_max_candidates <= 0:
            raise ValueError("QUESTION_CACHE_MAX_CANDIDATES must be positive")
        if self.question_cache_lock_seconds <= 0:
            raise ValueError("QUESTION_CACHE_LOCK_SECONDS must be positive")
        if self.redis_connect_timeout_seconds <= 0:
            raise ValueError("REDIS_CONNECT_TIMEOUT_SECONDS must be positive")
        if self.redis_socket_timeout_seconds <= 0:
            raise ValueError("REDIS_SOCKET_TIMEOUT_SECONDS must be positive")
        if self.redis_max_connections <= 0:
            raise ValueError("REDIS_MAX_CONNECTIONS must be positive")
        if self.redis_pool_timeout_seconds <= 0:
            raise ValueError("REDIS_POOL_TIMEOUT_SECONDS must be positive")
        if self.celery_broker_pool_limit <= 0:
            raise ValueError("CELERY_BROKER_POOL_LIMIT must be positive")
        if self.query_concurrency_limit <= 0:
            raise ValueError("QUERY_CONCURRENCY_LIMIT must be positive")
        if self.query_concurrency_wait_seconds <= 0:
            raise ValueError("QUERY_CONCURRENCY_WAIT_SECONDS must be positive")
        if self.query_concurrency_lease_seconds <= 0:
            raise ValueError("QUERY_CONCURRENCY_LEASE_SECONDS must be positive")
        if self.openai_max_retries < 0:
            raise ValueError("OPENAI_MAX_RETRIES must be non-negative")
        if self.openai_retry_backoff_seconds <= 0:
            raise ValueError("OPENAI_RETRY_BACKOFF_SECONDS must be positive")
        if self.query_understanding_max_tokens <= 0:
            raise ValueError("QUERY_UNDERSTANDING_MAX_TOKENS must be positive")
        return self

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
