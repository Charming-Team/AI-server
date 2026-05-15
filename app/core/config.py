from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "S-MAP AI Server"
    environment: str = "local"
    api_v1_prefix: str = "/api/v1"
    cors_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:3000", "http://127.0.0.1:3000"]
    )
    evidence_lookup_enabled: bool = False
    evidence_lookup_base_url: str = "http://localhost:8080"
    evidence_lookup_path: str = "/internal/chat/evidence"
    evidence_lookup_internal_token: str | None = None
    evidence_lookup_timeout_seconds: float = 10.0
    qdrant_search_enabled: bool = False
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection: str = "smap_internal_documents"
    qdrant_top_k: int = 5
    qdrant_score_threshold: float = 0.0
    qdrant_timeout_seconds: float = 10.0
    document_chunk_size: int = 800
    document_chunk_overlap: int = 80
    document_index_internal_token: str | None = None
    embedding_enabled: bool = False
    embedding_provider: str = "huggingface"
    embedding_base_url: str = "http://localhost:8002"
    embedding_path: str = "/embed"
    embedding_api_key: str | None = None
    embedding_model: str = "BAAI/bge-m3"
    embedding_dimension: int = 1024
    embedding_timeout_seconds: float = 30.0
    llm_enabled: bool = False
    llm_provider: str = "openai_compatible"
    llm_base_url: str = "http://localhost:8001/v1"
    llm_api_key: str | None = None
    llm_model: str = "local-open-source-model"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 1024
    llm_timeout_seconds: float = 60.0
    prompt_max_evidence_items: int = 5
    prompt_max_document_sources: int = 5
    prompt_max_summary_chars: int = 700
    prompt_max_data_chars: int = 1000

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, value: str | list[str]) -> list[str]:
        if isinstance(value, str):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()
