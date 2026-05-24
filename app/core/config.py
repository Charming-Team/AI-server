import json
from functools import lru_cache
from typing import Annotated

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "S-MAP AI Server"
    environment: str = "local"
    api_v1_prefix: str = "/api/v1"
    cors_origins: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["http://localhost:3000", "http://127.0.0.1:3000"]
    )
    chat_answer_internal_token: str | None = None
    chat_recommendation_internal_token: str | None = None
    evidence_lookup_enabled: bool = False
    evidence_lookup_base_url: str = "http://localhost:8080"
    evidence_lookup_path: str = "/internal/chat/evidence"
    evidence_lookup_internal_token: str | None = None
    evidence_lookup_timeout_seconds: float = 10.0
    rdb_evidence_enabled: bool = False
    rdb_evidence_dsn: str | None = None
    rdb_evidence_timeout_seconds: float = 5.0
    rdb_evidence_max_limit: int = Field(default=20, ge=1, le=100)
    qdrant_search_enabled: bool = False
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection: str = "smap_internal_documents"
    qdrant_top_k: int = Field(default=5, ge=1, le=20)
    qdrant_score_threshold: float = Field(default=0.0, ge=0.0, le=1.0)
    qdrant_timeout_seconds: float = 10.0
    document_content_max_chars: int = Field(default=100_000, ge=1_000, le=1_000_000)
    document_max_chunks: int = Field(default=200, ge=1, le=1_000)
    document_chunk_size: int = Field(default=800, ge=1, le=5_000)
    document_chunk_overlap: int = Field(default=80, ge=0, le=5_000)
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
    openai_api_key: str | None = None
    llm_model: str = "local-open-source-model"
    llm_allowed_models: Annotated[list[str], NoDecode] = Field(default_factory=list)
    llm_reasoning_effort: str = "minimal"
    llm_temperature: float = 0.1
    llm_max_tokens: int = Field(default=1024, ge=1, le=4096)
    llm_timeout_seconds: float = 60.0
    llm_response_cache_enabled: bool = True
    llm_response_cache_ttl_seconds: float = Field(default=60.0, ge=0.0, le=3600.0)
    llm_response_cache_max_entries: int = Field(default=128, ge=0, le=10_000)
    answer_max_chars: int = Field(default=1600, ge=100, le=5000)
    prompt_max_evidence_items: int = Field(default=3, ge=0, le=20)
    prompt_max_document_sources: int = Field(default=3, ge=0, le=20)
    prompt_max_summary_chars: int = Field(default=360, ge=1, le=2_000)
    prompt_max_data_chars: int = Field(default=500, ge=1, le=5_000)
    prompt_max_total_chars: int = Field(default=4_000, ge=1_000, le=20_000)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, value: str | list[str]) -> list[str]:
        return cls._parse_string_list(value, field_label="CORS_ORIGINS")

    @field_validator("llm_allowed_models", mode="before")
    @classmethod
    def parse_llm_allowed_models(cls, value: str | list[str] | None) -> list[str]:
        if value is None:
            return []
        return cls._parse_string_list(value, field_label="LLM_ALLOWED_MODELS")

    @staticmethod
    def _parse_string_list(value: str | list[str], field_label: str) -> list[str]:
        if isinstance(value, str):
            stripped_value = value.strip()
            if not stripped_value:
                return []
            if stripped_value.startswith("["):
                parsed_value = json.loads(stripped_value)
                if not isinstance(parsed_value, list):
                    raise ValueError(f"{field_label} JSON 값은 배열이어야 합니다.")
                return [
                    origin.strip()
                    for origin in parsed_value
                    if isinstance(origin, str) and origin.strip()
                ]
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return [item.strip() for item in value if item.strip()]

    @model_validator(mode="after")
    def validate_document_chunk_settings(self) -> "Settings":
        if self.document_chunk_overlap >= self.document_chunk_size:
            raise ValueError("document_chunk_overlap must be smaller than document_chunk_size")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
