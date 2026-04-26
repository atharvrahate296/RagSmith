"""
RAGSmith – Centralised configuration
Both Ollama AND Groq are available simultaneously.
The provider used for a query is determined by the project/session's `provider` field.
"""

from functools import lru_cache
from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── App ───────────────────────────────────────────────────────────────────
    app_env: Literal["development", "production"] = "development"
    app_secret_key: str = "change-me-in-production-use-openssl-rand-hex-32"
    log_level: str = "INFO"

    # ── Database ──────────────────────────────────────────────────────────────
    db_driver: Literal["sqlite", "postgres"] = "sqlite"
    database_url: str = ""
    sqlite_path: str = "data/ragsmith22.db"

    # ── LLM Providers — BOTH available simultaneously ─────────────────────────
    # The provider used per-query is determined by the project/session's `provider`
    # field stored in the DB — NOT by a global env switch.

    # Ollama (local)
    ollama_base_url: str = "http://localhost:11434"
    ollama_default_model: str = "mistral:7b"
    ollama_available_models: str = "mistral:7b,gemma:2b,llama2:7b,neural-chat:7b"

    # Groq (cloud) — get free key at https://console.groq.com
    groq_api_key: str = ""
    groq_default_model: str = "llama-3.1-8b-instant"
    groq_available_models: str = "llama-3.1-8b-instant,llama-3.3-70b-versatile,mixtral-8x7b-32768,gemma2-9b-it"

    # ── Embedding model ───────────────────────────────────────────────────────
    embedding_model: str = "all-MiniLM-L6-v2"

    # ── File Storage ──────────────────────────────────────────────────────────
    storage_backend: Literal["local", "s3"] = "local"
    local_upload_dir: str = "data/uploads"
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    aws_region: str = "us-east-1"
    s3_bucket_name: str = ""

    # ── FAISS / Vector indexes ────────────────────────────────────────────────
    faiss_index_dir: str = "data/indexes"
    faiss_chunks_dir: str = "data/chunks"

    # ── Server ────────────────────────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    reload: bool = False

    # ── Upload limits ─────────────────────────────────────────────────────────
    max_upload_mb: int = 50

    # ── CORS ──────────────────────────────────────────────────────────────────
    cors_origins: str = "*"

    @property
    def cors_origins_list(self) -> list[str]:
        if self.cors_origins == "*":
            return ["*"]
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    def default_model_for(self, provider: str) -> str:
        """Return the configured default model for the given provider."""
        if provider == "groq":
            return self.groq_default_model
        return self.ollama_default_model


@lru_cache
def get_settings() -> Settings:
    return Settings()