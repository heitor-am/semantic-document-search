from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    database_url: str = "sqlite+aiosqlite:///./jobs.db"
    log_level: str = "INFO"
    environment: Literal["development", "production", "test"] = "development"

    # OpenRouter (reused pattern from Q1)
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_chat_model: str = "openai/gpt-4o-mini"
    openrouter_embedding_model: str = "baai/bge-m3"
    openrouter_app_name: str = "Semantic Document Search"
    openrouter_app_url: str = "http://localhost:8000"

    # Qdrant
    qdrant_url: str = ""
    qdrant_api_key: str = ""
    # v2: named-vector schema (dense + bm25 sparse) for Qdrant native hybrid.
    # v1 was single-unnamed-dense and is incompatible; bump requires a
    # re-ingest into the new collection.
    qdrant_collection_version: str = "v2"

    git_sha: str = "dev"


@lru_cache
def get_settings() -> Settings:
    return Settings()
