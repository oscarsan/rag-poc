from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration loaded from environment / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    llm_provider: Literal["claude", "ollama"] = Field(default="claude")

    anthropic_api_key: str = Field(default="", description="Claude API key")
    claude_model: str = Field(default="claude-haiku-4-5")

    ollama_url: str = Field(default="http://localhost:11434")
    ollama_model: str = Field(
        default="hf.co/mradermacher/Llama-Poro-2-8B-Instruct-GGUF:Q6_K"
    )
    ollama_timeout_seconds: float = Field(default=300.0, gt=0)
    ollama_max_tokens: int = Field(default=256, ge=1, le=4096)

    embedding_model: str = Field(default="BAAI/bge-m3")
    embedding_dim: int = Field(default=1024)

    qdrant_url: str = Field(default="http://localhost:6333")
    qdrant_collection: str = Field(default="syote")

    top_k: int = Field(default=5, ge=1, le=50)
    max_history_turns: int = Field(default=4, ge=0, le=20)

    app_host: str = Field(default="0.0.0.0")
    app_port: int = Field(default=8000)
    log_level: str = Field(default="INFO")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
