from __future__ import annotations

from functools import lru_cache

from app.config import Settings, get_settings
from app.providers.embeddings import BgeM3EmbeddingProvider
from app.providers.llm import ClaudeProvider, LLMProvider, OllamaProvider
from app.providers.vectorstore import QdrantStore
from app.services import RagService


def _build_llm(settings: Settings) -> LLMProvider:
    if settings.llm_provider == "ollama":
        return OllamaProvider(
            url=settings.ollama_url,
            model=settings.ollama_model,
            timeout=settings.ollama_timeout_seconds,
            max_tokens=settings.ollama_max_tokens,
        )
    return ClaudeProvider(
        api_key=settings.anthropic_api_key, model=settings.claude_model
    )


@lru_cache(maxsize=1)
def get_rag_service() -> RagService:
    """Build the RagService once per process, wiring real providers.

    Tests can bypass this by using FastAPI's dependency_overrides.
    """
    settings = get_settings()
    embeddings = BgeM3EmbeddingProvider(model_name=settings.embedding_model)
    store = QdrantStore(url=settings.qdrant_url, collection=settings.qdrant_collection)
    llm = _build_llm(settings)
    return RagService(
        embeddings,
        store,
        llm,
        top_k=settings.top_k,
        max_history_turns=settings.max_history_turns,
    )
