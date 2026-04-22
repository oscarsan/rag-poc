from __future__ import annotations

from functools import lru_cache

from app.config import get_settings
from app.providers.embeddings import BgeM3EmbeddingProvider
from app.providers.llm import ClaudeProvider
from app.providers.vectorstore import QdrantStore
from app.services import RagService


@lru_cache(maxsize=1)
def get_rag_service() -> RagService:
    """Build the RagService once per process, wiring real providers.

    Tests can bypass this by using FastAPI's dependency_overrides.
    """
    settings = get_settings()
    embeddings = BgeM3EmbeddingProvider(model_name=settings.embedding_model)
    store = QdrantStore(url=settings.qdrant_url, collection=settings.qdrant_collection)
    llm = ClaudeProvider(api_key=settings.anthropic_api_key, model=settings.claude_model)
    return RagService(
        embeddings,
        store,
        llm,
        top_k=settings.top_k,
        max_history_turns=settings.max_history_turns,
    )
