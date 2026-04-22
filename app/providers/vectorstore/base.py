from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from app.domain import Chunk, Language, RetrievedChunk


class VectorStore(ABC):
    """Abstract vector store for RAG retrieval."""

    @abstractmethod
    def ensure_collection(self, dimension: int) -> None:
        """Create the collection if missing. Recreate if dimension mismatches."""

    @abstractmethod
    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> int:
        """Upsert chunks with their vectors. Returns number of points written."""

    @abstractmethod
    def search(
        self,
        vector: Sequence[float],
        *,
        top_k: int,
        language: Language | None = None,
    ) -> list[RetrievedChunk]:
        """Return top-k most similar chunks, optionally filtered by language."""
