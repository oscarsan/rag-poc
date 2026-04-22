from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from app.domain import Chunk
from app.ingestion.chunker import documents_to_chunks
from app.ingestion.parser import iter_markdown_files, parse_markdown_file
from app.providers.embeddings import EmbeddingProvider
from app.providers.vectorstore import VectorStore

log = logging.getLogger(__name__)


@dataclass
class IngestionReport:
    files_processed: int
    chunks_created: int
    points_upserted: int
    errors: list[str]


class IngestionPipeline:
    """Parse markdown → chunk → embed → upsert. Idempotent via chunk IDs."""

    def __init__(
        self,
        embeddings: EmbeddingProvider,
        store: VectorStore,
        *,
        max_words: int = 400,
        batch_size: int = 32,
    ) -> None:
        self._embeddings = embeddings
        self._store = store
        self._max_words = max_words
        self._batch_size = batch_size

    def run(self, content_root: Path) -> IngestionReport:
        errors: list[str] = []
        docs = []
        files = iter_markdown_files(content_root)
        for path in files:
            try:
                docs.append(parse_markdown_file(path))
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{path}: {exc}")
                log.exception("Failed to parse %s", path)

        chunks = documents_to_chunks(docs, max_words=self._max_words)
        log.info("Parsed %d files into %d chunks", len(docs), len(chunks))

        if not chunks:
            return IngestionReport(len(files), 0, 0, errors)

        self._store.ensure_collection(self._embeddings.dimension)

        upserted = 0
        for batch in _batched(chunks, self._batch_size):
            vectors = self._embeddings.embed([c.text for c in batch])
            upserted += self._store.upsert(batch, vectors)
            log.info("Upserted batch of %d (total=%d)", len(batch), upserted)

        return IngestionReport(len(files), len(chunks), upserted, errors)


def _batched(items: list[Chunk], size: int) -> list[list[Chunk]]:
    return [items[i : i + size] for i in range(0, len(items), size)]
