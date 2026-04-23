from __future__ import annotations

import logging
import uuid
from collections.abc import Sequence

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from app.domain import Chunk, Language, RetrievedChunk
from app.providers.vectorstore.base import VectorStore

log = logging.getLogger(__name__)

# Stable namespace so chunk_id -> point UUID is deterministic across runs.
_ID_NAMESPACE = uuid.UUID("6f4b2d6e-9a1e-4f2b-9a3e-7d1f2c3b4a55")


def _point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(_ID_NAMESPACE, chunk_id))


class QdrantStore(VectorStore):
    def __init__(self, url: str, collection: str) -> None:
        self._client = QdrantClient(url=url)
        self._collection = collection

    def ensure_collection(self, dimension: int) -> None:
        existing = {c.name for c in self._client.get_collections().collections}
        if self._collection in existing:
            info = self._client.get_collection(self._collection)
            current_dim = info.config.params.vectors.size
            if current_dim == dimension:
                return
            log.warning(
                "Recreating collection %s: dim %s -> %s",
                self._collection, current_dim, dimension,
            )
            self._client.delete_collection(self._collection)

        self._client.create_collection(
            collection_name=self._collection,
            vectors_config=qmodels.VectorParams(
                size=dimension, distance=qmodels.Distance.COSINE
            ),
        )
        # Payload index on language for cheap filtered search.
        self._client.create_payload_index(
            collection_name=self._collection,
            field_name="language",
            field_schema=qmodels.PayloadSchemaType.KEYWORD,
        )

    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> int:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        if not chunks:
            return 0

        points = [
            qmodels.PointStruct(
                id=_point_id(chunk.chunk_id),
                vector=list(vec),
                payload={
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "language": chunk.language,
                    "text": chunk.text,
                    **chunk.metadata,
                },
            )
            for chunk, vec in zip(chunks, vectors, strict=True)
        ]
        self._client.upsert(collection_name=self._collection, points=points, wait=True)
        return len(points)

    def search(
        self,
        vector: Sequence[float],
        *,
        top_k: int,
        language: Language | None = None,
    ) -> list[RetrievedChunk]:
        query_filter: qmodels.Filter | None = None
        if language is not None:
            query_filter = qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="language", match=qmodels.MatchValue(value=language)
                    )
                ]
            )

        if log.isEnabledFor(logging.DEBUG):
            head = ", ".join(f"{x:.4f}" for x in list(vector)[:4])
            log.debug(
                "qdrant.search collection=%s top_k=%d filter_lang=%s "
                "query_vec_head=[%s, ...]",
                self._collection, top_k, language, head,
            )

        response = self._client.query_points(
            collection_name=self._collection,
            query=list(vector),
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
        )
        hits = response.points

        log.debug(
            "qdrant.search hits=%d raw_scores=[%s]",
            len(hits),
            ", ".join(f"{h.score:.4f}" for h in hits) or "—",
        )

        results: list[RetrievedChunk] = []
        for hit in hits:
            payload = dict(hit.payload or {})
            text = payload.pop("text", "")
            chunk_id = payload.pop("chunk_id", str(hit.id))
            doc_id = payload.pop("doc_id", "")
            lang = payload.pop("language", "en")
            results.append(
                RetrievedChunk(
                    chunk=Chunk(
                        chunk_id=chunk_id,
                        doc_id=doc_id,
                        language=lang,
                        text=text,
                        metadata=payload,
                    ),
                    score=float(hit.score),
                )
            )
        return results
