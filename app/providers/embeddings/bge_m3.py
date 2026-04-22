from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

from app.providers.embeddings.base import EmbeddingProvider

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

log = logging.getLogger(__name__)


class BgeM3EmbeddingProvider(EmbeddingProvider):
    """Local sentence-transformers embeddings using BAAI/bge-m3 by default.

    The model is loaded lazily on first use so importing this module stays
    cheap (important for tests).
    """

    def __init__(self, model_name: str = "BAAI/bge-m3", device: str | None = None) -> None:
        self._model_name = model_name
        self._device = device
        self._model: SentenceTransformer | None = None

    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            log.info("Loading embedding model: %s", self._model_name)
            self._model = SentenceTransformer(self._model_name, device=self._device)
        return self._model

    @property
    def dimension(self) -> int:
        return self._get_model().get_sentence_embedding_dimension()

    def embed(self, texts: Sequence[str], *, normalize: bool = True) -> list[list[float]]:
        if not texts:
            return []
        model = self._get_model()
        vectors = model.encode(
            list(texts),
            normalize_embeddings=normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vectors.tolist()
