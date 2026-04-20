from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence


class EmbeddingProvider(ABC):
    """Abstract dense-embedding provider.

    Implementations must return unit-normalised vectors when `normalize=True`
    is set (default). All vectors in a single provider instance must share
    the dimension reported by `dimension`.
    """

    @property
    @abstractmethod
    def dimension(self) -> int: ...

    @abstractmethod
    def embed(self, texts: Sequence[str], *, normalize: bool = True) -> list[list[float]]: ...

    def embed_one(self, text: str, *, normalize: bool = True) -> list[float]:
        return self.embed([text], normalize=normalize)[0]
