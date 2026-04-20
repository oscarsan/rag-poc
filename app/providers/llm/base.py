from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from app.domain import ChatTurn


@dataclass(frozen=True)
class LLMRequest:
    system: str
    history: list[ChatTurn]
    user_message: str
    max_tokens: int = 1024
    temperature: float = 0.2


class LLMProvider(ABC):
    """Abstract provider for chat-completion style LLMs.

    Implementations must be stateless with respect to conversation: callers
    pass the full history in each request.
    """

    @abstractmethod
    def complete(self, request: LLMRequest) -> str:
        """Return the assistant's reply text."""
