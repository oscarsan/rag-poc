from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Language = Literal["fi", "en"]


@dataclass(frozen=True)
class Document:
    """A single source document parsed from markdown (before chunking)."""

    doc_id: str
    source_file: str
    frontmatter: dict[str, Any]
    sections: dict[Language, str]


@dataclass(frozen=True)
class Chunk:
    """A chunk of text ready to be embedded and stored."""

    chunk_id: str
    doc_id: str
    language: Language
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievedChunk:
    """A chunk returned from a vector search with its similarity score."""

    chunk: Chunk
    score: float


@dataclass(frozen=True)
class ChatTurn:
    role: Literal["user", "assistant"]
    content: str


@dataclass(frozen=True)
class Query:
    text: str
    language: Language
    history: list[ChatTurn] = field(default_factory=list)


@dataclass(frozen=True)
class Answer:
    reply: str
    language: Language
    sources: list[str]
