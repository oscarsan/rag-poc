from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    session_id: str | None = None
    history: list[ChatMessage] = Field(default_factory=list)
    language: Literal["fi", "en"] | None = None


class ChatResponse(BaseModel):
    reply: str
    sources: list[str]
    language: Literal["fi", "en"]


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
