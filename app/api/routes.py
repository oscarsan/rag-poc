from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import get_rag_service
from app.api.schemas import ChatRequest, ChatResponse, HealthResponse
from app.domain import ChatTurn
from app.services import RagService

log = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse()


@router.post("/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    rag: RagService = Depends(get_rag_service),
) -> ChatResponse:
    try:
        history = [ChatTurn(role=m.role, content=m.content) for m in request.history]
        answer = rag.answer(request.message, history, language=request.language)
    except Exception as exc:  # noqa: BLE001
        log.exception("Chat request failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return ChatResponse(
        reply=answer.reply,
        sources=answer.sources,
        language=answer.language,
    )
