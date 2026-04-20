from __future__ import annotations

import logging
from collections.abc import Sequence

from app.domain import Answer, ChatTurn, Language
from app.providers.embeddings import EmbeddingProvider
from app.providers.llm import LLMProvider, LLMRequest
from app.providers.vectorstore import VectorStore
from app.services.language import detect_language
from app.services.prompt import SYSTEM_PROMPT, build_user_message

log = logging.getLogger(__name__)


class RagService:
    def __init__(
        self,
        embeddings: EmbeddingProvider,
        store: VectorStore,
        llm: LLMProvider,
        *,
        top_k: int = 5,
        max_history_turns: int = 4,
    ) -> None:
        self._embeddings = embeddings
        self._store = store
        self._llm = llm
        self._top_k = top_k
        self._max_history_turns = max_history_turns

    def answer(
        self,
        question: str,
        history: Sequence[ChatTurn] = (),
        *,
        language: Language | None = None,
    ) -> Answer:
        lang = language or detect_language(question)
        log.info("RAG query lang=%s top_k=%d", lang, self._top_k)

        vector = self._embeddings.embed_one(question)
        retrieved = self._store.search(vector, top_k=self._top_k, language=lang)
        log.info("Retrieved %d chunks", len(retrieved))

        trimmed_history = list(history)[-self._max_history_turns * 2 :]
        user_message = build_user_message(question, lang, retrieved)

        reply = self._llm.complete(
            LLMRequest(
                system=SYSTEM_PROMPT,
                history=trimmed_history,
                user_message=user_message,
            )
        )

        sources = [rc.chunk.doc_id for rc in retrieved]
        return Answer(reply=reply, language=lang, sources=sources)
