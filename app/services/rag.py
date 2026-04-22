from __future__ import annotations

import logging
import math
from collections.abc import Sequence

from app.domain import Answer, ChatTurn, Language, RetrievedChunk
from app.providers.embeddings import EmbeddingProvider
from app.providers.llm import LLMProvider, LLMRequest
from app.providers.vectorstore import VectorStore
from app.services.language import detect_language
from app.services.prompt import SYSTEM_PROMPT, build_user_message

log = logging.getLogger(__name__)


def _l2_norm(vec: Sequence[float]) -> float:
    return math.sqrt(sum(x * x for x in vec))


def _preview(text: str, n: int = 72) -> str:
    collapsed = " ".join(text.split())
    return collapsed if len(collapsed) <= n else collapsed[: n - 1] + "…"


def _log_retrieval(
    question: str,
    language: Language,
    top_k: int,
    history_turns: int,
    vector: Sequence[float],
    retrieved: Sequence[RetrievedChunk],
) -> None:
    """Emit a self-contained log block showing the vector-proximity ranking.

    Intent: make it easy to *read* what RAG actually retrieved and how close
    the neighbours are in embedding space. Qdrant's COSINE distance returns a
    similarity score in [-1, 1]; with normalised embeddings (which bge-m3
    produces) the practical range is [0, 1]. Higher score = closer in
    meaning. The Δ-from-top column shows how much each hit drops off from
    the best match — a large gap means retrieval was confident, clustered
    scores mean the query was ambiguous or under-covered by the corpus.
    """
    log.info("=" * 78)
    log.info(
        "RAG retrieval  lang=%s  top_k=%d  history_turns=%d",
        language, top_k, history_turns,
    )
    log.info("  question: %r", _preview(question, 120))
    log.info(
        "  query_vector: dim=%d  L2_norm=%.4f  "
        "(bge-m3 returns unit-norm vectors; cosine compares direction)",
        len(vector), _l2_norm(vector),
    )
    log.info(
        "  returned %d chunk(s). score = cosine similarity; higher = more similar.",
        len(retrieved),
    )
    if not retrieved:
        log.info("  (no hits — collection empty or language filter eliminated all)")
        log.info("=" * 78)
        return

    top_score = retrieved[0].score
    log.info(
        "  %-4s  %-7s  %-10s  %-24s  %-4s  preview",
        "rank", "score", "Δ-vs-top", "doc_id", "lang",
    )
    log.info("  %s", "-" * 74)
    for i, rc in enumerate(retrieved, start=1):
        log.info(
            "  %-4d  %7.4f  %+9.4f  %-24s  %-4s  %s",
            i,
            rc.score,
            rc.score - top_score,
            rc.chunk.doc_id[:24],
            rc.chunk.language,
            _preview(rc.chunk.text),
        )
    log.info("=" * 78)


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

        vector = self._embeddings.embed_one(question)
        retrieved = self._store.search(vector, top_k=self._top_k, language=lang)

        _log_retrieval(
            question=question,
            language=lang,
            top_k=self._top_k,
            history_turns=len(history),
            vector=vector,
            retrieved=retrieved,
        )

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
