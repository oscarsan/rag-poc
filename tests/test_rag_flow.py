import logging
from collections.abc import Sequence

from app.domain import Chunk, Language, RetrievedChunk
from app.providers.embeddings import EmbeddingProvider
from app.providers.llm import LLMProvider, LLMRequest
from app.providers.vectorstore import VectorStore
from app.services import RagService


class FakeEmbeddings(EmbeddingProvider):
    @property
    def dimension(self) -> int:
        return 3

    def embed(self, texts, *, normalize=True):
        return [[1.0, 0.0, 0.0] for _ in texts]


class FakeStore(VectorStore):
    def __init__(self, chunks: list[RetrievedChunk]):
        self.chunks = chunks
        self.last_language: Language | None = None
        self.last_top_k: int | None = None

    def ensure_collection(self, dimension: int) -> None:  # pragma: no cover
        pass

    def upsert(self, chunks, vectors) -> int:  # pragma: no cover
        return 0

    def search(self, vector, *, top_k, language=None):
        self.last_language = language
        self.last_top_k = top_k
        return self.chunks[:top_k]


class FakeLLM(LLMProvider):
    def __init__(self, reply: str = "ok"):
        self.reply = reply
        self.last_request: LLMRequest | None = None

    def complete(self, request: LLMRequest) -> str:
        self.last_request = request
        return self.reply


def _rc(doc_id: str, lang: Language, text: str, score: float = 0.9) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(
            chunk_id=f"{doc_id}__{lang}__0",
            doc_id=doc_id,
            language=lang,
            text=text,
            metadata={},
        ),
        score=score,
    )


def test_rag_answer_detects_finnish_and_filters_by_language():
    store = FakeStore(
        [
            _rc("husky-safari-2h", "fi", "Hinta: 159 € aikuiselta."),
            _rc("snowshoe-hike", "fi", "Hinta: 65 € aikuiselta."),
        ]
    )
    llm = FakeLLM(reply="159 € aikuiselta.")
    svc = RagService(FakeEmbeddings(), store, llm, top_k=2, max_history_turns=4)

    answer = svc.answer("Paljonko husky safari maksaa?")

    assert answer.language == "fi"
    assert store.last_language == "fi"
    assert store.last_top_k == 2
    assert "husky-safari-2h" in answer.sources
    assert llm.last_request is not None
    assert "Hinta: 159" in llm.last_request.user_message


def test_rag_answer_english_path():
    store = FakeStore([_rc("husky-safari-2h", "en", "Price: 159 € per adult.")])
    llm = FakeLLM(reply="159 € per adult.")
    svc = RagService(FakeEmbeddings(), store, llm)

    answer = svc.answer("How much is the husky safari?")

    assert answer.language == "en"
    assert store.last_language == "en"
    assert answer.sources == ["husky-safari-2h"]


def test_rag_emits_retrieval_log_with_scores_and_doc_ids(caplog):
    store = FakeStore(
        [
            _rc("husky-safari-2h", "fi", "Hinta: 159 € aikuiselta.", score=0.87),
            _rc("snowshoe-hike", "fi", "Hinta: 65 € aikuiselta.", score=0.61),
        ]
    )
    svc = RagService(FakeEmbeddings(), store, FakeLLM(), top_k=2)

    with caplog.at_level(logging.INFO, logger="app.services.rag"):
        svc.answer("Paljonko husky safari maksaa?")

    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "RAG retrieval" in text
    assert "lang=fi" in text
    assert "query_vector" in text
    # Scores and doc ids appear in the table.
    assert "0.8700" in text and "0.6100" in text
    assert "husky-safari-2h" in text
    # Δ-vs-top shows 0 for rank 1 and a negative delta for rank 2.
    assert "+0.0000" in text
    assert "-0.2600" in text


def test_rag_logs_full_llm_prompt_before_completion(caplog):
    store = FakeStore(
        [
            _rc("husky-safari-2h", "fi", "Hinta: 159 € aikuiselta.", score=0.87),
        ]
    )
    llm = FakeLLM(reply="159 € aikuiselta.")
    svc = RagService(FakeEmbeddings(), store, llm, top_k=1, max_history_turns=1)

    from app.domain import ChatTurn

    history = [ChatTurn(role="user", content="Hei")]

    with caplog.at_level(logging.INFO, logger="app.services.rag"):
        svc.answer("Paljonko husky safari maksaa?", history)

    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "LLM prompt" in text
    assert "You are a helpful assistant for a Finnish resort" in text
    assert "[1] user:\nHei" in text
    assert "Context chunks:" in text
    assert "source=husky-safari-2h" in text
    assert "Hinta: 159 € aikuiselta." in text
    assert "Käyttäjän kysymys (fi): Paljonko husky safari maksaa?" in text


def test_rag_trims_history_to_last_n_turns():
    store = FakeStore([_rc("doc", "en", "context")])
    llm = FakeLLM()
    svc = RagService(FakeEmbeddings(), store, llm, top_k=1, max_history_turns=2)

    from app.domain import ChatTurn

    history: Sequence[ChatTurn] = [
        ChatTurn(role="user", content=f"msg {i}") for i in range(10)
    ]
    svc.answer("anything", history)

    assert llm.last_request is not None
    assert len(llm.last_request.history) == 4  # 2 turns == 4 messages
