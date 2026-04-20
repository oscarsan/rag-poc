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
