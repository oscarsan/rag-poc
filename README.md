# Resort RAG Chatbot PoC — Syöte

Proof-of-concept RAG chatbot answering guest questions about a Finnish resort
and national park (Syöte, Pudasjärvi region) in Finnish and English.

Status: **scaffolding** — see [commit log](#commit-log) for progress.

## Decisions log

Decisions made while building. Keep this updated so we remember why things are
the way they are.

- **Embeddings:** `sentence-transformers` loading `BAAI/bge-m3` in-process for
  the PoC, rather than a separate embedding server. Reason: simpler PoC (one
  fewer container), same model quality. The `EmbeddingProvider` abstraction
  means we can swap to a remote server later without touching services.
- **LLM:** Claude Haiku 4.5 via the official `anthropic` SDK, behind an
  `LLMProvider` abstract base class so a Poro/local model can be swapped in
  later.
- **Vector DB:** Qdrant in Docker. Collection is recreated if the configured
  dimension doesn't match.
- **Language detection:** simple character-set heuristic (presence of
  Finnish-specific characters + common word matching). Fine for a PoC; swap
  for `langdetect` or similar if accuracy matters.
- **Dep manager:** `uv` preferred (pyproject.toml). A `requirements.txt` can
  be generated with `uv pip compile` if needed.
- **Branch:** development happens on `claude/resort-rag-chatbot-JTZce`, not
  `main`, per task brief.

## Commit log

Filled in as we go.

1. chore: project scaffold
