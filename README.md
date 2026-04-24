# Resort RAG Chatbot PoC — Syöte

Proof-of-concept RAG chatbot answering guest questions about a Finnish resort
and national park (Syöte, Pudasjärvi region) in Finnish and English.

- **LLM:** Claude Haiku 4.5 via the official `anthropic` SDK, or any local
  model served by Ollama (e.g. `Llama-Poro-2-8B-Instruct`) — selectable via
  `LLM_PROVIDER`
- **Embeddings:** `BAAI/bge-m3` via `sentence-transformers` (in-process)
- **Vector DB:** Qdrant (Docker)
- **API:** FastAPI

## Architecture

For a full walkthrough with sequence diagrams and a tour of every file's
responsibility, see **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**. The
short version:

```
app/
  api/           FastAPI routes, request/response schemas
  services/      Orchestration: RagService, language detection, prompt builder
  providers/
    llm/         LLMProvider (ABC) + ClaudeProvider + OllamaProvider
    embeddings/  EmbeddingProvider (ABC) + BgeM3EmbeddingProvider
    vectorstore/ VectorStore (ABC) + QdrantStore
  ingestion/     Markdown parser, chunker, ingestion pipeline, CLI
  domain/        Pure domain models (Chunk, Document, Query, Answer)
  config/        pydantic-settings loader + logging
  main.py        FastAPI app entrypoint
content/syote/   Markdown content with YAML frontmatter
tests/           pytest suite
```

Every external integration is behind an ABC; swapping `ClaudeProvider` for a
Poro/local model, or swapping `BgeM3EmbeddingProvider` for a remote embedding
server, only touches `app/providers/` and dependency wiring.

## Decisions log

Kept here so we remember why things are the way they are.

- **Embeddings:** `sentence-transformers` loading `BAAI/bge-m3` in-process,
  not a separate embedding server. Simpler PoC (one fewer container), same
  model quality. `EmbeddingProvider` lets us switch to a remote server later
  without touching `services/`.
- **LLM:** Two providers behind `LLMProvider`:
  - `ClaudeProvider` — Claude Haiku 4.5 via the official `anthropic` SDK.
    Requires API billing on <https://console.anthropic.com> (your claude.ai
    Pro/Max subscription does **not** grant API access).
  - `OllamaProvider` — any model served by a local Ollama, default
    `hf.co/mradermacher/Llama-Poro-2-8B-Instruct-GGUF:Q6_K` (a Finnish-tuned
    Llama-3 8B). No API key needed, runs fully offline. Lower answer
    quality than Haiku 4.5 but free and private.
  - Toggle with `LLM_PROVIDER=claude` or `LLM_PROVIDER=ollama` in `.env`.
- **Vector DB:** Qdrant in Docker. Collection is recreated when the
  configured dimension doesn't match the current collection.
- **Language detection:** tiny heuristic in `app/services/language.py` —
  presence of Finnish-specific characters (`ä`, `ö`) plus a short list of
  common Finnish function words. Good enough for a PoC; swap for
  `langdetect` if accuracy matters.
- **Chunking:** max 400 words per chunk, split at paragraph boundaries,
  falling back to sentence then word boundaries for pathologically long
  paragraphs.
- **Point IDs:** deterministic — `{doc_id}__{language}__{chunk_idx}`, hashed
  to a UUIDv5 for Qdrant. Re-ingestion is idempotent.
- **Deps:** `uv` preferred (pyproject.toml). Generate `requirements.txt`
  with `uv pip compile pyproject.toml -o requirements.txt` if needed.
- **Branch:** development on `claude/resort-rag-chatbot-JTZce` per task
  brief.

## Quick start

Prereqs: Docker + Docker Compose, Python 3.11+, `uv` installed.

```bash
# 1. Configure
cp .env.example .env
# edit .env and set ANTHROPIC_API_KEY=...

# 2. Install local deps (for running tests, the ingest CLI, or the app
#    without Docker)
make install

# 3. Start Qdrant + the app
make up

# 4. In another terminal, ingest the example content. This embeds all
#    markdown chunks and upserts them to Qdrant. Re-runs are idempotent.
make ingest
# or: docker compose exec app python -m app.ingestion.cli --path content/syote

# 5. Try the chat endpoint
curl -s localhost:8000/health
curl -s -X POST localhost:8000/chat \
  -H 'content-type: application/json' \
  -d '{"message":"How much is the husky safari?"}' | jq

curl -s -X POST localhost:8000/chat \
  -H 'content-type: application/json' \
  -d '{"message":"Paljonko husky safari maksaa?"}' | jq
```

## Switching to a local Ollama model

The default `LLM_PROVIDER=claude` hits the Anthropic API. To run everything
locally with Ollama (e.g. Llama-Poro-2 for Finnish):

1. Install and start Ollama on the host: <https://ollama.com>.
2. Pull the model once:
   ```bash
   ollama pull hf.co/mradermacher/Llama-Poro-2-8B-Instruct-GGUF:Q6_K
   # sanity check outside the app:
   ollama run hf.co/mradermacher/Llama-Poro-2-8B-Instruct-GGUF:Q6_K \
     "Kerro lyhyesti Suomen pääkaupungista."
   ```
3. In `.env`:
   ```env
   LLM_PROVIDER=ollama
   OLLAMA_MODEL=hf.co/mradermacher/Llama-Poro-2-8B-Instruct-GGUF:Q6_K
   # Only relevant in mode B (API on the host). In mode A, docker-compose
   # overrides this to http://host.docker.internal:11434 automatically.
   OLLAMA_URL=http://localhost:11434
   ```
4. Restart the app:
   ```bash
   docker compose up -d            # mode A
   # or: make run                  # mode B
   ```

`host.docker.internal` resolves on Docker Desktop (macOS / Windows) by
default, and on Linux via the `extra_hosts: host-gateway` entry already in
`docker-compose.yml`.

No `ANTHROPIC_API_KEY` needed in this mode. Response quality on Finnish
is noticeably below Haiku 4.5, but the whole thing is private and free.

## Endpoints

### `POST /chat`

Request:

```json
{
  "message": "How much is the husky safari?",
  "session_id": "optional-opaque-string",
  "history": [
    {"role": "user", "content": "Hi"},
    {"role": "assistant", "content": "Hei! How can I help?"}
  ],
  "language": null
}
```

`language` is optional; omit to auto-detect.

Response:

```json
{
  "reply": "The 2-hour Husky Safari costs 159 € per adult and 95 € per child (ages 5–11). ...",
  "sources": ["husky-safari-2h"],
  "language": "en"
}
```

### `GET /health`

Returns `{"status":"ok"}`.

## Ingestion

Content is plain markdown with YAML frontmatter plus `## English` and
`## Suomi` sections:

```markdown
---
id: husky-safari-2h
type: activity
price_eur_adult: 159
price_eur_child: 95
duration_minutes: 120
updated: 2026-01-15
---

## English

English content...

## Suomi

Suomenkielinen sisältö...
```

The pipeline:
1. Walks `content/syote/` recursively.
2. Parses frontmatter and splits the body on `## English` / `## Suomi`.
3. Chunks each language section on paragraph boundaries (max 400 words).
4. Embeds chunks with `bge-m3`.
5. Upserts to Qdrant with payload = frontmatter + `{language, text,
   source_file, chunk_id, doc_id}`.

Deterministic chunk IDs make re-ingestion idempotent.

## Testing

```bash
make test
```

24 unit tests cover the parser, chunker, language detection, a mocked
end-to-end RAG flow (fake embeddings, store, and LLM), and the Ollama
provider (mocked HTTP layer). No network access needed; CI-safe.

## Security / production notes

This is a PoC, not a production service. Before deploying outside localhost,
at minimum:

- Lock down CORS — currently `allow_origins=["*"]` in `app/main.py`.
- Put the API behind authentication.
- Replace the permissive example phone number in
  `content/syote/practical/check-in-out.md` with the real on-call number.
- Rate-limit `/chat` and cap request size.
- Set a non-zero `temperature` only deliberately; the default 0.2 is fine
  for grounded Q&A.
- Never commit `.env` — `.gitignore` already excludes it.

## Project layout reference

See [Architecture](#architecture) above. The commit history on this branch
is atomic (one logical change per commit, conventional messages), in case
you want to read it as a tutorial.
